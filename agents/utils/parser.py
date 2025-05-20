import functools
import inspect
import logging
import re
from typing import Any, List, Callable, get_origin, Annotated, Dict, get_args, Type, Optional, Literal, ForwardRef, \
    Union, Tuple, Set

from docstring_parser import parse
from pydantic import BaseModel, Field, schema_of
from pydantic.json_schema import JsonSchemaValue

logger = logging.getLogger(__name__)


def extract_md_code_blocks(markdown_text: str):
    # Regex pattern to match code blocks and optional language specifiers
    pattern = r"```(\w+)?\n(.*?)```"

    # Find all matches (language and content)
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    # Parse results
    code_blocks = []
    for language, content in matches:
        language = (
            language.strip() if language else "plaintext"
        )  # Default to 'plaintext'
        code_blocks.append(
            {"language": language, "content": content.strip()}
        )

    return code_blocks


def extract_md_code(markdown_text: str, language: str = None):
    # Get all code blocks with detected languages
    code_blocks = extract_md_code_blocks(markdown_text)

    # If there are code blocks, extract their content first
    if code_blocks:
        # Filter by language if specified
        if language:
            code_blocks = [
                block["content"]
                for block in code_blocks
                if block["language"] == language
            ]
        else:
            code_blocks = [
                block["content"] for block in code_blocks
            ]  # Include all blocks
        # Return concatenated content
        return "\n\n".join(code_blocks) if code_blocks else ""
    else:
        # If there are no code blocks, try to extract the first valid JSON object using regex
        text = markdown_text.strip() if markdown_text else ""
        # Match the first content that starts with { and ends with }, supports multiline
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0)
        else:
            # If nothing is matched, return the original content
            return text


def func_to_str(function: dict[str, Any]) -> str:
    function_str = f"Function: {function['name']}\n"
    function_str += f"Description: {function['description']}\n"
    function_str += "Parameters:\n"

    for param, details in function["parameters"][
        "properties"
    ].items():
        function_str += f"  {param} ({details['type']}): {details.get('description', '')}\n"

    return function_str


def functions_to_str(functions: list[dict[str, Any]]) -> str:
    functions_str = ""
    for function in functions:
        functions_str += func_to_str(function) + "\n"

    return functions_str


def get_typed_annotation(
        annotation: Any, globalns: Dict[str, Any]
) -> Any:
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
    return annotation


def get_typed_signature(
        call: Callable[..., Any]
) -> inspect.Signature:
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(
                param.annotation, globalns
            ),
        )
        for param in signature.parameters.values()
    ]
    typed_signature = inspect.Signature(typed_params)
    return typed_signature


def get_typed_return_annotation(call: Callable[..., Any]) -> Any:
    signature = inspect.signature(call)
    annotation = signature.return_annotation

    if annotation is inspect.Signature.empty:
        return None

    globalns = getattr(call, "__globals__", {})
    return get_typed_annotation(annotation, globalns)


def get_param_annotations(
        typed_signature: inspect.Signature,
) -> Dict[str, Union[Annotated[Type[Any], str], Type[Any]]]:
    return {
        k: v.annotation
        for k, v in typed_signature.parameters.items()
        if v.annotation is not inspect.Signature.empty
    }


class Parameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, JsonSchemaValue]
    required: List[str]


class Function(BaseModel):
    description: Annotated[
        str, Field(description="Description of the function")
    ]
    name: Annotated[str, Field(description="Name of the function")]
    parameters: Annotated[
        Parameters, Field(description="Parameters of the function")
    ]


class ToolFunction(BaseModel):
    type: Literal["function"] = "function"
    function: Annotated[
        Function, Field(description="Function under tool")
    ]


def type2schema(t: Any) -> JsonSchemaValue:
    d = schema_of(t)
    if "title" in d:
        d.pop("title")
    if "description" in d:
        d.pop("description")

    return d


def get_parameter_json_schema(
        k: str, v: Any, default_values: Dict[str, Any]
) -> JsonSchemaValue:
    def type2description(
            k: str, v: Union[Annotated[Type[Any], str], Type[Any]]
    ) -> str:
        # handles Annotated
        if hasattr(v, "__metadata__"):
            retval = v.__metadata__[0]
            if isinstance(retval, str):
                return retval
            else:
                raise ValueError(
                    f"Invalid description {retval} for parameter {k}, should be a string."
                )
        else:
            return k

    schema = type2schema(v)
    if k in default_values:
        dv = default_values[k]
        schema["default"] = dv

    schema["description"] = type2description(k, v)

    return schema


def get_required_params(
        typed_signature: inspect.Signature,
) -> List[str]:
    return [
        k
        for k, v in typed_signature.parameters.items()
        if v.default == inspect.Signature.empty
    ]


def get_default_values(
        typed_signature: inspect.Signature,
) -> Dict[str, Any]:
    return {
        k: v.default
        for k, v in typed_signature.parameters.items()
        if v.default != inspect.Signature.empty
    }


def get_parameters(
        required: List[str],
        param_annotations: Dict[
            str, Union[Annotated[Type[Any], str], Type[Any]]
        ],
        default_values: Dict[str, Any],
) -> Parameters:
    return Parameters(
        properties={
            k: get_parameter_json_schema(k, v, default_values)
            for k, v in param_annotations.items()
            if v is not inspect.Signature.empty
        },
        required=required,
    )


def get_missing_annotations(
        typed_signature: inspect.Signature, required: List[str]
) -> Tuple[Set[str], Set[str]]:
    all_missing = {
        k
        for k, v in typed_signature.parameters.items()
        if v.annotation is inspect.Signature.empty
    }
    missing = all_missing.intersection(set(required))
    unannotated_with_default = all_missing.difference(missing)
    return missing, unannotated_with_default


def get_openai_function_schema_from_func(
        function: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: str = None,
) -> Dict[str, Any]:
    typed_signature = get_typed_signature(function)
    required = get_required_params(typed_signature)
    default_values = get_default_values(typed_signature)
    param_annotations = get_param_annotations(typed_signature)
    return_annotation = get_typed_return_annotation(function)
    missing, unannotated_with_default = get_missing_annotations(
        typed_signature, required
    )

    if return_annotation is None:
        logger.warning(
            f"The return type of the function '{function.__name__}' is not annotated. Although annotating it is "
            + "optional, the function should return either a string, a subclass of 'pydantic.BaseModel'."
        )

    if unannotated_with_default != set():
        unannotated_with_default_s = [
            f"'{k}'" for k in sorted(unannotated_with_default)
        ]
        logger.warning(
            f"The following parameters of the function '{function.__name__}' with default values are not annotated: "
            + f"{', '.join(unannotated_with_default_s)}."
        )

    if missing != set():
        missing_s = [f"'{k}'" for k in sorted(missing)]
        raise TypeError(
            f"All parameters of the function '{function.__name__}' without default values must be annotated. "
            + f"The annotations are missing for the following parameters: {', '.join(missing_s)}"
        )

    fname = name if name else function.__name__

    parameters = get_parameters(
        required, param_annotations, default_values=default_values
    )

    function = ToolFunction(
        function=Function(
            description=description,
            name=fname,
            parameters=parameters,
        )
    )

    return function.dict()


def get_load_param_if_needed_function(
        t: Any,
) -> Optional[Callable[[Dict[str, Any], Type[BaseModel]], BaseModel]]:
    if get_origin(t) is Annotated:
        return get_load_param_if_needed_function(get_args(t)[0])

    def load_base_model(
            v: Dict[str, Any], t: Type[BaseModel]
    ) -> BaseModel:
        return t(**v)

    return (
        load_base_model
        if isinstance(t, type) and issubclass(t, BaseModel)
        else None
    )


def load_basemodels_if_needed(
        func: Callable[..., Any]
) -> Callable[..., Any]:
    # get the type annotations of the parameters
    typed_signature = get_typed_signature(func)
    param_annotations = get_param_annotations(typed_signature)

    # get functions for loading BaseModels when needed based on the type annotations
    kwargs_mapping_with_nones = {
        k: get_load_param_if_needed_function(t)
        for k, t in param_annotations.items()
    }

    # remove the None values
    kwargs_mapping = {
        k: f
        for k, f in kwargs_mapping_with_nones.items()
        if f is not None
    }

    # a function that loads the parameters before calling the original function
    @functools.wraps(func)
    def _load_parameters_if_needed(*args: Any, **kwargs: Any) -> Any:
        # load the BaseModels if needed
        for k, f in kwargs_mapping.items():
            kwargs[k] = f(kwargs[k], param_annotations[k])

        # call the original function
        return func(*args, **kwargs)

    @functools.wraps(func)
    async def _a_load_parameters_if_needed(
            *args: Any, **kwargs: Any
    ) -> Any:
        # load the BaseModels if needed
        for k, f in kwargs_mapping.items():
            kwargs[k] = f(kwargs[k], param_annotations[k])

        # call the original function
        return await func(*args, **kwargs)

    if inspect.iscoroutinefunction(func):
        return _a_load_parameters_if_needed
    else:
        return _load_parameters_if_needed


def _remove_a_key(d: dict, remove_key: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key and "type" in d.keys():
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


def single_pydantic_to_openai_function(
        pydantic_type: type[BaseModel],
        output_str: bool = False,
) -> dict[str, Any]:
    schema = pydantic_type.model_json_schema()

    # Fetch the name of the class
    name = type(pydantic_type).__name__

    docstring = parse(pydantic_type.__doc__ or "")
    parameters = {
        k: v
        for k, v in schema.items()
        if k not in ("title", "description")
    }

    for param in docstring.params:
        if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
        ):
            if "description" not in parameters["properties"][name]:
                parameters["properties"][name][
                    "description"
                ] = description

    parameters["type"] = "object"

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            schema["description"] = (
                f"Correctly extracted `{name}` with all "
                f"the required parameters with correct types"
            )

    _remove_a_key(parameters, "title")
    _remove_a_key(parameters, "additionalProperties")

    if output_str:
        out = {
            "function_call": {
                "name": name,
            },
            "functions": [
                {
                    "name": name,
                    "description": schema["description"],
                    "parameters": parameters,
                },
            ],
        }
        return str(out)

    else:
        return {
            "function_call": {
                "name": name,
            },
            "functions": [
                {
                    "name": name,
                    "description": schema["description"],
                    "parameters": parameters,
                },
            ],
        }


def pydantic_to_function_call(
        pydantic_types: List[BaseModel] = None,
        output_str: bool = False,
) -> dict[str, Any]:
    functions: list[dict[str, Any]] = [
        single_pydantic_to_openai_function(pydantic_type, output_str)[
            "functions"
        ][0]
        for pydantic_type in pydantic_types
    ]

    return {
        "function_call": "auto",
        "functions": functions,
    }

import json
import asyncio
from typing import Any, Callable, List, Union, AsyncIterator

async def parse_and_execute_json(
    functions: List[Callable[..., Any]],
    json_string: str,
    parse_md: bool = False,
    verbose: bool = False,
    return_str: bool = False,
) -> AsyncIterator[Union[str, dict]]:
    """
    Asynchronously parses a JSON string, executes functions based on the JSON data,
    and yields results as an asynchronous iterator.

    Supports both synchronous and asynchronous functions.
    If a function returns an iterator, its items are yielded one by one (pass-through).
    If return_str is True, each yielded result is JSON serialized as a string.
    """
    # Ensure that functions and json_string are provided
    if not functions or not json_string:
        raise ValueError("Functions and JSON string are required")

    if parse_md:
        # Assume extract_md_code is defined elsewhere
        json_string = extract_md_code(json_string)

    try:
        # Build a mapping from function names to function objects
        function_dict = {func.__name__: func for func in functions}

        if verbose:
            logger.info(f"Available functions: {list(function_dict.keys())}")
            logger.info(f"Processing JSON: {json_string}")

        # Parse the JSON data
        data = json.loads(json_string)

        # Determine function list format:
        # Supports "functions", "function", or the entire object as a single function call.
        if "functions" in data:
            function_list = data["functions"]
        elif "function" in data:
            function_list = [data["function"]]
        else:
            function_list = [data]

        # Ensure function_list is a list and filter out any None values
        if isinstance(function_list, dict):
            function_list = [function_list]
        function_list = [f for f in function_list if f]

        if verbose:
            logger.info(f"Processing {len(function_list)} functions")

        # Iterate over each function specification and yield results
        for function_data in function_list:
            function_name = function_data.get("name")
            parameters = function_data.get("parameters", {})

            if not function_name:
                logger.warning("Function data missing name field")
                continue

            if verbose:
                logger.info(f"Executing {function_name} with params: {parameters}")

            if function_name not in function_dict:
                logger.warning(f"Function {function_name} not found")
                result = None
            else:
                func = function_dict[function_name]
                try:
                    # Check if the function is asynchronous; if so, await it
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**parameters)
                    else:
                        result = func(**parameters)
                        # If the result is a coroutine, await it
                        if asyncio.iscoroutine(result):
                            result = await result
                except Exception as e:
                    logger.error(f"Error executing {function_name}: {str(e)}")
                    result = f"Error: {str(e)}"

            # Yield results. If the result is an iterator, yield its items one by one.
            # For asynchronous iterators:
            if hasattr(result, '__aiter__'):
                async for item in result:
                    if return_str:
                        yield str(item)
                    else:
                        yield item
            # For synchronous iterators (but not str or bytes):
            elif hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                for item in result:
                    if return_str:
                        yield str(item)
                    else:
                        yield item
            else:
                if return_str:
                    yield str(result)
                else:
                    yield result

    except json.JSONDecodeError as e:
        error = f"Invalid JSON format: {str(e)}"
        logger.error(error)
        if return_str:
            yield json.dumps({"error": error})
        else:
            yield {"error": error}
    except Exception as e:
        error = f"Error parsing and executing JSON: {str(e)}"
        logger.error(error)
        if return_str:
            yield json.dumps({"error": error})
        else:
            yield {"error": error}