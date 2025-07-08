import json
import logging
from typing import Optional, AsyncIterator, List, Callable, Any, Union

import yaml
from langchain_core.messages import BaseMessageChunk
from mcp.types import CallToolResult, TextContent

from agents.agent.entity.agent_entity import DeepAgentExecutorOutput
from agents.agent.entity.inner.finish import FinishOutput
from agents.agent.entity.inner.inner_output import Output
from agents.agent.entity.inner.node_data import NodeMessage
from agents.agent.entity.inner.think_output import ThinkOutput
from agents.agent.entity.inner.tool_output import ToolOutput
from agents.agent.executor.executor import AgentExecutor
from agents.agent.executor.sliding_window import SlidingWindow
from agents.agent.prompts.default_prompt import ANSWER_PROMPT, CLARIFY_PROMPT, TOOLS_PROMPT, SYTHES_PROMPT
from agents.agent.prompts.tool_prompts import tool_prompt
from agents.agent.sensitive.sensitive_data_processor import SensitiveDataProcessor
from agents.agent.tokenizer.tiktoken_tokenizer import TikToken
from agents.common.context_scenarios import sensitive_config_map
from agents.models.entity import ToolInfo, ChatContext, ToolType
from agents.utils import tools_parser
from agents.utils.common import dict_to_csv, concat_strings
from agents.utils.http_client import async_client
from agents.utils.parser import parse_and_execute_json, extract_md_code

logger = logging.getLogger(__name__)


class DeepAgentExecutor(AgentExecutor):

    def __init__(
            self,
            chat_context: ChatContext,
            name: str,
            user_name: Optional[str] = "User",
            llm: Optional[Any] = None,
            system_prompt: Optional[str] = "You are a helpful assistant.",
            tool_system_prompt: str = tool_prompt(),
            description: str = "",
            role_settings: str = "",
            api_tools: Optional[List[ToolInfo]] = None,
            local_tools: Optional[List[Callable]] = None,
            node_massage_enabled: Optional[bool] = False,
            output_type: str = "str",
            output_detail_enabled: Optional[bool] = False,
            max_loops: Optional[int] = 1,
            retry: Optional[int] = 3,
            stop_func: Optional[Callable[[str], bool]] = None,
            tokenizer: Optional[Any] = TikToken(),
            long_term_memory: Optional[Any] = None,
            *args,
            **kwargs,
    ):

        super().__init__(
            chat_context=chat_context,
            name=name,
            user_name=user_name,
            llm=llm,
            system_prompt=system_prompt,
            tool_system_prompt=tool_system_prompt,
            description=description,
            role_settings=role_settings,
            api_tools=api_tools,
            local_tools=local_tools,
            node_massage_enabled=node_massage_enabled,
            output_type=output_type,
            output_detail_enabled=output_detail_enabled,
            max_loops=max_loops,
            retry=retry,
            stop_func=stop_func,
            tokenizer=tokenizer,
            long_term_memory=long_term_memory,
            *args,
            **kwargs,
        )

        # Initialize sensitive data processor
        self.sensitive_data_processor = SensitiveDataProcessor(chat_context.conversation_id)

        self.agent_output = DeepAgentExecutorOutput(
            agent_id=self.agent_executor_id,
            agent_name=self.agent_name,
            task="",
            max_loops=self.max_loops,
            steps=self.short_memory.to_dict(),
            full_history=self.short_memory.get_str(),
            total_tokens=self.tokenizer.count_tokens(
                self.short_memory.get_str()
            ),
        )

        # if self.description:
        #     self.short_memory.add(role="agent description", content=self.description)
        if self.role_settings:
            self.short_memory.add(role="Agent Settings", content=self.role_settings)

        self.short_memory.add(role="", content=SYTHES_PROMPT)

        self._initialize_tools()
        self._initialize_answer()
        # self._initialize_clarify()
        self.should_stop = False


    def _initialize_tools(self) -> None:

        # self.short_memory.add(role="system", content=self.tool_system_prompt)
        if self.api_tools:
            logger.info(
                "Tools provided: Accessing api: %d tools. Ensure functions have documentation and type hints.",
                 len(self.api_tools),
            )

            self.short_memory.add(role="", content=TOOLS_PROMPT)

            self.function_map = {}
            if self.api_tools:
                api_schemas, function_schemas, mcp_schemas = \
                    tools_parser.convert_tool_into_openai_schema(self.api_tools)
                tools_context = ""
                if api_schemas:
                    tools_context += self._get_tools_context(
                        tool_type="API",
                        description="The following API tools are available:",
                        schemas=api_schemas,
                    )
                if function_schemas:
                    tools_context += self._get_tools_context(
                        tool_type="Function",
                        description="The following Function tools are available:",
                        schemas=function_schemas,
                    )
                    self.function_map = {tool.__name__: tool for tool in self.local_tools}
                if mcp_schemas:
                    tools_context += self._get_tools_context(
                        tool_type="MCP",
                        description="The following MCP tools are available:",
                        schemas=mcp_schemas,
                    )
                self.short_memory.add(role="Available Tools", content=tools_context)

    def _initialize_answer(self):
        self.short_memory.add(role="", content=ANSWER_PROMPT)

    def _initialize_clarify(self):
        self.short_memory.add(role="", content=CLARIFY_PROMPT)

    def _get_tools_context(self, tool_type, description, schemas):
        return f"\n**{tool_type}**\n{description}\n{json.dumps(schemas, ensure_ascii=False, indent=4)}\n"


    async def stream(
            self,
            task: Optional[str] = None,
            img: Optional[str] = None,
            *args,
            **kwargs,
    ) -> AsyncIterator[str]:
        try:
            async for data in self.send_node_message("task understanding"): yield data

            self.agent_output.task = task

            # Add task to memory
            temporary = self.init_temporary()
            task = f"{task}\n{temporary}" if temporary else task
            self.short_memory.add(role="Now Question", content=task)


            # Set the loop count
            loop_count = 0
            # Clear the short memory
            response = None
            all_responses = []

            # Query the long term memory first for the context
            if self.long_term_memory is not None:
                async for data in self.send_node_message("load past context"): yield data
                self.memory_query(task)

            while (
                    self.max_loops == "auto"
                    or loop_count < self.max_loops
            ):
                loop_count += 1

                # Task prompt
                task_prompt = (
                    self.short_memory.get_history_as_string()
                )

                # Parameters
                attempt = 0
                success = False
                self.should_stop = False
                while attempt < self.retry_attempts and not success:
                    try:
                        if self.long_term_memory is not None:
                            logger.info(
                                "Querying RAG database for context..."
                            )
                            self.memory_query(task_prompt)

                        # Generate response using LLM
                        response_args = (
                            (task_prompt, *args)
                            if img is None
                            else (task_prompt, img, *args)
                        )
                        response = ""
                        whole_data = ""
                        logger.info(
                            f"Generating response with LLM... :{response_args}"
                        )
                        async for data in self.llm_astream(
                                *response_args, **kwargs
                        ):
                            if isinstance(data, ThinkOutput):
                                # Pass ThinkOutput object directly
                                yield data
                            elif isinstance(data, str):
                                whole_data += data
                                response += data
                                
                                # Check stopping condition
                                if not self.should_stop:
                                    if self.stop_func is not None and self._check_stopping_condition(response):
                                        async for node_data in self.send_node_message("generate response"):
                                            yield node_data
                                        self.should_stop = True
                                        yield self._get_stopping_condition_last_message(response)
                                        response = ""
                                else:
                                    yield response
                                    response = ""
                            else:
                                logger.error(
                                    f"Unexpected response format: {type(data)}"
                                )
                                raise ValueError(
                                    f"Unexpected response format: {type(data)}"
                                )

                        response = whole_data
                        logger.info(f"Response generated successfully. {response}")

                        # Convert to a str if the response is not a str
                        response = self.llm_output_parser(response)

                        # Check if response is a dictionary and has 'choices' key
                        if isinstance(response, dict) and "choices" in response:
                            response = response["choices"][0]["message"]["content"]
                        elif isinstance(response, str):
                            # If response is already a string, use it as is
                            pass
                        else:
                            raise ValueError(
                                f"Unexpected response format: {type(response)}"
                            )

                        # Add the response to the memory
                        self.short_memory.add(
                            role=self.agent_name, content=response
                        )

                        # Check and execute tools
                        if not self.should_stop:
                            async for data in self.parse_and_execute_tools(response):
                                yield data

                        # Add to all responses
                        all_responses.append(response)

                        # # TODO: Implement reliability check

                        success = True  # Mark as successful to exit the retry loop

                    except Exception as e:

                        logger.error(
                            f"Attempt {attempt + 1}: Error generating"
                            f" response: {e}"
                        )
                        attempt += 1

                if not success:
                    logger.error(
                        "Failed to generate a valid response after"
                        " retry attempts."
                    )
                    break
                if self.should_stop:
                    logger.warning(
                        "Stopping due to user input or other external"
                        " interruption."
                    )
                    break

                if (
                        self.stop_func is not None
                        and self._check_stopping_condition(response)
                ):
                    break

            # Merge all responses
            all_responses = [
                response
                for response in all_responses
                if response is not None
            ]

            self.agent_output.steps = self.short_memory.to_dict()
            self.agent_output.full_history = (
                self.short_memory.get_str()
            )
            self.agent_output.total_tokens = (
                self.tokenizer.count_tokens(
                    self.short_memory.get_str()
                )
            )

            # Handle artifacts
            # More flexible output types
            if (
                    self.output_type == "string"
                    or self.output_type == "str"
            ):
                yield concat_strings(all_responses)
            elif self.output_type == "list":
                yield all_responses
            elif (
                    self.output_type == "json"
                    or self.return_step_meta is True
            ):
                yield self.agent_output.model_dump_json(indent=4)
            elif self.output_type == "csv":
                yield dict_to_csv(
                    self.agent_output.model_dump()
                )
            elif self.output_type == "dict":
                yield self.agent_output.model_dump()
            elif self.output_type == "yaml":
                yield yaml.safe_dump(
                    self.agent_output.model_dump(), sort_keys=False
                )
            else:
                raise ValueError(
                    f"Invalid output type: {self.output_type}"
                )

        except Exception as error:
            self._handle_run_error(error)

        except KeyboardInterrupt as error:
            self._handle_run_error(error)

    async def llm_astream(self, input: str, *args, **kwargs) -> AsyncIterator[Union[str, ThinkOutput]]:
        if not isinstance(input, str):
            raise TypeError("Input must be a string")

        if not input.strip():
            raise ValueError("Input cannot be empty")

        if self.llm is None:
            raise TypeError("LLM object cannot be None")

        try:
            # Create sliding window instance
            sliding_window = SlidingWindow(window_size=10)
            
            async for out in self.llm.astream(input, *args, **kwargs):
                content = out.content if isinstance(out, BaseMessageChunk) else out
                
                if not isinstance(content, str):
                    logger.error(f"Unexpected response format: {type(content)}")
                    raise ValueError(f"Unexpected response format: {type(content)}")
                
                # Process content character by character
                for char in content:
                    # Use sliding window to process each character
                    result = sliding_window.process_char(char)
                    if result is not None:
                        yield result
            
            # Process remaining buffer content
            normal_output, think_output = sliding_window.get_remaining()
            if normal_output:
                yield normal_output
            if think_output:
                yield think_output
                
        except AttributeError as e:
            logger.error(
                f"Error calling LLM: {e} You need a class with a run(input: str) method"
            )
            raise e

    async def send_node_message(self, message: str) -> AsyncIterator[NodeMessage]:
        """Send a node message to the agent."""
        if self.should_send_node:
            yield NodeMessage(message=message)

    def _handle_run_error(self, error: any):
        logger.info(
            f"Error detected running your agent {self.agent_name} \n Error {error} \n) "
        )
        raise error

    def _check_stopping_condition(self, response: str) -> bool:
        """Check if the stopping condition is met."""
        try:
            if self.stop_func:
                return self.stop_func(response)
            return False
        except Exception as error:
            logger.error(
                f"Error checking stopping condition: {error}"
            )
    def _get_stopping_condition_last_message(self, response: str) -> str:
        """Get the last message sent to the user before the stopping condition was met."""
        messages = ""
        try:
            for stop in self.stop_condition:
                if stop in response:
                    splits = response.split(stop)
                    return splits[-1] if len(splits) > 1 else ""
        except Exception as error:
            logger.error(
                f"Error _get_stopping_condition_last_message: {error}"
            )
        return messages

    def memory_query(self, task: str = None, *args, **kwargs) -> None:
        try:
            # Query the long term memory
            if self.long_term_memory is not None:
                logger.info(f"Querying RAG for: {task}")

                memory_retrieval = self.long_term_memory.query(
                    task, *args, **kwargs
                )

                memory_retrieval = (
                    f"Documents Available: {str(memory_retrieval)}"
                )

                self.short_memory.add(
                    role="Database",
                    content=memory_retrieval,
                )

                return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def llm_output_parser(self, response: Any) -> str:
        """Parse the output from the LLM"""
        try:
            if isinstance(response, dict):
                if "choices" in response:
                    return response["choices"][0]["message"][
                        "content"
                    ]
                else:
                    return json.dumps(
                        response
                    )  # Convert dict to string
            elif isinstance(response, str):
                return response
            else:
                return str(
                    response
                )  # Convert any other type to string
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            return str(
                response
            )  # Return string representation as fallback

    async def parse_and_execute_tools(self, json_string) -> AsyncIterator[Output]:
        try:
            json_string = extract_md_code(json_string)
            tool_data = json.loads(json_string)
            result = self.dict_to_tool(tool_data)

            if result is None:
                return

            tool_info, parameters = result
            async for data in self.send_node_message(f"call {tool_info.name}"): yield data

            if tool_info.type == ToolType.FUNCTION.value:
                async for output in self.call_function(tool_info, parameters):
                    yield output
            elif tool_info.type == ToolType.OPENAPI.value:
                async for output in self.call_api(tool_info, parameters):
                    yield output
            elif tool_info.type == ToolType.MCP.value:
                await self.call_mcp(tool_info, parameters)
            else:
                logger.error(f"Unknown tool type: {tool_info.type}")
        except Exception as error:
            logger.error(f"Error executing tool: {error}", exc_info=True)

    def dict_to_tool(self, input_dict: dict) -> Optional[tuple[ToolInfo, dict]]:
        """
        Convert input dictionary to tool information and parameters
        
        Args:
            input_dict: Dictionary containing tool call information
            
        Returns:
            Tuple of (ToolInfo, parameters) if successful, None if no matching tool found
            
        Example input:
        {
            "type": "api",
            "function": {
                "name": "example_tool",
                "parameters": {
                    "header": {},
                    "query": {},
                    "path": {},
                    "body": {}
                }
            }
        }
        """
        try:
            if not isinstance(input_dict, dict):
                logger.error("Input must be a dictionary")
                self._add_tool_error("Input must be a dictionary")
                return None
                
            function_data = input_dict.get("function")
            if not function_data:
                logger.error("No function data found")
                self._add_tool_error("No function data found")
                return None
                
            tool_name = function_data.get("name")
            if not tool_name:
                logger.error("No tool name specified")
                self._add_tool_error("No tool name specified")
                return None
                
            # Find matching tool in self.api_tool
            matching_tool = None
            for tool in self.api_tools:
                if tool.name == tool_name:
                    matching_tool = tool
                    break
                    
            if not matching_tool:
                logger.error(f"No matching tool found for name: {tool_name}")
                return None

            extracted_params = {}
            if matching_tool.type == ToolType.FUNCTION.value:
                extracted_params = function_data.get("parameters", {})
            elif matching_tool.type == ToolType.OPENAPI.value:
                # Extract parameters
                parameters = function_data.get("parameters", {})
                extracted_params = {
                    "header": parameters.get("header", {}),
                    "query": parameters.get("query", {}),
                    "path": parameters.get("path", {}),
                    "body": parameters.get("body", {})
                }

                # Validate required parameters based on tool definition
                tool_params = matching_tool.parameters
                for param_type in ["header", "query", "path"]:
                    required_params = [
                        p["name"] for p in tool_params.get(param_type, [])
                        if p.get("required", False)
                    ]
                    if not isinstance(extracted_params[param_type], dict):
                        extracted_params[param_type] = {}

                    provided_params = extracted_params[param_type].keys()
                    missing_params = set(required_params) - set(provided_params)

                    if missing_params:
                        logger.error(f"Missing required {param_type} parameters: {missing_params}")
                        self.short_memory.add(
                            role="Tool missing params",
                        )
                        return None
            elif matching_tool.type == ToolType.MCP.value:
                # For MCP tools, we just pass the parameters directly
                extracted_params = function_data.get("parameters", {})
            
            # Return the matched tool and extracted parameters
            return matching_tool, extracted_params
            
        except Exception as e:
            logger.error(f"Error parsing tool data: {str(e)}", exc_info=True)
            self._add_tool_error("Tool call format error")
            return None

    def _add_tool_error(self, err_message: str):
        data = """Example input:
        {
            "type": "api",
            "function": {
                "name": "example_tool",
                "parameters": {
                    "header": {},
                    "query": {},
                    "path": {},
                    "body": {}
                }
            }
        }
        
        # For MCP tools:
        {
            "type": "mcp",
            "function": {
                "name": "listing-coins",
                "parameters": {
                    "limit": 10,
                    "convert": "USD"
                }
            }
        }"""
        self.short_memory.add(
            role="Tool format error",
            content=f"{err_message}, {data}"
        )

    async def call_api(self, tool_info: ToolInfo, parameters: dict):
        """
        Call the API tool with the given parameters
        Args:
        tool_info: ToolInfo object representing the tool to call
        parameters: Dictionary of parameter values to pass to the tool
        Returns:
        The output from the tool execution
        """
        # Process parameters to recover sensitive data if needed
        if tool_info.sensitive_data_config:
            processed_parameters = self.sensitive_data_processor.process_tool_parameters(
                tool_info.name, 
                parameters, 
                tool_info.sensitive_data_config
            )
        else:
            processed_parameters = parameters
            
        resp = async_client.request(
            method=tool_info.method,
            base_url=tool_info.origin,
            path=tool_info.path,
            params=processed_parameters["query"],
            headers=processed_parameters["header"],
            json_data=processed_parameters["body"],
            auth_config=tool_info.auth_config,
            stream=tool_info.is_stream
        )
        if tool_info.is_stream:
            async for response in resp:
                yield ToolOutput(response)
            self.should_stop = True
        else:
            answer = ""
            try:
                async for response in resp:
                    answer = response
                logger.info(f"call api response:{answer}")
            except Exception as e:
                logger.error("call_api Exception", exc_info=True)
                self.short_memory.add(
                    role="Tool Call failed",
                    content=f" tool name: {tool_info.name}. API call failed. Please try again. Exception: {str(e)}"
                )
                return
                
            # Process response to mask sensitive data if needed
            if tool_info.sensitive_data_config and answer and isinstance(answer, dict):
                processed_answer = self.sensitive_data_processor.process_tool_response(
                    tool_info.name,
                    answer,
                    tool_info.sensitive_data_config
                )
            else:
                processed_answer = answer
                
            self.short_memory.add(
                role="Tool Executor",
                content=processed_answer if isinstance(processed_answer, str) else json.dumps(processed_answer, ensure_ascii=False)
            )

    async def call_function(self, tool_info: ToolInfo, parameters: dict):
        """
        Call a Python function with the given parameters
        Args:
        tool_info: ToolInfo object representing the function to call
        parameters: Dictionary of parameter values to pass to the function
        Returns:
        The output from the function execution
        """
        # Process parameters to recover sensitive data if needed
        if tool_info.sensitive_data_config:
            processed_parameters = self.sensitive_data_processor.process_tool_parameters(
                tool_info.name,
                {"params": parameters},
                tool_info.sensitive_data_config
            ).get("params", {})
        else:
            processed_parameters = parameters
            
        data = {
            "type": "function",
            "function": {
                "name": tool_info.name,
                "parameters": processed_parameters
            }
        }
        function = self.function_map[tool_info.name]
        try:
            text_content = ""
            async for response in (
                    parse_and_execute_json([function], json.dumps(data, ensure_ascii=False))):
                if isinstance(response, FinishOutput):
                    self.should_stop = True
                    continue

                # Process response to mask sensitive data if needed
                if tool_info.sensitive_data_config and not isinstance(response, FinishOutput):
                    processed_response = self.sensitive_data_processor.process_tool_response(
                        tool_info.name,
                        response.content if hasattr(response, 'content') else response,
                        tool_info.sensitive_data_config
                    )

                    # Update response content if it has content attribute
                    if hasattr(response, 'content'):
                        response.content = processed_response
                    else:
                        response = processed_response

                if isinstance(response, Output):
                    yield response
                elif isinstance(response, str):
                    text_content += response
                else:
                    text_content += json.dumps(response, ensure_ascii=False)

            if text_content:
                self.short_memory.add(
                    role="Call function",
                    content=text_content
                )
        except Exception as e:
            logger.error("call_function Exception", exc_info=True)
            self.short_memory.add(
                role="Call function Error",
                content=f"error info:{str(e)}"
            )

    def init_temporary(self) -> str:
        # Add temporary data to short-term memory if available
        if hasattr(self.chat_context, 'temp_data') and self.chat_context.temp_data:
            for scenario, data in self.chat_context.temp_data.items():
                # Format the data as a string or JSON depending on its type
                if isinstance(data, dict) or isinstance(data, list):
                    if scenario in sensitive_config_map:
                        parameters = self.sensitive_data_processor \
                            .process_tool_response("", data, sensitive_config_map[scenario])
                        content = json.dumps(parameters, ensure_ascii=False)
                    else:
                        content = json.dumps(data, ensure_ascii=False)
                else:
                    content = str(data)

                return f"Tool Result Data ({scenario}): {content}"
        return ""

    async def call_mcp(self, tool_info: ToolInfo, parameters: dict):
        """
        Call an MCP (Model-Calling Protocol) tool
        
        Args:
            tool_info: Tool information with MCP details
            parameters: Parameters to pass to the MCP tool
            
        Yields:
            Generated output from the MCP tool
        """
        try:
            import json
            
            origin = tool_info.origin
            path = tool_info.path
            tool_name = tool_info.name
            
            # Process parameters to recover sensitive data if needed
            if tool_info.sensitive_data_config:
                processed_parameters = self.sensitive_data_processor.process_tool_parameters(
                    tool_info.name, 
                    {"params": parameters},
                    tool_info.sensitive_data_config
                ).get("params", {})
            else:
                processed_parameters = parameters
            
            # Extract API key from auth_config if available
            api_key = None
            if tool_info.auth_config and tool_info.auth_config.get('key') == 'api_key':
                api_key = tool_info.auth_config.get('value')
            
            # Check if origin is a URL (endpoint) or a file path
            if origin.startswith(('http://', 'https://')):
                # Remote MCP endpoint - Use mirascope.mcp sse_client
                from mirascope.mcp import sse_client

                url = origin + path
                # Use streaming or non-streaming based on configuration
                async with sse_client(url) as client:
                    result: CallToolResult = await client._session.call_tool(tool_name, processed_parameters)
                    
                    # Check if the result indicates an error
                    is_error = result.isError if hasattr(result, "isError") else False

                    content = ""
                    for data in result.content:
                        if isinstance(data, TextContent):
                            content += data.text if isinstance(data.text, str) else json.dumps(data.text, ensure_ascii=False)
                        else:
                            is_error = True
                            content = "This data format is not supported for parsing."

                    logger.error(f"call_mcp:{content}")
                    if is_error:
                        self.short_memory.add(
                            role="Tool Executor",
                            content=f"Response is ERROR: {content}"
                        )
                    else:
                        self.short_memory.add(
                            role="Tool Executor",
                            content=content
                        )

            else:
                self.short_memory.add(
                    role="Tool Executor",
                    content="tool config error!"
                )
                
        except Exception as e:
            logger.error(f"Error executing MCP tool: {e}", exc_info=True)
            self.short_memory.add(
                role="Tool Executor",
                content=f"Error executing MCP tool: {str(e)}"
            )
