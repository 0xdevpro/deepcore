import json
import logging
import uuid
from typing import Optional, AsyncIterator, List, Callable, Any

import yaml
from langchain_core.messages import BaseMessageChunk

from agents.agent.entity.agent_entity import DeepAgentExecutorOutput
from agents.agent.entity.inner.node_data import NodeMessage
from agents.agent.entity.inner.tool_output import ToolOutput
from agents.agent.memory.memory import MemoryObject
from agents.agent.memory.short_memory import ShortMemory
from agents.agent.prompts.default_prompt import ANSWER_PROMPT, CLARIFY_PROMPT, TOOLS_PROMPT, SYTHES_PROMPT
from agents.agent.prompts.tool_prompts import tool_prompt
from agents.agent.tokenizer.tiktoken_tokenizer import TikToken
from agents.agent.tools import BaseTool
from agents.agent.tools.tool_executor import async_execute
from agents.models.entity import ToolInfo
from agents.utils import tools_parser
from agents.utils.common import dict_to_csv, exists, concat_strings
from agents.utils.http_client import async_client
from agents.utils.parser import parse_and_execute_json, extract_md_code

logger = logging.getLogger(__name__)


def gen_agent_executor_id() -> str:
    return uuid.uuid4().hex


class DeepAgentExecutor(object):

    def __init__(
            self,
            name: str,
            user_name: Optional[str] = "User",
            llm: Optional[Any] = None,
            system_prompt: Optional[str] = "You are a helpful assistant.",
            tool_system_prompt: str = tool_prompt(),
            description: str = "",
            role_settings: str = "",
            api_tool: Optional[List[ToolInfo]] = None,
            tools: Optional[List[Callable]] = None,
            async_tools: Optional[List[Callable]] = None,
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
        self.agent_name = name
        self.llm = llm
        self.tool_system_prompt = tool_system_prompt
        self.user_name = user_name
        self.output_type = output_type
        self.return_step_meta = output_detail_enabled
        self.max_loops = max_loops
        self.retry_attempts = retry
        self.stop_func = stop_func
        self.tools = tools or []
        self.api_tool = api_tool or []
        self.async_tools = async_tools or []
        self.should_send_node = node_massage_enabled
        self.tokenizer = tokenizer
        self.long_term_memory = long_term_memory
        self.description = description
        self.role_settings = role_settings

        self.agent_executor_id = gen_agent_executor_id()

        self.short_memory = ShortMemory(
            system_prompt=system_prompt,
            user_name=user_name,
            *args,
            **kwargs,
        )

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

        if self.description:
            self.short_memory.add(role="agent description", content=self.description)
        if self.role_settings:
            self.short_memory.add(role="agent settings", content=self.role_settings)

        self.short_memory.add(role="", content=SYTHES_PROMPT)

        self._initialize_tools()
        self._initialize_answer()
        self._initialize_clarify()
        self.should_stop = False


    def _initialize_tools(self) -> None:
        function_tools = self.tools + self.async_tools

        if not exists(function_tools):
            return

        self.tool_struct = BaseTool(
            tools=function_tools,
            tool_system_prompt=self.tool_system_prompt,
        )

        # self.short_memory.add(role="system", content=self.tool_system_prompt)

        if function_tools or self.api_tool:
            logger.info(
                "Tools provided: Accessing fucntion: %d api: %d tools. Ensure functions have documentation and type hints.",
                len(function_tools), len(self.api_tool),
            )

            self.short_memory.add(role="", content=TOOLS_PROMPT)

            if self.api_tool:
                tool_dict = tools_parser.convert_tool_into_openai_schema(self.api_tool)
                self.short_memory.add(role="api tool", content=tool_dict)

            if function_tools:
                tool_dict = self.tool_struct.convert_tool_into_openai_schema()
                self.short_memory.add(role="function tool", content=tool_dict)
                self.function_map = {tool.__name__: tool for tool in function_tools}

    def _initialize_answer(self):
        self.short_memory.add(role="", content=ANSWER_PROMPT)

    def _initialize_clarify(self):
        self.short_memory.add(role="", content=CLARIFY_PROMPT)


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
            self.short_memory.add(role=self.user_name, content=task)

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
                            if isinstance(data, str):
                                whole_data += data
                                response += data
                            else:
                                logger.error(
                                    f"Unexpected response format: {type(data)}"
                                )
                                raise ValueError(
                                    f"Unexpected response format: {type(data)}"
                                )

                            if self.stop_func is not None and self._check_stopping_condition(response):
                                async for data in self.send_node_message("generate response"):
                                    yield data

                            if self.should_stop or (
                                    self.stop_func is not None
                                    and self._check_stopping_condition(response)
                            ):
                                self.should_stop = True
                                yield response
                                response = ""

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
                            async for data in self.parse_and_execute_api_tools(response):
                                yield ToolOutput(data)


                            # is_finished = False
                            # if self.async_tools is not None:
                            #     async for resp in self.execute_tools_astream(response, direct_output=True):
                            #         yield ToolOutput(resp)
                            #         is_finished = True
                            #     if is_finished:
                            #         self.should_stop = True
                            #         success = True
                            #         break
                            # try:
                            #     if self.tools is not None:
                            #         self.parse_and_execute_tools(response)
                            # except Exception as e:
                            #     logger.error(
                            #         f"Error executing tools: {e}"
                            #     )

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

    async def llm_astream(self, input: str, *args, **kwargs) -> AsyncIterator[str]:
        if not isinstance(input, str):
            raise TypeError("Input must be a string")

        if not input.strip():
            raise ValueError("Input cannot be empty")

        if self.llm is None:
            raise TypeError("LLM object cannot be None")

        try:
            async for out in self.llm.astream(input, *args, **kwargs):
                if isinstance(out, str):
                    yield out
                elif isinstance(out, BaseMessageChunk):
                    yield out.content
        except AttributeError as e:
            logger.error(
                f"Error calling LLM: {e} You need a class with a run(input: str) method"
            )
            raise e

    async def execute_tools_astream(self, response: str, *args, **kwargs):
        try:
            logger.info("Executing async tool...")
            direct_output = kwargs.get("direct_output", True)
            # try to Execute the tool and return a string
            whole_output = ""
            async for data in async_execute(
                    functions=self.async_tools,
                    json_string=response,
                    parse_md=True,
                    *args,
                    **kwargs,
            ):
                if direct_output:
                    yield data
                else:
                    whole_output += str(data)

            if direct_output:
                return

            # Add the output to the memory
            self.short_memory.add(
                role="Tool Executor",
                content=whole_output,
            )

        except Exception as error:
            logger.error(f"Error executing tool: {error}")
            raise error

    async def send_node_message(self, message: str) -> AsyncIterator[NodeMessage]:
        """Send a node message to the agent."""
        if self.should_send_node:
            yield NodeMessage(message=message)

    def add_memory_object(self, memory: MemoryObject):
        """Add a memory object to the agent's memory."""
        self.short_memory.add(
            role="History data",
            content=f"user: {memory.input}\n\nassistant: {memory.output}",
        )

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

    def parse_and_execute_tools(self, response: str, *args, **kwargs):
        try:
            logger.info("Executing tool...")

            # try to Execute the tool and return a string
            out = parse_and_execute_json(
                functions=self.tools,
                json_string=response,
                parse_md=True,
                *args,
                **kwargs,
            )

            out = str(out)

            logger.info(f"Tool Output: {out}")

            # Add the output to the memory
            self.short_memory.add(
                role="Tool Executor",
                content=out,
            )

        except Exception as error:
            logger.error(f"Error executing tool: {error}")
            raise error

    async def parse_and_execute_api_tools(self, json_string):
        try:
            json_string = extract_md_code(json_string)
            tool_data = json.loads(json_string)
            result = self.dict_to_tool(tool_data)

            if result is None:
                return

            tool_info, parameters = result
            resp = async_client.request(
                method=tool_info.method,
                base_url=tool_info.origin,
                path=tool_info.path,
                params=parameters["query"],
                headers=parameters["header"],
                json_data=parameters["body"],
                auth_config=tool_info.auth_config,
                stream=tool_info.is_stream
            )
            if tool_info.is_stream:
                async for response in resp:
                    yield response
                self.should_stop = True
            else:
                answer = ""
                async for response in resp:
                    answer = response
                self.short_memory.add(
                    role="Tool Executor",
                    content=answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
                )
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
                return None
                
            # Check if it's an API tool call
            if input_dict.get("type") != "api":
                logger.error("Only API type tools are supported")
                return None
                
            function_data = input_dict.get("function")
            if not function_data:
                logger.error("No function data found")
                return None
                
            tool_name = function_data.get("name")
            if not tool_name:
                logger.error("No tool name specified")
                return None
                
            # Find matching tool in self.api_tool
            matching_tool = None
            for tool in self.api_tool:
                if tool.name == tool_name:
                    matching_tool = tool
                    break
                    
            if not matching_tool:
                logger.error(f"No matching tool found for name: {tool_name}")
                return None
                
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
                provided_params = extracted_params[param_type].keys()
                missing_params = set(required_params) - set(provided_params)
                
                if missing_params:
                    logger.error(f"Missing required {param_type} parameters: {missing_params}")
                    return None
            
            # Return the matched tool and extracted parameters
            return matching_tool, extracted_params
            
        except Exception as e:
            logger.error(f"Error parsing tool data: {str(e)}")
            return None
