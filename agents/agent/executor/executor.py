import json
import uuid
from abc import ABC
from typing import Optional, AsyncIterator, Any, List, Callable

from agents.agent.memory.memory import MemoryObject
from agents.agent.memory.short_memory import ShortMemory
from agents.agent.prompts.tool_prompts import tool_prompt
from agents.agent.tokenizer.tiktoken_tokenizer import TikToken
from agents.models.entity import ToolInfo, ChatContext


def gen_agent_executor_id() -> str:
    return uuid.uuid4().hex

class AgentExecutor(ABC):

    def __init__(
            self,
            chat_context: ChatContext,
            name: str,
            user_name: Optional[str] = "User",
            llm: Optional[Any] = None,
            system_prompt: Optional[str] = None,
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
            stop_condition: Optional[str] = None,
            *args,
            **kwargs,
    ):
        self.chat_context = chat_context
        self.agent_name = name
        self.llm = llm
        self.tool_system_prompt = tool_system_prompt
        self.user_name = user_name
        self.output_type = output_type
        self.return_step_meta = output_detail_enabled
        self.max_loops = max_loops
        self.retry_attempts = retry
        self.stop_func = stop_func
        self.api_tools = api_tools or []
        self.local_tools = local_tools or []
        self.should_send_node = node_massage_enabled
        self.tokenizer = tokenizer
        self.long_term_memory = long_term_memory
        self.description = description
        self.role_settings = role_settings
        self.stop_condition = stop_condition or []

        self.agent_executor_id = gen_agent_executor_id()

        self.short_memory = ShortMemory(
            system_prompt=system_prompt,
            user_name=user_name,
            *args,
            **kwargs,
        )

    async def stream(
            self,
            task: Optional[str] = None,
            img: Optional[str] = None,
            *args,
            **kwargs,
    ) -> AsyncIterator[str]:
        pass


    def add_memory_object(self, memory_list: list[MemoryObject]):
        """Add a memory object to the agent's memory."""
        if not memory_list:
            return

        history = ''
        for index, memory in enumerate(memory_list):
            input_hint = ''
            if memory.temp_data and "wallet_signature" not in memory.temp_data:
                input_hint = f'User Input Hint: {json.dumps(memory.temp_data, ensure_ascii=False)}\n'

            history += (f'Question {index+1}, Time: {memory.time.strftime("%Y-%m-%d %H:%M:%S %Z")}\n'
                        f'User: {memory.input.strip()}\n'
                        f'{input_hint}'
                        f'Assistant: {memory.output.strip() if memory.output else "..."}\n\n')
        self.short_memory.add(
            role="History Question\n",
            content=history,
        )