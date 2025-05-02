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
            max_history_length: int = 25000,
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
        """Add a memory object to the agent's memory, with context trimming based on max_history_length."""
        if not memory_list:
            return

        # Sort memory by time ascending (oldest first)
        memory_list = sorted(memory_list, key=lambda m: m.time)
        max_tokens = getattr(self, 'max_history_length', 25000)
        tokenizer = self.tokenizer

        # Helper to build history string and count tokens
        def build_history(memories):
            history = ''
            for index, memory in enumerate(memories):
                input_hint = ''
                if hasattr(memory, 'temp_data') and memory.temp_data and "wallet_signature" not in memory.temp_data:
                    input_hint = f'User Input Hint: {json.dumps(memory.temp_data, ensure_ascii=False)}\n'
                output_str = memory.get_output_to_string() if hasattr(memory, 'get_output_to_string') else (memory.output if memory.output else "...")
                history += (f'Question {index+1}, Time: {memory.time.strftime("%Y-%m-%d %H:%M:%S %Z")}\n'
                            f'User: {memory.input.strip()}\n'
                            f'{input_hint}'
                            f'Assistant: {output_str.strip() if output_str else "..."}\n\n')
            return history

        # Step 1: Try with all original outputs
        history = build_history(memory_list)
        total_tokens = tokenizer.count_tokens(history)

        # Step 2: If over limit, start trimming outputs (oldest first)
        if total_tokens > max_tokens:
            trimmed = False
            for memory in memory_list:
                if memory.output and memory.output != "...":
                    memory.output = "..."  # Trim output
                    history = build_history(memory_list)
                    total_tokens = tokenizer.count_tokens(history)
                    if total_tokens <= max_tokens:
                        trimmed = True
                        break
            # Step 3: If still over limit, start dropping oldest memories
            if not trimmed and total_tokens > max_tokens:
                while memory_list and total_tokens > max_tokens:
                    memory_list.pop(0)  # Remove oldest
                    history = build_history(memory_list)
                    total_tokens = tokenizer.count_tokens(history)

        # Step 4: Add to short memory
        self.short_memory.add(
            role="History Question\n",
            content=history,
        )