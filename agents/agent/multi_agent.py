# MultiAgent implementation for handling multiple agents
# The specific implementation will be provided later

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, AsyncIterator, Optional

from agents.agent.chat_agent import ChatAgent
from agents.agent.entity.inner.node_data import NodeMessage
from agents.agent.tools.message_tool import send_message, send_markdown
from agents.models.entity import ChatContext, AgentInfo, ModelInfo
from agents.agent.llm.default_llm import openai
from agents.agent.memory.memory import MemoryObject
from agents.agent.memory.redis_memory import RedisMemoryStore
from agents.services import agent_service
from agents.services.model_service import get_model_with_key

logger = logging.getLogger(__name__)

class MultiAgent:
    """
    MultiAgent class for managing and collaborating with multiple agents.
    Supports LLM-based agent selection and multi-round agent collaboration with context-aware routing.
    """

    redis_memory: RedisMemoryStore = RedisMemoryStore()

    def __init__(self, agent_ids: List[str], user: Optional[dict], session, conversation_id: str, max_rounds: int = 1):
        """
        Initialize MultiAgent with a list of agent IDs, user info, db session, and conversation ID.
        """
        self.agent_ids = agent_ids
        self.user = user
        self.session = session
        self.conversation_id = conversation_id
        self.max_rounds = max_rounds
        self.agents: List[AgentInfo] = []
        self.global_memory: List[MemoryObject] = []  # Global memory for all agents

    async def load_agents(self):
        """
        Load agent information for all agent IDs.
        """
        self.agents = []
        for agent_id in self.agent_ids:
            agent_dto = await agent_service.get_agent(agent_id, self.user, self.session, is_full_config=True)
            agent_info = AgentInfo.from_dto(agent_dto)
            model_dto, api_key = await get_model_with_key(agent_dto.model.id, self.user, self.session)
            model_info = ModelInfo(**model_dto.model_dump())
            model_info.api_key = api_key
            agent_info.set_model(model_info)
            self.agents.append(agent_info)

    def get_recent_context(self, n: int = 5) -> str:
        """
        Get the recent n rounds of conversation context from RedisMemoryStore.
        Returns a string for LLM routing prompt.
        """
        memory_list = self.redis_memory.get_memory_by_conversation_id(self.conversation_id)
        if not memory_list:
            return ""
        # Only keep the last n rounds
        memory_list = memory_list[-n:]
        context = ""
        for m in memory_list:
            context += f"User: {m.input}\nAgent: {m.output}\n"
        return context

    async def llm_route(self, query: str, top_k: int = 1) -> List[AgentInfo]:
        """
        Use LLM to select the most suitable agents for the query, with context.
        If none of the agents are suitable, the model can output 'none'.
        """
        agent_infos = []
        for agent in self.agents:
            agent_infos.append(f"ID: {agent.id}\nName: {agent.name}\nDescription: {agent.description or ''}\n")
        agents_text = "\n".join(agent_infos)
        # Get recent context
        context_text = self.get_recent_context()
        context_prompt = f"\nConversation context:\n{context_text}\n" if context_text else ""
        prompt = (
            f"You are an agent router. Given a user query and a list of agents, "
            f"select the most relevant agent IDs (max {top_k}).\n"
            f"If none of the agents are suitable, output 'none'.\n"
            f"User Query: {query}\n"
            f"{context_prompt}"
            f"Agents:\n"
            f"{agents_text}\n"
            f"Please output a comma-separated list of agent IDs, or 'none' if no agent is suitable."
        )

        try:
            response = await openai.get_model().ainvoke(prompt)
            content = response.content.strip().lower()
            if content == 'none':
                return []
            agent_ids = [aid.strip() for aid in response.content.split(",") if aid.strip() in [a.id for a in self.agents]]
            selected = [agent for agent in self.agents if agent.id in agent_ids][:top_k]
            if selected:
                return selected
        except Exception:
            logger.error("Failed to route agents using LLM", exc_info=True)
        return self.agents[:top_k]

    async def select_agents(self, query: str, top_k: int = 3) -> List[AgentInfo]:
        """
        Select the most suitable agents for the query.
        Supports multiple strategies: keyword, tags, priority, and LLM-based (future).
        Args:
            query: User input query
            top_k: Maximum number of agents to select (default 3)
        Returns:
            List of selected AgentInfo objects
        """
        selected = await self.llm_route(query, top_k=top_k)
        return selected

    async def collab_stream(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the response from the most relevant agent selected by LLM.
        If the selected agent cannot answer (empty or error), fallback to a default LLM (e.g., OpenAI) for streaming response.
        Only the selected agent will respond; no aggregation of multiple agents' answers.
        Args:
            query (str): The user's input query.
        Yields:
            Dict[str, Any]: Streaming response dict with round, agent_id, and response text.
        """
        await self.load_agents()
        selected_agents = await self.select_agents(query, top_k=1)
        if not selected_agents:
            # No agent selected, use fallback model
            async for r in self._fallback_stream(query):
                yield r
            return

        agent = selected_agents[0]
        chat_context = ChatContext(
            conversation_id=self.conversation_id,
            user=self.user or {},
        )
        chat_agent = ChatAgent(agent, chat_context)
        answered = False
        yield NodeMessage(f"call {agent.name}").to_stream()
        try:
            async for out in chat_agent.stream(query, self.conversation_id):
                answered = True
                yield out
        except Exception as e:
            # If agent fails, fallback to default model
            logger.error(f"Agent {agent.id} failed, fallback to openai: {e}")
        if not answered:
            # If agent did not answer, use fallback model
            async for r in self._fallback_stream(query):
                yield r

    async def _fallback_stream(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream the response from a fallback LLM (e.g., OpenAI) when no agent can answer.
        Args:
            query (str): The user's input query.
        Yields:
            Dict[str, Any]: Streaming response dict with round, agent_id as 'fallback', and response text.
        """
        prompt = f"You are a helpful AI assistant. User Query: {query}"
        try:
            async for out in openai.get_model().astream(prompt):
                yield send_markdown(out.content)
        except Exception as e:
            logger.error(f"Fallback model failed: {e}")
            yield send_markdown("Sorry, I cannot answer your question right now.")