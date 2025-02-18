import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from agents.agent.factory.gen_agent import gen_agent
from agents.common.response import RestResponse
from agents.middleware.auth_middleware import get_current_user
from agents.models.db import get_db
from agents.protocol.schemas import AgentDTO, DialogueResponse, DialogueRequest, AgentStatus, \
    PaginationParams, AgentMode
from agents.services import agent_service

router = APIRouter()

defaults = {
    'id': uuid.uuid4().hex,
    'mode': AgentMode.REACT,
    'status': AgentStatus.ACTIVE,
    'max_loops': 3,
    'name': "",
    'description': "",
    'icon': "",
    'role_settings': "",
    'welcome_message': "",
    'twitter_link': "",
    'telegram_bot_id': "",
    'tool_prompt': "",
    'tools': [],
    'suggested_questions': [],
}


@router.post("/agents/create", summary="Create Agent", response_model=RestResponse[AgentDTO])
async def create_agent(
        agent: AgentDTO,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Create a new agent
    
    Parameters:
    - **name**: Name of the agent
    - **description**: Description of the agent
    - **mode**: Mode of the agent (ReAct or call)
    - **tools**: Optional list of tools to associate
    - **model_id**: Optional ID of the model to use
    - **suggested_questions**: Optional list of suggested questions
    """
    for key, value in defaults.items():
        if getattr(agent, key) is None:
            setattr(agent, key, value)

    agent.id = str(uuid.uuid4())
    agent = await agent_service.create_agent(agent, user, session)
    return RestResponse(data=agent)


@router.get("/agents/list", summary="List Personal Agents")
async def list_personal_agents(
        status: Optional[AgentStatus] = Query(None, description="Filter agents by status"),
        include_public: bool = Query(False, description="Include public agents along with personal agents"),
        pagination: PaginationParams = Depends(),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Retrieve a list of user's personal agents with pagination.
    
    - **status**: Filter agents by their status (active, inactive, or draft)
    - **include_public**: Whether to include public agents along with personal agents
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (1-100)
    """
    # Calculate offset from page number
    offset = (pagination.page - 1) * pagination.page_size
    
    agents = await agent_service.list_personal_agents(
        status=status,
        skip=offset,
        limit=pagination.page_size,
        user=user,
        include_public=include_public,
        session=session
    )
    return RestResponse(data=agents)


@router.get("/agents/public", summary="List Public Agents")
async def list_public_agents(
        status: Optional[AgentStatus] = Query(None, description="Filter agents by status"),
        only_official: bool = Query(False, description="Show only official agents"),
        pagination: PaginationParams = Depends(),
        session: AsyncSession = Depends(get_db)
):
    """
    Retrieve a list of public or official agents with pagination.
    
    - **status**: Filter agents by their status (active, inactive, or draft)
    - **only_official**: Whether to show only official agents
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (1-100)
    """
    # Calculate offset from page number
    offset = (pagination.page - 1) * pagination.page_size
    
    agents = await agent_service.list_public_agents(
        status=status,
        skip=offset,
        limit=pagination.page_size,
        only_official=only_official,
        session=session
    )
    return RestResponse(data=agents)


@router.get("/agents/get", summary="Get Agent Details")
async def get_agent(
        agent_id: str = Query(None, description="agent id"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    agents = await agent_service.get_agent(agent_id, user, session=session)
    return RestResponse(data=agents)


@router.post("/agents/update", summary="Update Agent")
async def update_agent(
        agent: AgentDTO,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Update an existing agent."""
    agent = await agent_service.update_agent(
        agent,
        user,
        session=session
    )
    return RestResponse(data=agent)


@router.delete("/agents/delete", summary="Delete Agent")
async def delete_agent(
        agent_id: str = Query(None, description="agent id"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Delete an agent by setting its is_deleted flag to True.
    
    - **agent_id**: ID of the agent to delete
    """
    await agent_service.delete_agent(agent_id, user, session)
    return RestResponse(data="ok")


@router.get("/agents/ai/create", summary="AI Create Agent")
async def ai_create_agent(
        description: Optional[str] = Query(None, description="description"),
        session: AsyncSession = Depends(get_db)):
    """
    Create a new agent.
    """
    resp = gen_agent(description)
    return StreamingResponse(content=resp, media_type="text/event-stream")


@router.post("/agents/{agent_id}/dialogue", response_model=DialogueResponse)
async def dialogue(
        agent_id: str,
        request: DialogueRequest,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Handle a dialogue between a user and an agent.
    
    - **agent_id**: ID of the agent to interact with
    - **user_id**: ID of the user
    - **message**: Message from the user
    """
    # Placeholder logic for generating a response
    resp = agent_service.dialogue(agent_id, request, user, session)
    return StreamingResponse(content=resp, media_type="text/event-stream")


@router.get("/agents/{agent_id}/dialogue", response_model=DialogueResponse)
async def dialogue_get(
        agent_id: str,
        query: Optional[str] = Query(None, description="Query message from the user"),
        conversation_id: Optional[str] = Query(
            None,
            alias="conversationId",
            description="ID of the conversation"
        ),
        session: AsyncSession = Depends(get_db)
):
    """
    Handle a dialogue between a user and an agent using GET method.

    - **agent_id**: ID of the agent to interact with
    - **query**: Query message from the user (optional)
    - **conversation_id**: ID of the conversation (optional, auto-generated if not provided)
    """
    request = DialogueRequest(query=query, conversation_id=conversation_id)
    resp = agent_service.dialogue(agent_id, request, session)
    return StreamingResponse(content=resp, media_type="text/event-stream")


@router.post("/agents/{agent_id}/publish", summary="Publish Agent")
async def publish_agent(
        agent_id: str,
        is_public: bool = Query(True, description="Set agent as public"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Publish or unpublish an agent
    """
    await agent_service.publish_agent(agent_id, is_public, user, session)
    return RestResponse(data="ok")
