import logging
import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, Query, Request, Body
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from agents.agent.factory.gen_agent import gen_agent
from agents.agent.memory.agent_context_manager import agent_context_manager
from agents.common.config import SETTINGS
from agents.common.context_scenarios import SUPPORTED_CONTEXT_SCENARIOS
from agents.common.error_messages import get_error_message
from agents.common.response import RestResponse
from agents.exceptions import CustomAgentException, ErrorCode
from agents.middleware.auth_middleware import get_current_user, get_optional_current_user
from agents.models.db import get_db
from agents.protocol.schemas import AgentDTO, DialogueRequest, AgentStatus, \
    PaginationParams, AgentMode, TelegramBotRequest, AgentSettingRequest, AgentContextStoreRequest, \
    PublishAgentToStoreRequest
from agents.services import agent_service, get_or_create_credentials

router = APIRouter()
logger = logging.getLogger(__name__)

defaults = {
    'id': uuid.uuid4().hex,
    'mode': AgentMode.REACT,  # Default to ReAct mode, but PROMPT is also valid
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
    'shouldInitializeDialog': False,
    'initializeDialogQuestion': None,
    'demo_video': None,
}


@router.post("/agents/create", summary="Create Agent", response_model=RestResponse[AgentDTO])
async def create_agent(
        agent: AgentDTO = Body(..., description="Agent configuration data"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Create a new agent

    Parameters:
    - **name**: Name of the agent
    - **description**: Description of the agent
    - **mode**: Mode of the agent (ReAct/Prompt/call)
    - **tools**: Optional list of tools to associate
    - **model_id**: Optional ID of the model to use
    - **suggested_questions**: Optional list of suggested questions
    - **shouldInitializeDialog**: Optional boolean to indicate whether to initialize dialog when creating the agent
    - **initializeDialogQuestion**: Optional string to specify the question to send when initializing dialog
    """
    try:
        logger.info(f"Creating agent with data: {agent.model_dump()}")
        
        # Set default values for missing fields
        for key, value in defaults.items():
            if getattr(agent, key) is None:
                setattr(agent, key, value)

        # Generate new UUID for the agent
        agent.id = str(uuid.uuid4())
        
        # Create agent
        result = await agent_service.create_agent(agent, user, session)
        logger.info(f"Agent created successfully with ID: {result.id}")
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error in agent creation: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in agent creation: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/agents/list", summary="List Personal Agents")
async def list_personal_agents(
        status: Optional[AgentStatus] = Query(None, description="Filter agents by status"),
        include_public: bool = Query(False, description="Include public agents along with personal agents"),
        category_id: Optional[int] = Query(None, description="Filter agents by category"),
        pagination: PaginationParams = Depends(),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Retrieve a list of user's personal agents with pagination.

    - **status**: Filter agents by their status (active, inactive, or draft)
    - **include_public**: Whether to include public agents along with personal agents
    - **category_id**: Optional filter for category ID
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (1-100)
    """
    try:
        # Calculate offset from page number
        offset = (pagination.page - 1) * pagination.page_size

        agents = await agent_service.list_personal_agents(
            status=status,
            skip=offset,
            limit=pagination.page_size,
            user=user,
            include_public=include_public,
            category_id=category_id,
            session=session
        )
        return RestResponse(data=agents)
    except CustomAgentException as e:
        logger.error(f"Error listing personal agents: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing personal agents: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/agents/public", summary="List Public Agents")
async def list_public_agents(
        status: Optional[AgentStatus] = Query(None, description="Filter agents by status"),
        only_official: bool = Query(False, description="Show only official agents"),
        only_hot: bool = Query(False, description="Show only hot agents"),
        category_id: Optional[int] = Query(None, description="Filter agents by category"),
        pagination: PaginationParams = Depends(),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Retrieve a list of public or official agents with pagination.

    - **status**: Filter agents by their status (active, inactive, or draft)
    - **only_official**: Whether to show only official agents
    - **only_hot**: Whether to show only hot agents
    - **category_id**: Optional filter for category ID
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (1-100)
    """
    try:
        # Calculate offset from page number
        offset = (pagination.page - 1) * pagination.page_size

        agents = await agent_service.list_public_agents(
            status=status,
            skip=offset,
            limit=pagination.page_size,
            only_official=only_official,
            only_hot=only_hot,
            category_id=category_id,
            user=user,
            session=session
        )
        return RestResponse(data=agents)
    except CustomAgentException as e:
        logger.error(f"Error listing public agents: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing public agents: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/agents/get", summary="Get Agent Details")
async def get_agent(
        agent_id: str = Query(None, description="agent id"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    try:
        agents = await agent_service.get_agent(agent_id, user, session=session)
        # Get API key for MCP URL if user is provided
        mcp_url = None
        if agents.enable_mcp and user and user.get("tenant_id"):
            try:
                async with session:
                    credentials = await get_or_create_credentials(user, session)
                    if credentials and credentials.get("token"):
                        mcp_url = f"{SETTINGS.API_BASE_URL}/mcp/assistant/{agents.id}?api-key={credentials['token']}"
            except Exception as e:
                logger.warning(f"Failed to generate MCP URL for agent {agents.id}: {e}")
        agents.mcp_url = mcp_url
        return RestResponse(data=agents)
    except CustomAgentException as e:
        logger.error(f"Error getting agent details: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting agent details: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agents/update", summary="Update Agent")
async def update_agent(
        agent: AgentDTO,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Update an existing agent.
    
    Parameters:
    - **id**: ID of the agent to update
    - **name**: Name of the agent
    - **description**: Description of the agent
    - **mode**: Mode of the agent (ReAct/Prompt/call)
    - **tools**: Optional list of tools to associate
    - **model_id**: Optional ID of the model to use
    - **suggested_questions**: Optional list of suggested questions
    - **shouldInitializeDialog**: Optional boolean to indicate whether to initialize dialog when creating the agent
    - **initializeDialogQuestion**: Optional string to specify the question to send when initializing dialog
    """
    try:
        agent = await agent_service.update_agent(
            agent,
            user,
            session=session
        )
        return RestResponse(data=agent)
    except CustomAgentException as e:
        logger.error(f"Error updating agent: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error updating agent: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


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
    try:
        await agent_service.delete_agent(agent_id, user, session)
        return RestResponse(data="ok")
    except CustomAgentException as e:
        logger.error(f"Error deleting agent: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error deleting agent: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/agents/ai/create", summary="AI Create Agent")
async def ai_create_agent(
        description: Optional[str] = Query(None, description="description"),
        session: AsyncSession = Depends(get_db)):
    """
    Create a new agent.
    """
    try:
        resp = gen_agent(description)
        return StreamingResponse(content=resp, media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in AI agent creation: {e}", exc_info=True)
        return RestResponse(
            code=ErrorCode.API_CALL_ERROR,
            msg=f"Failed to create AI"
        )

@router.post("/agents/{agent_id}/dialogue")
async def dialogue(
        agent_id: str,
        request: DialogueRequest,
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Handle a dialogue between a user and an agent.

    - **agent_id**: ID of the agent to interact with
    - **message**: Message from the user
    - **initFlag**: Flag to indicate if this is an initialization dialogue (optional, default: False)
    - **model_id**: Optional model ID to override the agent's default model
    """
    try:
        resp = agent_service.dialogue(agent_id, request, user, session)
        return StreamingResponse(content=resp, media_type="text/event-stream")
    except CustomAgentException as e:
        logger.error(f"Error in dialogue: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in dialogue: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/agents/{agent_id}/dialogue")
async def dialogue_get(
        request: Request,
        agent_id: str,
        query: str = Query(..., description="Query message from the user"),
        conversation_id: Optional[str] = Query(
            default=None,
            alias="conversationId",
            description="ID of the conversation"
        ),
        init_flag: bool = Query(
            default=False,
            alias="initFlag",
            description="Flag to indicate if this is an initialization dialogue"
        ),
        model_id: Optional[int] = Query(
            default=None,
            description="Optional model ID to override the agent's default model"
        ),
        user: Optional[dict] = Depends(get_optional_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Handle a dialogue between a user and an agent using GET method.

    - **agent_id**: ID of the agent to interact with
    - **query**: Query message from the user
    - **conversation_id**: ID of the conversation (optional, auto-generated if not provided)
    - **initFlag**: Flag to indicate if this is an initialization dialogue (optional, default: False)
    - **model_id**: Optional model ID to override the agent's default model
    """
    try:
        # Create a new DialogueRequest with default conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        dialogue_request = DialogueRequest(
            query=query,
            conversationId=conversation_id,
            initFlag=init_flag,
            model_id=model_id
        )
        
        resp = agent_service.dialogue(agent_id, dialogue_request, user, session)
        return StreamingResponse(content=resp, media_type="text/event-stream")
    except CustomAgentException as e:
        logger.error(f"Error in dialogue: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in dialogue: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agents/{agent_id}/publish", summary="Publish Agent")
async def publish_agent(
        agent_id: str,
        create_fee: float = Query(0.0, description="Fee for creating the agent (tips for creator)"),
        price: float = Query(0.0, description="Fee for using the agent"),
        enable_mcp: bool = Query(False, description="Whether to enable MCP for this agent"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Publish or unpublish an agent with pricing settings

    Parameters:
    - **agent_id**: ID of the agent to publish
    - **create_fee**: Fee for creating the agent (tips for creator)
    - **price**: Fee for using the agent
    - **enable_mcp**: Whether to enable MCP for this agent
    """
    try:
        await agent_service.publish_agent(agent_id, False, create_fee, price, enable_mcp, user, session)
        return RestResponse(data="ok")
    except CustomAgentException as e:
        logger.error(f"Error publishing agent: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error publishing agent: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agents/{agent_id}/telegram", summary="Register Telegram Bot")
async def register_telegram_bot(
        agent_id: str,
        bot_data: TelegramBotRequest,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Register an agent as a Telegram bot
    
    Parameters:
    - **agent_id**: ID of the agent to register
    - **bot_data**: JSON body containing:
      - **bot_name**: Name of the Telegram bot
      - **token**: Telegram bot token
    """
    try:
        result = await agent_service.register_telegram_bot(agent_id, bot_data.bot_name, bot_data.token, user, session)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error registering Telegram bot: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error registering Telegram bot: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agents/{agent_id}/setting", summary="Update Agent Settings")
async def update_agent_settings(
        agent_id: str,
        settings: AgentSettingRequest,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Update agent settings including token, symbol, photos, and telegram bot
    
    Parameters:
    - **agent_id**: ID of the agent to update
    - **settings**: JSON body containing:
      - **token**: (Optional) Token for the agent
      - **symbol**: (Optional) Symbol for the agent token
      - **photos**: (Optional) Photos for the agent
      - **telegram_bot_name**: (Optional) Name of the Telegram bot
      - **telegram_bot_token**: (Optional) Telegram bot token
    """
    try:
        result = await agent_service.update_agent_settings(
            agent_id, 
            settings.dict(exclude_unset=True), 
            user, 
            session
        )
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error in updating agent settings: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in updating agent settings: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agents/agent-context/store", summary="Store Agent Context Data")
async def store_agent_context(
        request: AgentContextStoreRequest,
        user: dict = Depends(get_current_user)
):
    """
    Store agent context data in Redis for later retrieval during conversations.
    
    Parameters:
    - **conversation_id**: ID of the conversation
    - **scenario**: Scenario identifier for the context data (currently only 'wallet_signature' is supported)
    - **data**: Context data to store
    - **ttl**: Time to live in seconds (default: 24 hours)
    - **metadata**: Additional metadata for the context data (optional)
    """
    try:
        # Validate scenario
        if request.scenario not in SUPPORTED_CONTEXT_SCENARIOS:
            return RestResponse(
                code=ErrorCode.INVALID_PARAMETERS,
                msg=f"Unsupported scenario: {request.scenario}. Supported scenarios: {', '.join(SUPPORTED_CONTEXT_SCENARIOS)}"
            )
        
        # Add user information to metadata
        metadata = request.metadata or {}
        metadata.update({
            "user_id": user.get("id"),
            "tenant_id": user.get("tenant_id"),
            "stored_by": user.get("username", "unknown")
        })

        success = agent_context_manager.store(
            conversation_id=request.conversation_id,
            scenario=request.scenario,
            data=request.data,
            ttl=request.ttl,
            **metadata
        )
        
        if success:
            return RestResponse(data="ok")
        else:
            return RestResponse(
                code=ErrorCode.API_CALL_ERROR,
                msg="Failed to store agent context data"
            )
    except Exception as e:
        logger.error(f"Error storing agent context data: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )

@router.post("/agents/cache/refresh", summary="Refresh Public Agents Cache")
async def refresh_public_agents_cache(
        user: dict = Depends(get_current_user)
):
    """
    Manually refresh the Redis cache for public agents by incrementing the cache version.
    
    This endpoint should be called after database updates that affect public agent listings.
    It requires administrator privileges.
    
    Instead of deleting existing cache entries, this approach simply increments a version number,
    causing new requests to use a different cache key, effectively invalidating the old cache.
    Old cache entries will expire naturally according to their TTL.
    
    Returns:
        Information about the cache version update
    """
    try:
        result = await agent_service.refresh_public_agents_cache()
        return RestResponse(data={
            "previous_version": result["previous_version"],
            "new_version": result["new_version"],
            "message": "Cache version updated successfully"
        })
    except CustomAgentException as e:
        logger.error(f"Error refreshing public agents cache: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error refreshing public agents cache: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )

@router.post("/agents/{agent_id}/publish-to-store", summary="Publish Agent to MCP Store")
async def publish_agent_to_store(
    agent_id: str,
    request: PublishAgentToStoreRequest,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Publish an agent to MCP Store
    
    Parameters:
    - **agent_id**: ID of the agent to publish
    - **icon**: (Optional) Icon URL for the agent
    - **tags**: (Optional) List of tags for the agent
    - **github_url**: (Optional) GitHub repository URL
    """
    try:
        store = await agent_service.publish_to_store(
            agent_id=agent_id,
            user=user,
            session=session,
            icon=request.icon,
            tags=request.tags,
            github_url=request.github_url
        )
        return RestResponse(data=store)
    except CustomAgentException as e:
        logger.error(f"Error publishing agent to store: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error publishing agent to store: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )
