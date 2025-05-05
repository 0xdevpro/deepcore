import json
import logging
from typing import Optional, AsyncIterator, List, Dict, Any

from fastapi import Depends
from sqlalchemy import func
from sqlalchemy import update, delete, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from agents.agent.chat_agent import ChatAgent
from agents.agent.memory.agent_context_manager import agent_context_manager
from agents.agent.tools.message_tool import send_markdown, send_message
from agents.common.config import SETTINGS
from agents.common.encryption_utils import encryption_utils
from agents.common.json_encoder import UniversalEncoder, universal_decoder
from agents.common.redis_utils import redis_utils
from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db
from agents.models.entity import AgentInfo, ModelInfo, ChatContext
from agents.models.models import App, Tool, AgentTool
from agents.protocol.schemas import AgentStatus, DialogueRequest, AgentDTO, ToolInfo, CategoryDTO, ModelDTO
from agents.services import mcp_service
from agents.services.model_service import get_model_with_key
from agents.services.profiles_service import get_balance, SpendChangeRequest, spend_balance, record_agent_usage
from decimal import Decimal

logger = logging.getLogger(__name__)

# Define cache constants
CACHE_PREFIX = "public_agents"
CACHE_VERSION_KEY = f"{CACHE_PREFIX}_version"
CACHE_TTL = 600  # Cache TTL in seconds (10 minutes)


async def dialogue(
        agent_id: str,
        request: DialogueRequest,
        user: Optional[dict],
        session: AsyncSession = Depends(get_db)
) -> AsyncIterator[str]:
    # Get agent info
    agent = await get_agent(agent_id, user, session, True)
    agent_info = AgentInfo.from_dto(agent)
    
    # Set up model info if specified
    # If request contains model_id, use it instead of agent's default model
    model_id_to_use = request.model_id if hasattr(request, 'model_id') and request.model_id is not None else agent.model_id
    
    if model_id_to_use:
        model_dto, api_key = await get_model_with_key(model_id_to_use, user, session)
        model_info = ModelInfo(**model_dto.model_dump())
        model_info.api_key = api_key
        agent_info.set_model(model_info)
    
    if not agent:
        raise CustomAgentException(
            ErrorCode.RESOURCE_NOT_FOUND,
            "Agent not found or no permission"
        )
    
    if agent.is_paused:
        yield send_markdown(agent.pause_message)
        return

    # Balance check before agent usage
    if user and SETTINGS.AGENT_BALANCE_CHECK_ENABLED and getattr(agent, "price", None):
        try:
            balance = get_balance(user)
            price = Decimal(str(agent.price))
            if balance < price:
                yield send_message(
                    "insufficient_balance",
                    {"message": "Insufficient balance, please recharge before using this agent."}
                )
                return
        except Exception as e:
            logger.error(f"Balance check failed: {e}", exc_info=True)
            yield send_markdown("Balance check failed, please try again later.")
            return

    # Get the initialization flag
    chat_context = ChatContext(
        conversation_id=request.conversation_id,
        initFlag=request.initFlag if hasattr(request, 'initFlag') else False,
        user=user or {},
    )
    
    # Retrieve all context data for the conversation
    context_data = agent_context_manager.get(request.conversation_id)
    
    # If context data is found, add it to the chat context
    if context_data:
        chat_context.temp_data = context_data
            
    # Create appropriate agent based on mode
    resp = ChatAgent(agent_info, chat_context)
    
    # Stream the response
    last_response = None
    async for response in resp.stream(request.query, request.conversation_id):
        last_response = response
        yield response

    # Spend balance and log usage after agent usage
    if user and getattr(agent, "price", None):
        try:
            price = Decimal(str(agent.price))
            # Use spend_balance to deduct balance
            spend_balance(SpendChangeRequest(
                tenant_id=user["tenant_id"],
                amount=price,
                requests_count=1
            ))
            # Use record_agent_usage to update stats and log
            record_agent_usage(
                agent_id=agent_id,
                user=user,
                price=float(price),
                query=getattr(request, "query", None),
                response=last_response,
                agent_name=getattr(agent, "name", None)
            )
        except Exception as e:
            logger.error(f"Spend balance or log usage failed: {e}")


async def get_agent(id: str, user: Optional[dict], session: AsyncSession, is_full_config=False):
    """
    Get agent with its associated tools
    """
    # Build base query conditions
    conditions = [App.id == id]
    
    # If user is logged in, add tenant filter or public agent condition
    if user and user.get('tenant_id'):
        conditions.append(or_(
            App.tenant_id == user.get('tenant_id'),
            App.is_public == True
        ))
    else:
        # Non-logged-in users can only access public agents
        conditions.append(App.is_public == True)

    # Execute query
    result = await session.execute(
        select(App).where(and_(*conditions)).options(
            selectinload(App.tools),
            selectinload(App.model),
            selectinload(App.category)
        )
    )
    agent = result.scalar_one_or_none()
    
    if agent is None:
        raise CustomAgentException(
            ErrorCode.RESOURCE_NOT_FOUND,
            "Agent not found or no permission"
        )

    try:
        # Convert to DTO using helper function
        agent_dto = await _convert_to_agent_dto(agent, user, is_full_config)
        return agent_dto
    except Exception as e:
        logger.error(f"Error converting agent to DTO: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.INTERNAL_ERROR,
            "Error processing agent data"
        )


async def verify_tool_permissions(
        tool_ids: List[int],
        user: dict,
        session: AsyncSession
) -> List[Tool]:
    """
    Verify if user has permission to use the specified tools
    Raises CustomAgentException if any tool is not accessible
    """
    if not tool_ids:
        return []

    tools = await session.execute(
        select(Tool).where(
            and_(
                Tool.id.in_(tool_ids),
                or_(
                    Tool.tenant_id == user.get('tenant_id'),
                    Tool.is_public == True
                )
            )
        )
    )
    found_tools = tools.scalars().all()

    if len(found_tools) != len(tool_ids):
        inaccessible_tools = set(tool_ids) - {tool.id for tool in found_tools}
        raise CustomAgentException(
            ErrorCode.PERMISSION_DENIED,
            f"No permission to access tools: {inaccessible_tools}"
        )

    return found_tools


async def create_agent(
        agent: AgentDTO,
        user: dict,
        session: AsyncSession = Depends(get_db)):
    """
    Create a new agent with user context and tools
    """
    if not user.get('tenant_id'):
        raise CustomAgentException(
            ErrorCode.UNAUTHORIZED,
            "User must belong to a tenant to create agents"
        )

    try:
        async with session.begin():
            # Verify tool permissions if tools are specified
            tool_ids = []
            if agent.tools:
                tool_ids = agent.tools
                await verify_tool_permissions(tool_ids, user, session)

            # Extract specific fields for model_json
            model_json_data = {}
            if agent.shouldInitializeDialog is not None:
                model_json_data["shouldInitializeDialog"] = agent.shouldInitializeDialog
            if agent.initializeDialogQuestion is not None:
                model_json_data["initializeDialogQuestion"] = agent.initializeDialogQuestion

            new_agent = App(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                mode=agent.mode,
                icon=agent.icon,
                status=agent.status,
                role_settings=agent.role_settings,
                welcome_message=agent.welcome_message,
                twitter_link=agent.twitter_link,
                telegram_bot_id=agent.telegram_bot_id,
                tool_prompt=agent.tool_prompt,
                max_loops=agent.max_loops,
                suggested_questions=agent.suggested_questions,
                model_id=agent.model_id,
                category_id=agent.category_id,
                model_json=json.dumps(model_json_data) if model_json_data else None,
                tenant_id=user.get('tenant_id'),
                dev=user.get('wallet_address')  # Add developer wallet address
            )
            session.add(new_agent)
            await session.flush()

            # Create tool associations
            for tool_id in tool_ids:
                agent_tool = AgentTool(
                    agent_id=new_agent.id,
                    tool_id=tool_id,
                    tenant_id=user.get('tenant_id')
                )
                session.add(agent_tool)

            return agent
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create agent: {str(e)}"
        )


async def list_personal_agents(
        status: Optional[AgentStatus],
        skip: int,
        limit: int,
        session: AsyncSession,
        user: dict,
        include_public: bool = False,
        category_id: Optional[int] = None
):
    """
    List user's personal agents

    Args:
        status: Optional filter for agent status
        skip: Number of records to skip ((page - 1) * page_size)
        limit: Number of records per page (page_size)
        session: Database session
        user: Current user info
        include_public: Whether to include public agents along with personal agents
        category_id: Optional filter for category ID

    Returns:
        dict: {
            "items": list of agents,
            "total": total number of records,
            "page": current page number,
            "page_size": number of items per page,
            "total_pages": total number of pages
        }
    """
    if not user or not user.get('tenant_id'):
        return {
            "items": [],
            "total": 0,
            "page": 1,
            "page_size": limit,
            "total_pages": 0
        }

    conditions = [App.tenant_id == user.get('tenant_id')]

    if include_public:
        conditions.append(App.tenant_id == user.get('tenant_id') or App.is_public == True)

    if status:
        conditions.append(App.status == status)
        
    if category_id:
        conditions.append(App.category_id == category_id)

    return await _get_paginated_agents(conditions, skip, limit, user, session)


async def list_public_agents(
        status: Optional[AgentStatus],
        skip: int,
        limit: int,
        session: AsyncSession,
        only_official: bool = False,
        only_hot: bool = False,
        category_id: Optional[int] = None,
        user: Optional[dict] = None
):
    """
    List public or official agents with pagination, using Redis cache with version control for improved performance.

    Args:
        status: Optional filter for agent status
        skip: Number of records to skip ((page - 1) * page_size)
        limit: Number of records per page (page_size)
        session: Database session
        only_official: Whether to only show official agents
        only_hot: Whether to only show hot agents
        category_id: Optional filter for category ID
        user: Optional user information for token decryption

    Returns:
        dict: {
            "items": list of agents,
            "total": total number of records,
            "page": current page number,
            "page_size": number of items per page,
            "total_pages": total number of pages
        }
    """
    try:
        # Calculate current page from skip and limit
        page = (skip // limit) + 1
        
        # Get current cache version
        current_version = redis_utils.get_value(CACHE_VERSION_KEY) or "0"
        
        # Generate versioned cache key based on parameters
        base_cache_key = f"{CACHE_PREFIX}:{status or 'all'}:{only_official}:{only_hot}:{category_id or 'all'}:{page}:{limit}"
        versioned_cache_key = f"{base_cache_key}:v{current_version}"
        
        # Try to get from cache first
        cached_data = redis_utils.get_value(versioned_cache_key)
        if cached_data:
            logger.info("list_public_agents, use cached_data!")
            try:
                # Deserialize and return cached data using universal_decoder
                return json.loads(cached_data, object_hook=universal_decoder)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in cache for key: {versioned_cache_key}")
                # Continue with database query if cache deserialization fails
        
        # If cache miss or invalid, query from database
        conditions = []

        if only_official:
            conditions.append(App.is_official == True)
        elif only_hot:
            conditions.append(App.is_hot == True)
        else:
            conditions.append(App.is_public == True)

        if status:
            conditions.append(App.status == status)
            
        if category_id:
            conditions.append(App.category_id == category_id)

        # Get data from database
        result = await _get_paginated_agents(conditions, skip, limit, user, session)
        
        # Cache the result with version in the key using UniversalEncoder
        redis_utils.set_value(
            versioned_cache_key, 
            json.dumps(result, cls=UniversalEncoder),
            ex=CACHE_TTL
        )
        
        return result
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error listing public agents: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to list public agents: {str(e)}"
        )


async def _get_paginated_agents(conditions: list, skip: int, limit: int, user: Optional[dict], session: AsyncSession):
    """
    Helper function to get paginated agents with given conditions
    """
    # Calculate total count for pagination info
    count_query = select(func.count()).select_from(App).where(and_(*conditions))
    total_count = await session.execute(count_query)
    total_count = total_count.scalar()

    # Get paginated results with ordering
    query = (
        select(App)
        .options(
            selectinload(App.category),
            selectinload(App.tools),
            selectinload(App.model)
        )
        .where(and_(*conditions))
        .order_by(App.create_time.desc())
    )

    result = await session.execute(
        query.offset(skip).limit(limit)
    )
    agents = result.scalars().all()
    results = []

    for agent in agents:
        # Convert to DTO using helper function
        agent_dto = await _convert_to_agent_dto(agent, user)
        results.append(agent_dto)

    # Calculate current page from skip and limit
    current_page = (skip // limit) + 1

    return {
        "items": results,
        "total": total_count,
        "page": current_page,
        "page_size": limit,
        "total_pages": (total_count + limit - 1) // limit
    }


async def update_agent(
        agent: AgentDTO,
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    """
    Update an existing agent
    """
    try:
        # Original update logic
        async with session.begin():
            # Check if agent exists and belongs to the user's tenant
            result = await session.execute(
                select(App).where(
                    App.id == agent.id,
                    App.tenant_id == user.get('tenant_id')
                )
            )
            existing_agent = result.scalar_one_or_none()
            
            if not existing_agent:
                raise CustomAgentException(
                    ErrorCode.RESOURCE_NOT_FOUND,
                    "Agent not found or no permission to update"
                )

            # Verify tool permissions if tools are specified
            if agent.tools:
                await verify_tool_permissions(agent.tools, user, session)

            # Extract specific fields for model_json
            model_json_data = {}
            if agent.shouldInitializeDialog is not None:
                model_json_data["shouldInitializeDialog"] = agent.shouldInitializeDialog
            if agent.initializeDialogQuestion is not None:
                model_json_data["initializeDialogQuestion"] = agent.initializeDialogQuestion
            
            # If there was existing model_json data, preserve it and update only what's needed
            if existing_agent.model_json:
                try:
                    existing_model_json = json.loads(existing_agent.model_json)
                    # Update with new values, keeping any existing values not being updated
                    existing_model_json.update(model_json_data)
                    model_json_data = existing_model_json
                except (json.JSONDecodeError, TypeError):
                    # If existing model_json is invalid, just use the new data
                    pass

            # Update agent fields
            existing_agent.name = agent.name
            existing_agent.description = agent.description
            existing_agent.mode = agent.mode
            existing_agent.icon = agent.icon
            existing_agent.status = agent.status
            existing_agent.role_settings = agent.role_settings
            existing_agent.welcome_message = agent.welcome_message
            existing_agent.twitter_link = agent.twitter_link
            existing_agent.tool_prompt = agent.tool_prompt
            existing_agent.max_loops = agent.max_loops
            existing_agent.suggested_questions = agent.suggested_questions
            existing_agent.model_id = agent.model_id
            existing_agent.category_id = agent.category_id
            existing_agent.model_json = json.dumps(model_json_data) if model_json_data else None

            # Update tool associations
            if agent.tools:
                # Remove existing tool associations
                await session.execute(
                    delete(AgentTool).where(
                        AgentTool.agent_id == agent.id,
                        AgentTool.tenant_id == user.get('tenant_id')
                    )
                )
                
                # Create new tool associations
                for tool_id in agent.tools:
                    agent_tool = AgentTool(
                        agent_id=agent.id,
                        tool_id=tool_id,
                        tenant_id=user.get('tenant_id')
                    )
                    session.add(agent_tool)

        # Refresh cache if the agent is public, official, or hot
        if existing_agent.is_public or existing_agent.is_official or existing_agent.is_hot:
            await refresh_public_agents_cache()

        return agent
    except CustomAgentException:
        logger.error(f"Error updating agent: ", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error updating agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to update agent: {str(e)}"
        )


async def delete_agent(
        agent_id: str,
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    try:
        async with session.begin():
            # Check if agent is public, official, or hot before deleting
            result = await session.execute(
                select(App).where(
                    App.id == agent_id,
                    App.tenant_id == user.get('tenant_id')
                )
            )
            agent = result.scalar_one_or_none()
            
            if not agent:
                raise CustomAgentException(
                    ErrorCode.RESOURCE_NOT_FOUND,
                    "Agent not found or no permission to delete"
                )
                
            is_cached = agent.is_public or agent.is_official or agent.is_hot
            
            # Delete agent
            await session.execute(
                delete(App).where(
                    App.id == agent_id,
                    App.tenant_id == user.get('tenant_id')
                )
            )
            
        # Refresh cache if the agent was public, official, or hot
        if is_cached:
            await refresh_public_agents_cache()
            
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to delete agent: {str(e)}"
        )


async def publish_agent(
        agent_id: str,
        is_public: bool,
        create_fee: float,
        price: float,
        enable_mcp: bool = False,
        user: dict = None,
        session: AsyncSession = None):
    """
    Publish or unpublish an agent

    Args:
        agent_id: ID of the agent
        is_public: Whether to make the agent public
        create_fee: Fee for creating the agent (tips for creator)
        price: Fee for using the agent
        enable_mcp: Whether to enable MCP for this agent
        user: Current user info
        session: Database session
    """
    try:
        async with session.begin():
            # First check if the agent exists and belongs to the user's tenant
            result = await session.execute(
                select(App).where(
                    App.id == agent_id,
                    App.tenant_id == user.get('tenant_id')
                )
            )
            agent = result.scalar_one_or_none()
            if not agent:
                raise CustomAgentException(
                    ErrorCode.RESOURCE_NOT_FOUND,
                    "Agent not found or no permission"
                )

            # Check if there's a change in public status
            needs_cache_refresh = agent.is_public != is_public

            # Update publish status and fees
            stmt = update(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            ).values(
                is_public=is_public,
                create_fee=create_fee,
                price=price,
                enable_mcp=enable_mcp
            )
            await session.execute(stmt)
            
        # Refresh cache if the public status changed
        if needs_cache_refresh:
            await refresh_public_agents_cache()
            
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error publishing agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to publish agent: {str(e)}"
        )

# Encryption function
def encrypt_token(token: str) -> str:
    """Encrypt Telegram bot token"""
    return encryption_utils.encrypt(token)

# Decryption function
def decrypt_token(encrypted_token: str) -> str:
    """Decrypt Telegram bot token"""
    return encryption_utils.decrypt(encrypted_token)

# Masking function, used to hide the middle part of the token
def mask_token(token: str) -> str:
    """Mask the middle part of the token with asterisks"""
    return encryption_utils.mask_token(token)

async def register_telegram_bot(
        agent_id: str,
        bot_name: str,
        token: str,
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    """
    Register an agent as a Telegram bot
    
    Args:
        agent_id: Agent ID
        bot_name: Telegram bot name
        token: Telegram bot token
        user: Current user information
        session: Database session
    """
    try:
        # Verify if agent exists and belongs to current user
        result = await session.execute(
            select(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            )
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Agent not found or no permission"
            )
        
        # Encrypt token
        encrypted_token = encrypt_token(token)
        
        # Update agent table with telegram bot info
        stmt = update(App).where(
            App.id == agent_id,
            App.tenant_id == user.get('tenant_id')
        ).values(
            telegram_bot_name=bot_name,
            telegram_bot_token=encrypted_token
        )
        await session.execute(stmt)
        await session.commit()
        
        # Update bot information in Redis
        await update_telegram_bots_redis(user, session)
        
        # Return masked token
        masked_token = mask_token(token)
        
        return {
            "agent_id": agent_id,
            "bot_name": bot_name,
            "token": masked_token
        }
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error registering Telegram bot: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to register Telegram bot: {str(e)}"
        )

async def update_telegram_bots_redis(user: dict,session: AsyncSession):
    """
    Update Telegram bot information in Redis
    """
    try:
        import json
        from agents.services.open_service import get_or_create_credentials
        
        # Query all agents with Telegram bot token
        result = await session.execute(
            select(App).where(
                App.telegram_bot_token.is_not(None)
            )
        )
        agents = result.scalars().all()
        
        bots_info = []
        for agent in agents:
            # Decrypt token
            token = decrypt_token(agent.telegram_bot_token)
            if not token:
                continue
                
            # Get open platform credentials for the agent's tenant
            try:
                credentials = await get_or_create_credentials(user, session)
                access_key = credentials.get("access_key")
                secret_key = credentials.get("secret_key")
                
                # Build agent URL
                agent_url = f"{SETTINGS.API_BASE_URL}/api/agents/{agent.id}/dialogue"
                
                bot_info = {
                    "token": token,
                    "access_key": access_key,
                    "secret_key": secret_key,
                    "agent_url": agent_url,
                    "agent_name": agent.name,
                    "agent_description": agent.description or ""
                }
                bots_info.append(bot_info)
            except Exception as e:
                logger.error(f"Error getting credentials for agent {agent.id}: {e}")
                continue
        
        # Store bot information in Redis
        if bots_info:
            redis_key = f"{SETTINGS.TELEGRAM_REDIS_KEY}_{SETTINGS.REDIS_PREFIX}"
            redis_utils.set_value(redis_key, json.dumps(bots_info))
            logger.info(f"Updated {len(bots_info)} Telegram bots in Redis")
        
    except Exception as e:
        logger.error(f"Error updating Telegram bots in Redis: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to update Telegram bots in Redis: {str(e)}"
        )

async def update_agent_settings(
        agent_id: str,
        settings: dict,
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    """
    Update agent settings including token, symbol, photos, and telegram bot
    
    Args:
        agent_id: Agent ID
        settings: Settings to update
        user: Current user information
        session: Database session
    """
    try:
        # Verify if agent exists and belongs to current user
        result = await session.execute(
            select(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            )
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Agent not found or no permission"
            )
        
        # Prepare update values
        update_values = {}
        
        # Update token, symbol, photos if provided
        if 'token' in settings and settings['token'] is not None:
            update_values['token'] = settings['token']
        
        if 'symbol' in settings and settings['symbol'] is not None:
            update_values['symbol'] = settings['symbol']
        
        if 'photos' in settings and settings['photos'] is not None:
            update_values['photos'] = settings['photos']
        
        # Handle Telegram bot registration if provided
        telegram_bot_name = settings.get('telegram_bot_name')
        telegram_bot_token = settings.get('telegram_bot_token')
        
        if telegram_bot_name and telegram_bot_token:
            # Encrypt token
            encrypted_token = encrypt_token(telegram_bot_token)
            update_values['telegram_bot_name'] = telegram_bot_name
            update_values['telegram_bot_token'] = encrypted_token
        
        # Update agent
        if update_values:
            stmt = update(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            ).values(**update_values)
            await session.execute(stmt)
            await session.commit()
            
            # Update bot information in Redis if telegram bot info was updated
            if telegram_bot_name and telegram_bot_token:
                await update_telegram_bots_redis(user, session)
        
        # Get updated agent
        result = await session.execute(
            select(App).where(
                App.id == agent_id
            ).options(
                selectinload(App.tools),
                selectinload(App.model),
                selectinload(App.category)
            )
        )
        updated_agent = result.scalar_one_or_none()
        
        # Convert to DTO
        agent_dto = await _convert_to_agent_dto(updated_agent, user)
        
        return agent_dto
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent settings: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to update agent settings: {str(e)}"
        )

async def _convert_to_agent_dto(agent: App, user: Optional[dict], is_full_config=False) -> AgentDTO:
    """
    Convert App model to AgentDTO
    
    Args:
        agent: App model instance
        user: Optional user information
        is_full_config: Whether to include full configuration
        
    Returns:
        AgentDTO: Converted DTO
    """
    # Parse model_json if exists
    shouldInitializeDialog = False
    initializeDialogQuestion = None
    is_paused = False
    pause_message = ""
    if agent.model_json:
        try:
            model_json_data = json.loads(agent.model_json)
            shouldInitializeDialog = model_json_data.get("shouldInitializeDialog", False)
            initializeDialogQuestion = model_json_data.get("initializeDialogQuestion")
            is_paused = model_json_data.get("isPaused", False)
            pause_message = model_json_data.get("pauseMessage", "")
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse model_json for agent {agent.id}")
    
    # Process telegram bot token if exists
    masked_token = None
    if agent.telegram_bot_token:
        masked_token = mask_token(decrypt_token(agent.telegram_bot_token))
    
    # Convert to DTO
    agent_dto = AgentDTO(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        mode=agent.mode,
        icon=agent.icon,
        status=agent.status,
        role_settings=agent.role_settings,
        welcome_message=agent.welcome_message,
        twitter_link=agent.twitter_link,
        telegram_bot_id=agent.telegram_bot_id,
        telegram_bot_name=agent.telegram_bot_name,
        token=agent.token,
        symbol=agent.symbol,
        photos=agent.photos,
        demo_video=agent.demo_video,
        tool_prompt=agent.tool_prompt,
        max_loops=agent.max_loops,
        custom_config=agent.custom_config,
        suggested_questions=agent.suggested_questions,
        model_id=agent.model_id,
        category_id=agent.category_id,
        is_public=agent.is_public,
        is_official=agent.is_official,
        is_hot=agent.is_hot,
        create_time=agent.create_time,
        update_time=agent.update_time,
        create_fee=float(agent.create_fee) if agent.create_fee else None,
        price=float(agent.price) if agent.price else None,
        vip_level=agent.vip_level,
        shouldInitializeDialog=shouldInitializeDialog,
        initializeDialogQuestion=initializeDialogQuestion,
        is_paused=is_paused,
        pause_message=pause_message,
        dev=agent.dev,
        tenant_id=agent.tenant_id,
        enable_mcp=agent.enable_mcp if hasattr(agent, 'enable_mcp') else False,
    )
    
    # Add tools to the DTO
    if agent.tools:
        agent_dto.tools = []
        for tool in agent.tools:
            should_include_auth = is_full_config or (
                    user is not None and
                    user.get('tenant_id') == tool.tenant_id
            )
            agent_dto.tools.append(ToolInfo(
                id=tool.id,
                name=tool.name,
                description=tool.description,
                type=tool.type,
                origin=tool.origin if should_include_auth else None,
                path=tool.path,
                method=tool.method,
                parameters=tool.parameters,
                auth_config=tool.auth_config if should_include_auth else None,
                icon=tool.icon or SETTINGS.DEFAULT_TOOL_ICON,
                is_public=tool.is_public,
                is_official=tool.is_official,
                tenant_id=tool.tenant_id,
                is_stream=tool.is_stream,
                output_format=tool.output_format,
                sensitive_data_config=tool.sensitive_data_config
            ))
    
    # Add model if exists
    if hasattr(agent, 'model') and agent.model:
        agent_dto.model = ModelDTO(
            id=agent.model.id,
            name=agent.model.name,
            model_name=agent.model.model_name,
            endpoint=agent.model.endpoint if is_full_config else None,
            is_official=agent.model.is_official,
            is_public=agent.model.is_public
        )
    
    # Add category if exists
    if hasattr(agent, 'category') and agent.category:
        agent_dto.category = CategoryDTO(
            id=agent.category.id,
            name=agent.category.name,
            type=agent.category.type,
            description=agent.category.description,
            tenant_id=agent.category.tenant_id,
            sort_order=agent.category.sort_order,
            create_time=agent.category.create_time.isoformat() if agent.category.create_time else None,
            update_time=agent.category.update_time.isoformat() if agent.category.update_time else None
        )
    
    return agent_dto

async def refresh_public_agents_cache():
    """
    Refresh the Redis cache for public agents by incrementing the cache version.
    
    This function should be called whenever there are changes to agent data 
    that would affect the results of the list_public_agents function.
    
    Instead of deleting existing cache keys, this approach simply increments a version number,
    causing new requests to use a different cache key, effectively invalidating the old cache.
    Old cache entries will expire naturally according to their TTL.
    
    Returns:
        dict: Information about the cache refresh operation
    """
    try:
        # Get current version
        current_version = redis_utils.get_value(CACHE_VERSION_KEY) or "0"
        
        # Increment version
        new_version = str(int(current_version) + 1)
        
        # Set new version
        redis_utils.set_value(CACHE_VERSION_KEY, new_version)
        
        logger.info(f"Successfully refreshed public agents cache: version incremented from {current_version} to {new_version}")
        
        return {
            "previous_version": current_version,
            "new_version": new_version
        }
    except Exception as e:
        logger.error(f"Error refreshing public agents cache version: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to refresh public agents cache: {str(e)}"
        )

async def publish_to_store(
    agent_id: str,
    user: dict,
    session: AsyncSession,
    name: Optional[str] = None,
    icon: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    author: Optional[str] = None,
    github_url: Optional[str] = None,
    is_public: bool = True
) -> Dict[str, Any]:
    """
    Publish an agent to MCP Store
    
    Args:
        agent_id: ID of the agent to publish
        user: Current user information
        session: Database session
        name: Optional custom name for the store (defaults to agent's name)
        icon: Optional icon URL for the agent
        description: Optional description of the agent
        tags: Optional list of tags for the agent
        author: Optional author name
        github_url: Optional GitHub repository URL
        is_public: Whether the store is public
        
    Returns:
        Dictionary containing store information
    """
    try:
        # Get agent information
        agent = await get_agent(agent_id, user, session)
        
        if not agent:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                f"Agent with ID '{agent_id}' not found"
            )
            
        # Check if agent belongs to user's tenant
        if agent.tenant_id != user["tenant_id"]:
            raise CustomAgentException(
                ErrorCode.PERMISSION_DENIED,
                "You don't have permission to publish this agent"
            )
            
        # Create store in MCP Store
        store = await mcp_service.create_agent_store(
            agent_id=agent_id,
            user=user,
            session=session,
            name=name or agent.name,
            icon=icon or agent.icon,
            description=description or agent.description,
            tags=tags or [],
            author=author or user.get("wallet_address", ""),
            github_url=github_url,
            is_public=is_public
        )
        
        return store
        
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error publishing agent to store: {str(e)}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to publish agent to store: {str(e)}"
        )

