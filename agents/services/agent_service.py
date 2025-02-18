import logging
from typing import Optional, AsyncIterator, List

from fastapi import Depends
from sqlalchemy import update, delete, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from agents.agent.chat_agent import ChatAgent
from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db
from agents.models.models import App, Tool, AgentTool
from agents.protocol.schemas import AgentStatus, DialogueRequest, AgentDTO, ToolInfo

logger = logging.getLogger(__name__)

async def dialogue(
        agent_id: str,
        request: DialogueRequest, 
        user: dict,
        session: AsyncSession = Depends(get_db)
) -> AsyncIterator[str]:
    # Add tenant filter
    result = await get_agent(agent_id, user, session)
    agent = result.scalar_one_or_none()
    if not agent:
        raise CustomAgentException(
            ErrorCode.RESOURCE_NOT_FOUND,
            "Agent not found or no permission"
        )
    agent = ChatAgent(agent)
    async for response in agent.stream(request.query, request.conversation_id):
        yield response


async def get_agent(id: str, user: dict, session: AsyncSession):
    """
    Get agent with its associated tools
    """
    if not user or not user.get('tenant_id'):
        raise CustomAgentException(
            ErrorCode.UNAUTHORIZED,
            "User authentication required"
        )

    # Add tenant filter
    result = await session.execute(
        select(App).where(
            App.id == id,
            App.tenant_id == user.get('tenant_id')
        )
    )
    agent = result.scalar_one_or_none()
    if agent is None:
        raise CustomAgentException(
            ErrorCode.RESOURCE_NOT_FOUND,
            "Agent not found or no permission"
        )

    # Get associated tools
    tools_result = await session.execute(
        select(Tool).join(AgentTool).where(
            AgentTool.agent_id == id,
            AgentTool.tenant_id == user.get('tenant_id')
        )
    )
    tools = tools_result.scalars().all()
    
    # Convert to DTO
    try:
        model = AgentDTO.model_validate_json(agent.model_json)
        model.tools = [ToolInfo(
            id=tool.id,
            name=tool.name,
            type=tool.type,
            content=tool.content
        ) for tool in tools]
        
        return model
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
                model_json=agent.model_dump_json(),
                tenant_id=user.get('tenant_id')
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
        include_public: bool = False
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
        conditions = [or_(
            App.tenant_id == user.get('tenant_id'),
            App.is_public == True
        )]

    if status:
        conditions.append(App.status == status)
    
    return await _get_paginated_agents(conditions, skip, limit, user, session)


async def list_public_agents(
        status: Optional[AgentStatus],
        skip: int,
        limit: int,
        session: AsyncSession,
        only_official: bool = False
):
    """
    List public or official agents
    
    Args:
        status: Optional filter for agent status
        skip: Number of records to skip ((page - 1) * page_size)
        limit: Number of records per page (page_size)
        session: Database session
        only_official: Whether to only show official agents
        
    Returns:
        dict: {
            "items": list of agents,
            "total": total number of records,
            "page": current page number,
            "page_size": number of items per page,
            "total_pages": total number of pages
        }
    """
    conditions = []
    
    if only_official:
        conditions.append(App.is_official == True)
    else:
        conditions.append(App.is_public == True)

    if status:
        conditions.append(App.status == status)
    
    return await _get_paginated_agents(conditions, skip, limit, None, session)


async def _get_paginated_agents(conditions: list, skip: int, limit: int, user: Optional[dict], session: AsyncSession):
    """
    Helper function to get paginated agents with given conditions
    """
    # Calculate total count for pagination info
    count_query = select(func.count()).select_from(App).where(and_(*conditions))
    total_count = await session.execute(count_query)
    total_count = total_count.scalar()
    
    # Get paginated results with ordering
    query = select(App).where(and_(*conditions)).order_by(App.create_time.desc())
    
    result = await session.execute(
        query.offset(skip).limit(limit)
    )
    agents = result.scalars().all()
    results = []
    
    for agent in agents:
        # Get associated tools for each agent
        tools_result = await session.execute(
            select(Tool).join(AgentTool).where(
                AgentTool.agent_id == agent.id,
                # Only filter by tenant_id for personal tools
                *([AgentTool.tenant_id == user.get('tenant_id')] if user else [])
            )
        )
        tools = tools_result.scalars().all()
        
        # Convert to DTO
        model = AgentDTO.model_validate_json(agent.model_json)
        model.tools = [ToolInfo(
            id=tool.id,
            name=tool.name,
            type=tool.type,
            content=tool.content
        ) for tool in tools]
        
        results.append(model)

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
    try:
        async with session.begin():
            # Verify agent ownership
            existing_agent = await get_agent(agent.id, user, session)
            if not existing_agent:
                raise CustomAgentException(
                    ErrorCode.PERMISSION_DENIED,
                    "Agent not found or no permission to update"
                )
                
            # Verify tool permissions if tools are being updated
            if agent.tools is not None:
                tool_ids = agent.tools
                await verify_tool_permissions(tool_ids, user, session)
                
                # Remove existing associations
                await session.execute(
                    delete(AgentTool).where(
                        AgentTool.agent_id == agent.id,
                        AgentTool.tenant_id == user.get('tenant_id')
                    )
                )
                
                # Create new associations
                for tool_id in tool_ids:
                    agent_tool = AgentTool(
                        agent_id=agent.id,
                        tool_id=tool_id,
                        tenant_id=user.get('tenant_id')
                    )
                    session.add(agent_tool)

            # Update agent fields
            update_values = {
                'name': agent.name,
                'description': agent.description,
                'mode': agent.mode,
                'icon': agent.icon,
                'status': agent.status,
                'role_settings': agent.role_settings,
                'welcome_message': agent.welcome_message,
                'twitter_link': agent.twitter_link,
                'telegram_bot_id': agent.telegram_bot_id,
                'tool_prompt': agent.tool_prompt,
                'max_loops': agent.max_loops,
                'suggested_questions': agent.suggested_questions,
                'model_json': agent.model_dump_json()
            }
            
            # Filter out None values
            update_values = {k: v for k, v in update_values.items() if v is not None}

            stmt = update(App).where(
                App.id == existing_agent.id,
                App.tenant_id == user.get('tenant_id')
            ).values(**update_values).execution_options(synchronize_session="fetch")
            
            await session.execute(stmt)

        return existing_agent
    except CustomAgentException:
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
            # Verify agent exists and belongs to user
            result = await session.execute(
                select(App).where(
                    App.id == agent_id,
                    App.tenant_id == user.get('tenant_id')
                )
            )
            if not result.scalar_one_or_none():
                raise CustomAgentException(
                    ErrorCode.RESOURCE_NOT_FOUND,
                    "Agent not found or no permission to delete"
                )
            
            # Delete agent
            await session.execute(
                delete(App).where(
                    App.id == agent_id,
                    App.tenant_id == user.get('tenant_id')
                )
            )
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
        user: dict,
        session: AsyncSession):
    """
    Publish or unpublish an agent
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

            # Update publish status
            stmt = update(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            ).values(
                is_public=is_public
            )
            await session.execute(stmt)
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error publishing agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to publish agent: {str(e)}"
        )
