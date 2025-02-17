from typing import Optional, AsyncIterator, List

from fastapi import Depends
from sqlalchemy import update, delete, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from agents.agent.chat_agent import ChatAgent
from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db
from agents.models.models import App, Tool, AgentTool
from agents.protocol.schemas import AgentStatus, DialogueRequest, AgentDTO, ToolInfo


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
        raise CustomAgentException(message=f'Agent not found or no permission')
    agent = ChatAgent(agent)
    async for response in agent.stream(request.query, request.conversation_id):
        yield response


async def get_agent(id: str, user: dict, session: AsyncSession):
    """
    Get agent with its associated tools
    """
    # Add tenant filter
    result = await session.execute(
        select(App).where(
            App.id == id,
            App.tenant_id == user.get('tenant_id')
        )
    )
    agent = result.scalar_one_or_none()
    if agent is None:
        raise CustomAgentException(message=f'Agent not found or no permission')

    # Get associated tools
    tools_result = await session.execute(
        select(Tool).join(AgentTool).where(
            AgentTool.agent_id == id,
            AgentTool.tenant_id == user.get('tenant_id')
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
    
    return model


async def verify_tool_permissions(
    tool_ids: List[int],
    user: dict,
    session: AsyncSession
) -> List[Tool]:
    """
    Verify if user has permission to use the specified tools
    Raises CustomAgentException if any tool is not accessible
    """
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
            ErrorCode.PERMISSION_DENIED,
            "User must belong to a tenant to create agents"
        )

    async with session.begin():
        # Verify tool permissions if tools are specified
        tool_ids = []
        if agent.tools:
            tool_ids = agent.tools  # Now tools is directly a list of IDs
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


async def list_agents(
        status: Optional[AgentStatus],
        skip: int,
        limit: int,
        session: AsyncSession,
        user: dict,
        include_public: bool = True,
        only_official: bool = False
):
    """
    List agents with filters for public and official agents, including their tools
    """
    conditions = []
    
    if only_official:
        conditions.append(App.is_official == True)
    else:
        # Show user's own agents and optionally public agents
        if user and user.get('tenant_id'):
            conditions.append(
                or_(
                    App.tenant_id == user.get('tenant_id'),
                    and_(App.is_public == True) if include_public else False
                )
            )
        else:
            conditions.append(App.is_public == True)

    if status:
        conditions.append(App.status == status)
    
    query = select(App).where(and_(*conditions))
    
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
                AgentTool.tenant_id == user.get('tenant_id')
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

    return results


async def update_agent(
        agent: AgentDTO,
        user: dict,
        session: AsyncSession = Depends(get_db)
):
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
            tool_ids = agent.tools  # Now tools is directly a list of IDs
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


async def delete_agent(
        agent_id: str, 
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    async with session.begin():  # Use transaction
        # Add tenant filter
        await session.execute(
            delete(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            )
        )
        # Transaction will auto commit or rollback when exiting async with


async def publish_agent(
        agent_id: str,
        is_public: bool,
        user: dict,
        session: AsyncSession):
    """
    Publish or unpublish an agent
    """
    async with session.begin():  # Use transaction
        # First check if the agent exists and belongs to the user's tenant
        result = await session.execute(
            select(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            )
        )
        agent = result.scalar_one_or_none()
        if not agent:
            raise CustomAgentException(message="Agent not found or no permission")

        # If agent exists and belongs to user's tenant, proceed with publish/unpublish
        stmt = update(App).where(
            App.id == agent_id,
            App.tenant_id == user.get('tenant_id')
        ).values(
            is_public=is_public
        )
        await session.execute(stmt)
        # Transaction will auto commit or rollback when exiting async with
