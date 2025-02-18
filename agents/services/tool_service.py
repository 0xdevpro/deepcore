from fastapi import Depends
from sqlalchemy import update, select, or_, and_, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db
from agents.models.models import Tool, App, AgentTool
from agents.protocol.response import ToolModel
from agents.protocol.schemas import ToolType, AuthConfig
from agents.utils import openapi

logger = logging.getLogger(__name__)

def tool_to_dto(tool: Tool) -> ToolModel:
    """Convert Tool ORM object to DTO"""
    try:
        return ToolModel(
            id=tool.id,
            name=tool.name,
            type=tool.type,
            content=tool.content,
            auth_config=tool.auth_config,
            is_public=tool.is_public,
            is_official=tool.is_official,
            tenant_id=tool.tenant_id,
            create_time=tool.create_time,
            update_time=tool.update_time
        )
    except Exception as e:
        logger.error(f"Error converting tool to DTO: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.INTERNAL_ERROR,
            "Error processing tool data"
        )

async def create_tool(
        name: str, 
        type: ToolType, 
        content: str,
        user: dict,
        session: AsyncSession,
        auth_config: Optional[AuthConfig] = None
):
    """
    Create tool with user context
    """
    if not user.get('tenant_id'):
        raise CustomAgentException(
            ErrorCode.UNAUTHORIZED,
            "User must belong to a tenant to create tools"
        )

    try:
        new_tool = Tool(
            name=name, 
            type=type.value, 
            content=content,
            is_public=False,
            is_official=False,
            auth_config=auth_config.model_dump() if auth_config else None,
            tenant_id=user.get('tenant_id')
        )
        await check_oepnapi_validity(type, name, content)
        session.add(new_tool)
        await session.commit()
        return tool_to_dto(new_tool)
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error creating tool: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create tool: {str(e)}"
        )

async def update_tool(
        tool_id: int,
        user: dict,
        session: AsyncSession,
        name: Optional[str] = None,
        type: Optional[ToolType] = None,
        content: Optional[str] = None,
        auth_config: Optional[AuthConfig] = None
):
    try:
        # Verify if the tool belongs to current user
        tool_result = await session.execute(
            select(Tool).where(
                Tool.id == tool_id,
                Tool.tenant_id == user.get('tenant_id')
            )
        )
        tool = tool_result.scalar_one_or_none()
        if not tool:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool not found or no permission"
            )

        values_to_update = {}
        if name is not None:
            values_to_update['name'] = name
        if type is not None:
            values_to_update['type'] = type.value
        if content is not None:
            await check_oepnapi_validity(type, name, content)
            values_to_update['content'] = content
        if auth_config is not None:
            values_to_update['auth_config'] = auth_config.model_dump()

        if values_to_update:
            stmt = update(Tool).where(
                Tool.id == tool_id,
                Tool.tenant_id == user.get('tenant_id')
            ).values(**values_to_update).execution_options(synchronize_session="fetch")
            await session.execute(stmt)
            await session.commit()
        return get_tool(tool_id, user, session)
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error updating tool: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to update tool: {str(e)}"
        )

async def delete_tool(
        tool_id: int, 
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    try:
        # Verify tool exists and belongs to user
        result = await session.execute(
            select(Tool).where(
                Tool.id == tool_id,
                Tool.tenant_id == user.get('tenant_id')
            )
        )
        if not result.scalar_one_or_none():
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool not found or no permission to delete"
            )
            
        stmt = update(Tool).where(
            Tool.id == tool_id,
            Tool.tenant_id == user.get('tenant_id')
        ).values(is_deleted=True).execution_options(synchronize_session="fetch")
        await session.execute(stmt)
        await session.commit()
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error deleting tool: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to delete tool: {str(e)}"
        )

async def get_tool(
        tool_id: int, 
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    try:
        result = await session.execute(
            select(Tool).where(
                Tool.id == tool_id,
                Tool.tenant_id == user.get('tenant_id')
            )
        )
        tool = result.scalar_one_or_none()
        if tool is None:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool not found or no permission"
            )
        return ToolModel.model_validate(tool)
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get tool: {str(e)}"
        )

async def get_tools(
        session: AsyncSession,
        user: dict,
        page: int = 1,
        page_size: int = 10,
        include_public: bool = True,
        only_official: bool = False
):
    """
    List tools with filters for public and official tools
    """
    try:
        conditions = []
        
        if only_official:
            conditions.append(Tool.is_official == True)
        else:
            if user and user.get('tenant_id'):
                conditions.append(
                    or_(
                        Tool.tenant_id == user.get('tenant_id'),
                        and_(Tool.is_public == True) if include_public else False
                    )
                )
            else:
                conditions.append(Tool.is_public == True)
                
        # Calculate total count for pagination info
        count_query = select(func.count()).select_from(Tool).where(and_(*conditions))
        total_count = await session.execute(count_query)
        total_count = total_count.scalar()
        
        # Calculate offset from page number
        offset = (page - 1) * page_size
        
        # Get paginated results
        result = await session.execute(
            select(Tool).where(and_(*conditions)).offset(offset).limit(page_size)
        )
        tools = result.scalars().all()
        
        return {
            "items": [ToolModel.model_validate(tool) for tool in tools],
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }
    except Exception as e:
        logger.error(f"Error getting tools: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get tools: {str(e)}"
        )

async def check_oepnapi_validity(type: ToolType, name: str, content: str):
    if type != ToolType.OPENAPI:
        return

    validated, error = openapi.validate_openapi(content)
    if not validated:
        raise CustomAgentException(
            ErrorCode.OPENAPI_ERROR,
            f"Invalid OpenAPI definition for {name}: {error}"
        )

async def publish_tool(
        tool_id: int,
        is_public: bool,
        user: dict,
        session: AsyncSession):
    """
    Publish or unpublish a tool
    """
    try:
        # First check if the tool exists and belongs to the user's tenant
        result = await session.execute(
            select(Tool).where(
                Tool.id == tool_id,
                Tool.tenant_id == user.get('tenant_id')
            )
        )
        tool = result.scalar_one_or_none()
        if not tool:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool not found or no permission"
            )

        # Update publish status
        stmt = update(Tool).where(
            Tool.id == tool_id,
            Tool.tenant_id == user.get('tenant_id')
        ).values(
            is_public=is_public
        )
        await session.execute(stmt)
        await session.commit()
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error publishing tool: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to publish tool: {str(e)}"
        )

async def assign_tool_to_agent(
        tool_id: int,
        agent_id: str,
        user: dict,
        session: AsyncSession
):
    """
    Assign a tool to an agent
    """
    try:
        # Check if tool exists and is accessible (owned or public)
        tool = await session.execute(
            select(Tool).where(
                or_(
                    Tool.tenant_id == user.get('tenant_id'),
                    Tool.is_public == True
                ),
                Tool.id == tool_id
            )
        )
        tool = tool.scalar_one_or_none()
        if not tool:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool not found or no permission"
            )

        # Check if agent belongs to user
        agent = await session.execute(
            select(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            )
        )
        agent = agent.scalar_one_or_none()
        if not agent:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Agent not found or no permission"
            )

        # Create association
        agent_tool = AgentTool(
            agent_id=agent_id,
            tool_id=tool_id,
            tenant_id=user.get('tenant_id')
        )
        session.add(agent_tool)
        await session.commit()
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error assigning tool to agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to assign tool to agent: {str(e)}"
        )

async def remove_tool_from_agent(
        tool_id: int,
        agent_id: str,
        user: dict,
        session: AsyncSession
):
    """
    Remove a tool from an agent
    """
    try:
        # Verify agent and tool exist and belong to user
        result = await session.execute(
            select(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tool_id == tool_id,
                AgentTool.tenant_id == user.get('tenant_id')
            )
        )
        if not result.scalar_one_or_none():
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool-agent association not found or no permission"
            )
            
        await session.execute(
            delete(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tool_id == tool_id,
                AgentTool.tenant_id == user.get('tenant_id')
            )
        )
        await session.commit()
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error removing tool from agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to remove tool from agent: {str(e)}"
        )

async def get_agent_tools(
        agent_id: str,
        user: dict,
        session: AsyncSession
):
    """
    Get all tools associated with an agent
    """
    try:
        result = await session.execute(
            select(Tool).join(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tenant_id == user.get('tenant_id')
            )
        )
        tools = result.scalars().all()
        return [ToolModel.model_validate(tool) for tool in tools]
    except Exception as e:
        logger.error(f"Error getting agent tools: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get agent tools: {str(e)}"
        )

async def assign_tools_to_agent(
        tool_ids: List[int],
        agent_id: str,
        user: dict,
        session: AsyncSession
):
    """
    Assign multiple tools to an agent
    """
    try:
        # Check if agent belongs to user
        agent = await session.execute(
            select(App).where(
                App.id == agent_id,
                App.tenant_id == user.get('tenant_id')
            )
        )
        agent = agent.scalar_one_or_none()
        if not agent:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Agent not found or no permission"
            )

        # Check if all tools exist and are accessible (owned or public)
        tools = await session.execute(
            select(Tool).where(
                or_(
                    Tool.tenant_id == user.get('tenant_id'),
                    Tool.is_public == True
                ),
                Tool.id.in_(tool_ids)
            )
        )
        found_tools = tools.scalars().all()
        if len(found_tools) != len(tool_ids):
            raise CustomAgentException(
                ErrorCode.PERMISSION_DENIED,
                "Some tools not found or no permission"
            )

        # Create associations
        for tool_id in tool_ids:
            agent_tool = AgentTool(
                agent_id=agent_id,
                tool_id=tool_id,
                tenant_id=user.get('tenant_id')
            )
            session.add(agent_tool)
        
        await session.commit()
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error assigning tools to agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to assign tools to agent: {str(e)}"
        )

async def remove_tools_from_agent(
        tool_ids: List[int],
        agent_id: str,
        user: dict,
        session: AsyncSession
):
    """
    Remove multiple tools from an agent
    """
    try:
        # Verify agent and tools exist and belong to user
        result = await session.execute(
            select(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tool_id.in_(tool_ids),
                AgentTool.tenant_id == user.get('tenant_id')
            )
        )
        found_associations = result.scalars().all()
        if len(found_associations) != len(tool_ids):
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Some tool-agent associations not found or no permission"
            )
            
        await session.execute(
            delete(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tool_id.in_(tool_ids),
                AgentTool.tenant_id == user.get('tenant_id')
            )
        )
        await session.commit()
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error removing tools from agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to remove tools from agent: {str(e)}"
        )

async def get_tools_by_agent(
        agent_id: str,
        session: AsyncSession,
        user: dict,
):
    """
    Get all tools associated with a specific agent
    """
    try:
        result = await session.execute(
            select(Tool).join(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tenant_id == user.get('tenant_id')
            )
        )
        tools = result.scalars().all()
        return [ToolModel.model_validate(tool) for tool in tools]
    except Exception as e:
        logger.error(f"Error getting tools by agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get tools by agent: {str(e)}"
        )
