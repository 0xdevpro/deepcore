import logging
from typing import List, Optional, Dict
from urllib.parse import urlparse

from fastapi import Depends
from sqlalchemy import update, select, or_, and_, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agents.common.config import SETTINGS
from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db
from agents.models.models import Tool, App, AgentTool
from agents.protocol.response import ToolModel
from agents.protocol.schemas import ToolType, AuthConfig, CategoryDTO
from agents.utils import openapi
from agents.utils.openapi_utils import extract_endpoints_info

logger = logging.getLogger(__name__)

def tool_to_dto(tool: Tool, user: Optional[dict] = None) -> ToolModel:
    """
    Convert Tool ORM object to DTO
    
    Args:
        tool: Tool ORM object
        user: Current user information. If provided, will check tenant_id match
             to determine whether to include auth_config
    """
    try:
        should_include_auth = (
            user is not None and 
            user.get('tenant_id') == tool.tenant_id
        )
        
        tool_dto = ToolModel(
            id=tool.id,
            name=tool.name,
            type=tool.type,
            origin=tool.origin if should_include_auth else None,
            path=tool.path,
            method=tool.method,
            parameters=tool.parameters,
            description=tool.description,
            auth_config=tool.auth_config if should_include_auth else None,
            icon=tool.icon or SETTINGS.DEFAULT_TOOL_ICON,
            is_public=tool.is_public,
            is_official=tool.is_official,
            tenant_id=tool.tenant_id,
            create_time=tool.create_time,
            update_time=tool.update_time,
            is_stream=tool.is_stream,
            output_format=tool.output_format,
            category_id=tool.category_id
        )
        
        if tool.category_id and hasattr(tool, 'category') and tool.category is not None:
            tool_dto.category = CategoryDTO(
                id=tool.category.id,
                name=tool.category.name,
                type=tool.category.type,
                description=tool.category.description,
                tenant_id=tool.category.tenant_id,
                sort_order=tool.category.sort_order,
                create_time=tool.category.create_time.isoformat() if tool.category.create_time else None,
                update_time=tool.category.update_time.isoformat() if tool.category.update_time else None
            )
        
        return tool_dto
    except Exception as e:
        logger.error(f"Error converting tool to DTO: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.INTERNAL_ERROR,
            "Error processing tool data"
        )

async def create_tool(
        tool_data: dict,
        user: dict,
        session: AsyncSession
):
    """
    Create a new tool
    
    Args:
        tool_data: API tool configuration data
        user: Current user information
        session: Database session
    """
    try:
        if not user.get('tenant_id'):
            raise CustomAgentException(
                ErrorCode.UNAUTHORIZED,
                "User must belong to a tenant to create tools"
            )

        tool_type = tool_data.get('type', ToolType.OPENAPI.value)
        
        new_tool = Tool(
            name=tool_data['name'],
            description=tool_data.get('description'),
            type=tool_type,
            origin=tool_data['origin'],
            path=tool_data['path'],
            method=tool_data['method'],
            parameters=tool_data['parameters'],
            auth_config=tool_data.get('auth_config'),
            icon=tool_data.get('icon') or SETTINGS.DEFAULT_TOOL_ICON,
            is_public=False,
            is_official=False,
            tenant_id=user.get('tenant_id'),
            is_stream=tool_data.get('is_stream', False),
            output_format=tool_data.get('output_format')
        )

        session.add(new_tool)
        await session.flush()
        return tool_to_dto(new_tool, user)
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error creating tool: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create tool: {str(e)}"
        )

async def create_tools_batch(
        tools: List[dict],
        user: dict,
        session: AsyncSession
):
    """
    Create multiple tools in batch
    
    Args:
        tools: List of API tool configurations
        user: Current user information
        session: Database session
    """
    try:
        created_tools = []
        
        for tool_data in tools:
            tool = await create_tool(
                tool_data=tool_data,
                user=user,
                session=session
            )
            created_tools.append(tool)
        return created_tools
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error creating tools in batch: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create tools in batch: {str(e)}"
        )

async def update_tool(
        tool_id: str,
        user: dict,
        session: AsyncSession,
        name: Optional[str] = None,
        origin: Optional[str] = None,
        path: Optional[str] = None,
        method: Optional[str] = None,
        parameters: Optional[Dict] = None,
        auth_config: Optional[AuthConfig | List[AuthConfig]] = None,
        icon: Optional[str] = None,
        is_stream: Optional[bool] = None,
        output_format: Optional[Dict] = None
):
    """
    Update an existing tool
    
    Args:
        tool_id: ID of the tool to update
        user: Current user information
        session: Database session
        name: Optional new name for the tool
        origin: Optional new API origin
        path: Optional new API path
        method: Optional new HTTP method
        parameters: Optional new API parameters
        auth_config: Optional new authentication configuration
        icon: Optional new icon URL for the tool
        is_stream: Optional boolean indicating if the API returns a stream response
        output_format: Optional JSON configuration for formatting API output
    """
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
        if origin is not None:
            values_to_update['origin'] = origin
        if path is not None:
            values_to_update['path'] = path
        if method is not None:
            values_to_update['method'] = method
        if parameters is not None:
            values_to_update['parameters'] = parameters
        if auth_config is not None:
            values_to_update['auth_config'] = auth_config.model_dump()
        if icon is not None:
            values_to_update['icon'] = icon or SETTINGS.DEFAULT_TOOL_ICON
        if is_stream is not None:
            values_to_update['is_stream'] = is_stream
        if output_format is not None:
            values_to_update['output_format'] = output_format

        if values_to_update:
            stmt = update(Tool).where(
                Tool.id == tool_id,
                Tool.tenant_id == user.get('tenant_id')
            ).values(**values_to_update).execution_options(synchronize_session="fetch")
            await session.execute(stmt)
            await session.commit()
        return await get_tool(tool_id, user, session)
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error updating tool: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to update tool: {str(e)}"
        )

async def delete_tool(
        tool_id: str, 
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
        tool_id: str, 
        user: dict,
        session: AsyncSession = Depends(get_db)
):
    try:
        result = await session.execute(
            select(Tool).options(selectinload(Tool.category)).where(
                Tool.id == tool_id,
                or_(
                    Tool.tenant_id == user.get('tenant_id'),
                    Tool.is_public == True
                ),
                Tool.is_deleted == False
            )
        )
        tool = result.scalar_one_or_none()
        if tool is None:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool not found or no permission"
            )
        return tool_to_dto(tool, user)
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
        only_official: bool = False,
        category_id: Optional[int] = None
):
    """
    List tools with filters for public and official tools
    """
    try:
        conditions = [Tool.is_deleted == False]
        
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
                
        if category_id:
            conditions.append(Tool.category_id == category_id)
                
        # Calculate total count for pagination info
        count_query = select(func.count()).select_from(Tool).where(and_(*conditions))
        total_count = await session.execute(count_query)
        total_count = total_count.scalar()
        
        # Calculate offset from page number
        offset = (page - 1) * page_size
        
        # Get paginated results with category join and preload
        query = (
            select(Tool)
            .options(selectinload(Tool.category))
            .where(and_(*conditions))
            .order_by(Tool.create_time.desc())
        )
        
        result = await session.execute(
            query.offset(offset).limit(page_size)
        )
        tools = result.scalars().all()
        
        tool_dtos = [tool_to_dto(tool, user) for tool in tools]
        
        return {
            "items": tool_dtos,
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
        tool_id: str,
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
        tool_id: str,
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
                Tool.id == tool_id,
                Tool.is_deleted == False
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
        tool_id: str,
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
            select(Tool).options(selectinload(Tool.category)).join(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tenant_id == user.get('tenant_id'),
                Tool.is_deleted == False
            )
        )
        tools = result.scalars().all()
        return [tool_to_dto(tool, user) for tool in tools]
    except Exception as e:
        logger.error(f"Error getting tools by agent: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get tools by agent: {str(e)}"
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
            select(Tool).options(selectinload(Tool.category)).join(AgentTool).where(
                AgentTool.agent_id == agent_id,
                AgentTool.tenant_id == user.get('tenant_id'),
                Tool.is_deleted == False
            )
        )
        tools = result.scalars().all()
        return [tool_to_dto(tool, user) for tool in tools]
    except Exception as e:
        logger.error(f"Error getting agent tools: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get agent tools: {str(e)}"
        )

async def assign_tools_to_agent(
        tool_ids: List[str],
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
        tool_ids: List[str],
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

def flatten_api_info(api_info: dict) -> list:
    """
    Flatten OpenAPI information into a list of API tools
    """
    flattened_apis = []
    origin = api_info.get('origin', '')
    
    for endpoint in api_info.get('endpoints', []):
        api_tool = {
            'name': endpoint.get('name', ''),
            'description': endpoint.get('description'),
            'path': endpoint.get('path', ''),
            'method': endpoint.get('method', ''),
            'origin': origin,
            'parameters': {
                'header': [],
                'query': [],
                'path': [],
                'body': None
            }
        }
        
        # Process parameters
        parameters = endpoint.get('parameters', {})
        for param_type in ['header', 'query', 'path']:
            for param in parameters.get(param_type, []):
                param_info = {
                    'name': param.get('name', ''),
                    'type': param.get('type', 'string'),
                }
                if param.get('required'):
                    param_info['required'] = True
                if 'default' in param:
                    param_info['default'] = param['default']
                if 'description' in param:
                    param_info['description'] = param['description']
                api_tool['parameters'][param_type].append(param_info)
        
        # Process body if exists
        if parameters.get('body'):
            api_tool['parameters']['body'] = parameters['body']
        
        flattened_apis.append(api_tool)
    
    return flattened_apis

async def parse_openapi_content(content: str) -> list:
    """
    Parse OpenAPI content and return flattened API information
    """
    try:
        api_info = extract_endpoints_info(content)
        return flatten_api_info(api_info)
    except Exception as e:
        logger.error(f"Error parsing OpenAPI content: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to parse OpenAPI content: {str(e)}"
        )

async def parse_mcp_content(url: str) -> list:
    """
    Parse MCP content from URL and return flattened API information
    
    Args:
        url: MCP service URL to fetch tools from
    
    Returns:
        List of flattened API tool information
    """
    try:
        from mirascope.mcp import sse_client
        
        async with sse_client(url) as client:
            tools_result = await client._session.list_tools()
            
        flattened_tools = []
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        path = parsed_url.path
        
        for item in tools_result:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tool_info = {
                        'name': tool.name,
                        'description': tool.description or '',
                        'path': path,  # Use a standardized path format
                        'method': 'SSE',  # MCP tools typically use SSE method
                        'origin': base_url,
                        'parameters': tool.inputSchema,  # Use the inputSchema as the body parameter
                        'type': ToolType.MCP.value,
                    }

                    flattened_tools.append(tool_info)
            
        return flattened_tools
    except Exception as e:
        logger.error(f"Error parsing MCP content from URL {url}: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to parse MCP content: {str(e)}"
        )

async def get_tools_by_ids(
        tool_ids: List[str], 
        user: dict,
        session: AsyncSession
):
    """
    Batch retrieve multiple tool objects (returns ORM objects directly, not DTOs)
    
    Args:
        tool_ids: List of tool IDs
        user: Current user information
        session: Database session
        
    Returns:
        Dictionary with tool IDs as keys and Tool objects as values
    """
    try:
        result = await session.execute(
            select(Tool).options(selectinload(Tool.category)).where(
                Tool.id.in_(tool_ids),
                or_(
                    Tool.tenant_id == user.get('tenant_id'),
                    Tool.is_public == True
                ),
                Tool.is_deleted == False
            )
        )
        tools = result.scalars().all()
        
        # Create mapping from ID to Tool object
        tool_map = {str(tool.id): tool for tool in tools}
        return tool_map
    except Exception as e:
        logger.error(f"Failed to batch retrieve tools: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to batch retrieve tools: {str(e)}"
        )
