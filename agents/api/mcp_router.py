import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Body, Path
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from agents.common.error_messages import get_error_message
from agents.common.response import RestResponse
from agents.exceptions import CustomAgentException, ErrorCode
from agents.middleware.auth_middleware import get_current_user
from agents.models.db import get_db
from agents.services import mcp_service

router = APIRouter()
logger = logging.getLogger(__name__)


class CreateMCPServerRequest(BaseModel):
    """Create MCP Server Request"""
    mcp_name: str
    tool_ids: List[str]
    description: Optional[str] = None


class AddPromptTemplateRequest(BaseModel):
    """Add Prompt Template Request"""
    prompt_name: str
    description: str
    arguments: List[dict]
    template: str


class AddResourceRequest(BaseModel):
    """Add Resource Request"""
    resource_uri: str
    content: str
    mime_type: str = "text/plain"


class MCPStoreRequest(BaseModel):
    """MCP Store Request"""
    store_name: str
    icon: Optional[str] = None
    description: Optional[str] = None
    store_type: str
    tags: Optional[List[str]] = None
    content: Optional[str] = None
    author: Optional[str] = None
    github_url: Optional[str] = None


class MCPStoreQueryParams(BaseModel):
    """MCP Store Query Parameters"""
    page: int = 1
    page_size: int = 10
    keyword: Optional[str] = None
    store_type: Optional[str] = None


class GenerateToolsRequest(BaseModel):
    """Generate Tools Request"""
    user_input: str
    conversation_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))


class CreateToolsAndMCPServerRequest(BaseModel):
    """Create Tools and MCP Server Request"""
    tools: List[dict]
    mcp_name: str
    description: Optional[str] = None
    icon: Optional[str] = None


@router.post("/mcp/create", summary="Create MCP Server")
async def create_mcp_server(
    request: CreateMCPServerRequest,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Create an MCP server from tools
    
    Parameters:
    - **mcp_name**: MCP server name
    - **tool_ids**: List of tool IDs to expose as MCP interface
    - **description**: Optional MCP service description
    """
    try:
        result = await mcp_service.create_mcp_server_from_tools(
            mcp_name=request.mcp_name,
            tool_ids=request.tool_ids,
            user=user,
            session=session,
            description=request.description
        )
        await session.commit()
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error creating MCP server: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating MCP server: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/mcp/{mcp_name}/prompts", summary="Add Prompt Template")
async def add_prompt_template(
    mcp_name: str = Path(..., description="MCP server name"),
    request: AddPromptTemplateRequest = Body(...),
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Add a prompt template to an MCP server
    
    Parameters:
    - **mcp_name**: MCP server name
    - **prompt_name**: Prompt template name
    - **description**: Prompt description
    - **arguments**: List of prompt arguments
    - **template**: Prompt template text
    """
    try:
        result = await mcp_service.add_prompt_template(
            mcp_name=mcp_name,
            prompt_name=request.prompt_name,
            description=request.description,
            arguments=request.arguments,
            template=request.template,
            session=session
        )
        
        if not result:
            return RestResponse(
                code=ErrorCode.RESOURCE_NOT_FOUND,
                msg=f"MCP server '{mcp_name}' not found"
            )
            
        return RestResponse(data={"success": True})
    except CustomAgentException as e:
        logger.error(f"Error adding prompt template: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error adding prompt template: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/mcp/{mcp_name}/resources", summary="Add Resource")
async def add_resource(
    mcp_name: str = Path(..., description="MCP server name"),
    request: AddResourceRequest = Body(...),
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Add a resource to an MCP server
    
    Parameters:
    - **mcp_name**: MCP server name
    - **resource_uri**: Resource URI
    - **content**: Resource content
    - **mime_type**: MIME type of the resource (default: text/plain)
    """
    try:
        result = await mcp_service.add_resource(
            mcp_name=mcp_name,
            resource_uri=request.resource_uri,
            content=request.content,
            mime_type=request.mime_type,
            session=session
        )
        
        if not result:
            return RestResponse(
                code=ErrorCode.RESOURCE_NOT_FOUND,
                msg=f"MCP server '{mcp_name}' not found"
            )
            
        return RestResponse(data={"success": True})
    except CustomAgentException as e:
        logger.error(f"Error adding resource: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error adding resource: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/mcp/list", summary="List All MCP Servers")
async def list_mcp_servers(
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    List all registered MCP servers
    """
    try:
        servers = await mcp_service.get_registered_mcp_servers(session)
        return RestResponse(data=servers)
    except CustomAgentException as e:
        logger.error(f"Error listing MCP servers: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing MCP servers: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.delete("/mcp/{mcp_name}", summary="Delete MCP Server")
async def delete_mcp_server(
    mcp_name: str = Path(..., description="MCP server name"),
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Delete a specific MCP server
    
    Parameters:
    - **mcp_name**: Name of the MCP server to delete
    """
    try:
        result = await mcp_service.delete_mcp_server(mcp_name, session)
        if not result:
            return RestResponse(
                code=ErrorCode.RESOURCE_NOT_FOUND,
                msg=f"MCP server '{mcp_name}' not found"
            )
        return RestResponse(data={"success": True})
    except CustomAgentException as e:
        logger.error(f"Error deleting MCP server: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error deleting MCP server: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/mcp/tools/{tool_id}", summary="Get Tool MCP Service Info")
async def get_tool_mcp_info(
    tool_id: str = Path(..., description="Tool ID"),
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Get MCP service information for a specific tool
    
    Parameters:
    - **tool_id**: Tool ID
    """
    try:
        mapping = await mcp_service.get_tool_mcp_mapping(session)
        if tool_id not in mapping:
            return RestResponse(
                code=ErrorCode.RESOURCE_NOT_FOUND,
                msg=f"Tool '{tool_id}' is not exposed as MCP"
            )
            
        mcp_name = mapping[tool_id]
        return RestResponse(data={
            "tool_id": tool_id,
            "mcp_name": mcp_name,
            "url": f"/mcp/{mcp_name}"
        })
    except CustomAgentException as e:
        logger.error(f"Error getting tool MCP info: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting tool MCP info: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/mcp/store", summary="Create MCP Store")
async def create_mcp_store(
    request: MCPStoreRequest,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Create a new MCP Store
    
    Parameters:
    - **store_name**: Name of the store
    - **icon**: Store icon URL (optional)
    - **description**: Store description (optional)
    - **store_type**: Type of the store
    - **tags**: List of store tags (optional)
    - **content**: Store content (optional)
    - **author**: Author name (optional)
    """
    try:
        result = await mcp_service.create_mcp_store(
            store_name=request.store_name,
            icon=request.icon,
            store_type=request.store_type,
            description=request.description,
            tags=request.tags,
            content=request.content,
            author=request.author,
            user=user,
            session=session
        )
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error creating MCP store: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating MCP store: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/mcp/stores", summary="List MCP Stores")
async def list_mcp_stores(
    page: int = 1,
    page_size: int = 10,
    keyword: Optional[str] = None,
    store_type: Optional[str] = None,
    is_public: Optional[bool] = None,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    List all registered MCP Stores with pagination and search support
    
    Parameters:
    - **page**: Page number (default: 1)
    - **page_size**: Number of items per page (default: 10)
    - **keyword**: Search keyword, matches name and description (optional)
    - **store_type**: Filter by store type (optional)
    - **is_public**: Filter by public status (optional)
    """
    try:
        result = await mcp_service.get_registered_mcp_stores(
            page=page,
            page_size=page_size,
            keyword=keyword,
            store_type=store_type,
            is_public=is_public,
            session=session,
            user=user
        )
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error listing MCP stores: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing MCP stores: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/mcp/store/{store_id}", summary="Get MCP Store Detail")
async def get_mcp_store(
    store_id: int,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific MCP Store
    
    Parameters:
    - **store_id**: ID of the store
    """
    try:
        result = await mcp_service.get_mcp_store_detail(store_id, session, user)
        if not result:
            return RestResponse(
                code=ErrorCode.RESOURCE_NOT_FOUND,
                msg=f"MCP store with ID {store_id} not found"
            )
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error getting MCP store detail: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting MCP store detail: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.delete("/mcp/store/{store_name}", summary="Delete MCP Store")
async def delete_mcp_store(
    store_name: str = Path(..., description="Store name"),
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Delete a specific MCP Store
    
    Parameters:
    - **store_name**: Name of the store to delete
    """
    try:
        result = await mcp_service.delete_mcp_store(store_name, session)
        if not result:
            return RestResponse(
                code=ErrorCode.RESOURCE_NOT_FOUND,
                msg=f"MCP store '{store_name}' not found"
            )
        return RestResponse(data={"success": True})
    except CustomAgentException as e:
        logger.error(f"Error deleting MCP store: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error deleting MCP store: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/mcp/generate-tools", summary="Generate Tools from User Input")
async def generate_tools(
    user_input: str,
    conversation_id: Optional[str] = None,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Generate tool list from user input using LLM
    
    Parameters:
    - **user_input**: User's input text
    - **conversation_id**: Conversation ID for context (optional)
    """
    try:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        result = mcp_service.generate_tools_from_input(
            user_input=user_input,
            conversation_id=conversation_id,
            user=user,
            session=session
        )
        return StreamingResponse(content=result, media_type="text/event-stream")
    except CustomAgentException as e:
        logger.error(f"Error generating tools: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error generating tools: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/mcp/create-with-tools", summary="Create Tools and MCP Server")
async def create_tools_and_mcp_server(
    request: CreateToolsAndMCPServerRequest,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Create multiple tools and an MCP server in one operation
    
    Parameters:
    - **tools**: List of tool configurations
    - **mcp_name**: Name for the new MCP server
    - **description**: Optional MCP service description
    """
    try:
        result = await mcp_service.create_tools_and_mcp_server(
            tools=request.tools,
            mcp_name=request.mcp_name,
            user=user,
            session=session,
            description=request.description,
            icon=request.icon
        )
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error creating tools and MCP server: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating tools and MCP server: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        ) 