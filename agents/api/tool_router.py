import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Query, Path, Body, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from agents.common.error_messages import get_error_message
from agents.common.response import RestResponse
from agents.exceptions import CustomAgentException, ErrorCode
from agents.middleware.auth_middleware import get_current_user
from agents.models.db import get_db
from agents.protocol.response import ToolModel
from agents.protocol.schemas import ToolCreate, ToolUpdate, AgentToolsRequest, CreateToolsBatchRequest, \
    OpenAPIParseRequest
from agents.services import tool_service
from agents.utils.http_client import debug_tool_api

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/tools/create", summary="Create Tool", response_model=RestResponse[ToolModel])
async def create_tool(
        request: ToolCreate,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Create a new API tool
    
    Parameters:
    - **tool_data**: API tool configuration including name, host, path, method, parameters and auth_config
    """
    try:
        tool = await tool_service.create_tool(
            tool_data=request.tool_data.model_dump(),
            user=user,
            session=session
        )
        await session.commit()
        return RestResponse(data=tool)
    except CustomAgentException as e:
        logger.error(f"Error creating tool: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating tool: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/tools/list", summary="List Tools")
async def list_tools(
        include_public: bool = Query(True, description="Include public tools"),
        only_official: bool = Query(False, description="Show only official tools"),
        category_id: Optional[int] = Query(None, description="Filter tools by category"),
        page: int = Query(1, description="Page number"),
        page_size: int = Query(10, description="Number of items per page"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    List all available tools
    
    Parameters:
    - **include_public**: Whether to include public tools
    - **only_official**: Whether to show only official tools
    - **category_id**: Optional filter for category ID
    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (1-100)
    """
    try:
        tools = await tool_service.get_tools(
            user=user,
            include_public=include_public,
            only_official=only_official,
            category_id=category_id,
            page=page,
            page_size=page_size,
            session=session
        )
        return RestResponse(data=tools)
    except CustomAgentException as e:
        logger.error(f"Error listing tools: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing tools: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/tools/{tool_id}", summary="Get Tool Details")
async def get_tool(
        tool_id: str,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Retrieve a tool's information by its ID.
    
    - **tool_id**: UUID of the tool to retrieve
    """
    try:
        return RestResponse(data=await tool_service.get_tool(tool_id, user, session))
    except CustomAgentException as e:
        logger.error(f"Error getting tool details: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting tool details: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.put("/tools/{tool_id}", summary="Update Tool")
async def update_tool(
        tool_id: str, 
        tool: ToolUpdate, 
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Update an existing API tool
    
    Parameters:
    - **tool_id**: ID of the tool to update
    - **name**: Optional new name for the tool
    - **origin**: Optional new API origin
    - **path**: Optional new API path
    - **method**: Optional new HTTP method
    - **parameters**: Optional new API parameters
    - **auth_config**: Optional new authentication configuration
    - **icon**: Optional new icon URL for the tool
    - **is_stream**: Optional boolean indicating if the API returns a stream response
    - **output_format**: Optional JSON configuration for formatting API output
    """
    try:
        tool = await tool_service.update_tool(
            tool_id=tool_id,
            user=user,
            session=session,
            name=tool.name,
            origin=tool.origin,
            path=tool.path,
            method=tool.method,
            parameters=tool.parameters,
            auth_config=tool.auth_config,
            icon=tool.icon,
            is_stream=tool.is_stream,
            output_format=tool.output_format
        )
        return RestResponse(data=tool)
    except CustomAgentException as e:
        logger.error(f"Error updating tool: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error updating tool: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.delete("/tools/{tool_id}", summary="Delete Tool")
async def delete_tool(
        tool_id: str, 
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Delete a tool by setting its is_deleted flag to True.
    
    - **tool_id**: ID of the tool to delete
    """
    try:
        await tool_service.delete_tool(tool_id, user, session)
        return RestResponse(data="ok")
    except CustomAgentException as e:
        logger.error(f"Error deleting tool: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error deleting tool: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/tools/{tool_id}/publish", summary="Publish Tool")
async def publish_tool(
        tool_id: str,
        is_public: bool = Query(True, description="Set tool as public"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Publish or unpublish a tool
    """
    try:
        await tool_service.publish_tool(tool_id, is_public, user, session)
        return RestResponse(data="ok")
    except CustomAgentException as e:
        logger.error(f"Error publishing tool: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error publishing tool: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/agents/{agent_id}/tools", summary="Assign Tools to Agent")
async def assign_tools(
        agent_id: str = Path(..., description="ID of the agent"),
        request: AgentToolsRequest = Body(..., description="Tool IDs to assign"),
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Assign multiple tools to an agent
    
    Parameters:
    - **agent_id**: ID of the agent
    - **tool_ids**: List of tool IDs to assign
    """
    try:
        await tool_service.assign_tools_to_agent(request.tool_ids, agent_id, user, session)
        return RestResponse(data="ok")
    except CustomAgentException as e:
        logger.error(f"Error assigning tools to agent: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error assigning tools to agent: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.delete("/agents/{agent_id}/tools", summary="Remove Tools from Agent")
async def remove_tools(
        agent_id: str,
        request: AgentToolsRequest,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Remove multiple tools from an agent
    
    - **agent_id**: ID of the agent
    - **tool_ids**: List of tool IDs to remove
    """
    try:
        await tool_service.remove_tools_from_agent(request.tool_ids, agent_id, user, session)
        return RestResponse(data="ok")
    except CustomAgentException as e:
        logger.error(f"Error removing tools from agent: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error removing tools from agent: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.get("/agents/{agent_id}/tools", summary="Get Agent Tools")
async def get_agent_tools(
        agent_id: str,
        user: dict = Depends(get_current_user),
        session: AsyncSession = Depends(get_db)
):
    """
    Get all tools associated with a specific agent
    """
    try:
        tools = await tool_service.get_tools_by_agent(
            agent_id=agent_id,
            user=user,
            session=session
        )
        return RestResponse(data=tools)
    except CustomAgentException as e:
        logger.error(f"Error getting agent tools: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting agent tools: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/tools/parse-openapi", summary="Parse OpenAPI Content")
async def parse_openapi(
    request: OpenAPIParseRequest,
    user: dict = Depends(get_current_user)
):
    """
    Parse OpenAPI content and return API information
    
    Parameters:
    - **request**: Request body containing OpenAPI specification content
    """
    try:
        api_info = await tool_service.parse_openapi_content(request.content)
        return RestResponse(data=api_info)
    except CustomAgentException as e:
        logger.error(f"Error parsing OpenAPI content: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error parsing OpenAPI content: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/tools/create-batch", summary="Create Tools in Batch", response_model=RestResponse[List[ToolModel]])
async def create_tools_batch(
    request: CreateToolsBatchRequest,
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Create multiple API tools in batch
    
    Parameters:
    - **tools**: List of API tool configurations, each containing name, host, path, method, parameters and auth_config
    """
    try:
        tools = await tool_service.create_tools_batch(
            tools=[tool.model_dump() for tool in request.tools],
            user=user,
            session=session
        )
        await session.commit()
        return RestResponse(data=tools)
    except CustomAgentException as e:
        logger.error(f"Error creating tools in batch: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating tools in batch: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/tools/upload-openapi", summary="Upload and Parse OpenAPI File")
async def upload_openapi(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    """
    Upload and parse OpenAPI file
    
    Parameters:
    - **file**: OpenAPI specification file (JSON or YAML format)
    """
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        api_info = await tool_service.parse_openapi_content(content_str)
        return RestResponse(data={
            "content": content_str,
            "api_info": api_info
        })
    except CustomAgentException as e:
        logger.error(f"Error uploading OpenAPI file: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error uploading OpenAPI file: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/tools/parse-mcp", summary="Parse MCP Service URL")
async def parse_mcp(
    mcp_url: str = Body(..., embed=True, description="MCP service URL to fetch tools from"),
    user: dict = Depends(get_current_user)
):
    """
    Parse MCP service URL and return tools information
    
    Parameters:
    - **mcp_url**: MCP service URL to fetch tools from (e.g., "http://0.0.0.0:8080/mcp/coin-api")
    """
    try:
        api_info = await tool_service.parse_mcp_content(mcp_url)
        return RestResponse(data={
            "url": mcp_url,
            "api_info": api_info
        })
    except CustomAgentException as e:
        logger.error(f"Error parsing MCP URL: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error parsing MCP URL: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )


@router.post("/tools/{tool_id}/debug", summary="Debug Tool API")
async def debug_tool(
    tool_id: str = Path(..., description="ID of the tool to debug"),
    input_params: Dict[str, Any] = Body({}, description="Input parameters for the API call"),
    session: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    """
    Debug a tool's API by sending a request with user-provided parameters
    
    This endpoint allows testing a tool's API with custom parameters to verify its functionality.
    It supports both streaming and non-streaming responses based on the tool's configuration.
    
    Parameters:
    - **tool_id**: ID of the tool to debug
    - **input_params**: JSON body containing parameters to use for the API call:
      - Query parameters as key-value pairs
      - Body data under the "body" key
      - Custom headers under the "headers" key as a nested object
    
    Example input_params:
    ```json
    {
        "query_param1": "value1",
        "query_param2": "value2",
        "body": {
            "key1": "value1",
            "key2": "value2"
        },
        "headers": {
            "X-Custom-Header": "custom-value"
        }
    }
    ```
    
    Returns the API response. For streaming APIs, a streaming response is returned.
    """
    try:
        # Get tool information
        tool = await tool_service.get_tool(tool_id, user, session)
        if not tool:
            raise CustomAgentException(
                ErrorCode.RESOURCE_NOT_FOUND,
                "Tool not found or no permission"
            )
        
        # Convert ToolModel to dict for the debug function
        tool_info = tool.model_dump()
        
        # Extract user headers if provided
        user_headers = input_params.pop('headers', None)
        
        # If tool is stream type, use StreamingResponse
        if tool.is_stream:
            response_stream = debug_tool_api(tool_info, input_params, user_headers)
            return StreamingResponse(
                content=response_stream, 
                media_type="text/event-stream"
            )
        else:
            # For non-streaming responses, collect the result
            async for response in debug_tool_api(tool_info, input_params, user_headers):
                return RestResponse(data=response)
    except CustomAgentException as e:
        logger.error(f"Error debugging tool: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error debugging tool: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )
