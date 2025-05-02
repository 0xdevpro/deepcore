import json
import logging
import re
from typing import Dict, List, Any, Optional

import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.sse import SseServerTransport
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.responses import Response

from agents.agent.entity.inner.think_output import ThinkOutput
from agents.agent.llm.llm_factory import LLMFactory
from agents.agent.prompts.mcp_prompt import generate_prompt
from agents.agent.tools.message_tool import send_message
from agents.common.config import SETTINGS
from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.models import MCPServer, MCPTool, MCPPrompt, MCPResource, MCPStore, App, Tool
from agents.services import tool_service
from agents.services.tool_service import get_tool, get_tools_by_ids
from agents.utils.http_client import async_client
from agents.utils.session import get_async_session_ctx

logger = logging.getLogger(__name__)

# MCP Store types
MCP_STORE_TYPE_LOCAL = "local"
MCP_STORE_TYPE_REMOTE = "remote"
MCP_STORE_TYPE_AGENT = "agent"
MCP_STORE_TYPE_TOOL = "tool"
MCP_STORE_TYPE_PROMPT = "prompt"
MCP_STORE_TYPE_RESOURCE = "resource"

async def create_mcp_server_from_tools(
    mcp_name: str,
    tool_ids: List[str],
    user: dict,
    session: AsyncSession,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an MCP server from a list of tool IDs
    
    Args:
        mcp_name: MCP server name
        tool_ids: List of tool IDs to expose as MCP interface
        user: Current user information
        session: Database session
        description: Optional MCP service description
        
    Returns:
        Dictionary containing service information
    """
    try:
        # Check if the name is already in use in database
        existing_server = await session.execute(
            select(MCPServer).where(MCPServer.name == mcp_name)
        )
        if existing_server.scalar_one_or_none():
            raise CustomAgentException(
                ErrorCode.RESOURCE_ALREADY_EXISTS,
                f"MCP server with name '{mcp_name}' already exists"
            )
            
        # Create MCP server database record instead of in-memory instance
        mcp_server = MCPServer(
            name=mcp_name,
            description=description or f"MCP server for {len(tool_ids)} tools",
            tenant_id=user["tenant_id"]
        )
        session.add(mcp_server)
        await session.flush()  # Ensure ID is assigned
            
        # Create tool associations
        for tool_id in tool_ids:
            mcp_tool = MCPTool(
                mcp_server_id=mcp_server.id,
                tool_id=tool_id
            )
            session.add(mcp_tool)
            
        # No longer need to register handlers or store server instance in memory
        
        return {
            "mcp_name": mcp_name,
            "mcp_id": mcp_server.id,
            "tool_count": len(tool_ids),
            "tool_ids": tool_ids,
            "url": f"{SETTINGS.API_BASE_URL}/mcp/{mcp_name}",
            "description": description or f"MCP server for {len(tool_ids)} tools"
        }
        
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error creating MCP server: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create MCP server: {str(e)}"
        )

async def _create_server_instance(mcp_name: str, user: dict) -> Server:
    """
    Dynamically create an MCP server instance for the given name
    
    Args:
        mcp_name: MCP server name
        user: User information for authorization
        
    Returns:
        Server instance configured with handlers
    """
    # Create a new server instance for this request
    server = Server(mcp_name)
    
    # Register tool list handler - now queries database directly
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools by querying database"""
        logger.info(f"MCP server '{mcp_name}' received list_tools request")
        # Get tools from database
        mcp_tools = []
        async with get_async_session_ctx() as db_session:
            # Get MCP server with associated tools
            db_server = await db_session.execute(
                select(MCPServer)
                .options(selectinload(MCPServer.tools).selectinload(MCPTool.tool))
                .where(MCPServer.name == mcp_name)
            )
            server_obj = db_server.scalar_one_or_none()
            
            if not server_obj:
                logger.warning(f"MCP server '{mcp_name}' not found")
                return []
            
            # Collect all tool IDs
            tool_ids = [str(mcp_tool.tool_id) for mcp_tool in server_obj.tools]
            
            # Batch retrieve tool objects
            tool_map = await get_tools_by_ids(tool_ids, user, db_session)
            
            # Create MCP tool format for each tool
            for mcp_tool in server_obj.tools:
                tool = tool_map.get(str(mcp_tool.tool_id))
                if not tool:
                    continue
                
                # Convert tool parameters to MCP input schema
                input_schema = _convert_parameters_to_schema(tool.parameters)
                
                mcp_tool = types.Tool(
                    name=tool.name,
                    description=tool.description or f"Tool {tool.name}",
                    inputSchema=input_schema
                )
                mcp_tools.append(mcp_tool)
        
        logger.info(f"MCP server '{mcp_name}' returned {len(mcp_tools)} tools")
        return mcp_tools
        
    # Register tool call handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent]:
        """Handle tool execution request"""
        logger.info(f"MCP server '{mcp_name}' received tool call request: {name}")
        # Get tools from database
        async with get_async_session_ctx() as db_session:
            # Get MCP server with associated tools
            db_server = await db_session.execute(
                select(MCPServer)
                .options(selectinload(MCPServer.tools).selectinload(MCPTool.tool))
                .where(MCPServer.name == mcp_name)
            )
            server_obj = db_server.scalar_one_or_none()
            
            if not server_obj:
                return [types.TextContent(type="text", text=f"MCP server not found: {mcp_name}")]
            
            # Collect all tool IDs
            tool_ids = [str(mcp_tool.tool_id) for mcp_tool in server_obj.tools]
            
            # Batch retrieve tool objects
            tool_map = await get_tools_by_ids(tool_ids, user, db_session)
            
            # Find matching tool
            matching_tool = None
            for mcp_tool in server_obj.tools:
                tool = tool_map.get(str(mcp_tool.tool_id))
                if tool and tool.name == name:
                    matching_tool = tool
                    break
        
        if not matching_tool:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            
        try:
            # Prepare request parameters
            params = {}
            headers = {}
            json_data = {}
            if arguments:
                if matching_tool.method == "GET":
                    params = arguments
                else:
                    json_data = arguments
                    headers = {'Content-Type': 'application/json'}
            # Execute API call
            resp = async_client.request(
                method=matching_tool.method,
                base_url=matching_tool.origin,
                path=matching_tool.path,
                params=params,
                headers=headers,
                json_data=json_data,
                auth_config=matching_tool.auth_config,
                stream=False
            )
            result = ""
            async for data in resp:
                result = data
            
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
            
        except Exception as e:
            logger.error(f"Error calling tool {name}: {str(e)}", exc_info=True)
            return [types.TextContent(type="text", text=f"Error calling tool: {str(e)}")]
    
    # Add Prompts support
    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        """List available prompts"""
        prompts = []
        
        # Get prompts from database
        async with get_async_session_ctx() as db_session:
            # Get MCP server
            db_server = await db_session.execute(
                select(MCPServer)
                .options(
                    selectinload(MCPServer.tools).selectinload(MCPTool.tool),
                    selectinload(MCPServer.prompts)
                )
                .where(MCPServer.name == mcp_name)
            )
            server_obj = db_server.scalar_one_or_none()
            
            if not server_obj:
                return []
            
            # Return default prompts for this MCP server
            prompts.append(types.Prompt(
                name=f"{mcp_name}-help", 
                description=f"Get help about how to use {mcp_name} tools",
                arguments=[]
            ))
            
            # Add stored prompts
            for prompt in server_obj.prompts:
                prompt_args = []
                if prompt.arguments:
                    for arg in prompt.arguments:
                        prompt_args.append(types.PromptArgument(
                            name=arg.get("name"),
                            description=arg.get("description", ""),
                            required=arg.get("required", False)
                        ))
                
                prompts.append(types.Prompt(
                    name=prompt.name,
                    description=prompt.description,
                    arguments=prompt_args
                ))
            
            # Add tool-specific prompts
            for mcp_tool in server_obj.tools:
                tool = mcp_tool.tool
                # Get tool details
                tool_data = await get_tool(str(tool.id), user, db_session)
                
                # Create a prompt for each tool with its parameters as arguments
                prompt_args = []
                
                if tool_data.parameters.get('body'):
                    # For body parameters, create a single argument for the JSON body
                    prompt_args.append(types.PromptArgument(
                        name="body", 
                        description="JSON body for the request",
                        required=True
                    ))
                else:
                    # For other parameter types, create an argument for each required parameter
                    for param_type in ['query', 'path', 'header']:
                        for param in tool_data.parameters.get(param_type, []):
                            if param.get('required'):
                                prompt_args.append(types.PromptArgument(
                                    name=param.get('name'),
                                    description=param.get('description') or f"{param_type} parameter",
                                    required=True
                                ))
                
                prompts.append(types.Prompt(
                    name=f"use-{tool_data.name}",
                    description=f"Create a prompt to use the {tool_data.name} tool",
                    arguments=prompt_args
                ))
        
        return prompts
        
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        """Get a specific prompt template"""
        async with get_async_session_ctx() as db_session:
            # Get MCP server
            db_server = await db_session.execute(
                select(MCPServer)
                .options(
                    selectinload(MCPServer.tools).selectinload(MCPTool.tool),
                    selectinload(MCPServer.prompts)
                )
                .where(MCPServer.name == mcp_name)
            )
            server_obj = db_server.scalar_one_or_none()
            
            if not server_obj:
                raise ValueError(f"MCP server not found: {mcp_name}")
            
            # Collect all tool IDs
            tool_ids = [str(mcp_tool.tool_id) for mcp_tool in server_obj.tools]
            
            # Batch retrieve tool objects
            tool_map = await get_tools_by_ids(tool_ids, user, db_session)
            
            # Check for custom prompt
            for prompt in server_obj.prompts:
                if prompt.name == name:
                    return types.GetPromptResult(
                        description=prompt.description,
                        messages=[
                            types.PromptMessage(
                                role="user",
                                content=types.TextContent(
                                    type="text", 
                                    text=prompt.template.format(**(arguments or {}))
                                )
                            )
                        ]
                    )
            
            # Handle the help prompt
            if name == f"{mcp_name}-help":
                tool_descriptions = []
                for mcp_tool in server_obj.tools:
                    tool = tool_map.get(str(mcp_tool.tool_id))
                    if not tool:
                        continue
                    tool_descriptions.append(f"- {tool.name}: {tool.description or 'No description available'}")
                
                tool_descriptions_text = "\n".join(tool_descriptions)
                
                return types.GetPromptResult(
                    description=f"Help information for {mcp_name} tools",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text", 
                                text=f"I need help with the {mcp_name} tools. Please provide information about the available tools and how to use them."
                            )
                        ),
                        types.PromptMessage(
                            role="assistant",
                            content=types.TextContent(
                                type="text", 
                                text=f"I'd be happy to help you with the {mcp_name} tools. Here are the available tools:\n\n{tool_descriptions_text}\n\nTo use these tools, you can call them directly or ask me to help you formulate the right parameters for each tool."
                            )
                        )
                    ]
                )
                
            # Handle tool-specific prompts
            if name.startswith("use-"):
                tool_name = name[4:]  # Remove 'use-' prefix
                
                # Find the matching tool
                matching_tool = None
                for mcp_tool in server_obj.tools:
                    tool = tool_map.get(str(mcp_tool.tool_id))
                    if tool and tool.name == tool_name:
                        matching_tool = tool
                        break
                        
                if not matching_tool:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                # Format parameters to show in the prompt
                params_text = ""
                if arguments:
                    params_text = "\n".join([f"- {k}: {v}" for k, v in arguments.items()])
                
                return types.GetPromptResult(
                    description=f"Prompt to use the {tool_name} tool",
                    messages=[
                        types.PromptMessage(
                            role="user",
                            content=types.TextContent(
                                type="text", 
                                text=f"I want to use the {tool_name} tool with the following parameters:\n{params_text}\n\nPlease help me format this correctly."
                            )
                        ),
                        types.PromptMessage(
                            role="assistant",
                            content=types.TextContent(
                                type="text", 
                                text=f"I'll help you use the {tool_name} tool. Based on the parameters you've provided, here's how you can call it:\n\n```json\n{json.dumps(arguments or {}, indent=2)}\n```\n\nYou can use this with the tool by asking me to 'Call the {tool_name} tool with these parameters.'"
                            )
                        )
                    ]
                )
            
        raise ValueError(f"Unknown prompt: {name}")
        
    # Add Resources support
    @server.list_resources()
    async def handle_list_resources() -> list[str]:
        """List available resources"""
        resources = []
        
        async with get_async_session_ctx() as db_session:
            # Get MCP server with resources and tools
            db_server = await db_session.execute(
                select(MCPServer)
                .options(
                    selectinload(MCPServer.tools).selectinload(MCPTool.tool),
                    selectinload(MCPServer.resources)
                )
                .where(MCPServer.name == mcp_name)
            )
            server_obj = db_server.scalar_one_or_none()
            
            if not server_obj:
                return []
            
            # Add documentation resources
            resources.append(f"doc://{mcp_name}/overview")
            
            # Add stored resources
            for resource in server_obj.resources:
                resources.append(resource.uri)
            
            # Add tool-specific resources
            for mcp_tool in server_obj.tools:
                tool = mcp_tool.tool
                # Get tool details
                tool_data = await get_tool(str(tool.id), user, db_session)
                resources.append(f"doc://{mcp_name}/tools/{tool_data.name}")
        
        return resources
        
    @server.read_resource()
    async def handle_read_resource(uri: str) -> tuple[str, str]:
        """Read a specific resource"""
        async with get_async_session_ctx() as db_session:
            # Get MCP server with resources and tools
            db_server = await db_session.execute(
                select(MCPServer)
                .options(
                    selectinload(MCPServer.tools).selectinload(MCPTool.tool),
                    selectinload(MCPServer.resources)
                )
                .where(MCPServer.name == mcp_name)
            )
            server_obj = db_server.scalar_one_or_none()
            
            if not server_obj:
                raise ValueError(f"MCP server not found: {mcp_name}")
            
            # Collect all tool IDs
            tool_ids = [str(mcp_tool.tool_id) for mcp_tool in server_obj.tools]
            
            # Batch retrieve tool objects
            tool_map = await get_tools_by_ids(tool_ids, user, db_session)
            
            # Check for stored resources
            for resource in server_obj.resources:
                if resource.uri == uri:
                    return resource.content, resource.mime_type
            
            # Handle documentation resources
            if uri.startswith(f"doc://{mcp_name}/overview"):
                # Create an overview of all tools
                content = f"# {mcp_name} Tools Overview\n\n"
                content += f"This MCP server provides {len(server_obj.tools)} tools:\n\n"
                
                for mcp_tool in server_obj.tools:
                    tool = tool_map.get(str(mcp_tool.tool_id))
                    if not tool:
                        continue
                    
                    content += f"## {tool.name}\n\n"
                    content += f"{tool.description or 'No description available'}\n\n"
                    content += "### Parameters\n\n"
                    
                    if tool.parameters.get('body'):
                        content += "This tool accepts a JSON body with the following schema:\n\n"
                        content += f"```json\n{json.dumps(tool.parameters['body'], indent=2)}\n```\n\n"
                    else:
                        for param_type in ['query', 'path', 'header']:
                            if tool.parameters.get(param_type):
                                content += f"#### {param_type.capitalize()} Parameters\n\n"
                                for param in tool.parameters[param_type]:
                                    content += f"- **{param.get('name')}**: {param.get('description') or 'No description'}"
                                    if param.get('required'):
                                        content += " (Required)"
                                    content += "\n"
                                content += "\n"
                
                return content, "text/markdown"
                
            if uri.startswith(f"doc://{mcp_name}/tools/"):
                # Extract tool name from URI
                tool_name = uri.split('/')[-1]
                
                # Find the matching tool
                matching_tool = None
                for mcp_tool in server_obj.tools:
                    tool = tool_map.get(str(mcp_tool.tool_id))
                    if tool and tool.name == tool_name:
                        matching_tool = tool
                        break
                        
                if not matching_tool:
                    raise ValueError(f"Unknown tool: {tool_name}")
                    
                # Create detailed documentation for this tool
                content = f"# {tool_name}\n\n"
                content += f"{matching_tool.description or 'No description available'}\n\n"
                content += f"Method: {matching_tool.method}\n"
                content += f"Endpoint: {matching_tool.origin}{matching_tool.path}\n\n"
                content += "## Parameters\n\n"
                
                if matching_tool.parameters.get('body'):
                    content += "This tool accepts a JSON body with the following schema:\n\n"
                    content += f"```json\n{json.dumps(matching_tool.parameters['body'], indent=2)}\n```\n\n"
                else:
                    for param_type in ['query', 'path', 'header']:
                        if matching_tool.parameters.get(param_type):
                            content += f"### {param_type.capitalize()} Parameters\n\n"
                            for param in matching_tool.parameters[param_type]:
                                content += f"- **{param.get('name')}**: {param.get('description') or 'No description'}"
                                if param.get('required'):
                                    content += " (Required)"
                                if param.get('default'):
                                    content += f" (Default: {param.get('default')})"
                                content += "\n"
                            content += "\n"
                    
                content += "## Example Usage\n\n"
                content += "```python\n"
                content += f"result = await client.call_tool(\"{tool_name}\", {{\n"
                
                example_params = {}
                if matching_tool.parameters.get('body'):
                    example_params = {"param1": "value1", "param2": "value2"}
                else:
                    for param_type in ['query', 'path', 'header']:
                        for param in matching_tool.parameters.get(param_type, []):
                            example_params[param.get('name')] = f"example_{param.get('name')}"
                            
                content += f"    # Example parameters\n"
                for k, v in example_params.items():
                    content += f"    \"{k}\": \"{v}\",\n"
                content += "}})\n"
                content += "```\n"
                
                return content, "text/markdown"
            
        raise ValueError(f"Unknown resource: {uri}")
    
    return server

def get_main_app():
    """
    Get the main application containing all MCP server routes
    
    Returns:
        ASGI application function for handling all MCP routes
    """
    async def dynamic_mcp_handler(scope, receive, send):
        """ASGI handler for MCP requests"""
        # Get MCP server name from URL
        path = scope["path"]
        path_parts = path.split('/')
        
        if len(path_parts) < 3:
            response = Response("Invalid MCP URL", status_code=400)
            await response(scope, receive, send)
            return
            
        # Create request object
        from starlette.requests import Request
        request = Request(scope=scope, receive=receive)
        
        # Extract user information (in a real implementation, this would be obtained from the request)
        user = {"tenant_id": "default"}
        
        # Check if MCP server exists
        mcp_name = path_parts[2]
        
        async with get_async_session_ctx() as session:
            result = await session.execute(
                select(MCPServer).where(
                    MCPServer.name == mcp_name,
                    MCPServer.is_active == True
                )
            )
            if not result.scalar_one_or_none():
                response = Response(f"MCP server '{mcp_name}' not found", status_code=404)
                await response(scope, receive, send)
                return
        
        # Create dynamic server instance
        server = await _create_server_instance(mcp_name, user)
        
        # Create SSE transmission - use empty path prefix
        sse = SseServerTransport("")
        
        # Directly handle SSE connection
        async with sse.connect_sse(scope, receive, send) as streams:
            await server.run(
                streams[0],
                streams[1],
                InitializationOptions(
                    server_name=mcp_name,
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    
    return dynamic_mcp_handler

def _convert_parameters_to_schema(parameters: Dict) -> Dict:
    """
    Convert tool parameters to MCP input schema
    
    Args:
        parameters: Tool parameter definition
        
    Returns:
        MCP-compliant input schema
    """
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Handle body parameters
    if parameters.get('body'):
        return parameters['body']
    
    # Handle other parameter types (query, path, header)
    for param_type in ['query', 'path', 'header']:
        for param in parameters.get(param_type, []):
            name = param.get('name')
            if not name:
                continue
                
            prop = {
                "type": param.get('type', 'string'),
                "description": param.get('description', f"{param_type} parameter")
            }
            
            # Add default value if available
            if 'default' in param:
                prop['default'] = param['default']
                
            schema['properties'][name] = prop
            
            # Add required parameters
            if param.get('required'):
                schema['required'].append(name)
    
    return schema

async def add_prompt_template(
    mcp_name: str,
    prompt_name: str,
    description: str,
    arguments: List[Dict[str, Any]],
    template: str,
    session: Optional[AsyncSession] = None
) -> bool:
    """
    Add a prompt template to an MCP server
    
    Args:
        mcp_name: MCP server name
        prompt_name: Prompt template name
        description: Prompt description
        arguments: List of prompt arguments
        template: Prompt template text
        session: Optional database session
        
    Returns:
        Success status
    """
    # If a session is provided, use it directly
    if session:
        return await _add_prompt_template_impl(
            mcp_name, prompt_name, description, arguments, template, session
        )
    
    # Otherwise use context manager to ensure session is properly closed
    try:
        async with get_async_session_ctx() as managed_session:
            return await _add_prompt_template_impl(
                mcp_name, prompt_name, description, arguments, template, managed_session
            )
    except Exception as e:
        logger.error(f"Error adding prompt template: {e}", exc_info=True)
        return False

async def _add_prompt_template_impl(
    mcp_name: str,
    prompt_name: str,
    description: str,
    arguments: List[Dict[str, Any]],
    template: str,
    session: AsyncSession
) -> bool:
    """Implementation of add_prompt_template with an existing session"""
    try:
        # Get the MCP server
        db_server = await session.execute(
            select(MCPServer).where(MCPServer.name == mcp_name)
        )
        server_obj = db_server.scalar_one_or_none()
        
        if not server_obj:
            return False
        
        # Check if prompt with the same name already exists
        existing_prompt = await session.execute(
            select(MCPPrompt).where(
                MCPPrompt.mcp_server_id == server_obj.id,
                MCPPrompt.name == prompt_name
            )
        )
        existing = existing_prompt.scalar_one_or_none()
        
        if existing:
            # Update existing prompt
            existing.description = description
            existing.arguments = arguments
            existing.template = template
        else:
            # Create new prompt
            prompt = MCPPrompt(
                mcp_server_id=server_obj.id,
                name=prompt_name,
                description=description,
                arguments=arguments,
                template=template
            )
            session.add(prompt)
        
        await session.commit()
        return True
    except Exception as e:
        logger.error(f"Error in add_prompt_template implementation: {e}", exc_info=True)
        await session.rollback()
        return False

async def add_resource(
    mcp_name: str,
    resource_uri: str,
    content: str,
    mime_type: str = "text/plain",
    session: Optional[AsyncSession] = None
) -> bool:
    """
    Add a resource to an MCP server
    
    Args:
        mcp_name: MCP server name
        resource_uri: Resource URI
        content: Resource content
        mime_type: MIME type of the resource, default is text/plain
        session: Optional database session
        
    Returns:
        Success status
    """
    # If a session is provided, use it directly
    if session:
        return await _add_resource_impl(mcp_name, resource_uri, content, mime_type, session)
    
    # Otherwise use context manager to ensure session is properly closed
    try:
        async with get_async_session_ctx() as managed_session:
            return await _add_resource_impl(mcp_name, resource_uri, content, mime_type, managed_session)
    except Exception as e:
        logger.error(f"Error adding resource: {e}", exc_info=True)
        return False

async def _add_resource_impl(
    mcp_name: str,
    resource_uri: str,
    content: str,
    mime_type: str,
    session: AsyncSession
) -> bool:
    """Implementation of add_resource with an existing session"""
    try:
        # Get the MCP server
        db_server = await session.execute(
            select(MCPServer).where(MCPServer.name == mcp_name)
        )
        server_obj = db_server.scalar_one_or_none()
        
        if not server_obj:
            return False
        
        # Check if resource with the same URI already exists
        existing_resource = await session.execute(
            select(MCPResource).where(
                MCPResource.mcp_server_id == server_obj.id,
                MCPResource.uri == resource_uri
            )
        )
        existing = existing_resource.scalar_one_or_none()
        
        if existing:
            # Update existing resource
            existing.content = content
            existing.mime_type = mime_type
        else:
            # Create new resource
            resource = MCPResource(
                mcp_server_id=server_obj.id,
                uri=resource_uri,
                content=content,
                mime_type=mime_type
            )
            session.add(resource)
        
        await session.commit()
        return True
    except Exception as e:
        logger.error(f"Error in add_resource implementation: {e}", exc_info=True)
        await session.rollback()
        return False

async def get_registered_mcp_servers(session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
    """
    Get all registered MCP servers
    
    Args:
        session: Optional database session
        
    Returns:
        List of registered MCP servers
    """
    # If a session is provided, use it directly
    if session:
        return await _get_registered_mcp_servers_impl(session)
    
    # Otherwise use context manager to ensure session is properly closed
    try:
        async with get_async_session_ctx() as managed_session:
            return await _get_registered_mcp_servers_impl(managed_session)
    except Exception as e:
        logger.error(f"Error getting registered MCP servers: {e}", exc_info=True)
        return []

async def _get_registered_mcp_servers_impl(session: AsyncSession) -> List[Dict[str, Any]]:
    """Implementation of get_registered_mcp_servers with an existing session"""
    try:
        # Get all MCP servers
        db_servers = await session.execute(select(MCPServer))
        servers = db_servers.scalars().all()
        
        result = []
        for server in servers:
            result.append({
                "id": server.id,
                "name": server.name,
                "description": server.description,
                "created_at": server.created_at,
                "updated_at": server.updated_at
            })
        
        return result
    except Exception as e:
        logger.error(f"Error in get_registered_mcp_servers implementation: {e}", exc_info=True)
        return []

async def delete_mcp_server(mcp_name: str, session: Optional[AsyncSession] = None) -> bool:
    """
    Delete an MCP server
    
    Args:
        mcp_name: MCP server name to delete
        session: Optional database session
        
    Returns:
        Success status
    """
    # If a session is provided, use it directly
    if session:
        return await _delete_mcp_server_impl(mcp_name, session)
    
    # Otherwise use context manager to ensure session is properly closed
    try:
        async with get_async_session_ctx() as managed_session:
            return await _delete_mcp_server_impl(mcp_name, managed_session)
    except Exception as e:
        logger.error(f"Error deleting MCP server: {e}", exc_info=True)
        return False

async def _delete_mcp_server_impl(mcp_name: str, session: AsyncSession) -> bool:
    """Implementation of delete_mcp_server with an existing session"""
    try:
        # Get the MCP server
        db_server = await session.execute(
            select(MCPServer).where(MCPServer.name == mcp_name)
        )
        server_obj = db_server.scalar_one_or_none()
        
        if not server_obj:
            return False
        
        # Delete the server
        await session.delete(server_obj)
        await session.commit()
        return True
    except Exception as e:
        logger.error(f"Error in delete_mcp_server implementation: {e}", exc_info=True)
        await session.rollback()
        return False

async def get_tool_mcp_mapping(session: Optional[AsyncSession] = None) -> Dict[str, str]:
    """
    Get the mapping from tool ID to MCP server name
    
    Args:
        session: Optional database session
        
    Returns:
        Mapping dictionary
    """
    # If a session is provided, use it directly
    if session:
        return await _get_tool_mcp_mapping_impl(session)
    
    # Otherwise use context manager to ensure session is properly closed
    try:
        async with get_async_session_ctx() as managed_session:
            return await _get_tool_mcp_mapping_impl(managed_session)
    except Exception as e:
        logger.error(f"Error getting tool MCP mapping: {e}", exc_info=True)
        return {}

async def _get_tool_mcp_mapping_impl(session: AsyncSession) -> Dict[str, str]:
    """Implementation of get_tool_mcp_mapping with an existing session"""
    try:
        # Get all MCP server tools
        result = await session.execute(
            select(MCPTool, MCPServer)
            .join(MCPServer, MCPTool.mcp_server_id == MCPServer.id)
        )
        server_tools = result.all()
        
        mapping = {}
        for server_tool, server in server_tools:
            mapping[server_tool.tool_id] = server.name
        
        return mapping
    except Exception as e:
        logger.error(f"Error in get_tool_mcp_mapping implementation: {e}", exc_info=True)
        return {}

# No longer need to initialize MCP servers on startup
# Instead, servers are created dynamically for each request 

def get_coin_api_mcp_service(user):
    """
    Create an MCP server instance for the built-in coin-api service
    
    Args:
        user: User information for authorization
        
    Returns:
        Server instance configured with handlers
    """
    from agents.agent.mcp import coin_api_mcp  # Local import to avoid circular dependency
    
    logger.info("Creating coin-api MCP service instance")
    return coin_api_mcp.server.server

async def create_mcp_store(
    store_name: str,
    store_type: str,
    user: dict,
    session: AsyncSession,
    icon: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    content: Optional[str] = None,
    author: Optional[str] = None,
    github_url: Optional[str] = None,
    is_public: Optional[bool] = False,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new MCP Store
    
    Args:
        store_name: Name of the store
        store_type: Type of the store
        user: Current user information
        session: Database session
        icon: Store icon URL (optional)
        description: Store description (optional)
        tags: List of store tags (optional)
        content: Store content (optional)
        author: Author name (optional)
        github_url: GitHub URL for the store (optional)
        is_public: Whether the store is public (optional)
        
    Returns:
        Dictionary containing store information
    """
    try:
        # Check if the name is already in use
        existing_store = await session.execute(
            select(MCPStore).where(MCPStore.name == store_name)
        )
        if existing_store.scalar_one_or_none():
            raise CustomAgentException(
                ErrorCode.RESOURCE_ALREADY_EXISTS,
                f"MCP store with name '{store_name}' already exists"
            )
            
        # Create MCP Store record
        mcp_store = MCPStore(
            name=store_name,
            icon=icon,
            description=description,
            store_type=store_type,
            tags=tags,
            content=content,
            author=author,
            creator_id=user["user_id"],
            github_url=github_url,
            tenant_id=user["tenant_id"],
            is_public=is_public,
            agent_id=agent_id
        )
        session.add(mcp_store)
        await session.commit()
        await session.refresh(mcp_store)
            
        return mcp_store.to_dict()
        
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error creating MCP store: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create MCP store: {str(e)}"
        )

async def get_registered_mcp_stores(
    page: int,
    page_size: int,
    keyword: Optional[str] = None,
    store_type: Optional[str] = None,
    is_public: Optional[bool] = None,
    session: Optional[AsyncSession] = None,
    user: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Get registered MCP Stores with pagination and search support
    
    Args:
        page: Page number
        page_size: Number of items per page
        keyword: Search keyword (optional)
        store_type: Filter by store type (optional)
        is_public: Filter by public status (optional)
        session: Optional database session
        user: Current user information (optional)
        
    Returns:
        Dictionary containing pagination info and store list
    """
    if session is None:
        async with get_async_session_ctx() as session:
            return await _get_registered_mcp_stores_impl(
                page, page_size, keyword, store_type, is_public, session, user
            )
    return await _get_registered_mcp_stores_impl(
        page, page_size, keyword, store_type, is_public, session, user
    )

async def _get_registered_mcp_stores_impl(
    page: int,
    page_size: int,
    keyword: Optional[str],
    store_type: Optional[str],
    is_public: Optional[bool],
    session: AsyncSession,
    user: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Implementation of get_registered_mcp_stores
    
    Args:
        page: Page number
        page_size: Number of items per page
        keyword: Search keyword (optional)
        store_type: Filter by store type (optional)
        is_public: Filter by public status (optional)
        session: Database session
        user: Current user information (optional)
        
    Returns:
        Dictionary containing pagination info and store list
    """
    try:
        # Build query
        query = select(MCPStore)
        
        # Add filters
        if keyword:
            query = query.where(
                or_(
                    MCPStore.name.ilike(f"%{keyword}%"),
                    MCPStore.description.ilike(f"%{keyword}%")
                )
            )
        
        if store_type:
            query = query.where(MCPStore.store_type == store_type)
        if is_public is not None:
            query = query.where(MCPStore.is_public == is_public)
            
        # Users can view their own stores or public stores
        if user and isinstance(user, dict):
            user_id = user.get("tenant_id")
            if user_id:
                query = query.where(
                    or_(
                        MCPStore.tenant_id == user_id,
                        MCPStore.is_public == True
                    )
                )
            else:
                # If user ID is not available, only show public stores
                query = query.where(MCPStore.is_public == True)
            
        # Calculate total count
        total_query = select(func.count()).select_from(query.subquery())
        total = await session.scalar(total_query)
        
        # Add pagination
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        # Execute query
        result = await session.execute(query)
        stores = result.scalars().all()
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [store.to_dict() for store in stores]
        }
        
    except Exception as e:
        logger.error(f"Error getting registered MCP stores: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get registered MCP stores: {str(e)}"
        )

async def _generate_tool_store_content(
    store: MCPStore,
    session: AsyncSession,
    user: Optional[dict] = None
) -> str:
    """
    Generate content for tool type store
    
    Args:
        store: MCP Store instance
        session: Database session
        user: Current user information (optional)
        
    Returns:
        Generated markdown content
    """
    # Get MCP server associated with this store
    mcp_server = await session.execute(
        select(MCPServer).where(MCPServer.id == store.agent_id)
    )
    mcp_server = mcp_server.scalar_one_or_none()
    
    if not mcp_server:
        return ""
        
    # Get tools associated with this MCP server
    tools = await session.execute(
        select(Tool).join(MCPTool).where(
            MCPTool.mcp_server_id == mcp_server.id
        )
    )
    tools = tools.scalars().all()
    
    # Get API Key based on user authentication
    api_key = "your-api-key"
    if user:
        try:
            from agents.services.open_service import get_or_create_credentials
            credentials = await get_or_create_credentials(user, session)
            if credentials and credentials.get("token"):
                api_key = credentials["token"]
        except Exception as e:
            logger.warning(f"Failed to get API Key for user: {str(e)}")
    
    # Generate markdown content with tool details
    content = f"""# DeepCore MCP Tools Guide
![DeepCore Logo](http://deepcore.top/deepcore.png)

## Introduction

This MCP server provides {len(tools)} tools for various tasks. This guide will explain how to use these tools through the MCP interface.

## Available Tools

"""
    
    # Add tool details
    for tool in tools:
        content += f"### {tool.name}\n\n"
        content += f"{tool.description or 'No description available'}\n\n"
        content += f"Endpoint: {tool.origin}{tool.path}\n"
        content += f"Method: {tool.method}\n\n"
        
        if tool.parameters:
            content += "#### Parameters\n\n"
            for param_type in ['query', 'path', 'header']:
                if tool.parameters.get(param_type):
                    content += f"##### {param_type.capitalize()} Parameters\n\n"
                    for param in tool.parameters[param_type]:
                        content += f"- **{param.get('name')}**: {param.get('description') or 'No description'}"
                        if param.get('required'):
                            content += " (Required)"
                        content += "\n"
                    content += "\n"
            
            if tool.parameters.get('body'):
                content += "##### Body Parameters\n\n"
                content += f"```json\n{json.dumps(tool.parameters['body'], indent=2)}\n```\n\n"
    
    content += f"""## Quick Start

### 1. Get API Key

Step 1: Log in to the DeepCore Platform
Visit https://deepcore.top and log in to your account.

Step 2: Get Your API Key
From your account settings, get your API key.

### 2. Configure MCP Client

Choose the appropriate configuration method based on your MCP client:

#### Claude Desktop

1. Open Claude Desktop Settings
2. Go to `Developer > Edit Config`
3. Add the following configuration to `claude_desktop_config.json`:

```json
{{
  "mcpServers": {{
    "deepcore-tools": {{
      "url": "{SETTINGS.API_BASE_URL}/mcp/{store.name}?api-key={api_key}"
    }}
  }}
}}
```

#### Cursor

1. Open Cursor Settings
2. Go to `Preferences > Cursor Settings > MCP`
3. Click `Add new global MCP Server`
4. Add the following configuration:

```json
{{
  "mcpServers": {{
    "deepcore-tools": {{
      "url": "{SETTINGS.API_BASE_URL}/mcp/{store.name}?api-key={api_key}"
    }}
  }}
}}
```

### 3. Usage Example

After configuration, you can directly use the tools in conversations. For example:

```
User: Help me use the tool to check the weather
AI: I'll help you use the weather tool with the appropriate parameters.
```

## Technical Support

If you encounter any issues during use, you can get support through the following channels:

1. Visit DeepCore Documentation Center: https://docs.deepcore.top
2. Join DeepCore Community: https://community.deepcore.top
3. Contact Technical Support: support@deepcore.top"""
    
    return content

async def _get_mcp_store_detail_impl(
    store_id: int,
    session: AsyncSession,
    user: Optional[dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Implementation of get_mcp_store_detail
    
    Args:
        store_id: ID of the store
        session: Database session
        user: Current user information (optional)
        
    Returns:
        Store details if found, None otherwise
    """
    try:
        # Query store
        result = await session.execute(
            select(MCPStore)
            .where(MCPStore.id == store_id)
        )
        store = result.scalar_one_or_none()
        
        if not store:
            return None
            
        # Generate content based on store type
        if store.store_type == MCP_STORE_TYPE_AGENT and store.agent_id:
            content = await get_store_content(str(store_id), session, user)
            if content:
                store.content = content
        elif store.store_type == MCP_STORE_TYPE_TOOL:
            content = await _generate_tool_store_content(store, session, user)
            if content:
                store.content = content
            
        return store.to_dict()
        
    except Exception as e:
        logger.error(f"Error getting MCP store detail: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to get MCP store detail: {str(e)}"
        )

async def get_mcp_store_detail(
    store_id: int,
    session: Optional[AsyncSession] = None,
    user: Optional[dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific MCP Store
    
    Args:
        store_id: ID of the store
        session: Optional database session
        user: Current user information (optional)
        
    Returns:
        Store details if found, None otherwise
    """
    if session is None:
        async with get_async_session_ctx() as session:
            return await _get_mcp_store_detail_impl(store_id, session, user)
    return await _get_mcp_store_detail_impl(store_id, session, user)

async def create_agent_store(
    agent_id: str,
    user: dict,
    session: AsyncSession,
    name: str,
    icon: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    author: Optional[str] = None,
    github_url: Optional[str] = None,
    is_public: bool = False
) -> Dict[str, Any]:
    """
    Create or update an agent store in the database
    
    Args:
        agent_id: ID of the agent to create store for
        user: Current user information
        session: Database session
        name: Name of the store
        icon: Optional icon URL
        description: Optional description
        tags: Optional list of tags
        author: Optional author name
        github_url: Optional GitHub repository URL
        is_public: Whether the store is public
        
    Returns:
        Dictionary containing store information
    """
    try:
        # Check if store with same agent_id already exists
        existing_store = await session.execute(
            select(MCPStore).where(MCPStore.agent_id == agent_id)
        )
        store = existing_store.scalar_one_or_none()
        
        # Check if there's another store with the same name (excluding current store if exists)
        name_check_query = select(MCPStore).where(MCPStore.name == name)
        if store:
            name_check_query = name_check_query.where(MCPStore.id != store.id)
        name_check_result = await session.execute(name_check_query)
        if name_check_result.scalar_one_or_none():
            raise CustomAgentException(
                ErrorCode.INVALID_PARAMETERS,
                f"Store with name '{name}' already exists"
            )
        
        if store:
            # Update existing store
            store.name = name
            store.icon = icon
            store.description = description
            store.tags = tags or []
            store.author = author or user.get("wallet_address", "")
            store.github_url = github_url
            store.is_public = is_public
            store.tenant_id = user["tenant_id"]
        else:
            # Create new store
            store = MCPStore(
                name=name,
                icon=icon,
                description=description,
                store_type="agent",
                tags=tags or [],
                content="",
                creator_id=user.get("id", 0),  # Use the integer user ID
                author=author or user.get("wallet_address", ""),
                github_url=github_url,
                tenant_id=user["tenant_id"],
                is_public=is_public,
                agent_id=agent_id
            )
            session.add(store)
        
        await session.commit()
        await session.refresh(store)
        
        return store.to_dict()
        
    except CustomAgentException:
        raise
    except Exception as e:
        logger.error(f"Error creating/updating agent store: {str(e)}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create/update agent store: {str(e)}"
        )

async def get_stores_by_name(name: str, session: AsyncSession):
    """
    Get stores by name
    
    Args:
        name: Store name to search for
        session: Database session
        
    Returns:
        List of stores with matching name
    """
    result = await session.execute(
        select(MCPStore).where(MCPStore.name == name)
    )
    stores = result.scalars().all()
    return stores

async def get_store_content(store_id: str, session: AsyncSession, user: Optional[dict] = None) -> str:
    """
    Get store content with special handling for agent stores
    
    Args:
        store_id: ID of the store
        session: Database session
        user: Current user information (optional)
        
    Returns:
        Formatted content string
    """
    result = await session.execute(
        select(MCPStore).where(MCPStore.id == store_id)
    )
    store = result.scalar_one_or_none()
    
    if not store:
        return ""
        
    if store.store_type == MCP_STORE_TYPE_AGENT and store.agent_id:
        # For agent stores, generate content from template
        agent_result = await session.execute(
            select(App).where(App.id == store.agent_id)
        )
        agent = agent_result.scalar_one_or_none()
        
        if agent:
            # Get API Key based on user authentication
            api_key = "your-api-key"
            if user:
                try:
                    from agents.services.open_service import get_or_create_credentials
                    credentials = await get_or_create_credentials(user, session)
                    if credentials and credentials.get("token"):
                        api_key = credentials["token"]
                except Exception as e:
                    logger.warning(f"Failed to get API Key for user: {str(e)}")
            
            # Generate markdown content with agent details
            content = f"""# DeepCore MCP Client Guide
![DeepCore Logo](http://deepcore.top/deepcore.png)

## Introduction

DeepCore provides AI Agent services based on MCP (Model Context Protocol), allowing AI models to access and operate various tools through standardized interfaces. This guide will explain how to configure and use DeepCore MCP services in different MCP clients.

## Quick Start

### 1. Get API Key

Step 1: Log in to the DeepCore Platform
Visit https://deepcore.top and log in to your account.

Step 2: Select Your Agent
From the dashboard, choose the Agent you want to work with.

Step 3: Find the MCP-Server Address
Go to the details page of the selected Agent and locate its current MCP-Server address.

### 2. Configure MCP Client

Choose the appropriate configuration method based on your MCP client:

#### Claude Desktop

1. Open Claude Desktop Settings
2. Go to `Developer > Edit Config`
3. Add the following configuration to `claude_desktop_config.json`:

```json
{{
  "mcpServers": {{
    "deepcore-agent": {{
      "url": "{SETTINGS.API_BASE_URL}/mcp/assistant/{agent.id}?api-key={api_key}"
    }}
  }}
}}
```

#### Cursor

1. Open Cursor Settings
2. Go to `Preferences > Cursor Settings > MCP`
3. Click `Add new global MCP Server`
4. Add the following configuration:

```json
{{
  "mcpServers": {{
    "deepcore-agent": {{
      "url": "{SETTINGS.API_BASE_URL}/mcp/assistant/{agent.id}?api-key={api_key}"
    }}
  }}
}}
```

### 3. Usage Example

After configuration, you can directly use DeepCore Agent's features in conversations. For example:

```
User: Help me check the weather in Beijing
AI: I'll use DeepCore Agent to query the weather information for Beijing.
```

## Technical Support

If you encounter any issues during use, you can get support through the following channels:

1. Visit DeepCore Documentation Center: https://docs.deepcore.top
2. Join DeepCore Community: https://community.deepcore.top
3. Contact Technical Support: support@deepcore.top"""
            return content
            
    return store.content

async def generate_tools_from_input(
    user_input: str,
    conversation_id: str,
    user: dict,
    session: AsyncSession
):
    """
    Generate tool list from user input using LLM
    
    Args:
        user_input: User's input text
        conversation_id: Conversation ID for context
        user: Current user information
        session: Database session
        
    Returns:
        List of tool configurations in flattened API format
    """
    try:
        prompt = generate_prompt(user_input, [])
        response = (await LLMFactory.get_llm(SETTINGS.MCP_MODEL_NAME, user, session)).astream(prompt)
        answer = ""
        async for data in response:
            if data.additional_kwargs and "reasoning_content" in data.additional_kwargs:
                yield ThinkOutput().write_text(data.additional_kwargs.get("reasoning_content")).to_stream()
            answer += data.text()
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, answer, re.DOTALL)
        if matches:
            for language, content in matches:
                yield send_message("tools", json.loads(content))
        else:
            yield send_message("tools", eval(answer))
    except Exception as e:
        logger.error(f"Error generating tools from input: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to generate tools from input: {str(e)}"
        )

async def create_tools_and_mcp_server(
    tools: List[dict],
    mcp_name: str,
    user: dict,
    session: AsyncSession,
    description: Optional[str] = None,
    icon: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create multiple tools and an MCP server in one operation with transaction support
    
    Args:
        tools: List of tool configurations
        mcp_name: Name for the new MCP server
        user: Current user information
        session: Database session
        description: Optional MCP service description
        icon: Optional icon URL for the MCP store
        
    Returns:
        Dictionary containing created tools, MCP server and store information
        
    Raises:
        CustomAgentException: If any error occurs during the operation
    """
    try:
        # Create tools in batch
        created_tools = await tool_service.create_tools_batch(
            tools=tools,
            user=user,
            session=session
        )
        
        # Extract tool IDs from created tools
        tool_ids = [str(tool.id) for tool in created_tools]
        
        # Create MCP server with the new tools
        mcp_server = await create_mcp_server_from_tools(
            mcp_name=mcp_name,
            tool_ids=tool_ids,
            user=user,
            session=session,
            description=description
        )

        from agents.services.open_service import get_or_create_credentials
        credentials = await get_or_create_credentials(user, session)
        api_key = ""
        if credentials and credentials.get("token"):
            api_key = credentials["token"]

        # Create MCP Store
        store = await create_mcp_store(
            store_name=mcp_name,
            store_type=MCP_STORE_TYPE_TOOL,
            user=user,
            session=session,
            description=description,
            content="",
            agent_id=mcp_server.get("mcp_id", None),
            icon=icon
        )

        # Commit the transaction
        await session.commit()
        return {
            "tools": created_tools,
            "mcp_server": mcp_server,
            "mcp_url": f"{SETTINGS.API_BASE_URL}/mcp/{mcp_name}?api-key={api_key}",
            "store": store
        }
            
    except CustomAgentException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Error creating tools and MCP server: {e}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.API_CALL_ERROR,
            f"Failed to create tools and MCP server: {str(e)}"
        )
