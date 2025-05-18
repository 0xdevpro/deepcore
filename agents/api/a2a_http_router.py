import json
import logging
import threading
import time
import asyncio
from queue import Empty, Queue
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from jinja2 import Template
from python_a2a import Conversation, Message, BaseA2AServer, Task, TaskState, TaskStatus
from python_a2a.server.ui_templates import JSON_HTML_TEMPLATE, AGENT_INDEX_HTML
from sqlalchemy.ext.asyncio import AsyncSession

from agents.middleware.auth_middleware import get_optional_current_user
from agents.models.db import get_db
from agents.services import a2a_agent_service

# Create base router and nested routers
base_router = APIRouter()
a2a_router = APIRouter()

logger = logging.getLogger(__name__)

# Helper dependency to get agent from path
async def get_agent_from_path(
    agent_id: str,
    user: Optional[dict] = Depends(get_optional_current_user), 
    session: AsyncSession = Depends(get_db)
) -> Tuple[object, dict]:
    """
    Dependency function to get agent by ID from the path parameter.
    
    Args:
        agent_id: ID of the agent from the URL path
        user: Optional current user from authentication
        session: Database session
        
    Returns:
        tuple: (agent, agent_data) where agent is the A2A agent instance and agent_data is its metadata
    """
    return await a2a_agent_service.get_agent(agent_id, user, session)

# Define routes with agent_id parameter
@a2a_router.get("/")
async def a2a_index(
        request: Request,
        agent_tuple: Tuple = Depends(get_agent_from_path),
        format_param: str = ''
):
    """
    A2A index with beautiful UI or JSON response based on request headers.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        format_param: Optional format parameter to force response type
        
    Returns:
        HTML or JSON response with agent information
    """
    agent, _ = agent_tuple
    
    # Check if this is a browser request by looking at headers
    user_agent = request.headers.get('User-Agent', '')
    accept_header = request.headers.get('Accept', '')

    # Return JSON if explicitly requested or doesn't look like a browser
    if format_param == 'json' or (
            'application/json' in accept_header and
            not any(browser in user_agent.lower() for browser in ['mozilla', 'chrome', 'safari', 'edge'])
    ):
        # Include Google A2A compatibility flag if available
        capabilities = {}
        if hasattr(agent, 'agent_card') and hasattr(agent.agent_card, 'capabilities'):
            capabilities = agent.agent_card.capabilities
        elif hasattr(agent, '_use_google_a2a'):
            capabilities = {
                "google_a2a_compatible": getattr(agent, '_use_google_a2a', False),
                "parts_array_format": getattr(agent, '_use_google_a2a', False)
            }

        return JSONResponse({
            "name": agent.agent_card.name if hasattr(agent, 'agent_card') else "A2A Agent",
            "description": agent.agent_card.description if hasattr(agent, 'agent_card') else "",
            "agent_card_url": f"/api/agents/{request.path_params['agent_id']}/a2a/agent.json",
            "protocol": "a2a",
            "capabilities": capabilities
        })

    # Otherwise serve HTML by default
    template = Template(AGENT_INDEX_HTML)
    rendered_html = template.render(
        agent=agent,
        request=request
    )
    response = HTMLResponse(content=rendered_html)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@a2a_router.get("/agent")
async def agent_index(
        request: Request,
        agent_tuple: Tuple = Depends(get_agent_from_path),
        format_param: str = ''
):
    """
    Agent endpoint with beautiful UI, redirects to a2a_index.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        format_param: Optional format parameter to force response type
        
    Returns:
        HTML or JSON response with agent information
    """
    return await a2a_index(request, agent_tuple, format_param)

@a2a_router.get("/agent.json")
async def a2a_agent_json(
        request: Request,
        agent_tuple: Tuple = Depends(get_agent_from_path),
        format_param: str = ''
):
    """
    Agent card JSON with beautiful UI or raw JSON based on request headers.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        format_param: Optional format parameter to force response type
        
    Returns:
        HTML or JSON response with agent card data
    """
    agent, agent_data = agent_tuple

    # Add Google A2A compatibility flag if available
    if hasattr(agent, '_use_google_a2a'):
        if "capabilities" not in agent_data:
            agent_data["capabilities"] = {}
        agent_data["capabilities"]["google_a2a_compatible"] = getattr(agent, '_use_google_a2a', False)
        agent_data["capabilities"]["parts_array_format"] = getattr(agent, '_use_google_a2a', False)

    # Check request format preferences
    user_agent = request.headers.get('User-Agent', '')
    accept_header = request.headers.get('Accept', '')

    # Return JSON if explicitly requested or doesn't look like a browser
    if format_param == 'json' or (
            'application/json' in accept_header and
            not any(browser in user_agent.lower() for browser in ['mozilla', 'chrome', 'safari', 'edge'])
    ):
        return JSONResponse(agent_data)

    # Otherwise serve HTML with pretty JSON visualization
    formatted_json = json.dumps(agent_data, indent=2)
    template = Template(JSON_HTML_TEMPLATE)
    rendered_html = template.render(
        title=agent_data.get('name', 'A2A Agent'),
        description="Agent Card JSON Data",
        json_data=formatted_json
    )
    response = HTMLResponse(content=rendered_html)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@a2a_router.post("/stream")
async def handle_streaming_request(
        request: Request,
        agent_tuple: Tuple = Depends(get_agent_from_path)
):
    """
    Handle streaming requests using Server-Sent Events (SSE).
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        
    Returns:
        StreamingResponse with agent's response or error JSON
    """
    try:
        # CORS for streaming - important for browser compatibility
        if request.method == 'OPTIONS':
            response = Response()
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

        agent, _ = agent_tuple

        # Check accept header for streaming support
        accept_header = request.headers.get('Accept', '')
        supports_sse = 'text/event-stream' in accept_header

        # Debug logging
        logger.info(f"Streaming request received with Accept: {accept_header}")
        logger.info(f"Request supports SSE: {supports_sse}")

        # Extract the message from the request
        data = await request.json()

        # Debug logging for request data
        logger.info(f"Streaming request data: {json.dumps(data)[:500]}")

        # Check if this is a direct message or wrapped
        if "message" in data and isinstance(data["message"], dict):
            message = Message.from_dict(data["message"])
        else:
            # Try parsing the entire request as a message
            message = Message.from_dict(data)

        # Debug logging for message
        logger.info(f"Extracted message: {message.content}")

        # Check if the agent supports streaming
        if not hasattr(agent, 'stream_response'):
            error_msg = "This agent does not support streaming"
            logger.error(f"Error: {error_msg}")
            return JSONResponse({"error": error_msg}, status_code=405)

        # Check if stream_response is implemented (not just inherited)
        if agent.stream_response == BaseA2AServer.stream_response:
            error_msg = "This agent inherits but does not implement stream_response"
            logger.error(f"Error: {error_msg}")
            return JSONResponse({"error": error_msg}, status_code=501)

        # Set up SSE streaming response
        async def generate():
            """Generator for streaming server-sent events."""
            async def process_stream():
                """Process the streaming response."""
                try:
                    logger.info("Starting streaming process")
                    # Get the stream generator from the agent
                    # Note: stream_response returns an async generator, not an awaitable
                    stream_gen = agent.stream_response(message)

                    # First heartbeat is sent from outside this function

                    # Process each chunk
                    index = 0
                    async for chunk in stream_gen:
                        logger.info(f"Received chunk from agent: {chunk}")

                        # Create chunk object with metadata
                        chunk_data = {
                            "content": chunk,
                            "index": index,
                            "append": True
                        }

                        yield chunk_data
                        logger.info(f"Put chunk {index} in queue")
                        index += 1

                    # Signal completion
                    yield {
                        "content": "",
                        "index": index,
                        "append": True,
                        "lastChunk": True
                    }
                    logger.info(f"Streaming complete, signaling with lastChunk")

                except Exception as e:
                    # Log the error
                    logger.error(f"Error in streaming process: {str(e)}", exc_info=True)
                    yield {"error": str(e)}

                finally:
                    logger.info("Set done_event")

            # Yield initial SSE comment to establish connection
            yield f": SSE stream established\n\n"

            total_chunks = 0
            try:
                async for chunk in process_stream():
                    total_chunks += 1

                    # Check if it's an error
                    if "error" in chunk:
                        error_event = f"event: error\ndata: {json.dumps(chunk)}\n\n"
                        logger.info(f"Yielding error event: {error_event}")
                        yield error_event
                        break

                    # Format as SSE event with proper newlines
                    data_event = f"data: {json.dumps(chunk)}\n\n"
                    logger.info(f"Yielding data event #{total_chunks}")
                    yield data_event

                    # Check if it's the last chunk
                    if chunk.get("lastChunk", False):
                        logger.info("Last chunk detected, ending stream")
                        break
                else:
                    # No data yet, sleep briefly
                    time.sleep(0.01)
            except Empty:
                # Queue was empty
                time.sleep(0.01)
            except Exception as e:
                # Other error
                logger.info(f"Error in queue processing: {e}")
                error_event = f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                yield error_event

            logger.info(f"Stream complete - yielded {total_chunks} chunks")

        # Create the streaming response
        response = StreamingResponse(generate(), media_type="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"  # Important for Nginx
        return response

    except Exception as e:
        # Log the exception
        logger.error(f"Exception in streaming request handler: {str(e)}", exc_info=True)

        # Return error response for any other exception
        return JSONResponse({"error": str(e)}, status_code=500)

@a2a_router.post("/")
async def handle_a2a_request(
        request: Request,
        agent_tuple: Tuple = Depends(get_agent_from_path)
) -> JSONResponse:
    """
    Handle A2A protocol requests for messages and conversations.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        
    Returns:
        JSON response with agent's response
    """
    try:
        data = await request.json()
        agent, _ = agent_tuple

        # Detect if this is Google A2A format
        is_google_format = False
        if "parts" in data and "role" in data and not "content" in data:
            is_google_format = True
        elif "messages" in data and data["messages"] and "parts" in data["messages"][0] and "role" in data["messages"][
            0]:
            is_google_format = True

        # Check if this is a single message or a conversation
        if "messages" in data:
            # This is a conversation
            if is_google_format:
                conversation = Conversation.from_google_a2a(data)
            else:
                conversation = Conversation.from_dict(data)

            response = await agent.handle_conversation(conversation)

            # Format response based on request format or agent preference
            use_google_format = is_google_format
            if hasattr(agent, '_use_google_a2a'):
                use_google_format = use_google_format or agent._use_google_a2a

            if use_google_format:
                return JSONResponse(response.to_google_a2a())
            else:
                return JSONResponse(response.to_dict())
        else:
            # This is a single message
            if is_google_format:
                message = Message.from_google_a2a(data)
            else:
                message = Message.from_dict(data)

            response = await agent.handle_message(message)

            # Format response based on request format or agent preference
            use_google_format = is_google_format
            if hasattr(agent, '_use_google_a2a'):
                use_google_format = use_google_format or agent._use_google_a2a

            if use_google_format:
                return JSONResponse(response.to_google_a2a())
            else:
                return JSONResponse(response.to_dict())

    except Exception as e:
        logger.error(f"handle_a2a_request {e}", exc_info=True)
        # Determine response format based on request
        is_google_format = False
        if 'data' in locals():
            if isinstance(data, dict):
                if "parts" in data and "role" in data and not "content" in data:
                    is_google_format = True
                elif "messages" in data and data["messages"] and "parts" in data["messages"][0] and "role" in \
                        data["messages"][0]:
                    is_google_format = True

        # Also consider agent preference
        if 'agent' in locals() and hasattr(agent, '_use_google_a2a'):
            is_google_format = is_google_format or agent._use_google_a2a

        # Return error in appropriate format
        error_msg = f"Error processing request: {str(e)}"
        if is_google_format:
            # Google A2A format
            return JSONResponse({
                "role": "agent",
                "parts": [
                    {
                        "type": "data",
                        "data": {"error": error_msg}
                    }
                ]
            }, status_code=500)
        else:
            # python_a2a format
            return JSONResponse({
                "content": {
                    "type": "error",
                    "message": error_msg
                },
                "role": "system"
            }, status_code=500)

@a2a_router.get("/metadata")
async def get_agent_metadata(
        agent_tuple: Tuple = Depends(get_agent_from_path)
) -> JSONResponse:
    """
    Return metadata about the agent.
    
    Args:
        agent_tuple: Tuple containing agent and its data from dependency
        
    Returns:
        JSON response with agent metadata
    """
    agent, _ = agent_tuple
    metadata = agent.get_metadata()

    # Add Google A2A compatibility flag if available
    if hasattr(agent, '_use_google_a2a'):
        metadata["google_a2a_compatible"] = getattr(agent, '_use_google_a2a', False)
        metadata["parts_array_format"] = getattr(agent, '_use_google_a2a', False)

    return JSONResponse(metadata)

@a2a_router.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint.
    
    Returns:
        JSON response with status
    """
    return JSONResponse({"status": "ok"})

@a2a_router.post("/tasks/send")
async def a2a_tasks_send(
    request: Request,
    agent_tuple: Tuple = Depends(get_agent_from_path)
):
    """
    Handle POST request to create or update a task.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        
    Returns:
        JSON response with task result or error
    """
    request_data = {}
    agent = None
    try:
        # Parse JSON data
        request_data = await request.json()
        agent, _ = agent_tuple

        # Handle as JSON-RPC if it follows that format
        if "jsonrpc" in request_data:
            rpc_id = request_data.get("id", 1)
            params = request_data.get("params", {})

            # Detect format from params
            is_google_format = False
            if isinstance(params, dict) and "message" in params:
                message_data = params.get("message", {})
                if isinstance(message_data, dict) and "parts" in message_data and "role" in message_data:
                    is_google_format = True

            # Process the task
            result_data, status_code = await agent.handle_task_request(params, is_google_format)

            # Return JSON-RPC response
            if status_code >= 400:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32603,
                        "message": result_data.get("status", {}).get("message", {}).get("error", "Unknown error")
                    }
                }, status_code=status_code)
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": result_data
                })
        else:
            # Direct task submission - detect format
            is_google_format = False
            if "message" in request_data:
                message_data = request_data.get("message", {})
                if isinstance(message_data, dict) and "parts" in message_data and "role" in message_data:
                    is_google_format = True

            # Handle the task request
            result_data, status_code = agent.handle_task_request(request_data, is_google_format)
            return JSONResponse(content=result_data, status_code=status_code)

    except Exception as e:
        # Handle error based on request format
        if "jsonrpc" in request_data:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_data.get("id", 1),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }, status_code=500)
        else:
            if hasattr(agent, '_use_google_a2a') and agent._use_google_a2a:
                return JSONResponse({
                    "role": "agent",
                    "parts": [
                        {
                            "type": "data",
                            "data": {"error": f"Error: {str(e)}"}
                        }
                    ]
                }, status_code=500)
            else:
                return JSONResponse({
                    "content": {
                        "type": "error",
                        "message": f"Error: {str(e)}"
                    },
                    "role": "system"
                }, status_code=500)

@a2a_router.post("/tasks/get")
async def a2a_tasks_get(
    request: Request,
    agent_tuple: Tuple = Depends(get_agent_from_path)
):
    """
    Handle POST request to get a task status.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        
    Returns:
        JSON response with task data or error
    """
    request_data = {}
    try:
        # Parse JSON data
        request_data = await request.json()
        agent, _ = agent_tuple

        # Handle as JSON-RPC if it follows that format
        if "jsonrpc" in request_data:
            rpc_id = request_data.get("id", 1)
            params = request_data.get("params", {})

            # Extract task ID
            task_id = params.get("id")
            history_length = params.get("historyLength", 0)

            # Get the task
            task = agent._get_task(task_id)
            if not task:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32000,
                        "message": f"Task not found: {task_id}"
                    }
                }, status_code=404)

            # Convert task to dict in appropriate format
            if hasattr(agent, '_use_google_a2a') and agent._use_google_a2a:
                task_dict = task.to_google_a2a()
            else:
                task_dict = task.to_dict()

            # Return the task
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": task_dict
            })
        else:
            # Handle as direct task request
            task_id = request_data.get("id")

            # Get the task
            task = agent._get_task(task_id)
            if not task:
                return JSONResponse({"error": f"Task not found: {task_id}"}, status_code=404)

            # Convert task to dict in appropriate format
            if hasattr(agent, '_use_google_a2a') and agent._use_google_a2a:
                task_dict = task.to_google_a2a()
            else:
                task_dict = task.to_dict()

            # Return the task
            return JSONResponse(task_dict)

    except Exception as e:
        # Handle error
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_data.get("id", 1) if 'request_data' in locals() else 1,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }, status_code=500)

@a2a_router.post("/tasks/cancel")
async def a2a_tasks_cancel(
    request: Request,
    agent_tuple: Tuple = Depends(get_agent_from_path)
):
    """
    Handle POST request to cancel a running task.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        
    Returns:
        JSON response with canceled task data or error
    """
    request_data = {}
    try:
        # Parse JSON data
        request_data = await request.json()
        agent, _ = agent_tuple

        # Handle as JSON-RPC if it follows that format
        if "jsonrpc" in request_data:
            rpc_id = request_data.get("id", 1)
            params = request_data.get("params", {})

            # Extract task ID
            task_id = params.get("id")

            # Get the task
            task = agent._get_task(task_id)
            if not task:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32000,
                        "message": f"Task not found: {task_id}"
                    }
                }, status_code=404)

            # Cancel the task
            task.status = TaskStatus(state=TaskState.CANCELED)
            
            # Save the updated task
            agent._save_task(task)

            # Convert task to dict in appropriate format
            if hasattr(agent, '_use_google_a2a') and agent._use_google_a2a:
                task_dict = task.to_google_a2a()
            else:
                task_dict = task.to_dict()

            # Return the task
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": task_dict
            })
        else:
            # Handle as direct task request
            task_id = request_data.get("id")

            # Get the task
            task = agent._get_task(task_id)
            if not task:
                return JSONResponse({"error": f"Task not found: {task_id}"}, status_code=404)

            # Cancel the task
            task.status = TaskStatus(state=TaskState.CANCELED)
            
            # Save the updated task
            agent._save_task(task)

            # Convert task to dict in appropriate format
            if hasattr(agent, '_use_google_a2a') and agent._use_google_a2a:
                task_dict = task.to_google_a2a()
            else:
                task_dict = task.to_dict()

            # Return the task
            return JSONResponse(task_dict)

    except Exception as e:
        # Handle error
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_data.get("id", 1) if 'request_data' in locals() else 1,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }, status_code=500)

@a2a_router.post("/tasks/stream")
async def a2a_tasks_stream(
    request: Request,
    agent_tuple: Tuple = Depends(get_agent_from_path)
):
    """
    Streaming endpoint for tasks/sendSubscribe and tasks/resubscribe operations.
    
    Args:
        request: FastAPI request object
        agent_tuple: Tuple containing agent and its data from dependency
        
    Returns:
        StreamingResponse for task updates or error JSON
    """
    request_data = {}
    try:
        # Parse JSON data
        request_data = await request.json()
        agent, _ = agent_tuple

        # Check if this is a JSON-RPC request
        if "jsonrpc" in request_data:
            method = request_data.get("method", "")
            params = request_data.get("params", {})
            rpc_id = request_data.get("id", 1)

            # Handle different streaming methods
            if method == "tasks/sendSubscribe":
                # Process tasks/sendSubscribe using SSE
                generator = agent.generate_tasks_send_subscribe_events(params, rpc_id)
                return StreamingResponse(
                    generator,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"  # Disable Nginx buffering
                    }
                )
            elif method == "tasks/resubscribe":
                # Process tasks/resubscribe using SSE
                generator = agent.generate_tasks_resubscribe_events(params, rpc_id)
                return StreamingResponse(
                    generator,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"  # Disable Nginx buffering
                    }
                )
            else:
                # Unknown method
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found"
                    }
                }, status_code=404)
        else:
            # Not a JSON-RPC request
            return JSONResponse({
                "error": "Expected JSON-RPC format for streaming requests"
            }, status_code=400)

    except Exception as e:
        # Handle error
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_data.get("id", 1) if 'request_data' in locals() else 1,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }, status_code=500)

# Include nested router with agent_id parameter
base_router.include_router(
    a2a_router,
    prefix="/A2A/{agent_id}",
)

# Expose the base_router as the main app
app = base_router