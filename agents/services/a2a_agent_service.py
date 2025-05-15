import json
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple

from python_a2a import AgentCard, A2AServer, Message, TextContent, MessageRole, Conversation, AgentSkill, Task, TaskState, TaskStatus
from python_a2a.discovery import enable_discovery
from sqlalchemy.ext.asyncio import AsyncSession

from agents.exceptions import CustomAgentException, ErrorCode
from agents.models.db import get_db
from agents.services import agent_service
from agents.common.config import SETTINGS
from agents.protocol.schemas import DialogueRequest
from agents.common.redis_utils import redis_utils

logger = logging.getLogger(__name__)

class DeepCoreA2AAgent(A2AServer):
    """
    Bridge class that converts DeepCore agents to A2A protocol agents.
    This class adapts the existing agent_service functionality to the A2A protocol.
    """
    
    def __init__(self, agent_id: str, agent_data: Dict[str, Any], user: Optional[dict] = None):
        """
        Initialize the A2A agent with data from the existing agent.
        
        Args:
            agent_id: The ID of the agent
            agent_data: Agent data from agent_service
            user: Optional user information
        """
        self.agent_id = agent_id
        self.user = user
        self.agent_data = agent_data
        self._use_google_a2a = True
        self.redis_task_prefix = f"a2a_agent:{agent_id}:tasks:"
        
        # Create an A2A agent card from the agent data
        agent_card = self._create_agent_card()
        
        # Initialize the A2A server
        super().__init__(agent_card=agent_card)
        
    def _create_agent_card(self) -> AgentCard:
        """
        Create an A2A AgentCard from agent data.
        
        Returns:
            AgentCard: The A2A agent card
        """
        # Base URL for the agent
        base_url = f"{SETTINGS.API_BASE_URL or 'http://localhost:8000'}/api/a2a"
        
        # Extract skills from agent data
        skills = []
        if self.agent_data.get("tools"):
            for tool in self.agent_data.get("tools", []):
                skills.append(AgentSkill(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                ))
        
        # Create and return the agent card
        return AgentCard(
            name=self.agent_data.get("name", "DeepCore Agent"),
            description=self.agent_data.get("description", ""),
            url=base_url,
            version="1.0.0",
            capabilities={
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": False,
                "google_a2a_compatible": True,
                "parts_array_format": True
            },
            skills=skills,
        )
    
    async def handle_message(self, message: Message) -> Message:
        """
        Handle a single message using the agent_service dialogue function.
        
        Args:
            message: The incoming A2A message
            
        Returns:
            Message: The response message
        """
        try:
            # Convert A2A message to DialogueRequest
            request = DialogueRequest(
                query=message.content.text,
                conversation_id=message.conversation_id or "new_conversation",
                initFlag=True if not message.conversation_id else False
            )
            # Collect responses from generator
            full_response = ""

            # Get the database session
            async for db in get_db():
                # Call the dialogue function
                response_generator = agent_service.dialogue(
                    agent_id=self.agent_id,
                    request=request,
                    user=self.user,
                    session=db
                )

                async for response_part in response_generator:
                    # Parse JSON-formatted response
                    if isinstance(response_part, str):
                        try:
                            response_json = json.loads(response_part)
                            if response_json.get("type") == "message":
                                full_response += response_json.get("content", "")
                            elif response_json.get("type") == "markdown":
                                full_response += response_json.get("content", "")
                        except json.JSONDecodeError:
                            full_response += response_part
                    else:
                        full_response += str(response_part)
            
            # Create and return A2A message
            return Message(
                content=TextContent(text=full_response),
                role=MessageRole.AGENT,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )
        
        except Exception as e:
            logger.error(f"Error in handle_message: {str(e)}", exc_info=True)
            return Message(
                content=TextContent(text=f"Error processing your request: {str(e)}"),
                role=MessageRole.SYSTEM,
                parent_message_id=message.message_id,
                conversation_id=message.conversation_id
            )
    
    async def handle_conversation(self, conversation: Conversation) -> Message:
        """
        Handle a conversation using the agent_service dialogue function.
        
        Args:
            conversation: The incoming A2A conversation
            
        Returns:
            Message: The response message
        """
        try:
            # Get the last user message from the conversation
            last_message = None
            for message in reversed(conversation.messages):
                if message.role == MessageRole.USER:
                    last_message = message
                    break
            
            if not last_message:
                raise ValueError("No user message found in conversation")
            
            # Handle the last message
            return await self.handle_message(last_message)
        
        except Exception as e:
            logger.error(f"Error in handle_conversation: {str(e)}", exc_info=True)
            return Message(
                content=TextContent(text=f"Error processing your conversation: {str(e)}"),
                role=MessageRole.SYSTEM,
                conversation_id=conversation.conversation_id
            )
    
    async def stream_response(self, message: Message):
        """
        Stream a response to a message using the agent_service dialogue function.
        
        Args:
            message: The incoming A2A message
            
        Yields:
            str: Response chunks
        """
        try:
            # Convert A2A message to DialogueRequest
            request = DialogueRequest(
                query=message.content.text,
                conversation_id=message.conversation_id or "new_conversation",
                initFlag=True if not message.conversation_id else False
            )
            
            # Get the database session
            async for db in get_db():
                # Call the dialogue function
                response_generator = agent_service.dialogue(
                    agent_id=self.agent_id,
                    request=request,
                    user=self.user,
                    session=db
                )

                # Stream responses
                async for response_part in response_generator:
                    # Parse JSON-formatted response
                    if isinstance(response_part, str):
                        try:
                            response_json = json.loads(response_part)
                            if response_json.get("type") == "message":
                                yield response_json.get("content", "")
                            elif response_json.get("type") == "markdown":
                                yield response_json.get("content", "")
                        except json.JSONDecodeError:
                            yield response_part
                    else:
                        yield str(response_part)
        
        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}", exc_info=True)
            yield f"Error streaming response: {str(e)}"
    
    def _get_redis_task_key(self, task_id: str) -> str:
        """
        Generate Redis key for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            str: Redis key
        """
        return f"{self.redis_task_prefix}{task_id}"
    
    def _save_task(self, task: Task, ttl: int = 60 * 60 * 24) -> None:
        """
        Save a task to Redis.
        
        Args:
            task: Task to save
            ttl: Time to live in seconds (default 24 hours)
        """
        key = self._get_redis_task_key(task.id)
        
        # Convert Task to dict and save to Redis
        task_dict = task.to_dict()
        redis_utils.set_value(key, json.dumps(task_dict), ex=ttl)
    
    def _get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task from Redis.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Task]: Task if found, None otherwise
        """
        key = self._get_redis_task_key(task_id)
        task_json = redis_utils.get_value(key)
        
        if not task_json:
            return None
        
        try:
            task_dict = json.loads(task_json)
            return Task.from_dict(task_dict)
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {str(e)}", exc_info=True)
            return None

    async def handle_task(self, task):
        """
        Process an A2A task

        Override this in your custom server implementation.

        Args:
            task: The incoming A2A task

        Returns:
            The processed task with response
        """
        logger.info(f"Processing task: {task.id}")
        
        # Extract message from task with careful handling to maintain format compatibility
        message_data = task.message or {}
        
        # IMPORTANT: Check if the subclass has overridden handle_message
        has_message_handler = hasattr(self, 'handle_message') and self.handle_message != A2AServer.handle_message
        
        if has_message_handler or hasattr(self, "_handle_message_impl") and self._handle_message_impl:
            try:
                # Convert to Message object if it's a dict
                message = None

                if isinstance(message_data, dict):
                    # First, check for Google A2A format
                    if "parts" in message_data and "role" in message_data and not "content" in message_data:
                        try:
                            message = Message.from_google_a2a(message_data)
                            logger.debug("Converted message from Google A2A format")
                        except Exception as e:
                            # If conversion fails, fall back to standard format
                            logger.warning(f"Failed to convert Google A2A message: {str(e)}")
                            pass

                    # If not Google A2A format or conversion failed, try standard format
                    if message is None:
                        try:
                            message = Message.from_dict(message_data)
                            logger.debug("Converted message from standard A2A format")
                        except Exception as e:
                            # If standard format fails too, create a basic message
                            logger.warning(f"Failed to convert standard A2A message: {str(e)}")
                            # Extract text directly from common formats to maintain compatibility
                            text = ""
                            if "content" in message_data and isinstance(message_data["content"], dict):
                                # python_a2a format
                                content = message_data["content"]
                                if "text" in content:
                                    text = content["text"]
                                elif "message" in content:
                                    text = content["message"]
                            elif "parts" in message_data:
                                # Google A2A format
                                for part in message_data["parts"]:
                                    if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                                        text = part["text"]
                                        break

                            logger.info(f"Created basic message with extracted text: {text[:100]}...")
                            message = Message(
                                content=TextContent(text=text),
                                role=MessageRole.USER
                            )
                else:
                    # If it's already a Message object, use it directly
                    message = message_data
                    logger.debug("Using provided Message object directly")

                # Call the appropriate message handler
                logger.info("Calling message handler")

                try:
                    if has_message_handler:
                        response = await self.handle_message(message)
                    else:
                        logger.debug("Creating new event loop")
                        response = await self._handle_message_impl(message)
                except Exception as e:
                    logger.error(f"Unexpected error handling message: {str(e)}", exc_info=True)
                    response = Message(
                        content=TextContent(text=f"Unexpected error: {str(e)}"),
                        role=MessageRole.SYSTEM
                    )

                # Create artifact based on response content type
                if hasattr(response, "content"):
                    content_type = getattr(response.content, "type", None)
                    logger.debug(f"Response content type: {content_type}")

                    if content_type == "text":
                        # Handle TextContent
                        task.artifacts = [{
                            "parts": [{
                                "type": "text",
                                "text": response.content.text
                            }]
                        }]
                    elif content_type == "function_response":
                        # Handle FunctionResponseContent
                        task.artifacts = [{
                            "parts": [{
                                "type": "function_response",
                                "name": response.content.name,
                                "response": response.content.response
                            }]
                        }]
                    elif content_type == "function_call":
                        # Handle FunctionCallContent
                        params = []
                        for param in response.content.parameters:
                            params.append({
                                "name": param.name,
                                "value": param.value
                            })

                        task.artifacts = [{
                            "parts": [{
                                "type": "function_call",
                                "name": response.content.name,
                                "parameters": params
                            }]
                        }]
                    elif content_type == "error":
                        # Handle ErrorContent
                        task.artifacts = [{
                            "parts": [{
                                "type": "error",
                                "message": response.content.message
                            }]
                        }]
                    else:
                        # Handle other content types
                        task.artifacts = [{
                            "parts": [{
                                "type": "text",
                                "text": str(response.content)
                            }]
                        }]
                else:
                    # Handle responses without content
                    logger.debug("Response has no content attribute, using string representation")
                    task.artifacts = [{
                        "parts": [{
                            "type": "text",
                            "text": str(response)
                        }]
                    }]
            except Exception as e:
                # Handle errors in message handler
                logger.error(f"Error in message handler: {str(e)}", exc_info=True)
                task.artifacts = [{
                    "parts": [{
                        "type": "error",
                        "message": f"Error in message handler: {str(e)}"
                    }]
                }]
        else:
            # Basic echo response when no message handler exists
            logger.warning("No message handler available, using echo response")
            content = message_data.get("content", {})

            # Handle different content types in passthrough mode
            if isinstance(content, dict):
                content_type = content.get("type")

                if content_type == "text":
                    # Text content
                    task.artifacts = [{
                        "parts": [{
                            "type": "text",
                            "text": content.get("text", "")
                        }]
                    }]
                elif content_type == "function_call":
                    # Function call - pass through
                    task.artifacts = [{
                        "parts": [{
                            "type": "text",
                            "text": f"Received function call '{content.get('name', '')}' without handler"
                        }]
                    }]
                else:
                    # Other content types
                    task.artifacts = [{
                        "parts": [{
                            "type": "text",
                            "text": f"Received message of type '{content_type}'"
                        }]
                    }]
            else:
                # For Google A2A format or other formats, try to extract text
                text = ""
                if isinstance(message_data, dict):
                    if "parts" in message_data and isinstance(message_data["parts"], list):
                        # Google A2A format
                        for part in message_data["parts"]:
                            if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                                text = part["text"]
                                break
                    elif "content" in message_data and isinstance(message_data["content"], dict) and "text" in message_data["content"]:
                        # Try to extract from nested content
                        text = message_data["content"]["text"]

                # Non-dict content or extracted text
                task.artifacts = [{
                    "parts": [{
                        "type": "text",
                        "text": text or str(content)
                    }]
                }]

        # Mark task as completed
        task.status = TaskStatus(state=TaskState.COMPLETED)
        logger.info(f"Task {task.id} completed successfully")
        return task
    
    async def handle_task_request(self, data, is_google_format=False) -> Tuple[Union[Dict, Task], int]:
        """
        Handle a task request in either format

        Args:
            data: Request data
            is_google_format: Whether the request is in Google A2A format

        Returns:
            Tuple containing the result data and status code
        """
        try:
            # Extract task ID and session ID
            task_id = data.get("id", str(uuid.uuid4()))
            session_id = data.get("sessionId")

            # Create task based on format
            if is_google_format:
                # Google A2A format - preserve the exact format of message
                task = Task.from_google_a2a(data)
            else:
                # Standard python_a2a format - preserve the exact format of message
                task = Task.from_dict(data)

            # Process the task
            result = await self.handle_task(task)

            # Store the task in Redis
            self._save_task(result)

            # Return result based on requested format
            if is_google_format or self._use_google_a2a:
                # Use Google A2A format for response
                return result.to_google_a2a(), 200
            else:
                # Use standard python_a2a format
                return result.to_dict(), 200
        except Exception as e:
            # Return an error in the appropriate format
            error_msg = f"Error processing task: {str(e)}"
            error_response = {
                "id": data.get("id", ""),
                "sessionId": data.get("sessionId", ""),
                "status": {
                    "state": "failed",
                    "message": {"error": error_msg},
                    "timestamp": datetime.now().isoformat()
                }
            }
            return error_response, 500

    async def generate_tasks_send_subscribe_events(self, params, rpc_id):
        """
        Generate SSE events for tasks/sendSubscribe

        Args:
            params: Task parameters
            rpc_id: JSON-RPC ID

        Yields:
            str: SSE event strings
        """
        try:
            # Extract task ID and session ID
            task_id = params.get("id", str(uuid.uuid4()))
            session_id = params.get("sessionId")

            # Create task from params
            task = Task.from_dict(params)

            # Send initial task state
            initial_task = task.to_dict() if not self._use_google_a2a else task.to_google_a2a()
            yield f"event: update\nid: {rpc_id}\ndata: {json.dumps(initial_task)}\n\n"

            # Process the task
            result_task = None
            try:
                result_task = await self.handle_task(task)
            except Exception as e:
                # Handle error
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"error": str(e)}
                )
                result_task = task

            # Store the task in Redis
            self._save_task(result_task)

            # Send complete event
            complete_task = result_task.to_dict() if not self._use_google_a2a else result_task.to_google_a2a()
            yield f"event: complete\nid: {rpc_id}\ndata: {json.dumps(complete_task)}\n\n"
        
        except Exception as e:
            # Handle error
            error_msg = f"Error in tasks/sendSubscribe: {str(e)}"
            yield f"event: error\nid: {rpc_id}\ndata: {json.dumps({'error': error_msg})}\n\n"

    async def generate_tasks_resubscribe_events(self, params, rpc_id):
        """
        Generate SSE events for tasks/resubscribe

        Args:
            params: Parameters with task ID
            rpc_id: JSON-RPC ID

        Yields:
            str: SSE event strings
        """
        try:
            # Extract task ID
            task_id = params.get("id")
            
            if not task_id:
                # Task ID is required
                yield f"event: error\nid: {rpc_id}\ndata: {json.dumps({'error': 'Missing task ID'})}\n\n"
                return

            # Get the task from Redis
            task = self._get_task(task_id)
            if not task:
                # Task not found
                yield f"event: error\nid: {rpc_id}\ndata: {json.dumps({'error': f'Task not found: {task_id}'})}\n\n"
                return

            # Send the current task state
            current_task = task.to_dict() if not self._use_google_a2a else task.to_google_a2a()
            yield f"event: update\nid: {rpc_id}\ndata: {json.dumps(current_task)}\n\n"

            # If the task is not completed, failed, or canceled, we should wait for updates
            if task.status.state not in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                # In a real implementation, this would wait for task updates
                # For this example, we'll just send a completion event after a delay
                await asyncio.sleep(0.5)  # Small delay to simulate processing

                # Update task status to completed
                task.status = TaskStatus(state=TaskState.COMPLETED)
                
                # Save updated task
                self._save_task(task)

                # Send complete event
                complete_task = task.to_dict() if not self._use_google_a2a else task.to_google_a2a()
                yield f"event: complete\nid: {rpc_id}\ndata: {json.dumps(complete_task)}\n\n"
            else:
                # If the task is already in a final state, send a complete event
                complete_task = task.to_dict() if not self._use_google_a2a else task.to_google_a2a()
                yield f"event: complete\nid: {rpc_id}\ndata: {json.dumps(complete_task)}\n\n"
        
        except Exception as e:
            # Handle error
            error_msg = f"Error in tasks/resubscribe: {str(e)}"
            yield f"event: error\nid: {rpc_id}\ndata: {json.dumps({'error': error_msg})}\n\n"


async def load_agent(agent_id: str, user: Optional[dict] = None, session: Optional[AsyncSession] = None) -> DeepCoreA2AAgent:
    """
    Load an agent from agent_service and convert it to an A2A agent.
    
    Args:
        agent_id: ID of the agent to load
        user: Optional user information
        session: Optional database session
        
    Returns:
        DeepCoreA2AAgent: The A2A agent instance
    """
    try:
        # Get agent data from agent_service
        agent_data = await agent_service.get_agent(agent_id, user, session)
        
        # Create and return the A2A agent
        return DeepCoreA2AAgent(agent_id, agent_data.dict(), user)
    
    except Exception as e:
        logger.error(f"Error loading agent {agent_id}: {str(e)}", exc_info=True)
        raise CustomAgentException(
            ErrorCode.RESOURCE_NOT_FOUND,
            f"Could not load agent {agent_id}: {str(e)}"
        )


async def get_agent_data(agent):
    """
    Get basic agent data for rendering.
    
    Returns:
        dict: Agent data with name, description, and other metadata
    """
    try:
        # If agent is available in global scope, use it
        if hasattr(agent, 'agent_card'):
            return agent.agent_card.to_dict()
        
        # Fallback for agents without agent_card
        return {
            "name": "DeepCore A2A Agent",
            "description": "DeepCore agent with A2A protocol support",
            "version": "1.0.0",
            "url": f"{SETTINGS.API_BASE_URL or 'http://localhost:8000'}/api/a2a",
            "capabilities": {
                "streaming": True,
                "google_a2a_compatible": True,
                "parts_array_format": True
            },
            "skills": []
        }
    except Exception as e:
        logger.error(f"Error getting agent data: {str(e)}", exc_info=True)
        return {
            "name": "DeepCore A2A Agent",
            "description": "Agent details not available",
            "version": "1.0.0",
            "error": str(e)
        }

async def get_agent(agent_id: str = None, user: Optional[dict] = None, session: Optional[AsyncSession] = None):
    """
    Get an A2A agent by ID or default agent if ID not provided.
    
    Args:
        agent_id: Optional ID of the agent to load
        user: Optional user information
        session: Optional database session
        
    Returns:
        tuple: (agent, agent_data) where agent is the DeepCoreA2AAgent instance and agent_data is its metadata
    """
    agent = await load_agent(agent_id, user, session)
    data = await get_agent_data(agent)
    return agent, data


async def register_with_registry(agent_id: str, registry_url: str, user: Optional[dict] = None, session: Optional[AsyncSession] = None):
    """
    Register an agent with an A2A registry server.
    
    Args:
        agent_id: ID of the agent to register
        registry_url: URL of the A2A registry
        user: Optional user information
        session: Optional database session
        
    Returns:
        dict: Registration result
    """
    try:
        # Load the agent
        agent = await load_agent(agent_id, user, session)
        
        # Enable discovery
        discovery_client = enable_discovery(agent, registry_url=registry_url)
        
        return {
            "status": "success",
            "message": f"Agent {agent_id} registered with registry at {registry_url}",
            "agent_card": agent.agent_card.to_dict()
        }
    
    except Exception as e:
        logger.error(f"Error registering agent {agent_id} with registry: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to register agent: {str(e)}"
        }