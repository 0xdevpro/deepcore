import logging
from typing import Optional

from fastapi import APIRouter, Depends, Body, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from agents.common.error_messages import get_error_message
from agents.common.response import RestResponse
from agents.exceptions import CustomAgentException, ErrorCode
from agents.middleware.auth_middleware import get_current_user, get_optional_current_user
from agents.models.db import get_db
from agents.protocol.schemas import DialogueRequest
from agents.services import open_service, agent_service

router = APIRouter()
logger = logging.getLogger(__name__)

class OpenAPIExample(BaseModel):
    """Code examples for open platform API integration"""
    curl: str
    python: str
    golang: str
    nodejs: str

class TokenRequest(BaseModel):
    """Request model for token generation"""
    access_key: str
    secret_key: str

@router.get("/credentials", summary="Get Open Platform API Credentials")
async def get_api_credentials(
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Retrieve or generate open platform API credentials.
    
    Returns:
        - access_key: The API access key
        - secret_key: The API secret key
    """
    try:
        result = await open_service.get_or_create_credentials(user, session)
        return RestResponse(data=result)
    except CustomAgentException as e:
        logger.error(f"Error getting open platform credentials: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting open platform credentials: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )

@router.get("/example", summary="Get API Usage Examples")
async def get_example(
    agent_id: Optional[str] = Query(None, description="Optional agent ID to use in examples"),
    user: Optional[dict] = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Retrieve code examples for API integration in multiple languages.
    
    If authenticated, examples will include actual credentials.
    If not authenticated, examples will use placeholder values.
    
    Parameters:
        - agent_id: Optional agent ID to use in examples
    
    Returns:
        - curl: Shell command example
        - python: Python code example
        - golang: Go code example
        - nodejs: Node.js code example
    """
    # Get token if user is authenticated
    token = "your_api_token"
    if user:
        try:
            # Get or create credentials for the user
            credentials = await open_service.get_or_create_credentials(user, session)
            token = credentials["token"]
        except Exception:
            logger.warning("Failed to get user token for examples", exc_info=True)
    
    # Get base URL from config
    from agents.common.config import SETTINGS
    base_url = SETTINGS.API_BASE_URL or "https://api.example.com"
    
    # Use provided agent_id or placeholder
    example_agent_id = agent_id or "your-agent-id"
    
    # Token authentication example
    curl_token_example = f'''# Using token authentication
# Get your API token from the platform dashboard or API settings

# Use token to call API (streaming response)
curl -X POST "{base_url}/api/open/agents/{example_agent_id}/dialogue" \\
    -H "X-API-Token: {token}" \\
    -H "Content-Type: application/json" \\
    -d '{{"message": "Hello, Agent!", "conversation_id": "optional-conversation-id", "init_flag": false}}'

# Response will be returned as a stream, with each line starting with "data: " containing the AI's reply
'''

    # Python example - Token authentication
    python_token_example = f'''# Using Token authentication
import requests

# Your API token from the platform dashboard or API settings
token = "{token}"

# Use streaming response
response = requests.post(
    "{base_url}/api/open/agents/{example_agent_id}/dialogue",
    headers={{
        "X-API-Token": token
    }},
    json={{
        "message": "Hello, Agent!",
        "conversation_id": "optional-conversation-id",
        "init_flag": False
    }},
    stream=True  # Enable streaming
)

# Process streaming response
for chunk in response.iter_lines():
    if chunk:
        # Filter out empty lines
        data = chunk.decode('utf-8')
        if data.startswith('data: '):
            # Remove 'data: ' prefix
            content = data[6:]
            print(content)
'''

    # Go example - Token authentication
    golang_token_example = f'''// Using Token authentication
package main

import (
    "bufio"
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
)

func main() {{
    // Your API token from the platform dashboard or API settings
    token := "{token}"
    
    // Use token to call API
    apiURL := "{base_url}/api/open/agents/{example_agent_id}/dialogue"
    apiReqBody := []byte(`{{
        "message": "Hello, Agent!",
        "conversation_id": "optional-conversation-id",
        "init_flag": false
    }}`)
    
    apiReq, _ := http.NewRequest("POST", apiURL, bytes.NewBuffer(apiReqBody))
    apiReq.Header.Set("Content-Type", "application/json")
    apiReq.Header.Set("X-API-Token", token)
    
    client := &http.Client{{}}
    apiResp, err := client.Do(apiReq)
    if err != nil {{
        fmt.Println("Error calling API:", err)
        return
    }}
    defer apiResp.Body.Close()
    
    // Process streaming response
    scanner := bufio.NewScanner(apiResp.Body)
    for scanner.Scan() {{
        line := scanner.Text()
        if strings.HasPrefix(line, "data: ") {{
            // Remove 'data: ' prefix
            content := line[6:]
            fmt.Println(content)
        }}
    }}
    
    if err := scanner.Err(); err != nil {{
        fmt.Println("Error reading response:", err)
    }}
}}
'''

    # Node.js example - Token authentication
    nodejs_token_example = f'''// TypeScript example
// First install required dependencies:
// npm install node-fetch@2 @types/node-fetch@2

import fetch from 'node-fetch';

async function callAPI(): Promise<void> {{
    // Your API token from the platform dashboard or API settings
    const token = "{token}";
    
    try {{
        const apiResponse = await fetch('{base_url}/api/open/agents/{example_agent_id}/dialogue', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
                'X-API-Token': token
            }},
            body: JSON.stringify({{
                message: 'Hello, Agent!',
                conversation_id: 'optional-conversation-id',
                init_flag: false
            }})
        }});
        
        // Process streaming response
        if (apiResponse.body) {{
            // Create a readable stream
            const reader = apiResponse.body;
            
            // Handle stream processing
            reader.on('readable', () => {{
                let chunk;
                while (null !== (chunk = reader.read())) {{
                    const lines = chunk.toString().split('\\n');
                    
                    for (const line of lines) {{
                        if (line.startsWith('data: ')) {{
                            // Remove 'data: ' prefix
                            const content = line.substring(6);
                            console.log(content);
                        }}
                    }}
                }}
            }});
            
            // Handle errors
            reader.on('error', (err) => {{
                console.error('Error reading stream:', err);
            }});
        }}
    }} catch (error) {{
        console.error('Error calling API:', error);
    }}
}}

// Execute function
callAPI().catch(console.error);
'''

    return RestResponse(data=OpenAPIExample(
        curl=curl_token_example,
        python=python_token_example,
        golang=golang_token_example,
        nodejs=nodejs_token_example
    ))

@router.post("/token", summary="Get Open Platform API Token")
async def get_api_token(
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Get Open Platform API Token.
    
    This endpoint allows users to obtain an API token using their user authentication.
    The token can be used for authentication via the X-API-Token header.
    
    If the user already has a stored token, it will be returned;
    otherwise, a new token will be generated and stored.
    
    Returns:
        - token: Simple token string
        - token_type: Bearer
    """
    try:
        # Get or create credentials for the user
        credentials = await open_service.get_or_create_credentials(user, session)
        stored_token = credentials["access_key"]

        return RestResponse(data={
            "token": stored_token,
            # "token_type": "Bearer"
        })
    except CustomAgentException as e:
        logger.error(f"Error generating open platform token: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error generating open platform token: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )

@router.post("/token/reset", summary="Reset Open Platform API Token")
async def reset_api_token(
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Reset Open Platform API Token.
    
    This endpoint allows users to reset their Open Platform API Token. After reset, the old token will be invalidated,
    and the new token must be used for authentication.
    
    Returns:
        - token: New encrypted token
        - expires_at: Token expiration timestamp
    """
    try:
        # Get the user's access_key
        credentials = await open_service.get_or_create_credentials(user, session)
        access_key = credentials["access_key"]
        
        # Reset token
        token_data = await open_service.reset_token(access_key, session)
        return RestResponse(data=token_data)
    except CustomAgentException as e:
        logger.error(f"Error resetting open platform token: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error resetting open platform token: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        )

@router.post("/agents/{agent_id}/dialogue", summary="Open API Agent Dialogue")
async def open_dialogue(
    agent_id: str,
    request_data: dict = Body(..., description="Dialogue request data"),
    user: Optional[dict] = Depends(get_optional_current_user),
    session: AsyncSession = Depends(get_db)
):
    """
    Open API endpoint for agent dialogue.
    
    This endpoint allows third-party applications to interact with agents through the Open API.
    Authentication is required via either token or signature method.
    
    Parameters:
        - agent_id: ID of the agent to interact with
        - request_data: JSON object containing dialogue parameters
            - message: Message from the user (required)
            - conversation_id: ID of the conversation (optional)
            - init_flag: Flag to indicate if this is an initialization dialogue (optional, default: False)
    
    Returns:
        Streaming response with agent's reply
    """
    try:
        # Convert the request data to DialogueRequest format
        dialogue_request = DialogueRequest(
            query=request_data.get("message", ""),
            conversationId=request_data.get("conversation_id", None),
            initFlag=request_data.get("init_flag", False)
        )
        
        # Call the agent service dialogue method
        resp = agent_service.dialogue(agent_id, dialogue_request, user, session)
        
        return StreamingResponse(content=resp, media_type="text/event-stream")
    except CustomAgentException as e:
        logger.error(f"Error in open dialogue: {str(e)}", exc_info=True)
        return RestResponse(code=e.error_code, msg=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in open dialogue: {str(e)}", exc_info=True)
        return RestResponse(
            code=ErrorCode.INTERNAL_ERROR,
            msg=get_error_message(ErrorCode.INTERNAL_ERROR)
        ) 