import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal, Union

from pydantic import BaseModel, Field, EmailStr


class ToolType(str, Enum):
    OPENAPI = "openapi"
    FUNCTION = "function"


class AgentMode(str, Enum):
    REACT = "ReAct"
    CALL = "call"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"  # New draft status added


class AuthLocationType(str, Enum):
    HEADER = "header"
    PARAM = "param"


class AuthConfig(BaseModel):
    location: AuthLocationType = Field(..., description="Where to add the auth parameter")
    key: str = Field(..., description="Name of the auth parameter")
    value: str = Field(..., description="Value of the auth parameter")


class ToolInfo(BaseModel):
    id: Optional[str] = Field(None, description="Tool UUID")
    name: str = Field(..., description="Name of the tool")
    type: ToolType = Field(default=ToolType.OPENAPI, description='Type of the tool')
    content: str = Field(..., description="Content or configuration details of the tool")
    is_public: Optional[bool] = Field(False, description="Whether the tool is public")
    tenant_id: Optional[str] = Field(None, description="Tenant ID that owns this tool")


class AgentDTO(BaseModel):
    name: Optional[str] = Field(None, description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent")
    mode: Optional[AgentMode] = Field(None, description='Mode of the agent')
    icon: Optional[str] = Field(None, description="Optional icon for the agent")
    role_settings: Optional[str] = Field(None, description="Optional roles for the agent")
    welcome_message: Optional[str] = Field(None, description="Optional welcome message for the agent")
    twitter_link: Optional[str] = Field(None, description="Optional twitter link for the agent")
    telegram_bot_id: Optional[str] = Field(None, description="Optional telegram bot id for the agent")
    status: Optional[AgentStatus] = Field(default=None,
                                          description="Status can be active, inactive, or draft")
    tool_prompt: Optional[str] = Field(None, description="Optional tool prompt for the agent")
    max_loops: Optional[int] = Field(default=None, description="Maximum number of loops the agent can perform")
    tools: Optional[List[str]] = Field(
        default=None, 
        description="List of tool UUIDs to associate with the agent"
    )
    id: Optional[str] = Field(default=None, description="Optional ID of the tool, used for identifying existing tools")
    suggested_questions: Optional[List[str]] = Field(
        default=None, 
        description="List of suggested questions for users to ask"
    )
    model_id: Optional[int] = Field(None, description="ID of the associated model")


class AICreateAgentDTO(BaseModel):
    description: str = Field(..., description="Description of the agent")


class ToolCreate(BaseModel):
    name: str = Field(..., description="Name of the tool")
    type: ToolType = Field(default=ToolType.OPENAPI, description='Type of the tool')
    content: str = Field(..., description="Content or configuration details of the tool")
    auth_config: Optional[AuthConfig] = Field(None, description="Authentication configuration")


class ToolUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Optional new name for the tool")
    type: ToolType = Field(default=ToolType.OPENAPI, description='Type of the tool')
    content: Optional[str] = Field(None, description="Optional new content for the tool")
    auth_config: Optional[AuthConfig] = Field(None, description="Authentication configuration")


class DialogueRequest(BaseModel):
    query: Optional[str] = None
    conversation_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), alias="conversationId")


class DialogueResponse(BaseModel):
    response: str = Field(..., description="Response message from the agent")


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1, description="Page number (starts from 1)")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of items per page")


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username or email for login")
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    message: str
    user: dict


class WalletLoginRequest(BaseModel):
    """Request for wallet login/registration"""
    wallet_address: str
    signature: Optional[str] = None


class NonceResponse(BaseModel):
    """Response containing nonce for wallet signature"""
    nonce: str
    message: str


class WalletLoginResponse(BaseModel):
    """Response for successful wallet login"""
    token: str
    user: dict
    is_new_user: bool


class AgentToolsRequest(BaseModel):
    tool_ids: List[str] = Field(..., description="List of tool UUIDs to assign/remove")


class ModelDTO(BaseModel):
    id: Optional[int] = Field(None, description="ID of the model")
    name: str = Field(..., description="Name of the model")
    endpoint: str = Field(..., description="API endpoint of the model")
    is_official: Optional[bool] = Field(False, description="Whether the model is official preset")
    is_public: Optional[bool] = Field(False, description="Whether the model is public")
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None


class ModelCreate(BaseModel):
    name: str = Field(..., description="Name of the model")
    endpoint: str = Field(..., description="API endpoint of the model")
    api_key: Optional[str] = Field(None, description="API key for the model")


class ModelUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the model")
    endpoint: Optional[str] = Field(None, description="API endpoint of the model")
    api_key: Optional[str] = Field(None, description="API key for the model")


class ToolModel(BaseModel):
    id: str
    name: str
    type: ToolType
    content: str
    auth_config: Optional[Dict] = None
    is_public: bool = False
    is_official: bool = False
    tenant_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
