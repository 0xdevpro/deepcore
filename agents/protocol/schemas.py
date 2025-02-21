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
    origin: str = Field(..., description="API origin")
    path: str = Field(..., description="API path")
    method: str = Field(..., description="HTTP method")
    auth_config: Optional[AuthConfig] = Field(None, description="Authentication configuration")
    parameters: Dict = Field(default_factory=dict, description="API parameters including header, query, path, and body")
    description: Optional[str] = Field(None, description="Description of the tool")
    is_public: Optional[bool] = Field(False, description="Whether the tool is public")
    tenant_id: Optional[str] = Field(None, description="Tenant ID that owns this tool")
    is_stream: Optional[bool] = Field(False, description="Whether the API returns a stream response")
    output_format: Optional[Dict] = Field(None, description="JSON configuration for formatting API output")


class CategoryType(str, Enum):
    AGENT = "agent"
    TOOL = "tool"


class CategoryCreate(BaseModel):
    name: str = Field(..., description="Name of the category")
    type: CategoryType = Field(..., description="Type of the category")
    description: Optional[str] = Field(None, description="Description of the category")
    sort_order: Optional[int] = Field(0, description="Sort order for display")


class CategoryUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the category")
    description: Optional[str] = Field(None, description="Description of the category")
    sort_order: Optional[int] = Field(None, description="Sort order for display")


class CategoryDTO(BaseModel):
    id: int = Field(..., description="ID of the category")
    name: str = Field(..., description="Name of the category")
    type: CategoryType = Field(..., description="Type of the category")
    description: Optional[str] = Field(None, description="Description of the category")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    sort_order: int = Field(0, description="Sort order for display")
    create_time: Optional[str] = Field(None, description="Creation time")
    update_time: Optional[str] = Field(None, description="Last update time")


class AgentDTO(BaseModel):
    id: Optional[str] = Field(default=None, description="ID of the agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    mode: AgentMode = Field(default=AgentMode.REACT, description='Mode of the agent')
    icon: Optional[str] = Field(None, description="Optional icon for the agent")
    role_settings: Optional[str] = Field(None, description="Optional roles for the agent")
    welcome_message: Optional[str] = Field(None, description="Optional welcome message for the agent")
    twitter_link: Optional[str] = Field(None, description="Optional twitter link for the agent")
    telegram_bot_id: Optional[str] = Field(None, description="Optional telegram bot id for the agent")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Status can be active, inactive, or draft")
    tool_prompt: Optional[str] = Field(None, description="Optional tool prompt for the agent")
    max_loops: int = Field(default=3, description="Maximum number of loops the agent can perform")
    tools: Optional[List[Union[str, ToolInfo]]] = Field(
        default_factory=list, 
        description="List of tool UUIDs to associate with the agent when creating/updating, or list of ToolInfo when getting agent details"
    )
    suggested_questions: Optional[List[str]] = Field(
        default_factory=list, 
        description="List of suggested questions for users to ask"
    )
    model_id: Optional[int] = Field(None, description="ID of the associated model")
    category_id: Optional[int] = Field(None, description="ID of the category")
    category: Optional[CategoryDTO] = Field(None, description="Category information")

    class Config:
        from_attributes = True


class AICreateAgentDTO(BaseModel):
    description: str = Field(..., description="Description of the agent")


class APIToolData(BaseModel):
    """Base model for API tool data"""
    name: str = Field(..., description="Name of the API tool")
    description: Optional[str] = Field(None, description="Description of the Tool")
    origin: str = Field(..., description="API origin")
    path: str = Field(..., description="API path")
    method: str = Field(..., description="HTTP method")
    parameters: Dict = Field(default_factory=dict, description="API parameters including header, query, path, and body")
    auth_config: Optional[AuthConfig] = Field(None, description="Authentication configuration")
    is_stream: Optional[bool] = Field(False, description="Whether the API returns a stream response")
    output_format: Optional[Dict] = Field(None, description="JSON configuration for formatting API output")


class ToolCreate(BaseModel):
    """Request model for creating a single tool"""
    tool_data: APIToolData = Field(..., description="API tool configuration data")


class ToolUpdate(BaseModel):
    """Request model for updating a tool"""
    name: Optional[str] = Field(None, description="Optional new name for the tool")
    description: Optional[str] = Field(None, description="Description of the Tool")
    origin: Optional[str] = Field(None, description="Optional new API origin")
    path: Optional[str] = Field(None, description="Optional new API path")
    method: Optional[str] = Field(None, description="Optional new HTTP method")
    parameters: Optional[Dict] = Field(None, description="Optional new API parameters")
    auth_config: Optional[AuthConfig] = Field(None, description="Optional new authentication configuration")
    is_stream: Optional[bool] = Field(None, description="Whether the API returns a stream response")
    output_format: Optional[Dict] = Field(None, description="JSON configuration for formatting API output")


class DialogueRequest(BaseModel):
    query: str = Field(..., description="Query message from the user")
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
    access_token: str
    refresh_token: str
    user: dict
    access_token_expires_in: int  # in seconds
    refresh_token_expires_in: int  # in seconds


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
    access_token: str
    refresh_token: str
    user: dict
    is_new_user: bool
    access_token_expires_in: int  # in seconds
    refresh_token_expires_in: int  # in seconds


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
    """Model for tool data"""
    id: str
    name: str
    type: ToolType = Field(default=ToolType.OPENAPI)
    origin: str
    path: str
    method: str
    parameters: Dict
    auth_config: Optional[Dict] = None
    is_public: bool = False
    is_official: bool = False
    tenant_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    category_id: Optional[int] = Field(None, description="ID of the category")
    category: Optional[CategoryDTO] = Field(None, description="Category information")


class RefreshTokenRequest(BaseModel):
    """Request for refreshing access token"""
    refresh_token: str


class TokenResponse(BaseModel):
    """Response containing new access token"""
    access_token: str
    refresh_token: str
    access_token_expires_in: int  # in seconds
    refresh_token_expires_in: int  # in seconds
    user: dict


class CreateOpenAPIToolRequest(BaseModel):
    """Request model for creating OpenAPI tools"""
    name: str = Field(..., description="Base name for the tools")
    api_list: List[dict] = Field(..., description="List of API endpoint information")
    auth_config: Optional[AuthConfig] = Field(None, description="Authentication configuration")


class CreateToolsBatchRequest(BaseModel):
    """Request model for creating multiple tools in batch"""
    tools: List[APIToolData] = Field(..., description="List of API tool configurations")
