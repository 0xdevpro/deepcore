import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Union, Any

from pydantic import BaseModel, Field, EmailStr


class ToolType(str, Enum):
    OPENAPI = "openapi"
    FUNCTION = "function"
    MCP = "mcp"


class AgentMode(str, Enum):
    REACT = "ReAct"  # ReAct mode for complex task decomposition and tool calling
    PROMPT = "Prompt"  # Simple prompt mode for direct conversation
    CALL = "call"  # Legacy mode for backward compatibility
    DEEP_THINKING = "DeepThinking"  # Advanced mode with sophisticated cognitive processing capabilities


class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"  # New draft status added


class AuthLocationType(str, Enum):
    HEADER = "header"
    PARAM = "param"


class ChainType(str, Enum):
    """Blockchain types supported for wallet authentication"""
    ETHEREUM = "ethereum"
    SOLANA = "solana"
    # BSC = "bsc"
    # POLYGON = "polygon"
    # AVALANCHE = "avalanche"


class AuthConfig(BaseModel):
    location: AuthLocationType = Field(..., description="Where to add the auth parameter")
    key: str = Field(..., description="Name of the auth parameter")
    value: str = Field(..., description="Value of the auth parameter")


class ToolInfo(BaseModel):
    id: Optional[str] = Field(None, description="Tool UUID")
    name: str = Field(..., description="Name of the tool")
    type: ToolType = Field(default=ToolType.OPENAPI, description='Type of the tool')
    origin: Optional[str] = Field(..., description="API origin")
    path: str = Field(..., description="API path")
    method: str = Field(..., description="HTTP method")
    auth_config: Optional[AuthConfig | List[AuthConfig]] = Field(None, description="Authentication configuration")
    parameters: Dict = Field(default_factory=dict, description="API parameters including header, query, path, and body")
    description: Optional[str] = Field(None, description="Description of the tool")
    icon: Optional[str] = Field(None, description="Icon URL of the tool")
    is_public: Optional[bool] = Field(False, description="Whether the tool is public", exclude=True)
    tenant_id: Optional[str] = Field(None, description="Tenant ID that owns this tool")
    is_stream: Optional[bool] = Field(False, description="Whether the API returns a stream response")
    output_format: Optional[Dict] = Field(None, description="JSON configuration for formatting API output")
    sensitive_data_config: Optional[Dict] = Field(None, description="Configuration for sensitive data handling")


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


class ModelDTO(BaseModel):
    id: Optional[int] = Field(None, description="ID of the model")
    name: str = Field(..., description="Name of the model")
    model_name: str = Field(..., description="Name of the underlying model (e.g. gpt-4, claude-3)")
    endpoint: Optional[str] = Field(None, description="API endpoint of the model")
    is_official: Optional[bool] = Field(False, description="Whether the model is official preset")
    is_public: Optional[bool] = Field(False, description="Whether the model is public", exclude=True)
    icon: Optional[str] = Field(None, description="Icon URL of the model")
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None


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
    telegram_bot_name: Optional[str] = Field(None, description="Optional telegram bot name for the agent")
    token: Optional[str] = Field(None, description="Optional token for the agent")
    symbol: Optional[str] = Field(None, description="Optional symbol for the agent token")
    photos: Optional[List[str]] = Field(default_factory=list, description="Optional photos for the agent")
    demo_video: Optional[str] = Field(None, description="Optional demo video URL for the agent")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Status can be active, inactive, or draft")
    is_paused: Optional[bool] = Field(False, description="Whether the agent's conversation is paused", exclude=True)
    pause_message: Optional[str] = Field(None, description="Message to display when the agent is paused", exclude=True)
    tool_prompt: Optional[str] = Field(None, description="Optional tool prompt for the agent")
    max_loops: int = Field(default=3, description="Maximum number of loops the agent can perform")
    custom_config: Optional[Dict] = Field(None, description="Custom configuration for the agent")
    create_time: Optional[datetime] = Field(None, description="Creation time")
    update_time: Optional[datetime] = Field(None, description="Last update time")
    tools: Optional[List[Union[str, ToolInfo]]] = Field(
        default_factory=list,
        description="List of tool UUIDs to associate with the agent when creating/updating, or list of ToolInfo when getting agent details"
    )
    suggested_questions: Optional[List[str]] = Field(
        default_factory=list,
        description="List of suggested questions for users to ask"
    )
    model_id: Optional[int] = Field(None, description="ID of the associated model")
    model: Optional[ModelDTO] = Field(None, description="Associated model information")
    category_id: Optional[int] = Field(None, description="ID of the category")
    category: Optional[CategoryDTO] = Field(None, description="Category information")
    is_public: Optional[bool] = Field(False, description="Whether the agent is public", exclude=True)
    is_official: Optional[bool] = Field(False, description="Whether the agent is official", exclude=True)
    is_hot: Optional[bool] = Field(False, description="Whether the agent is hot", exclude=True)
    create_fee: Optional[float] = Field(None, description="Creation fee for the agent")
    price: Optional[float] = Field(None, description="Price for the agent")
    mcp_url: Optional[str] = Field(None, description="MCP url for the agent")
    vip_level: Optional[int] = Field(0, description="VIP level required to access this agent")
    shouldInitializeDialog: Optional[bool] = Field(False,
                                                   description="Whether to initialize dialog when creating the agent")
    initializeDialogQuestion: Optional[str] = Field(None, description="Question to send when initializing dialog")
    dev: Optional[str] = Field(None, description="Developer wallet address")
    tenant_id: Optional[str] = None
    enable_mcp: Optional[bool] = Field(False, description="Whether MCP is enabled for this agent")

    class Config:
        from_attributes = True


class AICreateAgentDTO(BaseModel):
    description: str = Field(..., description="Description of the agent")


class APIToolData(BaseModel):
    """Base model for API tool data"""
    name: str = Field(..., description="Name of the API tool")
    type: ToolType = Field(default=ToolType.OPENAPI, description='Type of the tool')
    description: Optional[str] = Field(None, description="Description of the Tool")
    origin: str = Field(..., description="API origin")
    path: str = Field(..., description="API path")
    method: str = Field(..., description="HTTP method")
    parameters: Dict = Field(default_factory=dict, description="API parameters including header, query, path, and body")
    auth_config: Optional[AuthConfig | List[AuthConfig]] = Field(None, description="Authentication configuration")
    icon: Optional[str] = Field(None, description="Icon URL of the tool")
    is_stream: Optional[bool] = Field(False, description="Whether the API returns a stream response")
    output_format: Optional[Dict] = Field(None, description="JSON configuration for formatting API output")
    sensitive_data_config: Optional[Dict] = Field(None, description="Configuration for sensitive data handling")


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
    auth_config: Optional[AuthConfig | List[AuthConfig]] = Field(None,
                                                                 description="Optional new authentication configuration")
    icon: Optional[str] = Field(None, description="Icon URL of the tool")
    is_stream: Optional[bool] = Field(None, description="Whether the API returns a stream response")
    output_format: Optional[Dict] = Field(None, description="JSON configuration for formatting API output")
    sensitive_data_config: Optional[Dict] = Field(None, description="Configuration for sensitive data handling")


class DialogueRequest(BaseModel):
    query: str = Field(..., description="Query message from the user")
    conversation_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), alias="conversationId")
    initFlag: Optional[bool] = Field(False, description="Flag to indicate if this is an initialization dialogue")
    model_id: Optional[int] = Field(None, description="Optional model ID to override the agent's default model")


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
    chain_type: Optional[ChainType] = Field(ChainType.ETHEREUM, description="Blockchain type for wallet authentication")


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


class ModelCreate(BaseModel):
    name: str = Field(..., description="Name of the model")
    model_name: str = Field(..., description="Name of the underlying model (e.g. gpt-4, claude-3)")
    endpoint: str = Field(..., description="API endpoint of the model")
    api_key: Optional[str] = Field(None, description="API key for the model")
    icon: Optional[str] = Field(None, description="Icon URL of the model")


class ModelUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Name of the model")
    model_name: Optional[str] = Field(None, description="Name of the underlying model (e.g. gpt-4, claude-3)")
    endpoint: Optional[str] = Field(None, description="API endpoint of the model")
    api_key: Optional[str] = Field(None, description="API key for the model")
    icon: Optional[str] = Field(None, description="Icon URL of the model")


class ToolModel(BaseModel):
    """Model for tool data"""
    id: str
    name: str
    type: ToolType = Field(default=ToolType.OPENAPI)
    origin: str
    path: str
    method: str
    parameters: Dict
    auth_config: Optional[Dict | List] = None
    icon: Optional[str] = Field(None, description="Icon URL of the tool")
    is_public: bool = False
    is_official: bool = False
    tenant_id: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    category_id: Optional[int] = Field(None, description="ID of the category")
    category: Optional[CategoryDTO] = Field(None, description="Category information")
    sensitive_data_config: Optional[Dict] = Field(None, description="Configuration for sensitive data handling")


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
    auth_config: Optional[AuthConfig | List[AuthConfig]] = Field(None, description="Authentication configuration")


class CreateToolsBatchRequest(BaseModel):
    """Request model for creating multiple tools in batch"""
    tools: List[APIToolData] = Field(..., description="List of API tool configurations")


class OpenAPIParseRequest(BaseModel):
    """Request model for parsing OpenAPI content"""
    content: str = Field(..., description="OpenAPI specification content (JSON or YAML format)")


class TelegramBotRequest(BaseModel):
    """Request model for registering a Telegram bot"""
    bot_name: str = Field(..., description="Name of the Telegram bot")
    token: str = Field(..., description="Telegram bot token")


class AgentSettingRequest(BaseModel):
    """Request model for agent settings"""
    token: Optional[str] = Field(None, description="Token for the agent")
    symbol: Optional[str] = Field(None, description="Symbol for the agent token")
    photos: Optional[List[str]] = Field(default_factory=list, description="Photos for the agent")
    telegram_bot_name: Optional[str] = Field(None, description="Name of the Telegram bot")
    telegram_bot_token: Optional[str] = Field(None, description="Telegram bot token")


class AgentContextStoreRequest(BaseModel):
    """Request model for storing agent context data in Redis"""
    conversation_id: str = Field(..., description="Conversation ID")
    scenario: str = Field(..., description="Scenario identifier for the context data")
    data: Dict = Field(..., description="Context data to store")
    ttl: Optional[int] = Field(86400, description="Time to live in seconds (default: 24 hours)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict,
                                               description="Additional metadata for the context data")


class ImgProRemainingResp(BaseModel):
    """ImgProRemaining"""
    enabled: bool = Field(..., description="can upload img")


class ImgProTaskReq(BaseModel):
    """ImgProTaskReq"""
    img_url: str = Field(..., description="img url")
    gen_img_type: int = Field(0, description="gen img type")


class ImgProTaskResp(BaseModel):
    """ImgProTaskResp"""
    task_id: str = Field(..., description="task id")
    img_url: str = Field(..., description="img url")
    result_img_url: Optional[str] = Field("", description="Result img url")
    status: Optional[str] = Field("TODO", description="status")


class VipMembershipDTO(BaseModel):
    """VIP Membership DTO"""
    id: int
    user_id: int
    level: int
    start_time: datetime
    expire_time: datetime
    status: str
    create_time: datetime
    update_time: datetime


class VipPackageDTO(BaseModel):
    """VIP Package DTO"""
    id: int
    name: str
    level: int
    duration: int
    price: float
    description: Optional[str] = None
    features: Optional[dict] = None
    is_active: bool
    create_time: datetime
    update_time: datetime


class VipOrderDTO(BaseModel):
    """VIP Order DTO"""
    id: int
    order_no: str
    user_id: int
    package_id: int
    amount: float
    status: str
    payment_method: Optional[str] = None
    payment_time: Optional[datetime] = None
    create_time: datetime
    update_time: datetime


class VipPackageCreateDTO(BaseModel):
    """Create VIP Package DTO"""
    name: str
    level: int
    duration: int
    price: float
    description: Optional[str] = None
    features: Optional[dict] = None


class VipOrderCreateDTO(BaseModel):
    """Create VIP Order DTO"""
    package_id: int
    amount: float
    payment_method: str


class PublishAgentToStoreRequest(BaseModel):
    """Request model for publishing an agent to MCP Store"""
    name: Optional[str] = None
    icon: Optional[str] = None
    tags: Optional[List[str]] = None
    github_url: Optional[str] = None


class DepositRequest(BaseModel):
    wallet_type: str = ""
    tx_hash: str = ""
    amount: Decimal
    from_wallet: str = ""
    to_wallet: str = ""


class DepositInfo(DepositRequest):
    status: str = ""
    transaction_ts: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def give(amout: Decimal):
        return DepositInfo(
            amount=amout,
            wallet_type="give",
            status="give"
        )


class ProfileInfo(BaseModel):
    tenant_id: str
    api_key: str = ""
    deposit_history: List[DepositInfo] = []
    balance: Decimal = Field(default=Decimal("0.0"))
    total_spend: Decimal = Field(default=Decimal("0.0"))
    total_requests_count: int = Field(default=0)
    wallet_address: str = Field(default="", description="User wallet address")
    master_address: str = Field(default="", description="Master wallet address for transactions")
