from datetime import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel

from agents.protocol.schemas import ToolType, AgentMode, AgentStatus


class ToolModel(BaseModel):
    id: str  # UUID string
    name: str
    type: ToolType  # Enum type for tool type
    origin: str
    path: str
    method: str
    parameters: Dict
    description: Optional[str] = None
    auth_config: Optional[Dict] = None
    is_public: bool = False
    is_official: bool = False
    tenant_id: Optional[str] = None
    update_time: Optional[datetime] = None
    create_time: Optional[datetime] = None
    is_stream: bool = False
    output_format: Optional[Dict] = None

    class Config:
        from_attributes = True


class AppModel(BaseModel):
    id: str  # UUID string
    name: str
    description: Optional[str] = None
    mode: Optional[AgentMode] = None  # Enum type for agent mode
    icon: Optional[str] = None
    status: Optional[AgentStatus] = None  # Enum type for agent status
    role_settings: Optional[str] = None
    welcome_message: Optional[str] = None
    twitter_link: Optional[str] = None
    telegram_bot_id: Optional[str] = None
    tool_prompt: Optional[str] = None
    max_loops: int = 3
    is_deleted: bool = False
    tenant_id: Optional[str] = None
    is_public: bool = False
    is_official: bool = False
    model_id: Optional[int] = None
    suggested_questions: Optional[List[str]] = None
    update_time: Optional[datetime] = None
    create_time: Optional[datetime] = None
    tools: Optional[List[ToolModel]] = None

    class Config:
        from_attributes = True
