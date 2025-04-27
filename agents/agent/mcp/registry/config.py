"""
Configuration settings for MCP server registry.
"""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any
import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class MCPServerAuthSettings(BaseModel):
    """Authentication settings for MCP server."""
    api_key: Optional[str] = None
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_env(cls) -> "MCPServerAuthSettings":
        """Create auth settings from environment variables."""
        return cls(api_key=os.getenv("MCP_API_KEY"))


class MCPServerSettings(BaseModel):
    """Configuration for an individual MCP server."""
    name: Optional[str] = None
    description: Optional[str] = None
    transport: Literal["stdio", "websocket"] = "stdio"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    read_timeout_seconds: Optional[int] = None
    url: Optional[str] = None
    auth: Optional[MCPServerAuthSettings] = None
    headers: Optional[Dict[str, str]] = None
    env: Optional[Dict[str, str]] = None
    model_config = ConfigDict(extra="allow")

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Validate and merge environment variables."""
        if v is None:
            return {}
        return {**os.environ, **v}

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, v: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Validate headers."""
        if v is None:
            return {}
        return v

    def get_full_command(self) -> List[str]:
        """Get the full command with arguments."""
        if not self.command:
            return []
        return [self.command] + (self.args or [])


class MCPSettings(BaseModel):
    """Configuration for all MCP servers."""
    servers: Dict[str, MCPServerSettings] = {}
    model_config = ConfigDict(extra="allow")

    def get_server(self, name: str) -> Optional[MCPServerSettings]:
        """Get server configuration by name."""
        return self.servers.get(name)

    def add_server(self, name: str, config: MCPServerSettings) -> None:
        """Add a new server configuration."""
        self.servers[name] = config

    def remove_server(self, name: str) -> None:
        """Remove a server configuration."""
        self.servers.pop(name, None)


class Settings(BaseModel):
    """Main settings class for MCP registry."""
    mcp: MCPSettings = MCPSettings()
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_file(cls, path: str) -> "Settings":
        """Load settings from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return cls(**data)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls()

    def save(self, path: str) -> None:
        """Save settings to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def get_settings(config_path: Optional[str] = None) -> Settings:
    """Get settings from configuration file or environment."""
    if config_path:
        return Settings.from_file(config_path)
    
    # Try to find config file in current directory
    config_file = Path("mcp.config.yaml")
    if config_file.exists():
        return Settings.from_file(str(config_file))
    
    # Fallback to environment variables
    return Settings.from_env()
