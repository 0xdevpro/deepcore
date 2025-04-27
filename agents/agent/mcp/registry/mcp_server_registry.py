"""
Server registry for managing MCP server configurations and initialization.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Callable, Dict, AsyncGenerator, Optional, List, Any

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
import websockets

from .config import MCPServerSettings, MCPServerAuthSettings, Settings, get_settings
from ..core.server.connection import ClientSession


logger = logging.getLogger(__name__)

InitHookCallable = Callable[[Optional[ClientSession], Optional[MCPServerAuthSettings]], bool]


class ServerRegistry:
    """
    Registry for managing MCP server configurations and initialization.
    """

    def __init__(self, config: Optional[Settings] = None, config_path: Optional[str] = None):
        """
        Initialize the ServerRegistry with configuration.

        Args:
            config: The Settings object containing server configurations
            config_path: Path to the configuration file
        """
        self.registry = (
            self.load_registry_from_file(config_path)
            if config is None
            else config.mcp.servers
        )
        self.init_hooks: Dict[str, InitHookCallable] = {}
        self.active_sessions: Dict[str, ClientSession] = {}

    def load_registry_from_file(
        self, config_path: Optional[str] = None
    ) -> Dict[str, MCPServerSettings]:
        """
        Load server configurations from file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary of server configurations
        """
        servers = get_settings(config_path).mcp.servers or {}
        return servers

    async def _create_stdio_session(
        self,
        config: MCPServerSettings,
        client_session_factory: Callable[..., ClientSession],
        read_timeout: Optional[timedelta],
    ) -> ClientSession:
        """Create a stdio transport session."""
        process = await asyncio.create_subprocess_exec(
            *config.get_full_command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=config.env,
        )

        if not process.stdin or not process.stdout:
            raise RuntimeError("Failed to create subprocess pipes")

        read_stream = MemoryObjectReceiveStream()
        write_stream = MemoryObjectSendStream()

        # Start background tasks for reading/writing
        async def read_task():
            try:
                while True:
                    data = await process.stdout.read(4096)
                    if not data:
                        break
                    await read_stream.send(data)
            except Exception as e:
                logger.error(f"Error reading from process: {e}")
            finally:
                await read_stream.aclose()

        async def write_task():
            try:
                while True:
                    data = await write_stream.receive()
                    process.stdin.write(data)
                    await process.stdin.drain()
            except Exception as e:
                logger.error(f"Error writing to process: {e}")
            finally:
                process.stdin.close()

        asyncio.create_task(read_task())
        asyncio.create_task(write_task())

        return client_session_factory(read_stream, write_stream, read_timeout)

    async def _create_websocket_session(
        self,
        config: MCPServerSettings,
        client_session_factory: Callable[..., ClientSession],
        read_timeout: Optional[timedelta],
    ) -> ClientSession:
        """Create a websocket transport session."""
        if not config.url:
            raise ValueError("WebSocket URL is required")

        websocket = await websockets.connect(
            config.url,
            extra_headers=config.headers,
        )

        read_stream = MemoryObjectReceiveStream()
        write_stream = MemoryObjectSendStream()

        async def read_task():
            try:
                while True:
                    data = await websocket.recv()
                    await read_stream.send(data)
            except Exception as e:
                logger.error(f"Error reading from websocket: {e}")
            finally:
                await read_stream.aclose()

        async def write_task():
            try:
                while True:
                    data = await write_stream.receive()
                    await websocket.send(data)
            except Exception as e:
                logger.error(f"Error writing to websocket: {e}")
            finally:
                await websocket.close()

        asyncio.create_task(read_task())
        asyncio.create_task(write_task())

        return client_session_factory(read_stream, write_stream, read_timeout)

    @asynccontextmanager
    async def start_server(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] = ClientSession,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Start a server based on its configuration.

        Args:
            server_name: Name of the server to start
            client_session_factory: Factory function for creating client sessions

        Yields:
            ClientSession: The initialized client session

        Raises:
            ValueError: If server not found or transport not supported
        """
        if server_name not in self.registry:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        config = self.registry[server_name]
        read_timeout = (
            timedelta(seconds=config.read_timeout_seconds)
            if config.read_timeout_seconds
            else None
        )

        try:
            if config.transport == "stdio":
                session = await self._create_stdio_session(
                    config, client_session_factory, read_timeout
                )
            elif config.transport == "websocket":
                session = await self._create_websocket_session(
                    config, client_session_factory, read_timeout
                )
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

            self.active_sessions[server_name] = session
            yield session

        except Exception as e:
            logger.error(f"Failed to start server {server_name}: {e}")
            raise
        finally:
            self.active_sessions.pop(server_name, None)

    @asynccontextmanager
    async def initialize_server(
        self,
        server_name: str,
        client_session_factory: Callable[..., ClientSession] = ClientSession,
        init_hook: Optional[InitHookCallable] = None,
    ) -> AsyncGenerator[ClientSession, None]:
        """
        Initialize a server and execute initialization hooks.

        Args:
            server_name: Name of the server to initialize
            client_session_factory: Factory function for creating client sessions
            init_hook: Optional initialization hook to execute

        Yields:
            ClientSession: The initialized client session
        """
        async with self.start_server(server_name, client_session_factory) as session:
            try:
                await session.initialize()
                initialization_callback = init_hook or self.init_hooks.get(server_name)
                if initialization_callback:
                    initialization_callback(session, self.registry[server_name].auth)
                yield session
            except Exception as e:
                logger.error(f"Failed to initialize server {server_name}: {e}")
                raise RuntimeError(f"Failed to initialize server {server_name}: {str(e)}")

    def register_init_hook(self, server_name: str, hook: InitHookCallable) -> None:
        """
        Register an initialization hook for a server.

        Args:
            server_name: Name of the server
            hook: Initialization hook function
        """
        self.init_hooks[server_name] = hook

    def get_server_config(self, server_name: str) -> Optional[MCPServerSettings]:
        """
        Get configuration for a server.

        Args:
            server_name: Name of the server

        Returns:
            Server configuration if found, None otherwise
        """
        return self.registry.get(server_name)

    def get_active_session(self, server_name: str) -> Optional[ClientSession]:
        """
        Get the active session for a server.

        Args:
            server_name: Name of the server

        Returns:
            Active session if found, None otherwise
        """
        return self.active_sessions.get(server_name)

    def list_servers(self) -> List[str]:
        """
        List all registered servers.

        Returns:
            List of server names
        """
        return list(self.registry.keys())

    def list_active_servers(self) -> List[str]:
        """
        List all active servers.

        Returns:
            List of active server names
        """
        return list(self.active_sessions.keys())

    async def stop_server(self, server_name: str) -> None:
        """
        Stop a running server.

        Args:
            server_name: Name of the server to stop
        """
        session = self.active_sessions.get(server_name)
        if session:
            await session.close()
            self.active_sessions.pop(server_name)
