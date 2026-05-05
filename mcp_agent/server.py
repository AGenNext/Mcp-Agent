"""MCP Protocol Server implementation."""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import anyio
from pydantic import BaseModel

from mcp_agent.tool import ToolRegistry, ToolResult


class InitializeResult(BaseModel):
    """Result from initialize request."""
    
    protocolVersion: str = "2024-11-05"
    capabilities: dict[str, Any] = field(default_factory=dict)
    serverInfo: dict[str, str] = field(
        default_factory=lambda: {
            "name": "mcp-agent-server",
            "version": "0.1.0",
        }
    )


@dataclass
class MCPServerConfig:
    """Configuration for MCP server."""
    
    host: str = "0.0.0.0"
    port: int = 8080
    tool_registry: ToolRegistry | None = None


class MCPServer:
    """MCP Protocol Server for exposing tools."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.tool_registry = config.tool_registry or ToolRegistry()
        self._request_handlers: dict[str, Callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Set up JSON-RPC request handlers."""
        self._request_handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_list_tools,
            "tools/call": self._handle_call_tool,
            "tools/async/list": self._handle_list_tools,
            "tools/async/call": self._handle_async_call_tool,
        }
    
    async def _handle_initialize(self, params: dict) -> InitializeResult:
        """Handle initialize request."""
        return InitializeResult()
    
    async def _handle_list_tools(self, params: dict) -> dict[str, Any]:
        """Handle tools/list request."""
        schemas = self.tool_registry.get_schemas()
        return {
            "tools": [s.model_dump() for s in schemas]
        }
    
    async def _handle_call_tool(self, params: dict) -> dict[str, Any]:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        result = await self.tool_registry.execute(name, **arguments)
        
        if result.success:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result.result),
                    }
                ]
            }
        else:
            return {
                "content": [],
                "isError": True,
                "error": result.error,
            }
    
    async def _handle_async_call_tool(self, params: dict) -> dict[str, Any]:
        """Handle tools/async/call request."""
        return await self._handle_call_tool(params)
    
    async def _handle_request(self, request: dict) -> dict:
        """Handle incoming JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        handler = self._request_handlers.get(method)
        if handler is None:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,  # Method not found
                    "message": f"Method '{method}' not found",
                },
                "id": request_id,
            }
        
        try:
            result = await handler(params)
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e),
                },
                "id": request_id,
            }
    
    async def _handle_client(self, stream: anyio.abc.ByteStream) -> None:
        """Handle a single client connection."""
        async with stream:
            data = await stream.receive()
            request = json.loads(data.decode())
            
            if "method" in request:
                response = await self._handle_request(request)
                await stream.send(json.dumps(response).encode())
            else:
                # Batch request
                responses = []
                for req in request:
                    responses.append(await self._handle_request(req))
                await stream.send(json.dumps(responses).encode())
    
    async def start(self) -> None:
        """Start the MCP server."""
        print(f"Starting MCP server on {self.config.host}:{self.config.port}")
        print(f"Registered tools: {self.tool_registry.list_tools()}")
        
        async with anyio.create_tcp_listener(
            host=self.config.host,
            port=self.config.port,
        ) as listener:
            async for connection in listener:
                async with connection:
                    await self._handle_client(connection)
    
    def register_tool(self, tool) -> None:
        """Register a tool with the server."""
        self.tool_registry.register(tool)


async def create_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    tool_registry: ToolRegistry | None = None,
) -> MCPServer:
    """Create and configure an MCP server."""
    config = MCPServerConfig(
        host=host,
        port=port,
        tool_registry=tool_registry,
    )
    return MCPServer(config)