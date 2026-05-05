"""MCP Protocol Client implementation."""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from mcp_agent.tool import Tool, ToolResult, ToolSchema


class MCPMessage(BaseModel):
    """Base MCP message."""
    jsonrpc: Literal["2.0"] = "2.0"


class MCPRequest(MCPMessage):
    """MCP JSON-RPC request."""
    
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None


class MCPResponse(MCPMessage):
    """MCP JSON-RPC response."""
    
    id: str | int | None = None
    result: Any = None
    error: dict[str, Any] | None = None


class MCPError(BaseModel):
    """MCP error object."""
    
    code: int
    message: str
    data: Any = None


@dataclass
class MCPClientConfig:
    """Configuration for MCP client."""
    
    url: str
    timeout: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)


class MCPClient:
    """MCP Protocol Client for connecting to MCP servers."""
    
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self._request_id = 0
    
    def _next_id(self) -> int:
        """Generate next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def _request(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request."""
        request = MCPRequest(
            method=method,
            params=params or {},
            id=self._next_id(),
        )
        
        async with httpx.AsyncClient(
            timeout=self.config.timeout,
            headers=self.config.headers,
        ) as client:
            response = await client.post(
                self.config.url,
                content=request.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            
            mcp_response = MCPResponse.model_validate_json(response.content)
            
            if mcp_response.error:
                raise Exception(mcp_response.error["message"])
            
            return mcp_response.result
    
    async def initialize(self) -> dict[str, Any]:
        """Initialize connection with MCP server."""
        return await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "mcp-agent",
                "version": "0.1.0",
            },
        })
    
    async def list_tools(self) -> list[ToolSchema]:
        """List available tools from MCP server."""
        result = await self._request("tools/list")
        tools = result.get("tools", [])
        return [ToolSchema(**t) for t in tools]
    
    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """Call a tool on the MCP server."""
        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        
        # MCP returns tool results as content array
        content = result.get("content", [])
        if content:
            first = content[0]
            if first.get("type") == "text":
                return ToolResult(
                    success=True,
                    result=first.get("text"),
                )
        
        return ToolResult(success=True, result=result)
    
    async def close(self) -> None:
        """Close the client connection."""
        # httpx AsyncClient handles cleanup automatically
        pass


class MCPRemoteTool(Tool):
    """A tool that executes on a remote MCP server."""
    
    def __init__(self, client: MCPClient, schema: ToolSchema):
        self.client = client
        self.schema = schema
        super().__init__(
            name=schema.name,
            description=schema.description,
            parameters=[],  # Would parse from inputSchema
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool remotely."""
        return await self.client.call_tool(self.name, kwargs)