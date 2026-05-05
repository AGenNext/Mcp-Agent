"""
MCP Agent Framework
====================
A developer-friendly framework for building and orchestrating AI agents with MCP.
"""

__version__ = "0.1.0"

from mcp_agent.agent import MCPAgent
from mcp_agent.tool import tool, Tool, ToolResult
from mcp_agent.client import MCPClient
from mcp_agent.server import MCPServer

__all__ = [
    "MCPAgent",
    "tool",
    "Tool",
    "ToolResult",
    "MCPClient",
    "MCPServer",
    "__version__",
]