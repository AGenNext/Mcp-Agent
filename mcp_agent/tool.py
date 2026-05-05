"""Tool system for MCP Agent Framework."""

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TypeVar

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Represents a parameter for a tool."""
    
    name: str
    description: str = ""
    type: str = "string"
    required: bool = True
    default: Any = None


class ToolSchema(BaseModel):
    """JSON Schema representation of a tool."""
    
    name: str
    description: str
    inputSchema: dict = Field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from executing a tool."""
    
    success: bool
    result: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class Tool(ABC):
    """Base class for MCP tools."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: list[ToolParameter] | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or []
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""
        pass
    
    def get_schema(self) -> ToolSchema:
        """Get the JSON schema for this tool."""
        input_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for param in self.parameters:
            input_schema["properties"][param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                input_schema["properties"][param.name]["default"] = param.default
            if param.required:
                input_schema["required"].append(param.name)
        
        return ToolSchema(
            name=self.name,
            description=self.description,
            inputSchema=input_schema,
        )


class FunctionTool(Tool):
    """A tool backed by a Python function."""
    
    def __init__(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]] | Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        parameters: list[ToolParameter] | None = None,
    ):
        self.func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or ""
        
        if parameters is None:
            parameters = self._extract_parameters(func)
        
        super().__init__(self._name, self._description, parameters)
    
    def _extract_parameters(
        self, func: Callable[..., Any]
    ) -> list[ToolParameter]:
        """Extract parameters from function signature."""
        parameters = []
        sig = inspect.signature(func)
        
        for param_name, param in sig.parameters.items():
            # Skip self/cls parameters
            if param_name in ("self", "cls"):
                continue
            
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_name = param.annotation.__name__
                type_map = {
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean",
                    "list": "array",
                    "dict": "object",
                }
                param_type = type_map.get(type_name, "string")
            
            parameters.append(
                ToolParameter(
                    name=param_name,
                    description=f"Parameter {param_name}",
                    type=param_type,
                    required=param.default is inspect.Parameter.empty,
                    default=param.default,
                )
            )
        
        return parameters
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the function tool."""
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
            return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def tool(
    func: Callable[..., Coroutine[Any, Any, Any]] | Callable[..., Any]
) -> FunctionTool:
    """Decorator to convert a function into an MCP tool."""
    return FunctionTool(func=func)


T = TypeVar("T", bound=Tool)


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> Tool | None:
        """Unregister a tool by name."""
        return self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_schemas(self) -> list[ToolSchema]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found"
            )
        return await tool.execute(**kwargs)