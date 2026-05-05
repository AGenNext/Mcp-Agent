"""MCPAgent - Main agent implementation."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from mcp_agent.tool import FunctionTool, Tool, ToolRegistry, ToolResult


@dataclass
class Message:
    """A message in the conversation."""
    
    role: Literal["user", "assistant", "system"]
    content: str
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None


@dataclass
class AgentConfig:
    """Configuration for MCPAgent."""
    
    name: str = "assistant"
    instructions: str = "You are a helpful assistant."
    model: str = "gpt-4o"
    model_provider: str = "openai"  # openai, anthropic
    temperature: float = 1.0
    max_tokens: int = 4096


class LLMClient:
    """Base class for LLM clients."""
    
    async def chat(self, messages: list[dict], **kwargs) -> dict:
        """Send a chat request."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI LLM client."""
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    async def chat(self, messages: list[dict], **kwargs) -> dict:
        """Send chat request to OpenAI."""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                messages=messages,
                **kwargs,
            )
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "usage": response.usage.model_dump(),
            }
        except ImportError:
            return {
                "content": "OpenAI client not installed. Install with: pip install openai",
                "tool_calls": None,
            }


class AnthropicClient(LLMClient):
    """Anthropic LLM client."""
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    
    async def chat(self, messages: list[dict], **kwargs) -> dict:
        """Send chat request to Anthropic."""
        try:
            from anthropic import AsyncAnthropic
            
            client = AsyncAnthropic(api_key=self.api_key)
            
            # Convert messages to Anthropic format
            system = None
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            response = await client.messages.create(
                model=kwargs.get("model", "claude-3-opus-20240229"),
                max_tokens=kwargs.get("max_tokens", 4096),
                system=system,
                messages=anthropic_messages,
            )
            
            return {
                "content": response.content[0].text,
                "tool_calls": None,
                "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            }
        except ImportError:
            return {
                "content": "Anthropic client not installed. Install with: pip install anthropic",
                "tool_calls": None,
            }


class MCPAgent:
    """
    MCP Agent for building AI-powered applications.
    
    Example:
        agent = MCPAgent(
            name="assistant",
            instructions="You are a helpful coding assistant.",
            tools=[my_tool],
        )
        result = agent.run("Hello, how are you?")
    """
    
    def __init__(
        self,
        name: str = "assistant",
        instructions: str = "You are a helpful assistant.",
        tools: list[Tool | FunctionTool] | None = None,
        config: AgentConfig | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.name = name
        self.config = config or AgentConfig(name=name, instructions=instructions)
        self.tool_registry = ToolRegistry()
        self.messages: list[Message] = []
        
        # Register tools
        if tools:
            for t in tools:
                self.tool_registry.register(t)
        
        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        elif config:
            if config.model_provider == "anthropic":
                self.llm_client = AnthropicClient()
            else:
                self.llm_client = OpenAIClient()
        else:
            self.llm_client = OpenAIClient()
    
    def add_tool(self, tool: Tool | FunctionTool) -> None:
        """Add a tool to the agent."""
        self.tool_registry.register(tool)
    
    def get_tool_schemas(self) -> list[dict]:
        """Get tool schemas for LLM."""
        schemas = self.tool_registry.get_schemas()
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.inputSchema,
                },
            }
            for s in schemas
        ]
    
    def _format_messages(self) -> list[dict]:
        """Format messages for LLM."""
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": self.config.instructions,
        })
        
        # Conversation history
        for msg in self.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return messages
    
    async def _execute_tool_calls(
        self, tool_calls: list[dict]
    ) -> list[dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        
        for call in tool_calls:
            func = call.function
            name = func.name
            args = json.loads(func.arguments)
            
            result = await self.tool_registry.execute(name, **args)
            
            results.append({
                "tool_call_id": call.id,
                "name": name,
                "content": str(result.result) if result.success else f"Error: {result.error}",
            })
        
        return results
    
    async def run(self, prompt: str, **kwargs) -> str:
        """
        Run the agent with a prompt.
        
        Args:
            prompt: The user's input prompt
            
        Returns:
            The agent's response
            
        Example:
            result = await agent.run("What is the weather in Tokyo?")
        """
        import json
        
        # Add user message
        self.messages.append(Message(role="user", content=prompt))
        
        # Build messages with tools
        messages = self._format_messages()
        
        # Add tools if available
        tool_schemas = self.get_tool_schemas()
        if tool_schemas:
            messages.append({
                "role": "system",
                "tools": json.dumps(tool_schemas),
            })
        
        # Call LLM
        llm_kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        llm_kwargs.update(kwargs)
        
        response = await self.llm_client.chat(messages, **llm_kwargs)
        
        content = response.get("content", "")
        tool_calls = response.get("tool_calls")
        
        # Execute tool calls if present
        if tool_calls:
            tool_results = await self._execute_tool_calls(tool_calls)
            
            # Add assistant message with tool calls
            self.messages.append(Message(
                role="assistant",
                content=content,
                tool_calls=[tc.model_dump() for tc in tool_calls],
            ))
            
            # Add tool results to messages
            for result in tool_results:
                self.messages.append(Message(
                    role="user",
                    content=f"Tool {result['name']} result: {result['content']}",
                ))
            
            # Continue conversation
            messages = self._format_messages()
            response = await self.llm_client.chat(messages, **llm_kwargs)
            content = response.get("content", "")
        
        # Add assistant response
        self.messages.append(Message(role="assistant", content=content))
        
        return content
    
    async def chat(self, prompt: str) -> str:
        """Alias for run()."""
        return await self.run(prompt)