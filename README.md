# MCP Agent Framework

A developer-friendly, open-source framework for building and orchestrating AI agents with MCP (Model Context Protocol) as the core communication protocol.

## Features

- **MCP Protocol Support**: Full implementation of the Model Context Protocol for client/server communication
- **Tool Management**: Easy tool registration and execution
- **LLM Integration**: Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- **Agent Orchestration**: Build multi-agent systems with ease
- **Extensible**: Create custom tools and integrations

## Installation

```bash
pip install mcp-agent
```

## Quick Start

```python
from mcp_agent import MCPAgent, tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = MCPAgent(
    name="assistant",
    instructions="You are a helpful assistant.",
    tools=[calculator]
)

result = agent.run("What is 2 + 2?")
print(result)
```

## Documentation

See the `docs/` folder for detailed documentation.

## License

MIT