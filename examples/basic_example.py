"""Example MCP Agents and Tools."""

from mcp_agent import MCPAgent, tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result of the expression
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather(city: str) -> str:
    """
    Get the current weather for a city.
    
    Args:
        city: The city name
    
    Returns:
        Weather information
    """
    # Mock weather data
    weather_data = {
        "tokyo": "Sunny, 22°C",
        "new york": "Cloudy, 18°C",
        "london": "Rainy, 14°C",
        "san francisco": "Foggy, 16°C",
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"The weather in {city.title()} is {weather_data[city_lower]}"
    return f"Weather data not available for {city}"


@tool
def search(query: str) -> str:
    """
    Search for information.
    
    Args:
        query: The search query
    
    Returns:
        Search results (mock)
    """
    return f"Results for '{query}': Found 3 matching items."


async def main():
    """Run an example agent."""
    # Create an agent with tools
    agent = MCPAgent(
        name="assistant",
        instructions="""You are a helpful assistant that can use tools to answer questions.
        Always use tools when needed to get accurate information.""",
        tools=[calculator, weather, search],
    )
    
    # Test conversations
    print("=== Example 1: Calculator ===")
    result = await agent.run("What is 25 * 4?")
    print(f"User: What is 25 * 4?")
    print(f"Agent: {result}")
    print()
    
    print("=== Example 2: Weather ===")
    result = await agent.run("What's the weather in Tokyo?")
    print(f"User: What's the weather in Tokyo?")
    print(f"Agent: {result}")
    print()
    
    print("=== Example 3: Search ===")
    result = await agent.run("Search for Python tutorials")
    print(f"User: Search for Python tutorials")
    print(f"Agent: {result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())