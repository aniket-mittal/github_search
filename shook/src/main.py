"""
FastMCP quickstart example.

cd to the `examples/snippets/clients` directory and run:
    uv run server fastmcp_quickstart stdio
"""

from mcp.server.fastmcp import FastMCP
import json
import sys
from pathlib import Path

# Add parent directory to import retrieve module
sys.path.append(str(Path(__file__).parent.parent))
from retrieve import search_similar_vectors

# Create an MCP server
mcp = FastMCP("Shook")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

@mcp.tool()
def retrieve(query: str) -> str:
    """Retrieve similar code from the database given a string query"""
    return search_similar_vectors(query)


# Add a tool that demonstrates resource access
@mcp.tool()
def get_greeting_for_user(name: str) -> str:
    """Get a greeting for a specific user by accessing the greeting resource"""
    # This simulates accessing the greeting resource
    return f"Hello, {name}!"


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."


if __name__ == "__main__":
    mcp.run()
