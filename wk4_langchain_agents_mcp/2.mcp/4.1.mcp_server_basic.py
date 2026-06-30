import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from typing import Dict, Any
from requests import get

if "--http" in sys.argv:
    MCP_TRANSPORT = "streamable-http"
else:
    MCP_TRANSPORT = "stdio"


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mcp_server")

load_dotenv()
tavily_client = TavilyClient()


@asynccontextmanager
async def lifespan(server: FastMCP):
    logger.info("Client session opened.")
    try:
        yield
    finally:
        logger.info("Client session closed.")

MCP_HOST = "127.0.0.1"
MCP_PORT = 8000
mcp = FastMCP("mcp_server", lifespan=lifespan, host=MCP_HOST, port=MCP_PORT)

# Tool for searching the web
@mcp.tool()
def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information"""

    logger.info("Client request: search_web(query=%r)", query)
    results = tavily_client.search(query)

    return results


# Resources - provide access to langchain-ai repo files
@mcp.resource("github://langchain-ai/langchain-mcp-adapters/main/README.md")
def github_file():
    """
    Resource for accessing langchain-ai/langchain-mcp-adapters/README.md file

    """
    url = f"https://raw.githubusercontent.com/langchain-ai/langchain-mcp-adapters/main/README.md"
    try:
        resp = get(url)
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"

# Prompt template
@mcp.prompt()
def prompt():
    """Analyze data from a langchain-ai repo file with comprehensive insights"""
    return """
    You are a helpful assistant that answers user questions about LangChain, LangGraph and LangSmith.

    You can use the following tools/resources to answer user questions:
    - search_web: Search the web for information
    - github_file: Access the langchain-ai repo files

    If the user asks a question that is not related to LangChain, LangGraph or LangSmith, you should say "I'm sorry, I can only answer questions about LangChain, LangGraph and LangSmith."

    You may try multiple tool and resource calls to answer the user's question.

    You may also ask clarifying questions to the user to better understand their question.
    """

def _serve_http() -> None:
    import uvicorn

    config = uvicorn.Config(
        mcp.streamable_http_app(),
        host=MCP_HOST,
        port=MCP_PORT,
        log_level="info",
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    else:
        server.run()


if __name__ == "__main__":
    try:
        if MCP_TRANSPORT == "stdio":
            logger.info("Launching MCP server over stdio. Press Ctrl+C to stop.")
            mcp.run(transport="stdio")
        else:
            logger.info(
                "Launching MCP server over %s at http://%s:%d/mcp. Press Ctrl+C to stop.",
                MCP_TRANSPORT, MCP_HOST, MCP_PORT,
            )
            _serve_http()

    except KeyboardInterrupt:
        logger.info("Received interrupt (Ctrl+C) — stopping server.")
    except Exception:
        logger.exception("MCP server crashed unexpectedly.")
    finally:
        logger.info("MCP server process exited.")