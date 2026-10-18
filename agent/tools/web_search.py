import asyncio
import json
import logging
import os
from datetime import timedelta
from hashlib import sha256
from typing import Any

from exa_py import Exa
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_config

logger = logging.getLogger(__name__)

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
SEARCH_USER_AGENT = "OpenSWE/0.1.0 (+https://github.com/nikhilbajaj12/Jira_Git_SDLC)"
SEARCH_TIMEOUT_SECONDS = 30


async def _parallel_search(query: str, num_results: int, include_contents: bool) -> dict[str, Any]:
    arguments = {"objective": query, "search_queries": [query]}
    try:
        config = get_config()
    except RuntimeError:
        config = {}
    thread_id = config.get("configurable", {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        arguments["session_id"] = sha256(thread_id.encode()).hexdigest()
    client = MultiServerMCPClient(
        {
            "parallel": {
                "transport": "http",
                "url": PARALLEL_MCP_URL,
                "headers": {"User-Agent": SEARCH_USER_AGENT},
                "timeout": timedelta(seconds=SEARCH_TIMEOUT_SECONDS),
                "sse_read_timeout": timedelta(seconds=SEARCH_TIMEOUT_SECONDS),
            }
        }
    )
    async with asyncio.timeout(SEARCH_TIMEOUT_SECONDS), client.session("parallel") as session:
        response = await session.call_tool("web_search", arguments)
    text = "\n".join(block.text for block in response.content if block.type == "text")
    if response.isError:
        raise RuntimeError(text or "Parallel search failed")
    data = response.structuredContent
    if data is None:
        data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Parallel search returned an invalid results payload")
    results = data["results"][:num_results]
    if not all(isinstance(result, dict) for result in results):
        raise ValueError("Parallel search returned an invalid result")
    if not include_contents:
        results = [
            {key: value for key, value in result.items() if key != "excerpts"} for result in results
        ]
    return {"success": True, "results": json.dumps({**data, "results": results}), "error": None}


async def web_search(
    query: str,
    num_results: int = 5,
    include_contents: bool = True,
) -> dict[str, Any]:
    """Search the web for relevant information using the configured provider.

    Use this tool when you need to find documentation, code examples, GitHub repos,
    news, or research papers to help complete a task.

    Args:
        query: The search query
        num_results: Maximum number of results to return (default: 5)
        include_contents: Include page excerpts (Parallel) or full page text (Exa).

    Returns:
        Dictionary containing:
        - success: Whether the search succeeded
        - results: Search results as text, including source URLs
        - error: Error message if something failed
    """
    api_key = os.environ.get("EXA_API_KEY")
    provider = os.environ.get("WEB_SEARCH_PROVIDER", "").strip().lower()
    if not provider:
        provider = "exa" if api_key else "parallel"
    if provider == "disabled":
        return {"success": False, "results": None, "error": "Web search is disabled."}
    if provider not in {"exa", "parallel"}:
        return {
            "success": False,
            "results": None,
            "error": "WEB_SEARCH_PROVIDER must be exa, parallel, or disabled.",
        }
    if isinstance(num_results, bool) or not isinstance(num_results, int) or num_results < 1:
        return {
            "success": False,
            "results": None,
            "error": "num_results must be a positive integer.",
        }
    if provider == "exa" and not api_key:
        return {
            "success": False,
            "results": None,
            "error": "EXA_API_KEY is not configured. Please add it to your environment variables.",
        }

    async def _search() -> dict[str, Any]:
        client = Exa(api_key=api_key)
        if include_contents:
            result = await asyncio.to_thread(
                client.search_and_contents,
                query,
                text=True,
                num_results=num_results,
                type="auto",
            )
        else:
            result = await asyncio.to_thread(
                client.search,
                query,
                num_results=num_results,
                type="auto",
            )
        return {"success": True, "results": str(result), "error": None}

    try:
        if provider == "parallel":
            return await _parallel_search(query, num_results, include_contents)
        return await _search()
    except Exception as e:
        logger.exception("web_search failed")
        return {"success": False, "results": None, "error": f"{type(e).__name__}: {e}"}
