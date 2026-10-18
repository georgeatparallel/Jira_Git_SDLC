from __future__ import annotations

import asyncio
import importlib
import json
from hashlib import sha256
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import server
from agent.middleware.exclude_tools import ExcludeToolsMiddleware
from agent.tools import web_search

search_module = importlib.import_module("agent.tools.web_search")


@pytest.fixture(autouse=True)
def clean_search_environment(monkeypatch):
    for name in ("EXA_API_KEY", "WEB_SEARCH_PROVIDER", "PARALLEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mcp_server(monkeypatch):
    observed = []
    state = {"error": None, "structured": True, "payload": None, "wait": False, "disconnect": False}
    started = asyncio.Event()
    cancelled = asyncio.Event()
    payload = {
        "search_id": "search_fixture",
        "session_id": "session_fixture",
        "results": [
            {"url": "https://docs.python.org/3/", "title": "Python", "excerpts": ["Docs"]},
            {"url": "https://example.org/other", "title": "Other", "excerpts": ["Other"]},
        ],
    }

    async def handle(request):
        assert str(request.url) == search_module.PARALLEL_MCP_URL
        assert request.headers["User-Agent"] == search_module.SEARCH_USER_AGENT
        assert "authorization" not in request.headers
        if state["disconnect"]:
            raise httpx.ConnectError("Connection unavailable", request=request)
        if request.method != "POST":
            return httpx.Response(405)
        rpc = json.loads(request.content)
        observed.append(rpc)
        if "id" not in rpc:
            return httpx.Response(202)
        if rpc["method"] == "initialize":
            result = {
                "protocolVersion": rpc["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "search-fixture", "version": "1"},
            }
        elif rpc["method"] == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            assert rpc["method"] == "tools/call"
            assert rpc["params"]["name"] == "web_search"
            started.set()
            if state["wait"]:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            data = state["payload"] if state["payload"] is not None else payload
            text = state["error"] or json.dumps(data)
            result = {
                "content": [{"type": "text", "text": text}],
                "isError": bool(state["error"]),
            }
            if state["structured"] and not state["error"]:
                result["structuredContent"] = data
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": rpc["id"], "result": result})

    original = search_module.MultiServerMCPClient

    def client(connections):
        def http_client(headers=None, timeout=None, auth=None):
            return httpx.AsyncClient(
                headers=headers, timeout=timeout, auth=auth, transport=httpx.MockTransport(handle)
            )

        connections["parallel"]["httpx_client_factory"] = http_client
        return original(connections)

    monkeypatch.setattr(search_module, "MultiServerMCPClient", client)
    return observed, state, started, cancelled


@pytest.fixture
async def registered_tools(monkeypatch):
    captured = {}

    class Graph:
        def with_config(self, config):
            return self

    def capture(**kwargs):
        captured.update(kwargs)
        return Graph()

    for name, value in {
        "resolve_github_token": ("ghp", None),
        "ensure_sandbox_for_thread": MagicMock(),
        "aresolve_sandbox_work_dir": "/workspace",
        "get_team_default_model_pair": (("openai:gpt-5.5", "medium"), ("openai:gpt-5.5", "low")),
        "_load_observability_tools": [],
        "_observability_authorized": False,
        "_load_corridor_mcp_tools": [],
    }.items():
        monkeypatch.setattr(server, name, AsyncMock(return_value=value))
    monkeypatch.setattr(server, "resolve_triggering_user_identity", lambda *_: None)
    monkeypatch.setattr(server, "fallback_model_id_for", lambda *_: None)
    monkeypatch.setattr(server, "make_model", MagicMock())
    monkeypatch.setattr(server, "load_browser_tools", lambda: [])
    monkeypatch.setattr(server, "construct_system_prompt", lambda **_: "Find documentation.")
    monkeypatch.setattr(server, "create_deep_agent", capture)
    await server.get_agent(
        {"configurable": {"__is_for_execution__": True, "thread_id": "search-test"}, "metadata": {}}
    )
    assert web_search in captured["tools"]
    return captured["tools"]


class SearchConsumer(BaseChatModel):
    """Scripted model: dispatch search, then consume its returned URL and excerpt."""

    search_available: bool = False

    @property
    def _llm_type(self):
        return "scripted-search-consumer"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(
            update={
                "search_available": any(getattr(tool, "name", "") == "web_search" for tool in tools)
            }
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            response = json.loads(messages[-1].content)
            if response["success"]:
                result = json.loads(response["results"])["results"][0]
                content = result["url"] + " " + " ".join(result.get("excerpts", []))
            else:
                content = response["error"]
            message = AIMessage(content=content)
        elif not self.search_available:
            message = AIMessage(content="Search unavailable")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "Python asyncio documentation", "num_results": 1},
                        "id": "search-call",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_unconfigured_agent_dispatch_and_result_consumption(registered_tools, mcp_server):
    graph = create_deep_agent(model=SearchConsumer(), tools=registered_tools)
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Find Python docs"}]},
        config={"configurable": {"thread_id": "search-test"}},
    )
    assert result["messages"][-1].content == "https://docs.python.org/3/ Docs"
    observed = mcp_server[0]
    assert {rpc["method"] for rpc in observed} >= {"initialize", "tools/list", "tools/call"}
    arguments = next(rpc for rpc in observed if rpc["method"] == "tools/call")["params"][
        "arguments"
    ]
    assert arguments == {
        "objective": "Python asyncio documentation",
        "search_queries": ["Python asyncio documentation"],
        "session_id": sha256(b"search-test").hexdigest(),
    }


@pytest.mark.parametrize("exclude", [False, True])
async def test_search_tool_permissions(registered_tools, mcp_server, exclude):
    tools = registered_tools if exclude else [t for t in registered_tools if t is not web_search]
    middleware = [ExcludeToolsMiddleware(excluded=frozenset({"web_search"}))] if exclude else []
    graph = create_deep_agent(model=SearchConsumer(), tools=tools, middleware=middleware)
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Find docs"}]})
    assert result["messages"][-1].content == "Search unavailable"
    assert mcp_server[0] == []


@pytest.mark.parametrize("structured", [False, True])
async def test_parallel_result_limit_and_contents(mcp_server, structured):
    mcp_server[1]["structured"] = structured
    result = await web_search("Python docs", num_results=1, include_contents=False)
    assert result["success"] is True
    assert json.loads(result["results"])["results"] == [
        {"url": "https://docs.python.org/3/", "title": "Python"}
    ]


@pytest.mark.parametrize("include_contents", [False, True])
async def test_configured_exa_is_preserved(monkeypatch, mcp_server, include_contents):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    client = MagicMock()
    client.search.return_value = client.search_and_contents.return_value = "Exa results"
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(search_module, "Exa", factory)
    result = await web_search("Python docs", num_results=3, include_contents=include_contents)
    assert result == {"success": True, "results": "Exa results", "error": None}
    factory.assert_called_once_with(api_key="configured-exa-key")
    if include_contents:
        client.search_and_contents.assert_called_once_with(
            "Python docs", text=True, num_results=3, type="auto"
        )
        client.search.assert_not_called()
    else:
        client.search.assert_called_once_with("Python docs", num_results=3, type="auto")
        client.search_and_contents.assert_not_called()
    assert mcp_server[0] == []


async def test_explicit_parallel_overrides_exa_key(monkeypatch, mcp_server):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    assert (await web_search("Python docs"))["success"] is True
    assert any(rpc["method"] == "tools/call" for rpc in mcp_server[0])


@pytest.mark.parametrize("provider", ["exa", "disabled", "invalid"])
async def test_explicit_choice_does_not_fall_back(monkeypatch, mcp_server, provider):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", provider)
    result = await web_search("Python docs")
    assert result["success"] is False
    assert result["results"] is None
    assert mcp_server[0] == []


async def test_exa_failure_does_not_switch_provider(monkeypatch, mcp_server):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    client = MagicMock()
    client.search_and_contents.side_effect = RuntimeError("Exa failed")
    monkeypatch.setattr(search_module, "Exa", lambda **_: client)
    assert "Exa failed" in (await web_search("Python docs"))["error"]
    assert mcp_server[0] == []


async def test_disabled_with_saved_key_does_not_connect(monkeypatch, mcp_server):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "disabled")
    assert (await web_search("Python docs"))["error"] == "Web search is disabled."
    assert mcp_server[0] == []


async def test_connection_error_is_returned_without_exa_fallback(monkeypatch, mcp_server):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    exa = MagicMock()
    monkeypatch.setattr(search_module, "Exa", exa)
    mcp_server[1]["disconnect"] = True
    response = await web_search("Python docs")
    assert response["success"] is False
    assert response["results"] is None
    assert response["error"]
    exa.assert_not_called()


async def test_mcp_error_reaches_agent(registered_tools, mcp_server):
    mcp_server[1]["error"] = "Anonymous quota exceeded"
    graph = create_deep_agent(model=SearchConsumer(), tools=registered_tools)
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "Find docs"}]})
    assert "Anonymous quota exceeded" in result["messages"][-1].content


@pytest.mark.parametrize("payload", [{"unexpected": []}, {"results": ["invalid"]}])
async def test_malformed_response_fails(mcp_server, payload):
    mcp_server[1]["payload"] = payload
    assert (await web_search("Python docs"))["success"] is False


async def test_timeout_closes_inflight_request(monkeypatch, mcp_server):
    monkeypatch.setattr(search_module, "SEARCH_TIMEOUT_SECONDS", 0.1)
    mcp_server[1]["wait"] = True
    result = await web_search("Python docs")
    assert result["success"] is False
    assert "TimeoutError" in result["error"]
    assert mcp_server[3].is_set()


async def test_cancellation_propagates_and_closes_request(mcp_server):
    mcp_server[1]["wait"] = True
    task = asyncio.create_task(web_search("Python docs"))
    await asyncio.wait_for(mcp_server[2].wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mcp_server[3].is_set()


@pytest.mark.parametrize("count", [0, -1, True])
async def test_invalid_result_count_does_not_connect(mcp_server, count):
    assert (await web_search("Python docs", num_results=count))["success"] is False
    assert mcp_server[0] == []
