"""
TSAD Orchestra Agent — LangGraph + MCP
"""

import asyncio
import functools
import os
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast

from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import SecretStr

from src.agent.models import AnomalyReport
from src.agent.prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT

load_dotenv()

MODEL = "gpt-4o-mini"


class AgentState(TypedDict):
    """State container passed through the LangGraph agent graph.

    Attributes:
        messages: Accumulated conversation messages, merged via add_messages.
        result: The final structured anomaly report, or None if not yet produced.
    """

    messages: Annotated[list[Any], add_messages]
    result: AnomalyReport | None


def default_mcp_server_params() -> StdioServerParameters:
    """Build MCP server launch parameters rooted at the repository root.

    Ensures the repository root is prepended to PYTHONPATH so that the
    MCP server subprocess can resolve internal src.* imports correctly.

    Returns:
        StdioServerParameters configured to launch src.mcp_server via
        the current Python interpreter.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()

    pythonpath_parts = [str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server"],
        env=env,
    )


async def call_model(state: AgentState, llm_with_tools: Any) -> dict[str, list[Any]]:
    """Invoke the LLM with the current message history and bound tools.

    Args:
        state: The current agent state containing the message history.
        llm_with_tools: A ChatOpenAI instance with MCP tools already bound.

    Returns:
        A dict with a messages key containing the model's response.
    """
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route the graph to the next node based on whether tool calls were requested.

    Args:
        state: The current agent state containing the message history.

    Returns:
        "tools" if the last message contains tool calls, "final" otherwise.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "final"


async def finalize(state: AgentState, llm: ChatOpenAI) -> dict[str, AnomalyReport]:
    """Extract a structured AnomalyReport from the completed message history.

    Uses structured output binding so the model returns a validated
    AnomalyReport Pydantic object directly.

    Args:
        state: The current agent state containing the full message history.
        llm: A base ChatOpenAI instance (without tools bound) used for
            structured output extraction.

    Returns:
        A dict with a result key containing the parsed AnomalyReport.
    """
    structured = cast(
        AnomalyReport,
        await llm.with_structured_output(AnomalyReport).ainvoke(state["messages"]),
    )
    return {"result": structured}


def build_graph(llm_with_tools: Any, tool_node: ToolNode, llm: ChatOpenAI) -> Any:
    """Assemble and compile the LangGraph agent graph.

    Graph shape:

        Start -> agent -> tools -> agent -> ... -> final -> END

    Args:
        llm_with_tools: LLM instance with MCP tools bound.
        tool_node: A ToolNode wrapping the loaded MCP tools.
        llm: Base LLM instance used for structured output extraction.

    Returns:
        A compiled StateGraph ready to be invoked.
    """
    builder = StateGraph(AgentState)

    builder.add_node("agent", functools.partial(call_model, llm_with_tools=llm_with_tools))
    builder.add_node("tools", tool_node)
    builder.add_node("final", functools.partial(finalize, llm=llm))

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "final": "final",
        },
    )

    builder.add_edge("tools", "agent")
    builder.add_edge("final", END)

    return builder.compile()


async def run(series: list[float]) -> AnomalyReport:
    """Run the TSAD agent pipeline on a time series and return an anomaly report.

    Starts the MCP server subprocess, loads its tools, wires up the LangGraph
    agent graph, and invokes it with the provided series.

    Args:
        series: A list of float values representing the time series to analyse.

    Returns:
        An AnomalyReport describing any detected anomalies in the series.
    """
    server_params = default_mcp_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            logger.info("Loaded MCP tools: {}", [t.name for t in tools])

            llm = ChatOpenAI(
                api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                model=MODEL,
                temperature=0,
            )

            llm_with_tools = llm.bind_tools(tools)
            tool_node = ToolNode(tools)

            graph = build_graph(llm_with_tools, tool_node, llm)

            result = cast(
                AgentState,
                await graph.ainvoke(
                    {
                        "messages": [
                            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                            {"role": "user", "content": AGENT_USER_PROMPT.format(series=series)},
                        ],
                        "result": None,
                    }
                ),
            )

            return cast(AnomalyReport, result["result"])


if __name__ == "__main__":
    mock_series = [
        10.1,
        10.3,
        9.9,
        10.2,
        10.0,
        50.0,
        10.1,
        9.8,
        10.4,
        -30.0,
        10.2,
    ]

    try:
        result = asyncio.run(run(mock_series))
        logger.info("Anomaly Report: {}", result.model_dump())

    except Exception as e:
        logger.error("FAILED: {}", repr(e))
