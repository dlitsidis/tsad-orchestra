"""
TSAD Orchestra Agent — LangGraph + MCP
Dual-agent framework: primary analysis agent + validator/refinement agent.
"""

import asyncio
import functools
import json
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
from openai import RateLimitError
from pydantic import SecretStr

from src.agent.models import Anomaly, AnomalyReport, ValidationResult
from src.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_USER_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_USER_PROMPT,
)

load_dotenv()

MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 3


class AgentState(TypedDict):
    """State container passed through the LangGraph agent graph.

    Attributes:
        messages:  Messages for the current iteration. Cleared after each
                   finalize so the next pass starts fresh with the critique
                   injected into the opening user message.
        result:    Latest AnomalyReport produced by the primary agent.
        critique:  Validator feedback; empty on first pass or after acceptance.
        iteration: Number of completed primary→validator cycles.
        series_id: Time-series ID being analysed.
    """

    messages: Annotated[list[Any], add_messages]
    result: AnomalyReport | None
    critique: str
    iteration: int
    series_id: str


async def _invoke_with_backoff(runnable: Any, messages: list, retries: int = 5) -> Any:
    """Retry an LLM call on rate limit with exponential backoff."""
    for attempt in range(retries):
        try:
            return await runnable.ainvoke(messages)
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            logger.warning("Rate limit hit, retrying in {}s (attempt {}/{})", wait, attempt + 1, retries)
            await asyncio.sleep(wait)


def default_mcp_server_params() -> StdioServerParameters:
    """Build MCP server launch parameters rooted at the repository root."""
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


# ---------------------------------------------------------------------------
# Primary agent nodes
# ---------------------------------------------------------------------------


async def call_model(state: AgentState, llm_with_tools: Any) -> dict[str, Any]:
    """Invoke the primary LLM.

    Empty messages = start of a new iteration: build the opening prompt,
    appending the validator critique if this is a retry.
    Non-empty messages = mid-iteration tool loop: forward history as-is.
    """
    if not state["messages"]:
        user_content = AGENT_USER_PROMPT.format(series_id=state["series_id"])
        if state["critique"]:
            user_content += (
                f"\n\n---\nYour previous report was rejected. Address the following:\n\n"
                f"{state['critique']}\n\n"
                f"You MUST call the tools again to produce a revised analysis. Do not respond with text only."  # noqa: E501
            )
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = state["messages"]

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route: keep calling tools or move to finalize."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


def _collect_tool_results(messages: list[Any]) -> str:
    """Flatten all tool results and assistant text into a single string.

    Avoids passing raw tool-call messages to the structured output API,
    while preserving every detector result for the extraction step.
    """
    id_to_name: dict[str, str] = {}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            id_to_name[tc["id"]] = tc["name"]

    parts: list[str] = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            parts.append(f"[assistant]\n{msg.content.strip()}")
        if getattr(msg, "type", None) == "tool":
            name = id_to_name.get(getattr(msg, "tool_call_id", ""), "tool")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"[{name} result]\n{content.strip()}")

    return "\n\n".join(parts)


async def finalize(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    context = _collect_tool_results(state["messages"])

    tools_called = list(
        dict.fromkeys(tc["name"] for msg in state["messages"] if hasattr(msg, "tool_calls") for tc in (msg.tool_calls or []))
    )

    # Parse anomalies directly from tool results — don't trust the LLM to
    # faithfully copy 1000+ items out of a long context string.
    raw_anomalies: list[dict] = []
    last_detector: str = ""
    for msg in state["messages"]:
        if getattr(msg, "type", None) == "tool":
            tool_call_id = getattr(msg, "tool_call_id", "")
            tool_name = next(
                (
                    tc["name"]
                    for m in state["messages"]
                    for tc in (getattr(m, "tool_calls", None) or [])
                    if tc["id"] == tool_call_id
                ),
                "",
            )
            # Normalise content: LangChain may give a str or a list of blocks
            content = msg.content
            if isinstance(content, list):
                content = " ".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
            logger.debug("Tool '{}' raw content: {:.300}", tool_name, content)
            try:
                payload = json.loads(content)
                if "anomalies" in payload:
                    raw_anomalies = payload["anomalies"]
                    last_detector = tool_name
                    logger.debug("Parsed {} anomalies from '{}'", len(raw_anomalies), tool_name)
                else:
                    logger.debug("Tool '{}' has no 'anomalies' key. Keys: {}", tool_name, list(payload.keys()))
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug("Tool '{}' content not JSON: {}", tool_name, e)

    # Let the LLM only fill in summary + detector_used (small, safe task)
    structured = cast(
        AnomalyReport,
        await _invoke_with_backoff(
            llm.with_structured_output(AnomalyReport),
            [
                {
                    "role": "user",
                    "content": (
                        f"Given these detector outputs, write a concise summary and identify "
                        f"the single detector whose results are being reported.\n\n{context}\n\n"
                        f"For 'detector_used', set ONLY the one detector that produced the final anomaly list. "
                        f"Leave 'anomalies' as an empty list — it will be filled separately."
                    ),
                }
            ],
        ),
    )

    # Overwrite with the directly parsed values — no LLM hallucination risk
    structured.anomalies = [Anomaly(**a) for a in raw_anomalies]
    structured.detector_used = last_detector or structured.detector_used
    structured.tools_called = tools_called
    structured.anomaly_count = len(structured.anomalies)

    logger.info(
        "Primary agent produced report (iteration {}): {} anomalies from {}",
        state["iteration"] + 1,
        len(structured.anomalies),
        structured.detector_used,
    )
    return {"result": structured, "messages": []}


# ---------------------------------------------------------------------------
# Validator node
# ---------------------------------------------------------------------------


async def validate(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """Run the validator against the current AnomalyReport draft."""
    report = state["result"]
    if report is None:
        return {"critique": "No report was produced.", "iteration": state["iteration"] + 1}

    user_msg = VALIDATOR_USER_PROMPT.format(
        series_id=state["series_id"],
        iteration=state["iteration"] + 1,
        report_json=report.model_dump_json(indent=2),
    )

    validation = cast(
        ValidationResult,
        await _invoke_with_backoff(
            llm.with_structured_output(ValidationResult),
            [
                {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        ),
    )

    new_iteration = state["iteration"] + 1

    if validation.accepted:
        logger.info("Validator accepted report on iteration {}", new_iteration)
        return {"critique": "", "iteration": new_iteration}

    logger.warning(
        "Validator rejected report (iteration {}, severity={}): {}",
        new_iteration,
        validation.severity,
        validation.critique,
    )
    return {"critique": validation.critique, "iteration": new_iteration}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_validator(state: AgentState) -> str:
    """End if accepted or max iterations reached, otherwise retry."""
    if not state["critique"]:
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        logger.warning("Reached MAX_ITERATIONS ({}); returning best report so far.", MAX_ITERATIONS)
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(llm_with_tools: Any, tool_node: ToolNode, llm: ChatOpenAI) -> Any:
    """Assemble and compile the dual-agent LangGraph.

    Graph shape:

        primary_agent → tools → primary_agent → ... → finalize
                                                            ↓
                                                        validator
                                                            ↓
                                           ┌── accepted / max retries → END
                                           └── rejected ──────────────→ primary_agent
    """
    builder = StateGraph(AgentState)

    builder.add_node("primary_agent", functools.partial(call_model, llm_with_tools=llm_with_tools))
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", functools.partial(finalize, llm=llm))
    builder.add_node("validator", functools.partial(validate, llm=llm))

    builder.set_entry_point("primary_agent")

    builder.add_conditional_edges(
        "primary_agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    builder.add_edge("tools", "primary_agent")
    builder.add_edge("finalize", "validator")

    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {"end": END, "retry": "primary_agent"},
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run(series_id: str) -> AnomalyReport:
    """Run the dual-agent TSAD pipeline and return the final AnomalyReport."""
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

            final_state = cast(
                AgentState,
                await graph.ainvoke(
                    {
                        "messages": [],
                        "result": None,
                        "critique": "",
                        "iteration": 0,
                        "series_id": series_id,
                    }
                ),
            )

            return cast(AnomalyReport, final_state["result"])
