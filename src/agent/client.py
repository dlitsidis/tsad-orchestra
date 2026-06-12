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
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import SecretStr

from src.agent.models import Anomaly, AnomalyReport, LLMFinalReport, ValidationResult
from src.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_USER_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_USER_PROMPT,
)

class TokenTrackerCallback(BaseCallbackHandler):
    """Callback handler to track LLM token usage across all nodes in the graph."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        found_tokens = False
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message:
                    if hasattr(message, "usage_metadata") and message.usage_metadata:
                        usage = message.usage_metadata
                        self.prompt_tokens += usage.get("input_tokens", 0)
                        self.completion_tokens += usage.get("output_tokens", 0)
                        self.total_tokens += usage.get("total_tokens", 0)
                        found_tokens = True
                    elif hasattr(message, "response_metadata") and message.response_metadata:
                        token_usage = message.response_metadata.get("token_usage")
                        if token_usage:
                            self.prompt_tokens += token_usage.get("prompt_tokens", 0)
                            self.completion_tokens += token_usage.get("completion_tokens", 0)
                            self.total_tokens += token_usage.get("total_tokens", 0)
                            found_tokens = True
        
        if not found_tokens:
            llm_output = getattr(response, "llm_output", None)
            if llm_output and isinstance(llm_output, dict):
                token_usage = llm_output.get("token_usage")
                if token_usage and isinstance(token_usage, dict):
                    self.prompt_tokens += token_usage.get("prompt_tokens", 0)
                    self.completion_tokens += token_usage.get("completion_tokens", 0)
                    self.total_tokens += token_usage.get("total_tokens", 0)

load_dotenv()

# MODEL = "gpt-4o-mini"
MODEL = "gpt-4.1-mini"

MAX_ITERATIONS = 5


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
        detector_calls = [tc for tc in last.tool_calls if tc["name"].endswith("_detector")]
        
        previously_run = set()
        for msg in state["messages"][:-1]:
            for tc in getattr(msg, "tool_calls", None) or []:
                if tc["name"].endswith("_detector"):
                    previously_run.add(tc["name"])
                    
        total_unique_detectors = len(previously_run.union(tc["name"] for tc in detector_calls))
        
        if total_unique_detectors > 7:
            return "too_many_tools"
            
        return "tools"
    
    # Check if a detector tool was actually called
    detector_called = False
    for msg in state["messages"]:
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"].endswith("_detector"):
                    detector_called = True
                    break
        if detector_called:
            break
            
    if not detector_called:
        return "force_detector"

    return "finalize"


async def too_many_tools(state: AgentState) -> dict[str, Any]:
    """Intercept execution if the agent attempts to run more than 7 detectors cumulatively."""
    logger.warning("Agent attempted to run >7 detectors. Intercepting.")
    last = state["messages"][-1]
    
    responses = []
    for tc in last.tool_calls:
        if tc["name"].endswith("_detector"):
            responses.append({
                "role": "tool",
                "name": tc["name"],
                "tool_call_id": tc["id"],
                "content": "SYSTEM REJECTION: You have exceeded the strict limit of 7 anomaly detectors per dataset. Tool execution blocked to save compute. You MUST make your final decision based on the detectors you have already run.",
            })
        else:
            responses.append({
                "role": "tool",
                "name": tc["name"],
                "tool_call_id": tc["id"],
                "content": "SYSTEM REJECTION: Tool execution blocked because it was batched with an illegal number of anomaly detectors. Try again without the detectors.",
            })
        
    return {"messages": responses}


async def force_detector(state: AgentState) -> dict[str, Any]:
    """Intercept the agent if it tries to stop before using a detector."""
    logger.warning("Agent tried to finalize without calling a detector. Forcing it to continue.")
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "You have profiled the data but haven't run any anomaly detectors yet. "
                    "You MUST follow the two-phase workflow:\n"
                    "Phase 1: Run 2-3 detector tools (lof_detector, hbos_detector, iforest_detector, "
                    "pca_detector, poly_detector) to get their stat summaries and hot_segments.\n"
                    "Phase 2: Call drill_down_range on the most suspicious segments to get "
                    "per-point anomaly details and consensus_points.\n"
                    "Only then produce your final report."
                ),
            }
        ]
    }


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

    # Let the LLM consolidate the two-phase results into a structured report.
    # Point-level scoring is computed deterministically below — not by the LLM.
    llm_report = cast(
        LLMFinalReport,
        await _invoke_with_backoff(
            llm.with_structured_output(LLMFinalReport),
            [
                {
                    "role": "user",
                    "content": (
                        "Given these two-phase detector outputs (stat summaries + drill-down results), "
                        "produce the final report.\n\n"
                        f"{context}\n\n"
                        "For 'detectors_used': list the SHORT names of every detector you ran "
                        "(e.g. ['iforest', 'lof']). These must match what you passed to store_ensemble_scores.\n"
                        "For 'summary': explain your screening observations, which segments you drilled "
                        "into, what consensus you found, and your final reasoning."
                    ),
                }
            ],
        ),
    )

    import json
    extracted_anomalies_map = {}
    for msg in state["messages"]:
        if getattr(msg, "type", None) == "tool":
            id_to_name = {tc["id"]: tc["name"] for m in state["messages"] for tc in (getattr(m, "tool_calls", None) or [])}
            name = id_to_name.get(getattr(msg, "tool_call_id", ""), "")
            if name == "drill_down_range":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                try:
                    import ast
                    # The tool output might be a stringified Python dict, or a list of MCP contents
                    data = ast.literal_eval(content) if isinstance(content, str) else content
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            data = ast.literal_eval(data)
                            
                    # MCP tools return a list of text contents: [{"type": "text", "text": "..."}]
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get("type") == "text":
                                inner_text = item.get("text", "")
                                try:
                                    data = ast.literal_eval(inner_text)
                                except Exception:
                                    data = json.loads(inner_text)
                                break
                        
                    if isinstance(data, dict):
                        for pt in data.get("consensus_points", []):
                            extracted_anomalies_map[pt["index"]] = Anomaly(
                                index=pt["index"],
                                value=pt["value"],
                                score=pt.get("mean_score", 0.0)
                            )
                except Exception as e:
                    logger.debug(f"Could not parse drill_down_range output for anomalies: {e}")
                    
    extracted_anomalies = sorted(list(extracted_anomalies_map.values()), key=lambda x: x.index)

    tool_counts = {}
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if name:
                    tool_counts[name] = tool_counts.get(name, 0) + 1

    structured = AnomalyReport(
        detectors_used=llm_report.detectors_used,
        summary=llm_report.summary,
        anomalies=extracted_anomalies,
        tools_called=tools_called,
        anomaly_count=len(extracted_anomalies),
        tool_counts=tool_counts
    )

    # Read the fused per-point score vector that store_ensemble_scores wrote to disk.
    # This avoids re-running any detector — the MCP server cached everything.
    import json
    import tempfile
    score_path = os.path.join(tempfile.gettempdir(), f"tsad_ensemble_{state['series_id']}.json")
    if os.path.exists(score_path):
        try:
            with open(score_path) as f:
                point_scores: list[float] = json.load(f)
            logger.info("Loaded {} ensemble point scores from {}.", len(point_scores), score_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read ensemble score file: {}. Falling back to recompute.", exc)
            from src.utils.ensemble import compute_ensemble_scores
            point_scores = await asyncio.to_thread(
                compute_ensemble_scores, state["series_id"], structured.detectors_used
            )
    else:
        logger.warning(
            "Ensemble score file not found at {}. Agent may have skipped store_ensemble_scores. "
            "Falling back to recompute.",
            score_path,
        )
        from src.utils.ensemble import compute_ensemble_scores
        point_scores = await asyncio.to_thread(
            compute_ensemble_scores, state["series_id"], structured.detectors_used
        )

    structured.point_scores = point_scores

    logger.info(
        "Primary agent produced report (iteration {}): {} anomalies from ensemble {}",
        state["iteration"] + 1,
        len(structured.anomalies),
        structured.detectors_used,
    )
    return {"result": structured}




async def validate(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    """Run the validator against the current AnomalyReport draft."""
    report = state["result"]
    new_iteration = state["iteration"] + 1

    if report is None:
        return {"critique": "No report was produced.", "iteration": new_iteration}

    context = _collect_tool_results(state["messages"])

    # 1. Deterministic Rule Checks (prevents LLM counting hallucinations)
    tools_called = list(
        dict.fromkeys(
            tc["name"]
            for msg in state["messages"]
            if getattr(msg, "tool_calls", None)
            for tc in getattr(msg, "tool_calls", [])
        )
    )
    detector_calls = [name for name in tools_called if name.endswith("_detector")]

    critique = ""
    if len(detector_calls) < 2:
        critique = f"Fewer than 2 detectors were run. You only ran {len(detector_calls)}: {detector_calls}. You must run at least 2."
    elif "drill_down_range" not in tools_called:
        critique = "No drill-down was performed. You skipped Phase 2."
    elif "store_ensemble_scores" not in tools_called:
        critique = "store_ensemble_scores was NOT called. You skipped Phase 3."

    if critique:
        logger.warning(
            "Validator rejected report (iteration {}, severity=critical, deterministic): {}",
            new_iteration,
            critique,
        )
        retry_msg = {"role": "user", "content": f"SYSTEM/VALIDATOR: Your report was rejected.\n\nReason: {critique}\n\nYou must call the appropriate tools (e.g., drill_down_range) to fix this. Do not just reply with text."}
        return {"critique": critique, "iteration": new_iteration, "messages": [retry_msg]}

    # 2. LLM Checks (for logic, tracing anomalies, etc.)
    user_msg = VALIDATOR_USER_PROMPT.format(
        series_id=state["series_id"],
        iteration=new_iteration,
        context=context,
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

    if validation.accepted:
        logger.info("Validator accepted report on iteration {}", new_iteration)
        return {"critique": "", "iteration": new_iteration}

    logger.warning(
        "Validator rejected report (iteration {}, severity={}): {}",
        new_iteration,
        validation.severity,
        validation.critique,
    )
    retry_msg = {"role": "user", "content": f"SYSTEM/VALIDATOR: Your report was rejected.\n\nReason: {validation.critique}\n\nYou must call the appropriate tools to fix this. Do not just reply with text."}
    return {"critique": validation.critique, "iteration": new_iteration, "messages": [retry_msg]}




def route_after_validator(state: AgentState) -> str:
    """End if accepted or max iterations reached, otherwise retry."""
    if not state["critique"]:
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        logger.warning("Reached MAX_ITERATIONS ({}); returning best report so far.", MAX_ITERATIONS)
        return "end"
    return "retry"




def build_graph(
    llm_with_tools: Any,
    tool_node: ToolNode,
    llm: ChatOpenAI,
    *,
    skip_validator: bool = False,
) -> Any:
    """Assemble and compile the LangGraph.

    When *skip_validator* is False (default), the full dual-agent graph is used:

        primary_agent → tools → primary_agent → ... → finalize
                                                            ↓
                                                        validator
                                                            ↓
                                           ┌── accepted / max retries → END
                                           └── rejected ──────────────→ primary_agent

    When *skip_validator* is True (ablation mode), the validator is removed
    and finalize connects directly to END:

        primary_agent → tools → primary_agent → ... → finalize → END
    """
    builder = StateGraph(AgentState)

    builder.add_node("primary_agent", functools.partial(call_model, llm_with_tools=llm_with_tools))
    builder.add_node("tools", tool_node)
    builder.add_node("too_many_tools", too_many_tools)
    builder.add_node("force_detector", force_detector)
    builder.add_node("finalize", functools.partial(finalize, llm=llm))

    builder.set_entry_point("primary_agent")

    builder.add_conditional_edges(
        "primary_agent",
        should_continue,
        {"tools": "tools", "too_many_tools": "too_many_tools", "force_detector": "force_detector", "finalize": "finalize"},
    )
    builder.add_edge("tools", "primary_agent")
    builder.add_edge("too_many_tools", "primary_agent")
    builder.add_edge("force_detector", "primary_agent")

    if skip_validator:
        builder.add_edge("finalize", END)
    else:
        builder.add_node("validator", functools.partial(validate, llm=llm))
        builder.add_edge("finalize", "validator")
        builder.add_conditional_edges(
            "validator",
            route_after_validator,
            {"end": END, "retry": "primary_agent"},
        )

    return builder.compile()




async def run(series_id: str, *, skip_validator: bool = False) -> AnomalyReport:
    """Run the TSAD pipeline and return the final AnomalyReport.

    Args:
        series_id: Identifier for the time series to analyse.
        skip_validator: If True, skip the validator/refinement agent
            (single-agent ablation mode).
    """
    server_params = default_mcp_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            logger.info("Loaded MCP tools: {}", [t.name for t in tools])

            token_tracker = TokenTrackerCallback()
            llm = ChatOpenAI(
                api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                model=MODEL,
                temperature=0,
                callbacks=[token_tracker],
            )
            llm_with_tools = llm.bind_tools(tools)
            tool_node = ToolNode(tools)

            graph = build_graph(llm_with_tools, tool_node, llm, skip_validator=skip_validator)

            final_state = cast(
                AgentState,
                await graph.ainvoke(
                    {
                        "messages": [],
                        "result": None,
                        "critique": "",
                        "iteration": 0,
                        "series_id": series_id,
                    },
                    config={"callbacks": [token_tracker]}
                ),
            )

            report = cast(AnomalyReport, final_state["result"])
            if report is not None:
                report.prompt_tokens = token_tracker.prompt_tokens
                report.completion_tokens = token_tracker.completion_tokens
                report.total_tokens = token_tracker.total_tokens

                try:
                    from sqlalchemy import create_engine, text
                    from src.utils.db import get_db_url
                    
                    method_name = "agentic_no_validator" if skip_validator else "agentic_with_validator"
                    
                    engine = create_engine(get_db_url())
                    with engine.begin() as conn:
                        conn.execute(text("""
                            CREATE TABLE IF NOT EXISTS token_usage (
                                id SERIAL PRIMARY KEY,
                                dataset_name VARCHAR(255),
                                method VARCHAR(255),
                                prompt_tokens INTEGER,
                                completion_tokens INTEGER,
                                total_tokens INTEGER,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """))
                        
                        conn.execute(text("""
                            INSERT INTO token_usage (dataset_name, method, prompt_tokens, completion_tokens, total_tokens)
                            VALUES (:dataset, :method, :prompt, :completion, :total)
                        """), {
                            "dataset": series_id,
                            "method": method_name,
                            "prompt": token_tracker.prompt_tokens,
                            "completion": token_tracker.completion_tokens,
                            "total": token_tracker.total_tokens
                        })
                    logger.info("Recorded token usage for dataset {}: {} total tokens", series_id, token_tracker.total_tokens)
                except Exception as db_err:
                    logger.warning("Failed to record token usage to database: {}", db_err)

            return report
