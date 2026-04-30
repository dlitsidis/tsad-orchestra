"""TSAD Orchestra Agent — LangChain MCP pipeline (fully fixed)."""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agent.models import AnomalyReport
from src.agent.prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT

load_dotenv()

MODEL = "gpt-4o-mini"


def default_mcp_server_params() -> StdioServerParameters:
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


async def run(series: list[float]) -> AnomalyReport:
    llm_client = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=MODEL,
    )

    logger.info("Starting TSAD run with {} points.", len(series))

    server_params = default_mcp_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            tool_names = [tool.name for tool in tools]

            logger.info("Discovered {} MCP tools: {}", len(tool_names), tool_names)

            agent = create_agent(
                model=llm_client,
                tools=tools,
                system_prompt=AGENT_SYSTEM_PROMPT,
                response_format=AnomalyReport,
            )

            inputs = {
                "messages": [
                    {
                        "role": "user",
                        "content": AGENT_USER_PROMPT.format(series=series),
                    }
                ]
            }

            result = await agent.ainvoke(inputs)

            structured = result.get("structured_response")

            if structured is None:
                raise ValueError("Agent did not return structured ")

            return structured  # type: ignore[no-any-return]


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

    logger.info("Input series: {}", mock_series)

    try:
        answer = asyncio.run(run(mock_series))
        logger.info("Result: {}", answer.model_dump())

    except Exception as e:
        logger.error(e)
