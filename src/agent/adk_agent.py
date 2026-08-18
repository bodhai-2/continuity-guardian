"""
adk_agent.py

The Continuity Guardian conversational agent. Built with Google's Agent
Development Kit (ADK) and Gemini, this agent's primary tool is the official
ClickHouse MCP server (`mcp-clickhouse`), connected over stdio via ADK's
MCPToolset. This satisfies the hackathon's ClickHouse-track requirement:
the integration is called live, at runtime, not just referenced.

Run directly for a CLI smoke test:
    python src/agent/adk_agent.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types

load_dotenv()

AGENT_INSTRUCTIONS = """You are Continuity Guardian, an AI script supervisor
assistant. You have live, read-only access to a ClickHouse database
containing structured continuity data extracted from film dailies: shots,
detected objects/actors per take, camera notes, and continuity_flags
(known mismatches between takes).

When asked about continuity issues, props, costumes, or takes:
1. Use your ClickHouse tools to query the relevant tables directly —
   never guess or fabricate data.
2. Prefer querying `continuity_flags` first for known issues, then
   `detections` for raw per-take evidence if more detail is needed.
3. Answer concisely, citing scene and take numbers explicitly
   (e.g. "Take 2 vs Take 4 of SC01").
4. If the database has no relevant data, say so plainly rather than
   speculating about what a real set might look like.
"""


def build_clickhouse_mcp_toolset() -> MCPToolset:
    """
    Launches the official ClickHouse MCP server (`mcp-clickhouse`) as a
    subprocess over stdio and exposes its tools (query, list tables, etc.)
    to the ADK agent. Credentials are passed via environment variables,
    matching mcp-clickhouse's own configuration contract.
    """
    return MCPToolset(
        connection_params=StdioServerParameters(
            command="mcp-clickhouse",
            args=[],
            env={
                "CLICKHOUSE_HOST": os.environ["CLICKHOUSE_HOST"],
                "CLICKHOUSE_PORT": os.environ.get("CLICKHOUSE_PORT", "8443"),
                "CLICKHOUSE_USER": os.environ.get("CLICKHOUSE_USER", "default"),
                "CLICKHOUSE_PASSWORD": os.environ.get("CLICKHOUSE_PASSWORD", ""),
                "CLICKHOUSE_DATABASE": os.environ.get("CLICKHOUSE_DATABASE", "continuity_guardian"),
                "CLICKHOUSE_SECURE": os.environ.get("CLICKHOUSE_SECURE", "true"),
            },
        ),
    )


def build_agent() -> Agent:
    """
    ADK reads GOOGLE_API_KEY (free Gemini Developer API / AI Studio) or the
    GOOGLE_CLOUD_PROJECT + GOOGLE_GENAI_USE_VERTEXAI=true pair (Vertex AI)
    directly from the environment -- no explicit client wiring needed here.
    Set whichever one you're using in .env before running this.
    """
    return Agent(
        name="continuity_guardian",
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        instruction=AGENT_INSTRUCTIONS,
        tools=[build_clickhouse_mcp_toolset()],
        generate_content_config=types.GenerateContentConfig(temperature=0.1),
    )


async def run_cli() -> None:
    agent = build_agent()
    runner = InMemoryRunner(agent=agent, app_name="continuity_guardian")
    session = await runner.session_service.create_session(
        app_name="continuity_guardian", user_id="cli_user"
    )

    print("Continuity Guardian agent ready. Type a question (Ctrl+C to exit).")
    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input.strip():
            continue

        content = types.Content(role="user", parts=[types.Part(text=user_input)])
        async for event in runner.run_async(
            user_id="cli_user", session_id=session.id, new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="")
        print()


if __name__ == "__main__":
    asyncio.run(run_cli())
