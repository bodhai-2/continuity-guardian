# Continuity Guardian

An agentic AI Script Supervisor for film/TV productions. It watches raw dailies
(multi-take footage), uses Gemini multimodal analysis to detect continuity
errors across takes (prop position, costume, screen direction, lighting), and
lets script supervisors and editors query the entire shoot's continuity data
in natural language — grounded live against a ClickHouse analytics store via
the official ClickHouse MCP server.

Built for the **Google Cloud Agentic Cinema Hackathon** (ClickHouse track).

## Why this exists

Continuity errors are one of the most expensive, tedious problems on a film
set — a script supervisor manually cross-checks hundreds of takes by memory
and Polaroids. Nobody has shipped an accessible AI tool that automates this
end-to-end. Continuity Guardian turns raw footage into a queryable,
structured continuity database and puts a conversational agent in front of it.

## Architecture

```
Dailies (video) --> Gemini multimodal analysis --> structured per-shot JSON
                                                          |
                                                          v
                                              ClickHouse (shots, detections,
                                              continuity_flags tables)
                                                          |
                                                          v
                              ADK Agent (Gemini) + ClickHouse MCP tool
                                                          |
                                                          v
                                   Web dashboard (FastAPI, Cloud Run)
                       - flag review timeline
                       - chat-with-your-dailies interface
```

## Repo layout

```
src/
  ingestion/          # Gemini multimodal video analysis -> structured JSON
  db/                 # ClickHouse schema + loader
  agent/              # ADK agent wired to ClickHouse via MCP
  dashboard/          # FastAPI web app (upload, flag review, chat)
tests/
docs/
```

## Setup

### 1. Prerequisites
- Python 3.11+
- A Google Cloud project with Vertex AI enabled, and the $100 hackathon credit
  applied (see hackathon resources page)
- A ClickHouse Cloud instance (or self-hosted cluster)
- `uv` or `pip` for dependency install

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

This installs the ADK (`google-cloud-aiplatform[agent_engines,adk]`),
`google-genai`, `clickhouse-connect`, the `mcp` client library, and the
dashboard stack.

### 3. Install the ClickHouse MCP server

```bash
pip install mcp-clickhouse
```

`mcp-clickhouse` is the official ClickHouse MCP server — the ADK agent
launches it as a subprocess over stdio and calls its tools directly (see
`src/agent/adk_agent.py`).

### 4. Configure environment

```bash
cp .env.example .env
# fill in GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
# CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD,
# CLICKHOUSE_DATABASE
```

### 5. Create the ClickHouse schema

```bash
python src/db/clickhouse_loader.py --init-schema
```

### 6. Run the ingestion pipeline on a test clip

```bash
python src/ingestion/gemini_video_analysis.py \
  --video path/to/take.mp4 --scene SC01 --take 3 \
  --out data/SC01_take3.json

python src/db/clickhouse_loader.py --load data/SC01_take3.json
```

### 7. Run the agent (CLI test)

```bash
python src/agent/adk_agent.py
```

### 8. Run the dashboard

```bash
uvicorn src.dashboard.app:app --reload
```

## Hackathon compliance notes

- Uses `google-cloud-aiplatform` (ADK) and `google-genai` for all AI/agent
  logic — no non-Google AI vendor is called anywhere in this repo.
- Uses the official `mcp-clickhouse` MCP server, connected to a live
  ClickHouse cluster, called at runtime by the ADK agent (not just referenced
  in this README) — see `src/agent/adk_agent.py`.
- Licensed under MIT (see `LICENSE`) — open source, as required.

## Status

Early scaffold — see `docs/BUILD_PLAN.md` for the week-by-week plan this repo
is being built against.
