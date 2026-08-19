# Build Plan — Continuity Guardian

Target: Google Cloud Agentic Cinema Hackathon, ClickHouse track.
Deadline: Sep 7, 2026, 2:00 PM PT.

## Week 1 (Aug 9–16): Foundation & data pipeline
- [ ] GCP project set up, Vertex AI enabled, $100 hackathon credit applied
- [ ] ClickHouse Cloud instance provisioned; `mcp-clickhouse` installed and
      pointed at it
- [ ] ADK installed and a "hello world" agent runs locally
- [ ] 3–5 self-shot or CC0-licensed multi-take test scenes collected
- [ ] `gemini_video_analysis.py` producing correct structured JSON on a
      real clip

## Week 2 (Aug 17–23): Continuity detection logic
- [ ] ClickHouse schema finalized and applied (`clickhouse_schema.sql`)
- [ ] Loader (`clickhouse_loader.py`) tested end-to-end on real ingested data
- [ ] Cross-take diff logic validated against hand-labeled ground truth on
      at least 2 scenes
- [ ] False-positive rate on the diff logic measured and reduced

## Week 3 (Aug 24–30): Agent + dashboard
- [ ] `adk_agent.py` answering real natural-language questions grounded in
      live ClickHouse data via MCP
- [ ] FastAPI dashboard (`src/dashboard/app.py`) deployed locally, upload →
      flags → chat flow working end-to-end
- [ ] Basic auth / secret management via Secret Manager
- [ ] Gemini safety settings configured

## Week 4 (Aug 31–Sep 6): Polish, deploy, submit
- [ ] Deployed to Cloud Run with a public URL
- [ ] Text description written (features, tech stack, learnings)
- [ ] Demo video recorded and uploaded (≤3 min, YouTube/Vimeo)
- [ ] Repo cleaned: public, MIT license visible in GitHub "About", README
      verified against a clean clone + install
- [ ] Confirm `google-cloud-aiplatform`/ADK and `mcp-clickhouse` are
      genuinely imported and called (not just named) — this is checked
- [ ] Submitted on Devpost before Sep 7, 2:00 PM PT
