"""
app.py

FastAPI dashboard for Continuity Guardian:
  GET  /               - the web UI (upload, flag review, chat)
  POST /upload          - accept a take video, run it through the Gemini
                         ingestion pipeline, load results into ClickHouse
  GET  /flags/{scene}  - list continuity flags for a scene
  POST /chat           - forward a natural-language question to the ADK
                         agent (which queries ClickHouse via MCP) and
                         return the answer

Meant to be deployed on Cloud Run, Render, or Railway.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from src.agent.adk_agent import build_agent
from src.db.clickhouse_loader import (
    diff_takes_for_scene,
    get_client,
    get_prior_labels,
    load_shot_json,
)
from src.ingestion.gemini_video_analysis import analyze_take
from src.ingestion.thumbnail import extract_thumbnail_b64

load_dotenv()

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="Continuity Guardian")


def require_access_key(x_access_key: str | None = Header(None)):
    """
    Guards the two Gemini-quota-costly endpoints (/upload, /chat). If
    DASHBOARD_ACCESS_KEY is unset in the environment, protection is
    disabled (useful for local dev). Once deployed publicly, set it so
    random traffic can't drain your daily Gemini quota before judges
    test the demo.
    """
    required = os.environ.get("DASHBOARD_ACCESS_KEY")
    if not required:
        return  # protection disabled -- no key configured
    if x_access_key != required:
        raise HTTPException(401, "Missing or incorrect access key.")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serves the web dashboard UI."""
    return (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/upload", dependencies=[Depends(require_access_key)])
async def upload_take(
    scene_id: str,
    take_number: int,
    file: UploadFile = File(...),
):
    """Accept a video file, run Gemini analysis, and load it into ClickHouse."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project and not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            500,
            "Set either GOOGLE_API_KEY (free AI Studio key) or "
            "GOOGLE_CLOUD_PROJECT (Vertex AI) in your environment.",
        )

    client = get_client()

    # Feed the previous takes' object labels back into the prompt so Gemini
    # reuses consistent naming for the same real-world object across takes,
    # instead of us guessing at fuzzy text matches after the fact.
    prior_labels = get_prior_labels(client, scene_id)

    # Cap stored video size -- ClickHouse handles large strings fine, but
    # on a free-tier deployment (limited RAM, limited storage) it's worth
    # protecting against someone uploading a huge file and silently
    # bloating the database. Above this, we keep the thumbnail (for the
    # flags/scenes views) but skip storing the full video for playback.
    MAX_STORED_VIDEO_BYTES = 25 * 1024 * 1024  # 25 MB

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / file.filename
        with video_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        record = analyze_take(
            video_path=video_path,
            scene_id=scene_id,
            take_number=take_number,
            project=project,
            location=location,
            prior_labels=prior_labels,
        )

        thumbnail_b64 = extract_thumbnail_b64(video_path)

        video_bytes = video_path.read_bytes()
        video_b64 = ""
        if len(video_bytes) <= MAX_STORED_VIDEO_BYTES:
            video_b64 = base64.b64encode(video_bytes).decode("ascii")
        video_mime = "video/mp4" if video_path.suffix.lower() == ".mp4" else "video/webm"

        json_path = Path(tmp) / "record.json"
        json_path.write_text(json.dumps(asdict(record)))

        load_shot_json(client, json_path, thumbnail_b64=thumbnail_b64, video_b64=video_b64, video_mime=video_mime)
        flags_added = diff_takes_for_scene(client, scene_id)

    return {
        "scene_id": scene_id,
        "take_number": take_number,
        "shot_summary": record.shot_summary,
        "new_continuity_flags": flags_added,
        "video_stored": bool(video_b64),
    }


@app.get("/scenes")
async def list_scenes():
    """
    Returns every scene that has data, with take count and flag summary --
    lets someone landing on the dashboard cold browse what's there instead
    of needing to already know a Scene ID to type in.
    """
    client = get_client()

    shot_rows = client.query(
        "SELECT scene_id, count(DISTINCT take_number) AS takes "
        "FROM shots GROUP BY scene_id ORDER BY scene_id"
    ).result_rows

    flag_rows = client.query(
        "SELECT scene_id, count() AS total_flags, countIf(resolved = 0) AS open_flags, "
        "countIf(resolved = 0 AND severity = 'high') AS open_high_flags "
        "FROM continuity_flags GROUP BY scene_id"
    ).result_rows
    flag_map = {r[0]: {"total_flags": r[1], "open_flags": r[2], "open_high_flags": r[3]} for r in flag_rows}

    scenes = []
    for scene_id, takes in shot_rows:
        f = flag_map.get(scene_id, {"total_flags": 0, "open_flags": 0, "open_high_flags": 0})
        scenes.append({"scene_id": scene_id, "takes": takes, **f})

    return {"scenes": scenes}


@app.get("/flags/{scene_id}")
async def get_flags(scene_id: str):
    client = get_client()
    rows = client.query(
        """
        SELECT toString(flag_id), take_a, take_b, entity_label, flag_text, severity, resolved
        FROM continuity_flags
        WHERE scene_id = {scene_id:String}
        ORDER BY resolved ASC, severity DESC, ingested_at DESC
        """,
        parameters={"scene_id": scene_id},
    ).result_rows

    return {
        "scene_id": scene_id,
        "flags": [
            {
                "flag_id": r[0], "take_a": r[1], "take_b": r[2], "entity_label": r[3],
                "flag_text": r[4], "severity": r[5], "resolved": bool(r[6]),
            }
            for r in rows
        ],
    }


@app.post("/flags/{flag_id}/resolve")
async def resolve_flag(flag_id: str):
    """
    Marks a flag as resolved. Uses a ClickHouse ALTER ... UPDATE mutation --
    these apply asynchronously in ClickHouse, so on a busy cluster the
    change may take a moment to be visible on the next /flags read. Fine
    for this scale of usage.
    """
    client = get_client()
    client.command(
        "ALTER TABLE continuity_flags UPDATE resolved = 1 WHERE flag_id = {flag_id:String}",
        parameters={"flag_id": flag_id},
    )
    return {"flag_id": flag_id, "resolved": True}


@app.get("/shots/{scene_id}")
async def get_shots(scene_id: str):
    """Returns each take's summary and thumbnail for a scene, for the
    side-by-side visual comparison strip in the dashboard. Does NOT
    include the full video (that would bloat this response) -- see
    GET /shots/{scene_id}/video/{take_number} for actual playback."""
    client = get_client()
    rows = client.query(
        """
        SELECT take_number, shot_summary, thumbnail_b64, length(video_b64) > 0 AS has_video
        FROM shots
        WHERE scene_id = {scene_id:String}
        ORDER BY take_number ASC
        """,
        parameters={"scene_id": scene_id},
    ).result_rows

    return {
        "scene_id": scene_id,
        "shots": [
            {"take_number": r[0], "shot_summary": r[1], "thumbnail_b64": r[2], "has_video": bool(r[3])}
            for r in rows
        ],
    }


@app.get("/shots/{scene_id}/video/{take_number}")
async def get_shot_video(scene_id: str, take_number: int):
    """
    Streams the actual take footage back for in-browser playback. Video is
    stored base64-encoded in ClickHouse (see /upload) -- fine for a
    hackathon-scale number of demo takes, but worth knowing this isn't how
    you'd architect video storage at real production scale (a dedicated
    object store like GCS/S3 would be the real answer there).
    """
    client = get_client()
    rows = client.query(
        """
        SELECT video_b64, video_mime
        FROM shots
        WHERE scene_id = {scene_id:String} AND take_number = {take_number:UInt16}
        LIMIT 1
        """,
        parameters={"scene_id": scene_id, "take_number": take_number},
    ).result_rows

    if not rows or not rows[0][0]:
        raise HTTPException(404, "No video stored for this take.")

    video_b64, video_mime = rows[0]
    video_bytes = base64.b64decode(video_b64)
    return Response(content=video_bytes, media_type=video_mime or "video/mp4")


_agent = None
_runner = None


@app.post("/chat", dependencies=[Depends(require_access_key)])
async def chat(req: ChatRequest):
    """Forward a question to the ADK agent, which queries ClickHouse via MCP."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    global _agent, _runner
    if _agent is None:
        _agent = build_agent()
        _runner = InMemoryRunner(agent=_agent, app_name="continuity_guardian")

    session = await _runner.session_service.create_session(
        app_name="continuity_guardian", user_id="dashboard_user"
    )
    content = types.Content(role="user", parts=[types.Part(text=req.message)])

    reply_text = ""
    async for event in _runner.run_async(
        user_id="dashboard_user", session_id=session.id, new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    reply_text += part.text

    return {"reply": reply_text}


@app.get("/health")
async def health():
    return {"status": "ok"}
