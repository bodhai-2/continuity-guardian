"""
gemini_video_analysis.py

Ingests a single take (video clip) and uses Gemini's multimodal
understanding to extract structured, per-shot continuity-relevant metadata:
detected objects, their approximate screen position, actor pose/costume
notes, and lighting/framing descriptors.

This is the entry point of the Continuity Guardian pipeline: raw footage in,
structured JSON out. That JSON is later loaded into ClickHouse by
`src/db/clickhouse_loader.py`.

Uses the `google-genai` SDK against Vertex AI (Gemini). No non-Google AI
vendor is used anywhere in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Gemini model to use for multimodal video understanding. Swap for whichever
# current Gemini model your project has quota for.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

CONTINUITY_ANALYSIS_PROMPT = """You are an expert film script supervisor
analyzing a single take for continuity purposes.

Watch the provided clip and return STRICT JSON (no markdown, no commentary)
matching this shape:

{
  "shot_summary": "one sentence describing the action in this take",
  "detected_objects": [
    {
      "label": "e.g. coffee mug",
      "screen_position": "e.g. left hand, held at chest height",
      "state": "e.g. half full, handle facing camera"
    }
  ],
  "actor_notes": [
    {
      "actor_description": "e.g. actor in blue jacket",
      "costume_state": "e.g. jacket zipped, collar up",
      "pose_or_gesture": "e.g. right hand in pocket"
    }
  ],
  "camera_notes": {
    "framing": "e.g. medium close-up",
    "screen_direction": "e.g. actor exits frame left",
    "lighting": "e.g. warm key light from camera left"
  },
  "continuity_risk_flags": [
    "short natural-language notes on anything that looks like it could
     mismatch a previous or future take, if inferrable from this clip alone"
  ]
}

Be precise and concrete. If something is not visible or not determinable,
omit it rather than guessing.
"""


def build_prompt(prior_labels: list[str] | None = None) -> str:
    """
    Appends prior-take object labels to the base prompt, if any exist, so
    Gemini reuses consistent naming for the same real-world object across
    takes instead of re-describing it differently each time. This is the
    main fix for the label-matching ambiguity found during testing --
    letting the model itself maintain consistency (it can see the object)
    is more reliable than reconstructing it from text similarity alone
    after the fact.
    """
    if not prior_labels:
        return CONTINUITY_ANALYSIS_PROMPT
    labels_list = ", ".join(f'"{label}"' for label in prior_labels)
    return (
        CONTINUITY_ANALYSIS_PROMPT
        + f"\n\nIMPORTANT: earlier takes of this same scene already described "
        f"objects/actors using these exact labels: {labels_list}. If you see "
        f"what appears to be the same real-world object or person, REUSE the "
        f"exact same label text rather than describing it differently. Only "
        f"introduce a new label for something genuinely not in that list."
    )


@dataclass
class ShotRecord:
    scene_id: str
    take_number: int
    source_file: str
    shot_summary: str = ""
    detected_objects: list[dict[str, Any]] = field(default_factory=list)
    actor_notes: list[dict[str, Any]] = field(default_factory=list)
    camera_notes: dict[str, Any] = field(default_factory=dict)
    continuity_risk_flags: list[str] = field(default_factory=list)


def build_client(project: str | None = None, location: str | None = None) -> genai.Client:
    """
    Builds a Gemini client. Uses the free Gemini Developer API (AI Studio key)
    if GOOGLE_API_KEY is set, otherwise falls back to Vertex AI (requires a
    GCP project with billing enabled).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client(vertexai=True, project=project, location=location)


def analyze_take(
    video_path: Path,
    scene_id: str,
    take_number: int,
    project: str | None = None,
    location: str | None = None,
    prior_labels: list[str] | None = None,
) -> ShotRecord:
    """Send a single take to Gemini and parse the structured response."""
    client = build_client(project=project, location=location)

    video_bytes = video_path.read_bytes()
    video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[video_part, build_prompt(prior_labels)],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    try:
        parsed = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"Gemini did not return valid JSON for {video_path}: {exc}\n"
            f"Raw response: {getattr(response, 'text', response)}"
        ) from exc

    return ShotRecord(
        scene_id=scene_id,
        take_number=take_number,
        source_file=str(video_path),
        shot_summary=parsed.get("shot_summary", ""),
        detected_objects=parsed.get("detected_objects", []),
        actor_notes=parsed.get("actor_notes", []),
        camera_notes=parsed.get("camera_notes", {}),
        continuity_risk_flags=parsed.get("continuity_risk_flags", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Path to the take's video file")
    parser.add_argument("--scene", required=True, help="Scene identifier, e.g. SC01")
    parser.add_argument("--take", required=True, type=int, help="Take number, e.g. 3")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the resulting JSON")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    args = parser.parse_args()

    if not args.project and not os.environ.get("GOOGLE_API_KEY"):
        sys.exit(
            "Set either GOOGLE_API_KEY (free AI Studio key) or "
            "GOOGLE_CLOUD_PROJECT (env or --project) for Vertex AI."
        )
    if not args.video.exists():
        sys.exit(f"Video file not found: {args.video}")

    prior_labels: list[str] = []
    try:
        from src.db.clickhouse_loader import get_client, get_prior_labels
        prior_labels = get_prior_labels(get_client(), args.scene)
    except Exception:
        pass  # ClickHouse not reachable yet (e.g. schema not initialized) -- fine, proceed without it

    record = analyze_take(
        video_path=args.video,
        scene_id=args.scene,
        take_number=args.take,
        project=args.project,
        location=args.location,
        prior_labels=prior_labels,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(record), indent=2))
    print(f"Wrote continuity analysis to {args.out}")


if __name__ == "__main__":
    main()
