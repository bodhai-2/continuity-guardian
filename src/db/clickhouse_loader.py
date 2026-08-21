"""
clickhouse_loader.py

Loads structured continuity JSON (produced by
src/ingestion/gemini_video_analysis.py) into ClickHouse, and runs a simple
cross-take diff to generate continuity_flags rows.

Uses `clickhouse-connect`, the standard Python client for ClickHouse. This
is separate from the MCP server used by the agent (src/agent/adk_agent.py) —
this script is the write path; the agent is the read/query path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

SCHEMA_PATH = Path(__file__).parent / "clickhouse_schema.sql"


def get_client(database: str | None = None):
    """
    database=None connects using CLICKHOUSE_DATABASE from the environment
    (the normal case, once the DB exists). Pass database="" to connect
    without selecting a database -- required the very first time, before
    `CREATE DATABASE` has run.
    """
    if database is None:
        database = os.environ.get("CLICKHOUSE_DATABASE", "continuity_guardian")
    return clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        port=int(os.environ.get("CLICKHOUSE_PORT", 8443)),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=database,
        secure=os.environ.get("CLICKHOUSE_SECURE", "true").lower() == "true",
    )


def init_schema(client) -> None:
    statements = [s.strip() for s in SCHEMA_PATH.read_text().split(";") if s.strip()]
    for stmt in statements:
        client.command(stmt)
    print(f"Applied {len(statements)} schema statements.")


def get_prior_labels(client, scene_id: str) -> list[str]:
    """
    Returns the distinct object/actor labels already recorded for this
    scene from earlier takes. Feeding this back into the Gemini prompt for
    the next take (see gemini_video_analysis.py) lets the model reuse
    consistent naming for the same real-world object across takes, rather
    than us reconstructing that after the fact via fuzzy text matching.
    Returns [] (not an error) if the scene/table doesn't exist yet or the
    query fails -- this is a best-effort quality improvement, not
    something that should ever block ingestion.
    """
    try:
        rows = client.query(
            "SELECT DISTINCT label FROM detections WHERE scene_id = {scene_id:String}",
            parameters={"scene_id": scene_id},
        ).result_rows
        return [r[0] for r in rows]
    except Exception:
        return []


def load_shot_json(client, json_path: Path, thumbnail_b64: str = "") -> str:
    record = json.loads(json_path.read_text())
    scene_id = record["scene_id"]
    take_number = record["take_number"]

    client.insert(
        "shots",
        [[scene_id, take_number, record["source_file"], record.get("shot_summary", ""), thumbnail_b64]],
        column_names=["scene_id", "take_number", "source_file", "shot_summary", "thumbnail_b64"],
    )

    detection_rows = []
    for obj in record.get("detected_objects", []):
        detection_rows.append(
            [scene_id, take_number, "object", obj.get("label", ""),
             obj.get("screen_position", ""), obj.get("state", "")]
        )
    for actor in record.get("actor_notes", []):
        detection_rows.append(
            [scene_id, take_number, "actor", actor.get("actor_description", ""),
             actor.get("pose_or_gesture", ""), actor.get("costume_state", "")]
        )
    if detection_rows:
        client.insert(
            "detections",
            detection_rows,
            column_names=["scene_id", "take_number", "entity_type", "label",
                           "screen_position", "state"],
        )

    cam = record.get("camera_notes", {})
    if cam:
        client.insert(
            "camera_notes",
            [[scene_id, take_number, cam.get("framing", ""),
              cam.get("screen_direction", ""), cam.get("lighting", "")]],
            column_names=["scene_id", "take_number", "framing",
                           "screen_direction", "lighting"],
        )

    for flag_text in record.get("continuity_risk_flags", []):
        client.insert(
            "continuity_flags",
            [[scene_id, take_number, take_number, "unspecified", flag_text, "medium"]],
            column_names=["scene_id", "take_a", "take_b", "entity_label",
                           "flag_text", "severity"],
        )

    print(f"Loaded shot {scene_id} take {take_number} into ClickHouse.")
    return scene_id


def diff_takes_for_scene(client, scene_id: str) -> int:
    """
    Compare detections across all loaded takes of a scene and insert
    continuity_flags rows where the same labeled object/actor has a
    different state or screen_position between takes.
    """
    rows = client.query(
        """
        SELECT take_number, entity_type, label, screen_position, state
        FROM detections
        WHERE scene_id = {scene_id:String}
        ORDER BY label, take_number
        """,
        parameters={"scene_id": scene_id},
    ).result_rows

FILLER_WORDS = {"small", "medium", "large", "the", "a", "an", "object", "shape"}

STATE_STOPWORDS = {
    "and", "with", "the", "a", "an", "of", "in", "on", "at", "to", "is",
    "are", "facing", "visible",
}


def _tokenize_state(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STATE_STOPWORDS}


def _states_differ(a: str, b: str) -> bool:
    """
    Word-set similarity, not exact-text equality. The same real-world
    state described in a different word order (e.g. "yellow handles and
    silver metal blades" vs "silver metal blades with yellow plastic
    handles" -- both plainly yellow) should NOT count as a change just
    because Gemini phrased it differently between calls. A genuine color
    or condition change (yellow -> green) still drops similarity well
    below the threshold, since the differing word is exactly the one that
    matters. Threshold tuned against real test data, not synthetic
    examples -- see docs/BUILD_PLAN.md testing notes.
    """
    if a.lower() == b.lower():
        return False
    tokens_a, tokens_b = _tokenize_state(a), _tokenize_state(b)
    if not tokens_a or not tokens_b:
        return a.lower() != b.lower()
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return jaccard < 0.6


def _normalize_label(label: str) -> str:
    """Strip only generic size/article noise so labels like 'small outlined
    blue circle' and 'Outlined blue circle' compare as the same entity --
    but deliberately keeps shape/texture words (circular, rectangular,
    outlined, solid) since those are what distinguish genuinely different
    objects, not noise."""
    words = re.findall(r"[a-z]+", label.lower())
    core = [w for w in words if w not in FILLER_WORDS]
    return " ".join(core) if core else label.lower()


def _labels_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() > 0.65


def diff_takes_for_scene(client, scene_id: str) -> int:
    """
    Compare detections across all loaded takes of a scene and insert
    continuity_flags rows where the same real-world object/actor has a
    different state or screen_position between takes.

    Gemini rarely re-describes the same object with identical wording
    across separate calls, so this groups detections by *fuzzy* label
    similarity rather than requiring an exact string match. Crucially, a
    cluster is only allowed ONE entry per take -- two objects detected in
    the SAME take can never be the same real-world object, even if their
    labels happen to be textually similar (e.g. two blue circular objects
    in one frame), so same-take collisions are never merged.

    Known limitation: two genuinely different objects whose descriptions
    differ by only one word (e.g. two similarly-colored props in the same
    scene) can occasionally have their identities swapped between takes --
    pure text similarity can't always distinguish "different wording, same
    object" from "different wording, different object". Detection that
    *something* changed stays reliable even then; only the specific
    object attribution can be off in that edge case. A stronger fix would
    feed the previous take's object list back into the Gemini prompt in
    gemini_video_analysis.py so the model itself maintains consistent
    naming across takes -- worth building if this proves noisy on real
    footage with several visually similar props in frame.
    """
    rows = client.query(
        """
        SELECT take_number, entity_type, label, screen_position, state
        FROM detections
        WHERE scene_id = {scene_id:String}
        ORDER BY take_number
        """,
        parameters={"scene_id": scene_id},
    ).result_rows

    # Cluster detections into entities across takes using fuzzy label match.
    entities: list[dict] = []
    for take_number, entity_type, label, position, state in rows:
        norm = _normalize_label(label)
        match = next(
            (
                e for e in entities
                if e["entries"][-1][0] != take_number  # never merge same-take detections
                and any(_labels_match(norm, n) for n in e["norms"])
            ),
            None,
        )
        if match is None:
            match = {"norms": set(), "entries": []}
            entities.append(match)
        match["norms"].add(norm)
        match["entries"].append((take_number, label, position, state))

    new_flags = []

    # All take numbers that actually have data for this scene -- needed to
    # detect a "gap" (an object present before AND after a take where it
    # simply wasn't detected at all).
    all_takes = sorted({r[0] for r in rows})

    for entity in entities:
        entries = sorted(entity["entries"], key=lambda e: e[0])

        # 1. Position/state changes between the object's consecutive
        #    *appearances*. Severity is differentiated, not hardcoded:
        #    a STATE change (e.g. zipped -> unzipped, full -> empty) is a
        #    more concrete, verifiable continuity break than a POSITION-only
        #    difference, which can sometimes just be an artifact of
        #    Gemini's phrasing rather than something an actor actually
        #    changed. Both differing at once is the clearest case, so it
        #    stays high; state-only is high (a costume/prop condition
        #    changed); position-only alone is medium (worth a look, less
        #    certain to be a real error).
        for i in range(len(entries) - 1):
            take_a, label_a, pos_a, state_a = entries[i]
            take_b, label_b, pos_b, state_b = entries[i + 1]
            pos_changed = _states_differ(pos_a, pos_b)
            state_changed = _states_differ(state_a, state_b)
            if pos_changed or state_changed:
                severity = "high" if state_changed else "medium"
                flag_text = (
                    f"'{label_a}' differs between take {take_a} "
                    f"({pos_a}, {state_a}) and take {take_b} ({pos_b}, {state_b})"
                )
                new_flags.append([scene_id, take_a, take_b, label_b, flag_text, severity])

        # 2. Gaps: the object appears, is completely absent from an
        #    intermediate take, then reappears -- flagged even if its
        #    position/state matches before and after, since disappearing
        #    for an entire take is itself the continuity error.
        entity_takes = {e[0] for e in entries}
        for i in range(len(entries) - 1):
            take_a, label_a, _, _ = entries[i]
            take_b, label_b, _, _ = entries[i + 1]
            missing_takes = [
                t for t in all_takes
                if take_a < t < take_b and t not in entity_takes
            ]
            for missing_take in missing_takes:
                flag_text = (
                    f"'{label_a}' is present in take {take_a} and take {take_b}, "
                    f"but was not detected at all in take {missing_take} -- "
                    f"check whether it genuinely left frame or was missed"
                )
                new_flags.append([scene_id, take_a, missing_take, label_a, flag_text, "high"])

    if new_flags:
        client.insert(
            "continuity_flags",
            new_flags,
            column_names=["scene_id", "take_a", "take_b", "entity_label",
                           "flag_text", "severity"],
        )
    print(f"Generated {len(new_flags)} cross-take continuity flags for {scene_id}.")
    return len(new_flags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-schema", action="store_true")
    parser.add_argument("--load", type=Path, help="Path to a shot JSON file to load")
    parser.add_argument("--diff-scene", help="Run cross-take diff for this scene_id")
    args = parser.parse_args()

    if args.init_schema:
        # No database selected yet -- the schema script itself creates it.
        client = get_client(database="")
        init_schema(client)
    if args.load:
        client = get_client()
        scene_id = load_shot_json(client, args.load)
        diff_takes_for_scene(client, scene_id)
    if args.diff_scene:
        client = get_client()
        diff_takes_for_scene(client, args.diff_scene)


if __name__ == "__main__":
    main()
