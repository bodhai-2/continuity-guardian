"""
thumbnail.py

Extracts a single representative frame from a take's video as a base64
JPEG string, for the dashboard's side-by-side take comparison strip.
Deliberately lightweight (OpenCV only, no Gemini call) since this runs on
every upload and shouldn't add meaningful latency or cost.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2


def extract_thumbnail_b64(video_path: Path, max_width: int = 480) -> str:
    """
    Grabs the middle frame of the video (a reasonable default -- avoids
    black/blank opening or closing frames common in raw takes), resizes it
    to keep the stored payload small, and returns it as a base64-encoded
    JPEG string. Returns "" if the video can't be read, so a thumbnail
    failure never blocks the rest of the ingestion pipeline.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = max(frame_count // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return ""

    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)))

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""

    return base64.b64encode(buffer).decode("ascii")
