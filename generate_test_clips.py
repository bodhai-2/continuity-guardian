"""
Generates two short synthetic 'takes' of a fake scene, for testing the
Continuity Guardian ingestion pipeline without needing real filmed footage.

Take 1: a blue circle ("mug") sits on the LEFT side of the frame.
Take 2: the same blue circle sits on the RIGHT side of the frame.

This is a deliberate, obvious continuity error (prop moved sides between
takes) so you can confirm the whole pipeline -- Gemini analysis -> ClickHouse
load -> cross-take diff -> flag generation -- works end to end.

Run: python generate_test_clips.py
Outputs: clips/sc01_take1.mp4, clips/sc01_take2.mp4
"""

import cv2
import numpy as np
from pathlib import Path

OUT_DIR = Path("clips")
OUT_DIR.mkdir(exist_ok=True)

WIDTH, HEIGHT = 640, 360
FPS = 24
DURATION_SEC = 4
N_FRAMES = FPS * DURATION_SEC


def make_take(filename: str, mug_x_frac: float):
    """mug_x_frac: horizontal position of the 'mug' as a fraction of width."""
    path = OUT_DIR / filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))

    mug_x = int(WIDTH * mug_x_frac)
    mug_y = int(HEIGHT * 0.65)

    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), (40, 40, 40), dtype=np.uint8)  # dark bg

        # "actor" -- a simple rectangle, centered, slightly bobbing
        actor_x = WIDTH // 2
        actor_y = int(HEIGHT * 0.45 + 3 * np.sin(i / 5))
        cv2.rectangle(frame, (actor_x - 40, actor_y - 60), (actor_x + 40, actor_y + 60),
                      (180, 140, 100), -1)  # skin-tone-ish torso block
        cv2.circle(frame, (actor_x, actor_y - 80), 25, (200, 180, 160), -1)  # head

        # "mug" -- a blue circle at a fixed position (left in take1, right in take2)
        cv2.circle(frame, (mug_x, mug_y), 18, (200, 100, 30), -1)  # blue-ish mug
        cv2.circle(frame, (mug_x, mug_y), 18, (255, 255, 255), 2)  # outline

        cv2.putText(frame, filename.replace(".mp4", ""), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    print(f"Wrote {path} ({N_FRAMES} frames, {DURATION_SEC}s)")


if __name__ == "__main__":
    make_take("sc01_take1.mp4", mug_x_frac=0.25)   # mug on the left
    make_take("sc01_take2.mp4", mug_x_frac=0.75)   # mug on the right -- continuity error
    print("\nDone. Copy the clips/ folder into your project and run the ingestion "
          "pipeline on both files.")
