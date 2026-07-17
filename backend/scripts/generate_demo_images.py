"""Generate skin-like synthetic before/after demo images + a 4-column cases.csv.

These are *placeholder* images so the app demos believably out of the box. The
Before image shows an inflamed (red, textured) region; the After image shows a
calmer version. Response strength varies per case. Replace data/cases.csv and
data/images/ with real de-identified photos for real predictions.
"""

from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"
IMAGE_DIR = DATA_ROOT / "images"
CASES_CSV = DATA_ROOT / "cases.csv"

WIDTH, HEIGHT = 640, 512

# (case_id, skin_tone RGB, biologic, age, before_severity, after_severity)
CASES = [
    ("DM-001", (232, 190, 168), "Dupixent", 29, 0.60, 0.18),
    ("DM-002", (214, 170, 146), "Dupixent", 41, 0.55, 0.25),
    ("DM-003", (196, 150, 126), "Dupixent", 36, 0.62, 0.42),
    ("DM-004", (170, 120, 96), "Dupixent", 52, 0.58, 0.15),
    ("DM-005", (238, 200, 180), "Dupixent", 24, 0.50, 0.14),
    ("DM-006", (224, 182, 158), "Ebglyss", 33, 0.60, 0.22),
    ("DM-007", (150, 104, 82), "Ebglyss", 46, 0.56, 0.40),
    ("DM-008", (208, 164, 138), "Ebglyss", 58, 0.54, 0.18),
    ("DM-009", (190, 146, 120), "Ebglyss", 31, 0.52, 0.32),
    ("DM-010", (128, 88, 70), "Ebglyss", 67, 0.60, 0.20),
]


def _render(skin: tuple, severity: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.array(skin, dtype=np.float32)
    canvas = np.tile(base, (HEIGHT, WIDTH, 1))
    canvas += rng.normal(0.0, 5.0, size=canvas.shape)  # skin micro-texture

    # Inflamed region: a localized cluster of red blobs (kept away from edges so
    # surrounding normal skin remains for the erythema baseline).
    cx = int(WIDTH * (0.42 + 0.16 * rng.random()))
    cy = int(HEIGHT * (0.42 + 0.16 * rng.random()))
    spread = 55 + int(55 * severity)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]

    intensity = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for _ in range(4):
        bx = cx + int(rng.normal(0, spread * 0.4))
        by = cy + int(rng.normal(0, spread * 0.4))
        rx = spread * (0.5 + 0.5 * rng.random())
        ry = spread * (0.5 + 0.5 * rng.random())
        blob = np.exp(-(((xx - bx) ** 2) / (2 * rx**2) + ((yy - by) ** 2) / (2 * ry**2)))
        intensity = np.maximum(intensity, blob.astype(np.float32))

    intensity *= severity
    # Redness: raise R, lower G/B where inflamed (increases LAB a*).
    canvas[:, :, 0] += intensity * 60.0
    canvas[:, :, 1] -= intensity * 40.0
    canvas[:, :, 2] -= intensity * 35.0

    # Dry scaling: bright flecks inside the inflamed area.
    scale_mask = (intensity > 0.35) & (rng.random((HEIGHT, WIDTH)) < 0.05 * severity)
    canvas[scale_mask] += 40.0

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def main() -> None:
    rows = []
    for index, (case_id, skin, biologic, age, before_sev, after_sev) in enumerate(CASES):
        case_dir = IMAGE_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _render(skin, before_sev, seed=index * 2 + 1).save(
            case_dir / "before.jpg", format="JPEG", quality=90
        )
        _render(skin, after_sev, seed=index * 2 + 1).save(  # same seed = same layout
            case_dir / "after.jpg", format="JPEG", quality=90
        )
        rows.append(
            (f"images/{case_id}/before.jpg", f"images/{case_id}/after.jpg", biologic, age)
        )
        print(f"generated {case_id}: {biologic} age={age}")

    with CASES_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Before", "After", "Biologic", "Age"])
        writer.writerows(rows)
    print(f"wrote {CASES_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    main()
