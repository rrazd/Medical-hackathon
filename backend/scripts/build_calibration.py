"""Precompute per-biologic recommendation calibration offsets.

The reference dataset's two biologic cohorts have different cluster shapes, so the
raw nearest-neighbor similarity is not directly comparable between them: over the
distribution of real uploaded photos one biologic tends to score higher purely as
a dataset artifact, biasing every recommendation toward it.

This script estimates that bias empirically. It generates a large, deterministic
sample of diverse synthetic skin photos, runs them through the real biomarker
extractor + matcher, and measures each biologic's mean nearest-neighbor similarity
over that realistic distribution. The calibration offset shifts both biologics to
their shared mean so a typical upload scores them equally — after which the
patient's own biomarkers (and intake nudges) decide the winner.

Run from the backend directory:

    .venv/bin/python scripts/build_calibration.py

It writes data/calibration.json, which the API loads at startup. Re-run whenever
the reference dataset or the scoring code changes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

# Ensure the app package is importable when run as a script.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.biomarker_extraction import extract_biomarkers  # noqa: E402
from app.services.image_dataset import ImageReferenceRepository  # noqa: E402
from app.services.image_predict import BIOLOGICS, _score_cases  # noqa: E402

SAMPLE_SIZE = 200
SEED = 42
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
OUTPUT = DATA_ROOT / "calibration.json"


def _to_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.clip(0, 255).astype("uint8")).save(buffer, format="PNG")
    return buffer.getvalue()


def _diverse_photo(rng: np.random.Generator) -> bytes:
    """A random skin-tone base with 0–6 red lesion patches and texture noise.

    Spans the range of plausible real uploads: skin tone, inflamed-patch coverage,
    and image noise all vary, so the resulting biomarker vectors cover the space
    the extractor actually produces for real photos.
    """
    tone = rng.integers(120, 230, 3)
    img = np.full((150, 150, 3), tone, dtype=float)
    for _ in range(rng.integers(0, 7)):
        x, y = rng.integers(10, 120, 2)
        size = rng.integers(15, 50)
        img[y : y + size, x : x + size] = [
            rng.integers(150, 230),
            rng.integers(20, 110),
            rng.integers(20, 110),
        ]
    img += rng.normal(0, rng.integers(5, 45), img.shape)
    return _to_png(img)


def _nearest_similarity(biologic: str, scored) -> float:
    matches = [item for item in scored if item.case.biologic == biologic]
    return matches[0].similarity if matches else 0.0


def build_offsets() -> dict:
    repo = ImageReferenceRepository()
    cases = repo.list_cases()
    rng = np.random.default_rng(SEED)

    sums = {biologic: 0.0 for biologic in BIOLOGICS}
    counts = {biologic: 0 for biologic in BIOLOGICS}
    for _ in range(SAMPLE_SIZE):
        features, _ = extract_biomarkers(_diverse_photo(rng))
        scored = _score_cases(features, 35, cases)
        for biologic in BIOLOGICS:
            sums[biologic] += _nearest_similarity(biologic, scored)
            counts[biologic] += 1

    means = {b: sums[b] / counts[b] for b in BIOLOGICS if counts[b]}
    grand_mean = mean(means.values())
    offsets = {b: round(grand_mean - means[b], 6) for b in BIOLOGICS}
    return {
        "offsets": offsets,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "mean_similarity": {b: round(v, 6) for b, v in means.items()},
    }


def main() -> None:
    payload = build_offsets()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
