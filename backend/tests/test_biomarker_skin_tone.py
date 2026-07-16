from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.biomarker_extraction import extract_biomarkers


TONE_CASES = [
    ("light", np.array([218, 170, 145], dtype=np.uint8), np.array([35, -52, -32], dtype=np.int16)),
    ("medium", np.array([186, 132, 108], dtype=np.uint8), np.array([34, -48, -30], dtype=np.int16)),
    ("dark", np.array([122, 84, 66], dtype=np.uint8), np.array([30, -32, -18], dtype=np.int16)),
    ("deep", np.array([84, 58, 48], dtype=np.uint8), np.array([26, -24, -12], dtype=np.int16)),
]


def _png_bytes(arr):
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def _structured_lesion(base, shift, red=True, scaly=True):
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :] = base
    lesion = np.clip(base.astype(np.int16) + (shift if red else np.array([-8, 2, 3], dtype=np.int16)), 0, 255).astype(np.uint8)
    arr[160:352, 160:352] = lesion
    rng = np.random.default_rng(11)
    noise = rng.integers(-22, 23, size=(192, 192, 1), dtype=np.int16)
    patch = np.clip(arr[160:352, 160:352].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    arr[160:352, 160:352] = patch
    if scaly:
        flake = np.clip(base.astype(np.int16) + np.array([72, 68, 62], dtype=np.int16), 0, 255).astype(np.uint8)
        for row in range(178, 338, 22):
            arr[row : row + 2, 178:338] = flake
        for col in range(190, 330, 40):
            arr[180:336, col : col + 1] = flake
    return arr


def test_equivalent_relative_red_shift_scores_stay_in_tolerance_across_synthetic_tones():
    results = []
    for _, base, shift in TONE_CASES:
        features, _ = extract_biomarkers(_png_bytes(_structured_lesion(base, shift, red=True, scaly=False)))
        results.append(features)

    erythema_scores = [features.erythema_score for features in results]
    assert min(erythema_scores) > 0.05
    assert max(erythema_scores) - min(erythema_scores) < 0.55


def test_same_synthetic_lesion_structure_keeps_non_red_scores_similar_on_light_and_dark_skin():
    light_features, _ = extract_biomarkers(_png_bytes(_structured_lesion(TONE_CASES[0][1], TONE_CASES[0][2])))
    dark_features, _ = extract_biomarkers(_png_bytes(_structured_lesion(TONE_CASES[-1][1], TONE_CASES[-1][2])))

    assert abs(light_features.lesion_coverage_pct - dark_features.lesion_coverage_pct) < 8.0
    assert abs(light_features.texture_score - dark_features.texture_score) < 0.35
    assert abs(light_features.dryness_scaling_score - dark_features.dryness_scaling_score) < 0.35
    assert abs(light_features.inflammation_score - dark_features.inflammation_score) < 0.45


def test_dark_non_red_textured_scaly_lesion_drives_inflammation_without_redness_alone():
    features, quality = extract_biomarkers(
        _png_bytes(_structured_lesion(TONE_CASES[-1][1], TONE_CASES[-1][2], red=False, scaly=True))
    )

    assert features.erythema_score < 0.20
    assert features.lesion_coverage_pct > 1.0
    assert features.texture_score > 0.10
    assert features.dryness_scaling_score > 0.10
    assert features.inflammation_score > 0.08
    assert "dark_tone_multicue_inflammation" in quality.warnings


@pytest.mark.parametrize("image_path", sorted(Path("data/images").glob("DM-*/before.jpg"))[:10])
def test_phase_2_synthetic_placeholder_images_return_finite_in_range_vectors(image_path):
    # Phase 2 placeholders are synthetic thumbnails, so this structural smoke test
    # upscales them in memory to satisfy the upload minimum dimensions.
    with Image.open(image_path) as image:
        resized = image.convert("RGB").resize((768, 528), Image.Resampling.NEAREST)
    features, _ = extract_biomarkers(_png_bytes(np.asarray(resized, dtype=np.uint8)))
    values = features.model_dump()

    assert values
    assert all(np.isfinite(value) for value in values.values())
    assert 0.0 <= features.erythema_score <= 1.0
    assert 0.0 <= features.texture_score <= 1.0
    assert 0.0 <= features.dryness_scaling_score <= 1.0
    assert 0.0 <= features.inflammation_score <= 1.0
    assert 0.0 <= features.lesion_coverage_pct <= 100.0
    assert 0.0 <= features.affected_body_area_pct <= 100.0
