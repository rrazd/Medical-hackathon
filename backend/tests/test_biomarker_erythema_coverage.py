from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.services.biomarker_extraction import extract_biomarkers


SKIN = np.array([186, 132, 108], dtype=np.uint8)
RED_LESION = np.array([215, 75, 75], dtype=np.uint8)
NEUTRAL_PATCH = np.array([170, 125, 105], dtype=np.uint8)


def _png_bytes(arr):
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def _skin_image(patch_color=None, patch=(200, 200, 300, 300), base=SKIN):
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :] = base
    if patch_color is not None:
        y1, x1, y2, x2 = patch
        arr[y1:y2, x1:x2] = patch_color
    return arr


def test_relative_erythema_is_high_for_red_patch_and_low_for_uniform_skin():
    uniform_features, uniform_quality = extract_biomarkers(_png_bytes(_skin_image()))
    red_features, red_quality = extract_biomarkers(_png_bytes(_skin_image(RED_LESION)))

    assert 0.0 <= uniform_features.erythema_score <= 1.0
    assert 0.0 <= red_features.erythema_score <= 1.0
    assert uniform_features.erythema_score <= 0.05
    assert red_features.erythema_score > 0.5
    assert red_features.erythema_score > uniform_features.erythema_score
    assert uniform_quality.skin_pixel_fraction > 0.95
    assert red_quality.skin_pixel_fraction > 0.95


def test_neutral_patch_does_not_score_like_relative_redness():
    neutral_features, _ = extract_biomarkers(_png_bytes(_skin_image(NEUTRAL_PATCH)))
    red_features, _ = extract_biomarkers(_png_bytes(_skin_image(RED_LESION)))

    assert neutral_features.erythema_score < red_features.erythema_score
    assert neutral_features.erythema_score <= 0.2


def test_lesion_coverage_pct_uses_skin_roi_denominator_and_mirrors_affected_area():
    patch = (192, 192, 320, 320)
    expected_pct = 100.0 * ((320 - 192) * (320 - 192)) / float(512 * 512)

    features, quality = extract_biomarkers(_png_bytes(_skin_image(RED_LESION, patch=patch)))

    assert features.lesion_coverage_pct == pytest.approx(expected_pct, abs=1.2)
    assert features.affected_body_area_pct == pytest.approx(features.lesion_coverage_pct)
    assert 0.0 <= features.lesion_coverage_pct <= 100.0
    assert "visible_roi_area_only" in quality.warnings


def test_same_relative_red_shift_scores_on_light_and_dark_skin():
    light_base = np.array([206, 155, 130], dtype=np.uint8)
    dark_base = np.array([105, 73, 58], dtype=np.uint8)
    light_lesion = np.clip(light_base + np.array([35, -55, -35]), 0, 255).astype(np.uint8)
    dark_lesion = np.clip(dark_base + np.array([35, -35, -20]), 0, 255).astype(np.uint8)

    light_features, _ = extract_biomarkers(_png_bytes(_skin_image(light_lesion, base=light_base)))
    dark_features, _ = extract_biomarkers(_png_bytes(_skin_image(dark_lesion, base=dark_base)))

    assert light_features.erythema_score > 0.2
    assert dark_features.erythema_score > 0.2
    assert abs(light_features.erythema_score - dark_features.erythema_score) < 0.45
