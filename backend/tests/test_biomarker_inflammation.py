from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.schemas.predict import PatientFeatures
from app.services.biomarker_extraction import compute_inflammation_score, extract_biomarkers
from app.services.reference_cases import BIOMARKER_FIELDS


SKIN = np.array([186, 132, 108], dtype=np.uint8)
LESION = np.array([215, 75, 75], dtype=np.uint8)


def _png_bytes(arr):
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def test_compute_inflammation_score_uses_transparent_weighted_composite():
    score = compute_inflammation_score(
        erythema_score=0.6,
        lesion_coverage_pct=30.0,
        texture_score=0.4,
        dryness_scaling_score=0.2,
    )

    expected = 0.35 * 0.6 + 0.25 * (30.0 / 60.0) + 0.20 * 0.4 + 0.20 * 0.2
    assert score == pytest.approx(expected)


def test_compute_inflammation_score_clips_components_and_area_score():
    assert compute_inflammation_score(2.0, 120.0, 2.0, 2.0) == pytest.approx(1.0)
    assert compute_inflammation_score(-1.0, -5.0, -0.5, -0.2) == pytest.approx(0.0)


def test_extract_biomarkers_returns_complete_finite_patient_feature_vector():
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :] = SKIN
    arr[176:336, 176:336] = LESION
    for row in range(188, 324, 20):
        arr[row : row + 2, 190:322] = (235, 220, 205)

    features, quality = extract_biomarkers(_png_bytes(arr))
    values = features.model_dump()

    assert isinstance(features, PatientFeatures)
    assert list(values) == list(PatientFeatures.model_fields) == list(BIOMARKER_FIELDS)
    assert all(np.isfinite(value) for value in values.values())
    assert 0.0 <= features.erythema_score <= 1.0
    assert 0.0 <= features.texture_score <= 1.0
    assert 0.0 <= features.dryness_scaling_score <= 1.0
    assert 0.0 <= features.inflammation_score <= 1.0
    assert 0.0 <= features.lesion_coverage_pct <= 100.0
    assert features.affected_body_area_pct == pytest.approx(features.lesion_coverage_pct)
    assert 0.0 <= features.affected_body_area_pct <= 100.0
    assert "visible_roi_area_only" in quality.warnings
