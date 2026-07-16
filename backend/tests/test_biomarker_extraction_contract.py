from io import BytesIO

from PIL import Image

from app.schemas.biomarker import QualityReport
from app.schemas.predict import PatientFeatures
from app.services.biomarker_extraction import extract_biomarkers
from app.services.reference_cases import BIOMARKER_FIELDS


def _synthetic_png_bytes():
    image = Image.new("RGB", (512, 512), (186, 132, 108))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_biomarkers_returns_patient_features_and_quality_report_contract():
    features, quality = extract_biomarkers(_synthetic_png_bytes())

    assert isinstance(features, PatientFeatures)
    assert isinstance(quality, QualityReport)
    assert list(PatientFeatures.model_fields) == list(BIOMARKER_FIELDS)
    assert list(features.model_dump()) == list(BIOMARKER_FIELDS)


def test_extract_biomarkers_clips_extracted_values_to_canonical_ranges():
    features, _ = extract_biomarkers(_synthetic_png_bytes())

    assert 0.0 <= features.erythema_score <= 1.0
    assert 0.0 <= features.texture_score <= 1.0
    assert 0.0 <= features.dryness_scaling_score <= 1.0
    assert 0.0 <= features.inflammation_score <= 1.0
    assert 0.0 <= features.lesion_coverage_pct <= 100.0
    assert 0.0 <= features.affected_body_area_pct <= 100.0


def test_extract_biomarkers_populates_relative_erythema_and_coverage_fields():
    import numpy as np

    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :] = (186, 132, 108)
    arr[200:300, 200:300] = (215, 75, 75)
    image = Image.fromarray(arr)

    features, quality = extract_biomarkers(_image_bytes_from_pil(image))

    assert features.erythema_score > 0.5
    assert features.lesion_coverage_pct > 0.0
    assert features.affected_body_area_pct == features.lesion_coverage_pct
    assert "visible_roi_area_only" in quality.warnings


def _image_bytes_from_pil(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
