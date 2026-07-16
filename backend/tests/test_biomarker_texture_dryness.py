from io import BytesIO

import numpy as np
from PIL import Image

from app.services.biomarker_extraction import extract_biomarkers


SKIN = np.array([186, 132, 108], dtype=np.uint8)
LESION = np.array([174, 102, 96], dtype=np.uint8)


def _png_bytes(arr):
    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def _base_image(lesion_color=LESION):
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :] = SKIN
    arr[160:352, 160:352] = lesion_color
    return arr


def _textured_lesion_image():
    arr = _base_image()
    rng = np.random.default_rng(7)
    noise = rng.integers(-34, 35, size=(192, 192, 1), dtype=np.int16)
    patch = arr[160:352, 160:352].astype(np.int16) + noise
    checker = ((np.indices((192, 192)).sum(axis=0) % 2) * 28 - 14).astype(np.int16)
    patch[:, :, 0] += checker
    patch[:, :, 1] -= checker // 2
    arr[160:352, 160:352] = np.clip(patch, 0, 255).astype(np.uint8)
    return arr


def _scaly_lesion_image(overexposed=False):
    arr = _base_image()
    for offset in range(174, 340, 18):
        arr[offset : offset + 2, 176:336] = (232, 225, 210)
    for offset in range(184, 330, 36):
        arr[176:336, offset : offset + 1] = (220, 214, 202)
    if overexposed:
        arr[180:230, 180:230] = (255, 255, 255)
    return arr


def test_textured_lesion_has_higher_texture_score_than_smooth_lesion():
    smooth_features, _ = extract_biomarkers(_png_bytes(_base_image()))
    textured_features, _ = extract_biomarkers(_png_bytes(_textured_lesion_image()))

    assert 0.0 <= smooth_features.texture_score <= 1.0
    assert 0.0 <= textured_features.texture_score <= 1.0
    assert textured_features.texture_score > smooth_features.texture_score + 0.15


def test_flaky_scaly_lesion_has_higher_dryness_scaling_score_than_smooth_lesion():
    smooth_features, _ = extract_biomarkers(_png_bytes(_base_image()))
    scaly_features, _ = extract_biomarkers(_png_bytes(_scaly_lesion_image()))

    assert 0.0 <= smooth_features.dryness_scaling_score <= 1.0
    assert 0.0 <= scaly_features.dryness_scaling_score <= 1.0
    assert scaly_features.dryness_scaling_score > smooth_features.dryness_scaling_score + 0.10


def test_dryness_scaling_warns_when_overexposure_can_mimic_scale():
    features, quality = extract_biomarkers(_png_bytes(_scaly_lesion_image(overexposed=True)))

    assert 0.0 <= features.dryness_scaling_score <= 1.0
    assert "overexposed" in quality.warnings or "scale_mimic_overexposure" in quality.warnings
