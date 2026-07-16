import numpy as np

from app.services.biomarker_masks import (
    MaskResult,
    build_lesion_mask,
    build_masks,
    build_skin_mask,
    estimate_normal_skin_mask,
)


SKIN = (186, 132, 108)
BACKGROUND = (25, 90, 170)
LESION = (215, 75, 75)


def _skin_roi_image(size=256, lesion=True):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = BACKGROUND
    arr[48:208, 48:208] = SKIN
    if lesion:
        arr[104:152, 112:160] = LESION
    return arr


def test_build_skin_mask_covers_skin_roi_and_excludes_background():
    image = _skin_roi_image(lesion=False)

    skin_mask, fraction, ita = build_skin_mask(image)

    assert skin_mask.dtype == bool
    assert skin_mask.shape == image.shape[:2]
    assert fraction == np.mean(skin_mask)
    assert skin_mask[64:192, 64:192].mean() > 0.95
    assert skin_mask[:32, :32].mean() == 0.0
    assert ita is not None


def test_build_lesion_mask_is_inside_skin_mask_for_colored_patch():
    image = _skin_roi_image(lesion=True)
    skin_mask, _, _ = build_skin_mask(image)
    normal_mask = estimate_normal_skin_mask(image, skin_mask)

    lesion_mask, saliency, confidence = build_lesion_mask(image, skin_mask, normal_mask)

    assert lesion_mask.dtype == bool
    assert lesion_mask.shape == image.shape[:2]
    assert saliency.shape == image.shape[:2]
    assert lesion_mask.sum() > 1000
    assert not np.any(lesion_mask & ~skin_mask)
    assert lesion_mask[110:146, 118:154].mean() > 0.75
    assert confidence > 0.5


def test_normal_skin_baseline_non_empty_and_falls_back_for_small_lesion_candidate():
    image = _skin_roi_image(lesion=True)
    skin_mask, _, _ = build_skin_mask(image)
    tiny_candidate = np.zeros(skin_mask.shape, dtype=bool)
    tiny_candidate[128, 128] = True

    normal_mask = estimate_normal_skin_mask(image, skin_mask, tiny_candidate)

    assert normal_mask.dtype == bool
    assert normal_mask.shape == skin_mask.shape
    assert normal_mask.sum() > 0
    assert np.all(normal_mask <= skin_mask)


def test_build_masks_orchestration_reports_low_confidence_for_no_skin_image():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND

    result = build_masks(image)

    assert isinstance(result, MaskResult)
    assert result.skin_pixel_fraction < 0.05
    assert result.segmentation_confidence <= 0.2
    assert result.skin_mask.sum() == 0
    assert result.lesion_mask.sum() == 0
    assert result.normal_skin_mask.sum() == 0
