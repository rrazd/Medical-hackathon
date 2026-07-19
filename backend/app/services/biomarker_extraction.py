from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from app.schemas.biomarker import QualityReport
from app.schemas.predict import PatientFeatures
from app.services.biomarker_masks import MaskResult, build_masks
from app.services.preprocessing import preprocess_image

INFLAMMATION_ERYTHEMA_WEIGHT = 0.35
INFLAMMATION_AREA_WEIGHT = 0.25
INFLAMMATION_TEXTURE_WEIGHT = 0.20
INFLAMMATION_DRYNESS_WEIGHT = 0.20
INFLAMMATION_AREA_DENOMINATOR_PCT = 60.0

MIN_LESION_PIXELS = 25
TEXTURE_LBP_POINTS = 8
TEXTURE_LBP_RADIUS = 1
TEXTURE_GLCM_LEVELS = 16


def extract_biomarkers(image_bytes: bytes) -> Tuple[PatientFeatures, QualityReport]:
    preprocessed = preprocess_image(image_bytes)
    masks = build_masks(preprocessed.rgb_uint8)
    erythema = compute_relative_erythema(preprocessed.rgb_uint8, masks)
    coverage = compute_lesion_coverage_pct(masks)
    texture = compute_texture_score(preprocessed.rgb_uint8, masks)
    dryness = compute_dryness_scaling_score(preprocessed.rgb_uint8, masks)
    inflammation = compute_inflammation_score(
        erythema_score=erythema,
        lesion_coverage_pct=coverage,
        texture_score=texture,
        dryness_scaling_score=dryness,
    )

    features = PatientFeatures(
        erythema_score=_clip_unit(erythema),
        lesion_coverage_pct=_clip_pct(coverage),
        texture_score=_clip_unit(texture),
        dryness_scaling_score=_clip_unit(dryness),
        inflammation_score=_clip_unit(inflammation),
        affected_body_area_pct=_clip_pct(coverage),
    )
    quality = _quality_with_mask_metrics(
        preprocessed.quality,
        masks,
        erythema_score=features.erythema_score,
        texture_score=features.texture_score,
        dryness_scaling_score=features.dryness_scaling_score,
    )
    return features, quality


def compute_relative_erythema(rgb: np.ndarray, masks: MaskResult) -> float:
    import cv2

    rgb_uint8 = _as_uint8_rgb(rgb)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_channel = lab[:, :, 1] - 128.0

    normal_mask = masks.normal_skin_mask & masks.skin_mask
    if np.any(normal_mask):
        baseline_a = float(np.median(a_channel[normal_mask]))
    elif np.any(masks.skin_mask):
        baseline_a = float(np.median(a_channel[masks.skin_mask]))
    else:
        return 0.0

    lesion_mask = masks.lesion_mask & masks.skin_mask
    min_lesion_pixels = max(10, int(0.002 * masks.skin_mask.sum())) if np.any(masks.skin_mask) else 10
    if int(lesion_mask.sum()) >= min_lesion_pixels:
        lesion_a = float(np.median(a_channel[lesion_mask]))
    else:
        relative_a = a_channel[masks.skin_mask] - baseline_a
        if relative_a.size == 0:
            return 0.0
        top_count = max(1, int(np.ceil(0.10 * relative_a.size)))
        lesion_a = baseline_a + float(np.median(np.partition(relative_a, -top_count)[-top_count:]))

    delta_a = lesion_a - baseline_a
    return _clip_unit((delta_a - 1.0) / 18.0)


def compute_lesion_coverage_pct(masks: MaskResult) -> float:
    skin_pixels = int(masks.skin_mask.sum())
    if skin_pixels <= 0:
        return 0.0
    lesion_pixels = int((masks.lesion_mask & masks.skin_mask).sum())
    return _clip_pct(100.0 * lesion_pixels / float(skin_pixels))


def compute_texture_score(rgb: np.ndarray, masks: MaskResult) -> float:
    lesion_mask = masks.lesion_mask & masks.skin_mask
    if int(lesion_mask.sum()) < MIN_LESION_PIXELS:
        return 0.0

    rgb_uint8 = _as_uint8_rgb(rgb)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    lesion_values = gray[lesion_mask]
    if lesion_values.size < MIN_LESION_PIXELS or float(np.std(lesion_values)) < 1e-6:
        return 0.0

    lbp = local_binary_pattern(
        gray,
        P=TEXTURE_LBP_POINTS,
        R=TEXTURE_LBP_RADIUS,
        method="uniform",
    )
    bins = TEXTURE_LBP_POINTS + 2
    hist, _ = np.histogram(lbp[lesion_mask], bins=bins, range=(0, bins), density=False)
    probs = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    probs = probs[probs > 0]
    lbp_entropy = float(-(probs * np.log2(probs)).sum() / np.log2(bins)) if probs.size else 0.0

    y1, y2, x1, x2 = _mask_bbox(lesion_mask)
    crop_gray = gray[y1:y2, x1:x2]
    crop_mask = lesion_mask[y1:y2, x1:x2]
    fill_value = int(np.median(lesion_values))
    glcm_input = np.where(crop_mask, crop_gray, fill_value)
    quantized = np.clip((glcm_input.astype(np.float32) / 256.0) * TEXTURE_GLCM_LEVELS, 0, TEXTURE_GLCM_LEVELS - 1).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1],  # Use only distance=1 (was [1,2]) for faster computation
        angles=[0, np.pi / 2],  # Use only 0° and 90° (was 4 angles) for faster computation
        levels=TEXTURE_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    contrast_score = _clip_unit(contrast / 24.0)
    inverse_homogeneity = _clip_unit(1.0 - homogeneity)

    return _clip_unit(0.45 * lbp_entropy + 0.35 * contrast_score + 0.20 * inverse_homogeneity)


def compute_dryness_scaling_score(rgb: np.ndarray, masks: MaskResult) -> float:
    lesion_mask = masks.lesion_mask & masks.skin_mask
    if int(lesion_mask.sum()) < MIN_LESION_PIXELS:
        return 0.0

    rgb_uint8 = _as_uint8_rgb(rgb)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    lesion_gray = gray[lesion_mask]
    if lesion_gray.size < MIN_LESION_PIXELS:
        return 0.0

    median_gray = float(np.median(lesion_gray))
    dark_hairlike = lesion_mask & (gray < max(35.0, median_gray - 45.0))
    saturated_specular = lesion_mask & (value > 248) & (saturation < 18)
    valid_scale_mask = lesion_mask & ~dark_hairlike & ~saturated_specular
    if int(valid_scale_mask.sum()) < MIN_LESION_PIXELS:
        valid_scale_mask = lesion_mask & ~saturated_specular
    if int(valid_scale_mask.sum()) < MIN_LESION_PIXELS:
        return 0.0

    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=1.1)
    high_pass = np.abs(gray.astype(np.float32) - blurred)
    hp_values = high_pass[valid_scale_mask]
    high_pass_score = _clip_unit(float(np.percentile(hp_values, 90.0)) / 22.0) if hp_values.size else 0.0

    edges = cv2.Canny(gray, threshold1=35, threshold2=95)
    edge_density = float(((edges > 0) & valid_scale_mask).sum() / max(int(valid_scale_mask.sum()), 1))
    edge_score = _clip_unit(edge_density / 0.16)

    lesion_luminance_p60 = float(np.percentile(lesion_gray, 60.0))
    bright_flakes = (
        valid_scale_mask
        & (saturation < 70)
        & (gray > min(245.0, max(lesion_luminance_p60 + 18.0, median_gray + 14.0)))
        & ~(value > 252)
    )
    flake_fraction = float(bright_flakes.sum() / max(int(valid_scale_mask.sum()), 1))
    flake_score = _clip_unit(flake_fraction / 0.10)

    return _clip_unit(0.35 * edge_score + 0.35 * high_pass_score + 0.30 * flake_score)


def compute_inflammation_score(
    erythema_score: float,
    lesion_coverage_pct: float,
    texture_score: float,
    dryness_scaling_score: float,
) -> float:
    area_score = _clip_unit(_clip_pct(lesion_coverage_pct) / INFLAMMATION_AREA_DENOMINATOR_PCT)
    return _clip_unit(
        INFLAMMATION_ERYTHEMA_WEIGHT * _clip_unit(erythema_score)
        + INFLAMMATION_AREA_WEIGHT * area_score
        + INFLAMMATION_TEXTURE_WEIGHT * _clip_unit(texture_score)
        + INFLAMMATION_DRYNESS_WEIGHT * _clip_unit(dryness_scaling_score)
    )


def _quality_with_mask_metrics(
    quality: QualityReport,
    masks: MaskResult,
    erythema_score: float,
    texture_score: float,
    dryness_scaling_score: float,
) -> QualityReport:
    warnings = list(quality.warnings)
    if masks.skin_pixel_fraction < 0.15 and "low_skin_fraction" not in warnings:
        warnings.append("low_skin_fraction")
    if masks.segmentation_confidence < 0.5 and "low_segmentation_confidence" not in warnings:
        warnings.append("low_segmentation_confidence")
    caveat = "visible_roi_area_only"
    if caveat not in warnings:
        warnings.append(caveat)
    warnings.extend(_dryness_quality_warnings(quality, masks))
    if (
        masks.estimated_skin_tone_ita is not None
        and masks.estimated_skin_tone_ita < 10.0
        and erythema_score < 0.20
        and (texture_score >= 0.20 or dryness_scaling_score >= 0.20 or compute_lesion_coverage_pct(masks) >= 2.0)
        and "dark_tone_multicue_inflammation" not in warnings
    ):
        warnings.append("dark_tone_multicue_inflammation")

    return quality.model_copy(
        update={
            "warnings": _dedupe(warnings),
            "skin_pixel_fraction": float(masks.skin_pixel_fraction),
            "estimated_skin_tone_ita": masks.estimated_skin_tone_ita,
            "segmentation_confidence": float(masks.segmentation_confidence),
        }
    )


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return 0, 1, 0, 1
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _dryness_quality_warnings(quality: QualityReport, masks: MaskResult) -> List[str]:
    warnings: List[str] = []
    if (
        quality.overexposed_pct > 0.5
        and int((masks.lesion_mask & masks.skin_mask).sum()) >= MIN_LESION_PIXELS
    ):
        warnings.append("scale_mimic_overexposure")
    return warnings


def _dedupe(values: List[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _clip_unit(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _clip_pct(value: float) -> float:
    return float(min(100.0, max(0.0, value)))
