from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskResult:
    skin_mask: np.ndarray
    lesion_mask: np.ndarray
    normal_skin_mask: np.ndarray
    saliency: np.ndarray
    skin_pixel_fraction: float
    segmentation_confidence: float
    estimated_skin_tone_ita: Optional[float]


def build_skin_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float, Optional[float]]:
    rgb_uint8 = _as_uint8_rgb(rgb)
    h, w = rgb_uint8.shape[:2]
    ycrcb = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    ycrcb_skin = (cr >= 125) & (cr <= 188) & (cb >= 70) & (cb <= 150) & (y >= 35)
    hsv_skin = (hue <= 38) & (sat >= 12) & (sat <= 210) & (val >= 45) & (val <= 250)
    not_extreme = (gray >= 35) & (gray <= 245) & ~((val > 240) & (sat < 18))
    not_green_blue = ~(((hue >= 45) & (hue <= 135) & (sat > 60)) | ((hue >= 90) & (hue <= 140) & (sat > 45)))
    mask = (ycrcb_skin | hsv_skin) & not_extreme & not_green_blue

    mask = _clean_mask(mask, min_component_pixels=max(16, int(h * w * 0.005)), kernel_size=5)
    fraction = float(mask.mean())
    ita = _estimate_ita(rgb_uint8, mask)
    return mask, fraction, ita


def estimate_normal_skin_mask(rgb: np.ndarray, skin_mask: np.ndarray, lesion_candidate_mask: Optional[np.ndarray] = None) -> np.ndarray:
    skin_mask = skin_mask.astype(bool)
    if not np.any(skin_mask):
        return np.zeros_like(skin_mask, dtype=bool)

    rgb_uint8 = _as_uint8_rgb(rgb)
    lab = _lab_float(rgb_uint8)
    skin_lab = lab[skin_mask]
    median_lab = np.median(skin_lab, axis=0)
    delta_e = np.linalg.norm(lab - median_lab.reshape(1, 1, 3), axis=2)
    skin_delta = delta_e[skin_mask]
    cutoff = float(np.percentile(skin_delta, 40.0)) if skin_delta.size else 0.0
    low_saliency = skin_mask & (delta_e <= cutoff)

    if lesion_candidate_mask is not None and np.any(lesion_candidate_mask & skin_mask):
        lesion_u8 = (lesion_candidate_mask & skin_mask).astype(np.uint8)
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
        inner = cv2.dilate(lesion_u8, k_inner, iterations=1) > 0
        outer = cv2.dilate(lesion_u8, k_outer, iterations=1) > 0
        ring = outer & ~inner & skin_mask
        ring_low_saliency = ring & low_saliency
        if int(ring_low_saliency.sum()) >= max(25, int(0.02 * skin_mask.sum())):
            return ring_low_saliency.astype(bool)
        if int(ring.sum()) >= max(25, int(0.02 * skin_mask.sum())):
            return ring.astype(bool)

    if int(low_saliency.sum()) < max(25, int(0.05 * skin_mask.sum())):
        return skin_mask.copy()
    return low_saliency.astype(bool)


def build_lesion_mask(rgb: np.ndarray, skin_mask: np.ndarray, normal_skin_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    skin_mask = skin_mask.astype(bool)
    normal_skin_mask = normal_skin_mask.astype(bool) & skin_mask
    saliency = np.zeros(skin_mask.shape, dtype=np.float32)
    if not np.any(skin_mask):
        return np.zeros_like(skin_mask, dtype=bool), saliency, 0.0

    rgb_uint8 = _as_uint8_rgb(rgb)
    lab = _lab_float(rgb_uint8)
    if not np.any(normal_skin_mask):
        normal_skin_mask = estimate_normal_skin_mask(rgb_uint8, skin_mask)
    baseline_lab = np.median(lab[normal_skin_mask], axis=0) if np.any(normal_skin_mask) else np.median(lab[skin_mask], axis=0)

    delta_e = np.linalg.norm(lab - baseline_lab.reshape(1, 1, 3), axis=2)
    relative_a = np.maximum(lab[:, :, 1] - baseline_lab[1], 0.0)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=3.0)
    mean_sq = cv2.GaussianBlur(gray_f * gray_f, (0, 0), sigmaX=3.0)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    edge = cv2.Laplacian(gray, cv2.CV_32F)
    edge_energy = np.abs(edge)

    saliency = (
        0.45 * _robust_norm(delta_e, skin_mask)
        + 0.25 * _robust_norm(relative_a, skin_mask)
        + 0.20 * _robust_norm(variance, skin_mask)
        + 0.10 * _robust_norm(edge_energy, skin_mask)
    ).astype(np.float32)
    saliency[~skin_mask] = 0.0

    skin_values = saliency[skin_mask]
    if skin_values.size == 0:
        return np.zeros_like(skin_mask, dtype=bool), saliency, 0.0
    if float(np.max(skin_values) - np.min(skin_values)) < 1e-6:
        return np.zeros_like(skin_mask, dtype=bool), saliency, 0.0

    otsu_input = np.clip(skin_values * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(otsu_input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_norm = float(np.clip(threshold / 255.0, 0.35, 0.70))
    lesion = skin_mask & (saliency >= threshold_norm)

    lesion = _clean_mask(lesion, min_component_pixels=max(9, int(0.0025 * skin_mask.sum())), kernel_size=3)
    lesion &= skin_mask

    if lesion.sum() > 0.85 * skin_mask.sum():
        high_cutoff = np.percentile(skin_values, 90.0)
        lesion = skin_mask & (saliency >= high_cutoff)
        lesion = _clean_mask(lesion, min_component_pixels=max(9, int(0.0025 * skin_mask.sum())), kernel_size=3) & skin_mask

    lesion_values = saliency[lesion] if np.any(lesion) else np.asarray([], dtype=np.float32)
    background_values = saliency[skin_mask & ~lesion] if np.any(skin_mask & ~lesion) else np.asarray([], dtype=np.float32)
    if lesion_values.size and background_values.size:
        confidence = float(np.clip((float(np.median(lesion_values)) - float(np.median(background_values))) / 0.35, 0.0, 1.0))
    elif lesion_values.size:
        confidence = 0.4
    else:
        confidence = 0.0
    return lesion.astype(bool), saliency, confidence


def build_masks(rgb: np.ndarray) -> MaskResult:
    skin_mask, skin_fraction, ita = build_skin_mask(rgb)
    initial_normal = estimate_normal_skin_mask(rgb, skin_mask)
    preliminary_lesion, preliminary_saliency, _ = build_lesion_mask(rgb, skin_mask, initial_normal)
    normal_skin = estimate_normal_skin_mask(rgb, skin_mask, preliminary_lesion)
    lesion_mask, saliency, lesion_confidence = build_lesion_mask(rgb, skin_mask, normal_skin)

    confidence = float(np.clip(0.65 * lesion_confidence + 0.35 * min(skin_fraction / 0.15, 1.0), 0.0, 1.0))
    if not np.any(lesion_mask):
        confidence = min(confidence, 0.45)
    if skin_fraction < 0.05:
        confidence = min(confidence, 0.2)

    return MaskResult(
        skin_mask=skin_mask.astype(bool),
        lesion_mask=lesion_mask.astype(bool),
        normal_skin_mask=normal_skin.astype(bool),
        saliency=saliency.astype(np.float32),
        skin_pixel_fraction=float(skin_fraction),
        segmentation_confidence=confidence,
        estimated_skin_tone_ita=ita,
    )


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("rgb must be an HxWx3 array")
    if arr.dtype == np.uint8:
        return arr
    return (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _lab_float(rgb_uint8: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def _estimate_ita(rgb_uint8: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if not np.any(mask):
        return None
    lab = _lab_float(rgb_uint8)
    l_values = lab[:, :, 0][mask]
    b_values = lab[:, :, 2][mask]
    ita_values = np.degrees(np.arctan2(l_values - 50.0, np.maximum(b_values, 1e-6)))
    return float(np.median(ita_values))


def _robust_norm(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float32)
    selected = values[mask]
    if selected.size == 0:
        return result
    low, high = np.percentile(selected, [10.0, 95.0])
    if high - low < 1e-6:
        high = float(np.max(selected))
        low = float(np.min(selected))
    if high - low < 1e-6:
        return result
    result = np.clip((values.astype(np.float32) - float(low)) / float(high - low), 0.0, 1.0)
    result[~mask] = 0.0
    return result.astype(np.float32)


def _clean_mask(mask: np.ndarray, min_component_pixels: int, kernel_size: int) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    u8 = (mask.astype(np.uint8) * 255)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)

    flood = u8.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    u8 = cv2.bitwise_or(u8, holes)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((u8 > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_pixels:
            cleaned |= labels == label
    return cleaned
