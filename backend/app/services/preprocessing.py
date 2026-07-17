from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas.biomarker import QualityReport


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MIN_SHORT_EDGE = 256
MIN_LONG_EDGE = 512
LOW_RES_LONG_EDGE = 768
MAX_LONG_EDGE = 1024


class InvalidImageError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreprocessedImage:
    normalized_rgb: np.ndarray
    rgb_uint8: np.ndarray
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    quality: QualityReport


def preprocess_image(image_bytes: bytes, content_type: Optional[str] = None) -> PreprocessedImage:
    image = _decode_image(image_bytes, content_type)
    original_size = image.size

    image = ImageOps.exif_transpose(image).convert("RGB")
    # Small photos are upscaled to the minimum analyzable size rather than
    # rejected, so low-resolution uploads (e.g. cropped screenshots) still work.
    # Genuine low resolution is surfaced as a soft "low_resolution" quality
    # warning by assess_quality (which uses the *original* size).
    image = _upscale_to_minimum(image)
    image = _resize_long_edge(image, MAX_LONG_EDGE)
    rgb = np.asarray(image, dtype=np.uint8)
    quality = assess_quality(rgb, original_size)
    normalized = _normalize_rgb(rgb)
    return PreprocessedImage(
        normalized_rgb=normalized,
        rgb_uint8=rgb,
        original_size=original_size,
        processed_size=image.size,
        quality=quality,
    )


def assess_quality(rgb: np.ndarray, original_size: Tuple[int, int]) -> QualityReport:
    float_rgb = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luminance = gray.astype(np.float32) / 255.0

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure_mean = float(luminance.mean())
    under_pct = float((luminance < 0.05).mean() * 100.0)
    over_pct = float((luminance > 0.95).mean() * 100.0)
    scales = _gray_world_scales(float_rgb)
    color_cast_score = float(max(abs(scale - 1.0) for scale in scales))
    skin_fraction, ita = _estimate_skin_fraction_and_ita(rgb)
    segmentation_confidence = float(np.clip(skin_fraction / 0.15, 0.0, 1.0))

    warnings = []
    if blur_var < 80.0:
        warnings.append("blur_low")
    if exposure_mean < 0.18 or under_pct > 20.0:
        warnings.append("underexposed")
    if exposure_mean > 0.85 or over_pct > 20.0:
        warnings.append("overexposed")
    if max(original_size) < LOW_RES_LONG_EDGE:
        warnings.append("low_resolution")
    if any(scale < 0.65 or scale > 1.55 for scale in scales):
        warnings.append("color_cast")
    if skin_fraction < 0.15:
        warnings.append("low_skin_fraction")
    if segmentation_confidence < 0.5:
        warnings.append("low_segmentation_confidence")

    return QualityReport(
        warnings=warnings,
        blur_laplacian_var=blur_var,
        exposure_mean=exposure_mean,
        underexposed_pct=under_pct,
        overexposed_pct=over_pct,
        color_cast_score=color_cast_score,
        skin_pixel_fraction=float(skin_fraction),
        estimated_skin_tone_ita=ita,
        segmentation_confidence=segmentation_confidence,
    )


def _decode_image(image_bytes: bytes, content_type: Optional[str]) -> Image.Image:
    if not image_bytes:
        raise InvalidImageError("empty", "Image upload is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("too_large", "Image upload exceeds the 8 MB limit.")
    if content_type is not None and content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError("unsupported_content_type", "Upload must be a JPEG or PNG image.")
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("invalid_bytes", "Image bytes could not be decoded.") from exc
    if image.format not in ALLOWED_FORMATS:
        raise InvalidImageError("unsupported_format", "Decoded image must be JPEG or PNG.")
    return image


def _upscale_to_minimum(image: Image.Image) -> Image.Image:
    """Enlarge images below the minimum analyzable size, preserving aspect ratio.

    Scales up so the long edge is at least MIN_LONG_EDGE and the short edge is at
    least MIN_SHORT_EDGE. Images already large enough are returned unchanged.
    """
    width, height = image.size
    long_edge = max(width, height)
    short_edge = min(width, height)
    if long_edge <= 0 or short_edge <= 0:
        return image
    scale_long = MIN_LONG_EDGE / long_edge if long_edge < MIN_LONG_EDGE else 1.0
    scale_short = MIN_SHORT_EDGE / short_edge if short_edge < MIN_SHORT_EDGE else 1.0
    scale = max(scale_long, scale_short)
    if scale <= 1.0:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _resize_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / float(long_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    scales = np.asarray(_gray_world_scales(arr), dtype=np.float32)
    balanced = np.clip(arr * scales.reshape(1, 1, 3), 0.0, 1.0)
    normalized = np.empty_like(balanced)
    for channel in range(3):
        plane = balanced[:, :, channel]
        p1, p99 = np.percentile(plane, [1, 99])
        if p99 - p1 < 1e-6:
            normalized[:, :, channel] = np.clip(plane, 0.0, 1.0)
        else:
            normalized[:, :, channel] = np.clip((plane - p1) / (p99 - p1), 0.0, 1.0)
    return normalized.astype(np.float32)


def _gray_world_scales(arr: np.ndarray) -> Tuple[float, float, float]:
    mask = np.all((arr > 0.03) & (arr < 0.97), axis=2)
    pixels = arr[mask] if np.any(mask) else arr.reshape(-1, 3)
    means = np.maximum(pixels.mean(axis=0), 1e-6)
    target = float(means.mean())
    scales = np.clip(target / means, 0.25, 4.0)
    return float(scales[0]), float(scales[1]), float(scales[2])


def _estimate_skin_fraction_and_ita(rgb: np.ndarray) -> Tuple[float, Optional[float]]:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    ycrcb_skin = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135) & (y > 35)
    hsv_skin = (h <= 35) & (s >= 25) & (s <= 220) & (v >= 50) & (v <= 245)
    mask = (ycrcb_skin | hsv_skin) & ~((v > 245) & (s < 25))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask = mask_uint8 > 0
    fraction = float(mask.mean())
    if not np.any(mask):
        return fraction, None

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_values = lab[:, :, 0][mask] * (100.0 / 255.0)
    b_values = lab[:, :, 2][mask] - 128.0
    ita_values = np.degrees(np.arctan2(l_values - 50.0, np.maximum(b_values, 1e-6)))
    return fraction, float(np.median(ita_values))
