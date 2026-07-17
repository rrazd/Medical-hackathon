from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.services.preprocessing import (
    MAX_UPLOAD_BYTES,
    MIN_LONG_EDGE,
    MIN_SHORT_EDGE,
    InvalidImageError,
    preprocess_image,
)


def _image_bytes(image, fmt="PNG", orientation=None):
    buffer = BytesIO()
    if orientation is not None:
        exif_data = image.getexif()
        exif_data[274] = orientation
        image.save(buffer, format=fmt, exif=exif_data.tobytes())
    else:
        image.save(buffer, format=fmt)
    return buffer.getvalue()


def _solid(size=(512, 512), color=(186, 132, 108)):
    return Image.new("RGB", size, color)


def _gradient(size=(1200, 600)):
    width, height = size
    x = np.linspace(70, 210, width, dtype=np.uint8)
    arr = np.tile(x, (height, 1))
    rgb = np.stack([arr, np.flip(arr, axis=1), np.full_like(arr, 128)], axis=2)
    return Image.fromarray(rgb)


def _checker(size=(512, 512), block=8):
    y, x = np.indices(size)
    pattern = ((x // block + y // block) % 2 * 160 + 50).astype(np.uint8)
    rgb = np.stack([pattern, np.full_like(pattern, 128), np.full_like(pattern, 110)], axis=2)
    return Image.fromarray(rgb)


def test_decodes_jpeg_and_png_to_deterministic_rgb_float_array():
    for fmt, content_type in [("JPEG", "image/jpeg"), ("PNG", "image/png")]:
        result = preprocess_image(_image_bytes(_gradient(), fmt=fmt), content_type=content_type)

        assert result.normalized_rgb.dtype == np.float32
        assert result.normalized_rgb.min() >= 0.0
        assert result.normalized_rgb.max() <= 1.0
        assert result.normalized_rgb.shape[2] == 3
        assert max(result.processed_size) <= 1024
        assert result.processed_size == (1024, 512)
        assert result.quality.exposure_mean == pytest.approx(0.54, abs=0.05)


def test_rejects_empty_invalid_oversize_unsupported_and_bad_format_images():
    cases = [
        (b"", None, "empty"),
        (b"not an image", None, "invalid_bytes"),
        (b"0" * (MAX_UPLOAD_BYTES + 1), None, "too_large"),
        (_image_bytes(_solid(), fmt="PNG"), "text/plain", "unsupported_content_type"),
        (_image_bytes(_solid(), fmt="GIF"), "image/png", "unsupported_format"),
    ]

    for payload, content_type, code in cases:
        with pytest.raises(InvalidImageError) as error:
            preprocess_image(payload, content_type=content_type)
        assert error.value.code == code


def test_upscales_small_images_to_minimum_analyzable_size():
    result = preprocess_image(
        _image_bytes(_gradient((120, 90)), fmt="PNG"), content_type="image/png"
    )

    # Original size is preserved for reporting; processed size is enlarged so the
    # long edge >= 512 and the short edge >= 256, with the low-res warning raised.
    assert result.original_size == (120, 90)
    assert max(result.processed_size) >= MIN_LONG_EDGE
    assert min(result.processed_size) >= MIN_SHORT_EDGE
    assert "low_resolution" in result.quality.warnings


def test_applies_exif_orientation_before_resize_and_preserves_aspect_ratio():
    image = _gradient((600, 300))

    result = preprocess_image(_image_bytes(image, fmt="JPEG", orientation=6), content_type="image/jpeg")

    assert result.original_size == (600, 300)
    assert result.processed_size == (300, 600)
    assert result.normalized_rgb.shape[:2] == (600, 300)


def test_quality_warnings_for_blur_exposure_low_resolution_and_color_cast():
    blurred = _checker().filter(ImageFilter.GaussianBlur(radius=8))
    underexposed = _solid(color=(5, 4, 4))
    overexposed = _solid(color=(252, 252, 250))
    low_resolution = _solid(size=(320, 512))
    color_cast = _solid(color=(250, 70, 70))

    assert "blur_low" in preprocess_image(_image_bytes(blurred)).quality.warnings
    assert "underexposed" in preprocess_image(_image_bytes(underexposed)).quality.warnings
    assert "overexposed" in preprocess_image(_image_bytes(overexposed)).quality.warnings
    assert "low_resolution" in preprocess_image(_image_bytes(low_resolution)).quality.warnings
    cast_report = preprocess_image(_image_bytes(color_cast)).quality
    assert "color_cast" in cast_report.warnings
    assert cast_report.color_cast_score > 0.55


def test_quality_report_populates_numeric_metrics_for_textured_image():
    report = preprocess_image(_image_bytes(_checker())).quality

    assert report.blur_laplacian_var > 80.0
    assert 0.0 <= report.exposure_mean <= 1.0
    assert 0.0 <= report.underexposed_pct <= 100.0
    assert 0.0 <= report.overexposed_pct <= 100.0
    assert report.color_cast_score >= 0.0
    assert 0.0 <= report.skin_pixel_fraction <= 1.0
    assert 0.0 <= report.segmentation_confidence <= 1.0


def test_framing_low_skin_fraction_warning_for_tiny_skin_roi():
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    arr[:, :] = (25, 90, 170)
    arr[240:272, 240:272] = (186, 132, 108)
    image = Image.fromarray(arr)

    report = preprocess_image(_image_bytes(image)).quality

    assert report.skin_pixel_fraction < 0.15
    assert "low_skin_fraction" in report.warnings
    assert "low_segmentation_confidence" in report.warnings
