"""Pure-Pillow image preprocessing for OCR.

No system dependencies and no numpy/opencv — Pillow only, so the pipeline stays
portable across machines. Reduces OCR noise (grayscale + Otsu binarization) and
prevents the render from exceeding the OCR engine's size limit (downscale cap).
"""

import logging

from PIL import Image

logger = logging.getLogger(__name__)


def flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite any transparency onto a white background, returning RGB."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def otsu_threshold(histogram: list[int]) -> int:
    """Return the Otsu threshold (0-255) for a 256-bin grayscale histogram."""
    total = sum(histogram)
    if total == 0:
        return 127
    sum_all = sum(i * histogram[i] for i in range(256))
    weight_bg = 0
    sum_bg = 0.0
    best_threshold = 127
    best_variance = -1.0
    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_variance:
            best_variance = between
            best_threshold = t
    return best_threshold


def binarize(image: Image.Image) -> Image.Image:
    """Grayscale then Otsu-threshold to black/white, kept in 'L' mode.

    'L' (not '1') avoids img2pdf encoding surprises while still being bilevel.
    """
    gray = image.convert("L")
    threshold = otsu_threshold(gray.histogram())
    return gray.point(lambda p: 255 if p > threshold else 0, mode="L")


def downscale_to_cap(image: Image.Image, max_dimension: int) -> Image.Image:
    """Downscale (never upscale) so the longest side is at most max_dimension."""
    longest = max(image.size)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    logger.debug("Downscaling image %s -> %s (cap=%d)", image.size, new_size, max_dimension)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def preprocess_image(
    image: Image.Image, max_dimension: int, binarize_image: bool = True
) -> Image.Image:
    """Flatten alpha, cap size, and optionally binarize an image for OCR."""
    result = flatten_alpha(image)
    result = downscale_to_cap(result, max_dimension)
    if binarize_image:
        result = binarize(result)
    else:
        result = result.convert("L")
    return result
