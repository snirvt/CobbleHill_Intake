from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from classifier.extractors.easyocr_extractor import EasyOcrExtractor
from classifier.models import ExtractedText


def _write_png(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    """Write a real PNG so Pillow can decode it during extraction."""
    Image.new("RGB", size, "white").save(path)


async def test_easyocr_extractor_returns_joined_text(tmp_path: Path) -> None:
    img = tmp_path / "note.png"
    _write_png(img)

    reader = MagicMock()
    reader.readtext.return_value = ["line one", "line two"]

    extractor = EasyOcrExtractor(reader)
    result = await extractor.extract(img)

    assert isinstance(result, ExtractedText)
    assert result.text == "line one\nline two"
    assert result.num_pages == 1
    assert result.file_path == img


async def test_easyocr_extractor_passes_decoded_array(tmp_path: Path) -> None:
    img = tmp_path / "note.png"
    _write_png(img)

    reader = MagicMock()
    reader.readtext.return_value = ["text"]

    extractor = EasyOcrExtractor(reader, paragraph=True)
    await extractor.extract(img)

    # easyocr receives a decoded RGB numpy array, not the raw path.
    call = reader.readtext.call_args
    assert isinstance(call.args[0], np.ndarray)
    assert call.args[0].shape[2] == 3  # RGB channels
    assert call.kwargs == {"detail": 0, "paragraph": True}


async def test_easyocr_extractor_coerces_non_string_items(tmp_path: Path) -> None:
    img = tmp_path / "note.png"
    _write_png(img)

    reader = MagicMock()
    reader.readtext.return_value = ["a", 42]

    extractor = EasyOcrExtractor(reader)
    result = await extractor.extract(img)

    assert result.text == "a\n42"


async def test_easyocr_extractor_raises_on_unreadable_image(tmp_path: Path) -> None:
    img = tmp_path / "bad.png"
    img.write_bytes(b"not an image")

    reader = MagicMock()

    extractor = EasyOcrExtractor(reader)
    with pytest.raises(RuntimeError, match="easyocr failed"):
        await extractor.extract(img)

    reader.readtext.assert_not_called()
