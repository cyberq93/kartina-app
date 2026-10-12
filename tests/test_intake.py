"""Тесты безопасного чтения фото (pbn_engine/intake.py)."""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from pbn_engine import intake
from pbn_engine.intake import IntakeError, read_photo


def _jpeg_bytes(img: Image.Image, exif=None) -> bytes:
    buf = io.BytesIO()
    if exif is not None:
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


def _expect_error(data: bytes, status: int):
    try:
        read_photo(data)
    except IntakeError as e:
        assert e.status == status, f"ожидали {status}, получили {e.status}: {e.message}"
        return
    raise AssertionError("ожидали IntakeError, его не было")


def test_plain_jpeg_is_read_as_rgb():
    img = Image.fromarray(np.full((40, 60, 3), 128, dtype=np.uint8))
    out = read_photo(_jpeg_bytes(img))
    assert out.mode == "RGB" and out.size == (60, 40)


def test_exif_rotation_is_applied():
    """Фото 60x40 с EXIF-ориентацией 6 (поворот на 90°) должно стать 40x60:
    именно так оно выглядит на экране телефона."""
    img = Image.fromarray(np.full((40, 60, 3), 128, dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6
    out = read_photo(_jpeg_bytes(img, exif=exif))
    assert out.size == (40, 60), f"поворот по EXIF не применён: {out.size}"


def test_png_with_transparency_becomes_rgb():
    img = Image.new("RGBA", (30, 30), (255, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert read_photo(buf.getvalue()).mode == "RGB"


def test_empty_and_garbage_are_400():
    _expect_error(b"", 400)
    _expect_error(b"definitely not an image", 400)


def test_truncated_jpeg_is_400_not_crash():
    img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    data = _jpeg_bytes(img)
    _expect_error(data[: len(data) // 3], 400)


def test_unsupported_format_is_415():
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, format="BMP")
    _expect_error(buf.getvalue(), 415)


def test_huge_dimensions_rejected_before_decoding():
    """«Бомба»: крошечный файл, заявляющий огромный размер. Должна
    отклоняться по заголовку, без разворачивания в память."""
    old = intake.MAX_PIXELS
    intake.MAX_PIXELS = 1000
    try:
        img = Image.new("RGB", (100, 100))  # 10 000 пикселей > лимита 1000
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        _expect_error(buf.getvalue(), 413)
    finally:
        intake.MAX_PIXELS = old


def test_heic_without_library_gives_clear_415():
    """Если pillow-heif не установлен, HEIC должен давать понятный отказ,
    а не 500. (Если библиотека есть, этот тест пропускается: настоящее
    декодирование HEIC нужно проверять на реальном файле с iPhone.)"""
    if intake.HEIC_SUPPORTED:
        print("  (pillow-heif установлен, тест пропущен)")
        return
    fake_heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"
    _expect_error(fake_heic, 415)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"OK   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} тестов прошли")
    sys.exit(1 if failed else 0)
