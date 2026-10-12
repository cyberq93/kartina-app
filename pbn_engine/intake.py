"""
Безопасное чтение загруженного фото.

Что делает:
- проверяет реальный формат по содержимому (не по имени и не по
  заявленному типу): принимаются JPEG, PNG, WEBP и, если установлена
  библиотека pillow-heif, HEIC/HEIF (формат фото с iPhone);
- применяет поворот по EXIF — без этого фото с телефона открывается
  «на боку»;
- отсекает «бомбы декомпрессии» — маленькие файлы, которые разворачиваются
  в гигантское изображение и забивают память (лимит MAX_PIXELS);
- переводит в RGB (убирает прозрачность и палитровые режимы).

HEIC: библиотека pillow-heif ставится отдельно (pip install pillow-heif).
Если её нет, HEIC-файл получает понятную ошибку UnsupportedFormat, а не
падение. На боевой машине её нужно установить и проверить на настоящем
фото с iPhone: в среде разработки этого модуля декодирование HEIC
проверить не удалось (библиотека там недоступна).
"""

import io

from PIL import Image, ImageOps

try:  # необязательная зависимость
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except Exception:
    HEIC_SUPPORTED = False

MAX_PIXELS = 50_000_000  # 50 МП: с запасом для любых телефонов и камер
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"} | ({"HEIF", "HEIC"} if HEIC_SUPPORTED else set())

# Сигнатуры HEIC/HEIF (ftyp-контейнер), чтобы отличить «не поддерживаем
# HEIC» от «это вообще не картинка».
_HEIC_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1")


class IntakeError(Exception):
    """Ошибка чтения фото. status — рекомендуемый HTTP-код."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _looks_like_heic(data: bytes) -> bool:
    return len(data) > 12 and data[4:8] == b"ftyp" and data[8:12] in _HEIC_BRANDS


def read_photo(data: bytes) -> Image.Image:
    """Прочитать фото из байтов. Возвращает RGB-изображение с учётом EXIF."""
    if not data:
        raise IntakeError("пустой файл")

    if _looks_like_heic(data) and not HEIC_SUPPORTED:
        raise IntakeError(
            "формат HEIC пока не поддерживается, пришлите JPEG или PNG", status=415
        )

    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        width, height = probe.size  # размер читается из заголовка, без разворачивания
    except Exception:
        raise IntakeError("не удалось прочитать изображение")

    if fmt not in ALLOWED_FORMATS:
        raise IntakeError("неподдерживаемый формат изображения", status=415)
    if width * height > MAX_PIXELS:
        raise IntakeError("изображение слишком большое по размеру в пикселях", status=413)

    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # реально декодируем здесь, чтобы поймать битые файлы
        img = ImageOps.exif_transpose(img)  # поворот по метаданным камеры
        return img.convert("RGB")
    except Exception:
        raise IntakeError("файл повреждён или не является изображением")
