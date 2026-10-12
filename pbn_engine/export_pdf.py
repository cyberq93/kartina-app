"""
Печатный PDF: лист разметки в реальном размере холста + лист палитры с
рецептами смесей (это НЕ то, что уходит на сайт — только для производства).

Два листа:
1. Разметка — контуры + номера, растянутые ровно под физический размер
   холста в мм (чтобы при печати на плоттере получить правильный масштаб).
2. Палитра — для каждого номера: образец цвета (результат смеси),
   название краски(ок) и пропорция. Это и есть инструкция для маляра/печати.

Ограничения оборудования (из брифа): HP DesignJet T520 — макс. ширина
печати 61 см; Epson L18050 — макс. формат А3+. Если холст шире — печатаем
предупреждение прямо на листе, чтобы не напечатать "как есть" то, что
реально имеет смысл только через подрядчика.
"""

import io
import os

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas

from pbn_engine.pipeline import PBNResult

MAX_OWN_PRODUCTION_WIDTH_CM = 61.0  # HP DesignJet T520

# ВАЖНО: встроенные базовые шрифты reportlab (Helvetica и т.п.) НЕ
# поддерживают кириллицу — русский текст печатался бы нечитаемыми
# квадратами. Регистрируем системный TTF-шрифт с кириллицей один раз при
# импорте модуля и используем его имя вместо "Helvetica" везде ниже.
_FONT_REGULAR = "DejaVuSans"
_FONT_BOLD = "DejaVuSans-Bold"
_DEJAVU_PATHS = {
    _FONT_REGULAR: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    _FONT_BOLD: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}
for _font_name, _path in _DEJAVU_PATHS.items():
    if _font_name not in pdfmetrics.getRegisteredFontNames():
        if not os.path.exists(_path):
            raise RuntimeError(
                f"Не найден шрифт с поддержкой кириллицы по пути {_path}. "
                f"Без него весь русский текст в PDF будет нечитаем. "
                f"Укажите путь к доступному TTF-шрифту (например Arial/DejaVu) вручную."
            )
        pdfmetrics.registerFont(TTFont(_font_name, _path))


def _np_image_to_reader(image_rgb: np.ndarray) -> ImageReader:
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def export_outline_sheet(
    result: PBNResult,
    canvas_width_cm: float,
    canvas_height_cm: float,
    output_path: str,
    margin_mm: float = 10.0,
) -> None:
    """Лист разметки в физическом размере холста."""
    margin = margin_mm * mm  # перевод мм в points — единицы координат reportlab
    content_w = canvas_width_cm * 10 * mm
    content_h = canvas_height_cm * 10 * mm
    page_w = content_w + 2 * margin
    page_h = content_h + 2 * margin

    c = pdf_canvas.Canvas(output_path, pagesize=(page_w, page_h))

    if canvas_width_cm > MAX_OWN_PRODUCTION_WIDTH_CM:
        c.setFont(_FONT_BOLD, 10)
        c.setFillColorRGB(0.8, 0, 0)
        c.drawString(
            margin,
            page_h - margin,
            f"ВНИМАНИЕ: ширина {canvas_width_cm:.0f} см больше {MAX_OWN_PRODUCTION_WIDTH_CM:.0f} см "
            f"(макс. своя печать) — только через подрядчика",
        )
        c.setFillColorRGB(0, 0, 0)

    img_reader = _np_image_to_reader(result.outline_image)
    c.drawImage(
        img_reader,
        margin,
        margin,
        width=content_w,
        height=content_h,
        preserveAspectRatio=False,  # предполагается, что фото уже обрезано под пропорции холста
    )
    c.save()


def export_palette_sheet(result: PBNResult, output_path: str) -> None:
    """Лист палитры: номер, образец цвета, название краски(ок) и пропорция."""
    c = pdf_canvas.Canvas(output_path, pagesize=A4)
    page_w, page_h = A4

    margin = 15 * mm
    row_h = 16 * mm
    swatch_size = 10 * mm
    y = page_h - margin

    c.setFont(_FONT_BOLD, 14)
    c.drawString(margin, y, "Палитра — рецепты смесей")
    y -= 10 * mm

    c.setFont(_FONT_REGULAR, 9)
    for entry in sorted(result.legend, key=lambda e: e.number):
        if y < margin + row_h:
            c.showPage()
            y = page_h - margin
            c.setFont(_FONT_REGULAR, 9)

        # Номер
        c.setFont(_FONT_BOLD, 11)
        c.drawString(margin, y - swatch_size / 2 - 3, f"№{entry.number}")

        # Образец цвета (результат смеси)
        r, g, b = entry.recipe.result_rgb
        c.setFillColorRGB(r / 255, g / 255, b / 255)
        c.rect(margin + 15 * mm, y - swatch_size, swatch_size, swatch_size, fill=1, stroke=1)
        c.setFillColorRGB(0, 0, 0)

        # Рецепт текстом
        c.setFont(_FONT_REGULAR, 9)
        c.drawString(margin + 30 * mm, y - 5 * mm, entry.recipe.recipe_text())
        c.setFont(_FONT_REGULAR, 7)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(margin + 30 * mm, y - 10 * mm, f"ΔE={entry.recipe.delta_e:.1f} (0=идеально, <2 не отличить глазом)")
        c.setFillColorRGB(0, 0, 0)

        y -= row_h

    c.save()


def export_pbn_pdf(
    result: PBNResult,
    canvas_width_cm: float,
    canvas_height_cm: float,
    outline_path: str,
    palette_path: str,
) -> None:
    """Удобный враппер: сделать оба листа за один вызов."""
    export_outline_sheet(result, canvas_width_cm, canvas_height_cm, outline_path)
    export_palette_sheet(result, palette_path)
