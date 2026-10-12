import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pbn_engine.pipeline import generate_pbn
from pbn_engine.palette import PLACEHOLDER_PALETTE
from pbn_engine.export_pdf import export_pbn_pdf, MAX_OWN_PRODUCTION_WIDTH_CM
from pypdf import PdfReader

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def _sample_result():
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    img[:75, :] = (220, 170, 140)
    img[75:, :] = (80, 60, 50)
    return generate_pbn(img, PLACEHOLDER_PALETTE, n_colors=6, min_region_fraction=0.01)


def test_pdf_files_are_created_and_valid():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = _sample_result()
    outline_path = os.path.join(OUTPUT_DIR, "test_outline.pdf")
    palette_path = os.path.join(OUTPUT_DIR, "test_palette.pdf")

    export_pbn_pdf(result, canvas_width_cm=40, canvas_height_cm=60, outline_path=outline_path, palette_path=palette_path)

    assert os.path.exists(outline_path)
    assert os.path.exists(palette_path)

    outline_pdf = PdfReader(outline_path)
    assert len(outline_pdf.pages) == 1

    palette_pdf = PdfReader(palette_path)
    assert len(palette_pdf.pages) >= 1


def test_outline_page_size_matches_canvas_dimensions():
    from reportlab.lib.units import mm

    result = _sample_result()
    outline_path = os.path.join(OUTPUT_DIR, "test_outline_size.pdf")
    from pbn_engine.export_pdf import export_outline_sheet

    export_outline_sheet(result, canvas_width_cm=40, canvas_height_cm=60, output_path=outline_path, margin_mm=10)

    pdf = PdfReader(outline_path)
    page = pdf.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)

    expected_w_mm = 40 * 10 + 20
    expected_h_mm = 60 * 10 + 20
    # reportlab mm constant -> points; mediabox уже в points
    width_mm = width_pt / mm
    height_mm = height_pt / mm

    assert abs(width_mm - expected_w_mm) < 1, f"ширина листа {width_mm:.1f}мм, ожидали {expected_w_mm}мм"
    assert abs(height_mm - expected_h_mm) < 1, f"высота листа {height_mm:.1f}мм, ожидали {expected_h_mm}мм"


def test_warning_shown_for_oversized_canvas():
    """Холст шире 61см (макс. своя печать) должен давать предупреждение на
    листе — проверяем это косвенно через текстовое содержимое PDF."""
    result = _sample_result()
    outline_path = os.path.join(OUTPUT_DIR, "test_outline_oversized.pdf")
    from pbn_engine.export_pdf import export_outline_sheet

    export_outline_sheet(result, canvas_width_cm=150, canvas_height_cm=150, output_path=outline_path)

    pdf = PdfReader(outline_path)
    text = pdf.pages[0].extract_text()
    assert "подрядчика" in text, f"не нашли предупреждение про подрядчика в тексте PDF: {text!r}"
    assert str(int(MAX_OWN_PRODUCTION_WIDTH_CM)) in text


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
