"""
Тест полного пайплайна на синтетическом фото + сохранение результата в
output/ для визуальной проверки глазами (автотесты проверяют структуру,
но финальную "читаемость" картинки надо один раз посмотреть руками).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from pbn_engine.pipeline import generate_pbn
from pbn_engine.palette import PLACEHOLDER_PALETTE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def make_face_like_gradient(size=300):
    """Не настоящее фото (его пока нет), но изображение с плавным градиентом
    телесного тона + резкими цветными зонами — стресс-тест на то, что и
    плавные переходы, и чёткие границы обрабатываются разумно."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    # плавный телесный градиент
    base = np.stack([
        210 + (xx / size) * 30,
        170 + (yy / size) * 20,
        140 + ((xx + yy) / (2 * size)) * 25,
    ], axis=-1)
    img[:] = np.clip(base, 0, 255).astype(np.uint8)
    # глаза (тёмные пятна)
    cv2.circle(img, (size // 3, size // 2), size // 20, (40, 30, 30), -1)
    cv2.circle(img, (2 * size // 3, size // 2), size // 20, (40, 30, 30), -1)
    # волосы (тёмная зона сверху)
    img[: size // 5, :] = (60, 40, 30)
    return img


def test_pipeline_runs_end_to_end_without_errors():
    img = make_face_like_gradient(200)
    result = generate_pbn(img, PLACEHOLDER_PALETTE, n_colors=16, min_region_fraction=0.003)
    assert result.outline_image.shape == img.shape
    assert result.colorized_image.shape == img.shape
    assert len(result.legend) > 0
    assert len(result.legend) <= 16, "номеров не должно быть больше n_colors"


def test_numbers_are_reused_across_regions_with_same_mix():
    """Это и есть смысл "по номерам" — проверяем, что число уникальных
    номеров в легенде меньше или равно числу регионов (иначе группировка
    по рецепту не работает и каждый регион получает свой номер)."""
    img = make_face_like_gradient(200)
    result = generate_pbn(img, PLACEHOLDER_PALETTE, n_colors=16, min_region_fraction=0.003)
    n_regions = len(result.region_to_number)
    n_numbers = len(result.legend)
    assert n_numbers <= n_regions, f"номеров ({n_numbers}) больше регионов ({n_regions}) — группировка не работает"


def test_outline_image_has_visible_lines():
    """Изображение разметки не должно быть пустым белым листом — должны
    быть видны чёрные пиксели контуров."""
    img = make_face_like_gradient(200)
    result = generate_pbn(img, PLACEHOLDER_PALETTE, n_colors=16, min_region_fraction=0.003)
    black_pixels = np.sum(np.all(result.outline_image < 50, axis=-1))
    assert black_pixels > 50, f"слишком мало тёмных пикселей в разметке: {black_pixels}"


def test_boundaries_drawn_between_all_neighbours_including_nested():
    """Регрессия: раньше рисовались только внешние контуры, и границы
    внутри других пятен терялись."""
    from pbn_engine.pipeline import _draw_boundaries
    labels = np.ones((60, 60), dtype=np.int32)
    labels[20:40, 20:40] = 2          # пятно целиком внутри другого
    labels[:, 50:] = 3                # третье пятно у края
    edge = _draw_boundaries(labels, thickness=1)
    assert edge[19, 30] or edge[20, 30], "нет границы вокруг вложенного пятна"
    assert edge[30, 49] or edge[30, 50], "нет границы между соседними пятнами"
    assert not edge[5, 5], "линия там, где границы нет"


def test_number_is_placed_away_from_image_edge():
    """Регрессия: у пятен, касающихся края кадра, номер садился на сам край
    и обрезался."""
    from pbn_engine.pipeline import _find_label_point
    mask = np.ones((100, 100), dtype=bool)   # пятно на весь кадр
    (x, y), radius = _find_label_point(mask)
    assert 30 < x < 70 and 30 < y < 70, f"номер у края: {(x, y)}"
    assert radius > 30


def test_portrait_end_to_end_has_one_number_per_color():
    from skimage import data
    result = generate_pbn(data.astronaut(), PLACEHOLDER_PALETTE, n_colors=24, min_region_fraction=0.002)
    assert len(result.legend) <= 24
    assert len(result.region_to_number) > len(result.legend), (
        "номера должны повторяться у разных пятен одного цвета"
    )


def save_visual_samples():
    """Не тест в строгом смысле — сохраняет файлы, чтобы посмотреть глазами."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = make_face_like_gradient(300)
    result = generate_pbn(img, PLACEHOLDER_PALETTE, n_colors=18, min_region_fraction=0.003)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "sample_source.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sample_outline.png"), cv2.cvtColor(result.outline_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "sample_colorized.png"), cv2.cvtColor(result.colorized_image, cv2.COLOR_RGB2BGR))

    print(f"\nСохранено в {OUTPUT_DIR}: sample_source.png, sample_outline.png, sample_colorized.png")
    print(f"Легенда ({len(result.legend)} номеров):")
    for entry in result.legend:
        print(f"  №{entry.number}: {entry.recipe.recipe_text()} (ΔE={entry.recipe.delta_e:.2f})")


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
    print(f"\n{len(tests) - failed}/{len(tests)} тестов прошли")

    save_visual_samples()
    sys.exit(1 if failed else 0)
