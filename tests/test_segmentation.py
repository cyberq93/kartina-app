"""
Контрольные тесты сегментации (pbn_engine/segmentation.py) на синтетических
изображениях с заранее известным "правильным ответом" — так можно проверить
правильность независимо от того, есть ли уже реальные фото клиента.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pbn_engine.segmentation import segment_image


def make_four_quadrant_image(size=200):
    """Четыре чётких цветных квадранта — простейший случай: ожидаем, что
    сегментация найдёт 4 крупных региона примерно по 25% площади каждый."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    h = size // 2
    img[:h, :h] = (220, 60, 60)     # красный
    img[:h, h:] = (60, 200, 90)     # зелёный
    img[h:, :h] = (50, 90, 220)     # синий
    img[h:, h:] = (230, 210, 60)    # жёлтый
    return img


def make_image_with_tiny_speck(size=200):
    """Большое однородное поле с одним маленьким пятном другого цвета —
    проверка, что min_region_fraction действительно убирает мелкие пятна."""
    img = np.full((size, size, 3), (200, 180, 160), dtype=np.uint8)
    img[10:13, 10:13] = (10, 10, 200)  # пятно 3x3 пикселя = 9/40000 = 0.02%
    return img


def test_four_quadrants_segmented_into_four_regions():
    img = make_four_quadrant_image()
    result = segment_image(img, n_colors=4, min_region_fraction=0.01)
    n_regions = len(result.region_area)
    assert n_regions == 4, f"ожидали 4 региона на 4-квадрантном фото, получили {n_regions}"

    total = img.shape[0] * img.shape[1]
    for area in result.region_area.values():
        fraction = area / total
        assert 0.2 < fraction < 0.3, f"регион занимает {fraction:.2%}, ожидали ~25%"


def test_quadrant_colors_are_correctly_recovered():
    img = make_four_quadrant_image()
    result = segment_image(img, n_colors=4, min_region_fraction=0.01)
    expected_colors = [(220, 60, 60), (60, 200, 90), (50, 90, 220), (230, 210, 60)]
    found_colors = list(result.region_mean_rgb.values())

    for expected in expected_colors:
        closest = min(
            found_colors,
            key=lambda c: sum((a - b) ** 2 for a, b in zip(c, expected)),
        )
        dist = sum((a - b) ** 2 for a, b in zip(closest, expected)) ** 0.5
        assert dist < 15, f"не нашли региона близкого к {expected}, ближайший {closest} (dist={dist:.1f})"


def test_tiny_speck_gets_merged_away():
    img = make_image_with_tiny_speck()
    result = segment_image(img, n_colors=8, min_region_fraction=0.01)
    total = img.shape[0] * img.shape[1]
    min_pixels = total * 0.01
    too_small = [a for a in result.region_area.values() if a < min_pixels]
    assert not too_small, (
        f"найдены регионы мельче порога min_region_fraction: {too_small} "
        f"(min_region_fraction должен был их слить)"
    )


def test_more_n_colors_gives_more_or_equal_regions():
    """Не строгая монотонность (бинарный поиск по порогу не идеален), но
    направление должно быть верным: больше n_colors -> не меньше регионов."""
    img = make_four_quadrant_image()
    few = segment_image(img, n_colors=3, min_region_fraction=0.01)
    many = segment_image(img, n_colors=4, min_region_fraction=0.01)
    assert len(many.region_area) >= len(few.region_area), (
        f"n_colors=4 дал {len(many.region_area)} регионов, "
        f"n_colors=3 дал {len(few.region_area)} — направление неверное"
    )


def test_real_portrait_face_is_not_merged_into_background():
    """Регрессия на реальный дефект: на настоящем портрете кожа лица
    сливалась со стеной и пропадала. Берём фото-портрет из scikit-image
    (лежит внутри пакета, интернет не нужен)."""
    from skimage import data
    img = data.astronaut()
    result = segment_image(img, n_colors=24, min_region_fraction=0.002)

    face_id = result.labels[130, 225]   # лоб/щека
    wall_id = result.labels[100, 335]   # стена справа от головы
    assert face_id != wall_id, "лицо и стена оказались в одном регионе"

    r, g, b = result.region_mean_rgb[int(face_id)]
    assert r > g > b and 150 < r < 245, f"цвет лица не похож на кожу: {(r, g, b)}"


def test_requested_color_count_is_respected_on_real_photo():
    from skimage import data
    result = segment_image(data.astronaut(), n_colors=16, min_region_fraction=0.002)
    distinct_colors = len(set(result.region_mean_rgb.values()))
    assert 12 <= distinct_colors <= 16, f"цветов {distinct_colors}, просили 16"


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
    sys.exit(1 if failed else 0)
