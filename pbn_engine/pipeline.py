"""
Единый пайплайн: фото -> (размеченное изображение с номерами, раскрашенное
изображение смешанными цветами).

Важный принцип: номер на картинке соответствует РЕЦЕПТУ СМЕСИ, а не региону.
Если два разных по форме региона используют одну и ту же смесь красок —
у них один и тот же номер (как и положено в настоящей картине по номерам:
число красок на палитре обычно меньше числа регионов на холсте).

Раскрашенное превью красится именно подобранными СМЕСЯМИ, а не исходными
цветами фото — так результат на сайте честно показывает, что реально
получится при закрашивании этой палитрой, а не идеализированную фотографию.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from pbn_engine.segmentation import segment_image, SegmentationResult
from pbn_engine.mixing import find_best_mix, MixResult
from pbn_engine.palette import PigmentColor


@dataclass
class LegendEntry:
    number: int
    recipe: MixResult


@dataclass
class PBNResult:
    outline_image: np.ndarray  # HxWx3 uint8, белый фон, чёрные контуры + номера
    colorized_image: np.ndarray  # HxWx3 uint8, регионы закрашены смесями
    legend: list[LegendEntry] = field(default_factory=list)  # для печати, НЕ для сайта
    region_to_number: dict[int, int] = field(default_factory=dict)  # для отладки/печати


def _recipe_key(mix: MixResult) -> tuple:
    """Ключ для группировки регионов с одинаковым рецептом смеси (с учётом
    округления доли, чтобы 32.4% и 32.6% не считались разными рецептами)."""
    return tuple(sorted((p.name, round(frac, 2)) for p, frac in mix.components if frac >= 0.02))


def _find_label_point(mask: np.ndarray) -> tuple[tuple[int, int], float]:
    """Точка внутри пятна, наиболее удалённая от его границы, и эта
    удалённость в пикселях (радиус вписанного круга). Пятно окружаем нулевой
    рамкой, иначе у пятен, касающихся края кадра, номер сажается на сам край
    и обрезается."""
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return (max_loc[0] - 1, max_loc[1] - 1), float(max_val)


def _draw_boundaries(labels: np.ndarray, thickness: int) -> np.ndarray:
    """Границы между ВСЕМИ соседними пятнами (маска True там, где линия).
    Строится по карте пятен, поэтому видны и границы внутри других пятен,
    которые терялись при рисовании только внешних контуров."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge = edge.astype(np.uint8)
    if thickness > 1:
        edge = cv2.dilate(edge, np.ones((thickness, thickness), np.uint8))
    return edge.astype(bool)


def _place_number(outline: np.ndarray, text: str, point: tuple[int, int], radius: float, base_scale: float) -> None:
    """Вписать номер в пятно: шрифт уменьшается, пока цифры не помещаются в
    вписанный круг. Если не помещаются даже в минимальном размере, всё равно
    ставим минимальный (лучше мелкий номер, чем незакрашенное пятно без номера)."""
    scale = base_scale
    for _ in range(5):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if (tw * tw + th * th) ** 0.5 / 2 + 1 <= radius:
            break
        scale *= 0.8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    origin = (point[0] - tw // 2, point[1] + th // 2)
    cv2.putText(outline, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def generate_pbn(
    image_rgb: np.ndarray,
    palette: list[PigmentColor],
    n_colors: int = 24,
    min_region_fraction: float = 0.002,
    max_mix_components: int = 2,
) -> PBNResult:
    seg: SegmentationResult = segment_image(
        image_rgb, n_colors=n_colors, min_region_fraction=min_region_fraction
    )

    # Шаг 1: для каждого цвета подобрать смесь (пятна одного кластера имеют
    # одинаковый цвет, поэтому считаем один раз), сгруппировать по рецепту.
    recipe_to_number: dict[tuple, int] = {}
    mix_cache: dict[tuple, MixResult] = {}
    region_mix: dict[int, MixResult] = {}
    region_to_number: dict[int, int] = {}
    legend: list[LegendEntry] = []

    for region_id, mean_rgb in seg.region_mean_rgb.items():
        if mean_rgb not in mix_cache:
            mix_cache[mean_rgb] = find_best_mix(mean_rgb, palette, max_components=max_mix_components)
        mix = mix_cache[mean_rgb]
        region_mix[region_id] = mix
        key = _recipe_key(mix)
        if key not in recipe_to_number:
            number = len(recipe_to_number) + 1
            recipe_to_number[key] = number
            legend.append(LegendEntry(number=number, recipe=mix))
        region_to_number[region_id] = recipe_to_number[key]

    # Шаг 2: раскрашенное изображение. Цвет = результат смеси красок, а не
    # цвет фото: превью честно показывает, что получится этой палитрой.
    h, w = seg.labels.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    for region_id, mix in region_mix.items():
        colorized[seg.labels == region_id] = mix.result_rgb

    # Шаг 3: разметка. Линии и номера масштабируются вместе с размером кадра.
    side = max(h, w)
    outline = np.full((h, w, 3), 255, dtype=np.uint8)
    outline[_draw_boundaries(seg.labels, thickness=max(1, round(side / 800)))] = (0, 0, 0)

    base_scale = max(0.35, side / 1200.0)
    for region_id, number in region_to_number.items():
        mask = seg.labels == region_id
        point, radius = _find_label_point(mask)
        if radius < 3:  # пятно тоньше трёх пикселей: цифра не поместится
            continue
        _place_number(outline, str(number), point, radius, base_scale)

    return PBNResult(
        outline_image=outline,
        colorized_image=colorized,
        legend=legend,
        region_to_number=region_to_number,
    )
