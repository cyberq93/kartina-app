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


def _find_label_point(mask: np.ndarray) -> tuple[int, int] | None:
    """Точка внутри региона, наиболее удалённая от границы — лучшее место
    для номера (не попадёт на тонкий "хвост" региона сложной формы)."""
    mask_u8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val < 3:  # регион слишком тонкий, чтобы в него поместилась цифра
        return None
    return max_loc  # (x, y)


def generate_pbn(
    image_rgb: np.ndarray,
    palette: list[PigmentColor],
    n_colors: int = 24,
    min_region_fraction: float = 0.002,
    max_mix_components: int = 2,
    min_blob_area_for_number: int = 150,
) -> PBNResult:
    seg: SegmentationResult = segment_image(
        image_rgb, n_colors=n_colors, min_region_fraction=min_region_fraction
    )

    # Шаг 1: для каждого региона подобрать смесь, сгруппировать по рецепту.
    recipe_to_number: dict[tuple, int] = {}
    region_mix: dict[int, MixResult] = {}
    region_to_number: dict[int, int] = {}
    legend: list[LegendEntry] = []

    for region_id, mean_rgb in seg.region_mean_rgb.items():
        mix = find_best_mix(mean_rgb, palette, max_components=max_mix_components)
        region_mix[region_id] = mix
        key = _recipe_key(mix)
        if key not in recipe_to_number:
            number = len(recipe_to_number) + 1
            recipe_to_number[key] = number
            legend.append(LegendEntry(number=number, recipe=mix))
        region_to_number[region_id] = recipe_to_number[key]

    # Шаг 2: раскрашенное изображение — цвет = результат смеси (не фото!).
    h, w = seg.labels.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    for region_id, mix in region_mix.items():
        colorized[seg.labels == region_id] = mix.result_rgb

    # Шаг 3: контуры + номера на белом фоне.
    outline = np.full((h, w, 3), 255, dtype=np.uint8)
    for region_id in np.unique(seg.labels):
        mask = (seg.labels == region_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline, contours, -1, (0, 0, 0), 1)

        number = region_to_number[int(region_id)]
        # Регион мог распасться на несколько несвязных кусков при слиянии —
        # подписываем номер в каждом достаточно крупном куске отдельно.
        n_blobs, blob_labels = cv2.connectedComponents(mask)
        for blob_id in range(1, n_blobs):
            blob_mask = blob_labels == blob_id
            if blob_mask.sum() < min_blob_area_for_number:
                continue
            point = _find_label_point(blob_mask)
            if point is None:
                continue
            text = str(number)
            font_scale = 0.4
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            origin = (point[0] - text_w // 2, point[1] + text_h // 2)
            cv2.putText(
                outline, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA
            )

    return PBNResult(
        outline_image=outline,
        colorized_image=colorized,
        legend=legend,
        region_to_number=region_to_number,
    )
