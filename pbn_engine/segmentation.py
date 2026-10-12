"""
Сегментация фото на регионы одного цвета.

Пайплайн:
1. SLIC-суперпиксели в Lab-пространстве — группируют пиксели одновременно
   по цвету и по близости на фото (а не как плоский k-means по всем пикселям
   без учёта положения — так получаются рваные острова одного цвета
   вперемешку по всему фото).
2. Region Adjacency Graph (RAG) + иерархическое слияние соседних регионов
   по близости среднего цвета, пока не останется нужное число регионов
   (n_colors) — это и есть "подбор количества цветов".
3. Пост-слияние мелких по площади регионов (меньше min_region_fraction от
   кадра) в ближайшего по цвету соседа — убирает нечитаемые мелкие кусочки
   (это была жалоба и на "Раскраска 2.0": "мелкие нечитаемые области").

Параметр сложности (n_colors + min_region_fraction) — то, чем управляется
уровень сложности под конкретный размер холста.
"""

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import slic
from skimage import graph
from skimage.measure import regionprops


@dataclass
class SegmentationResult:
    labels: np.ndarray  # 2D int-массив, label региона на пиксель
    region_mean_rgb: dict[int, tuple[int, int, int]]  # label -> средний цвет региона (0-255)
    region_area: dict[int, int]  # label -> площадь в пикселях


def segment_image(
    image_rgb: np.ndarray,
    n_colors: int = 24,
    min_region_fraction: float = 0.002,
    slic_segments: int = 800,
    compactness: float = 12.0,
) -> SegmentationResult:
    """
    image_rgb: HxWx3 uint8 массив (0-255)
    n_colors: целевое число регионов после слияния (= число "номеров" на
        итоговой картине по номерам — ключевой параметр сложности)
    min_region_fraction: регионы меньше этой доли площади кадра
        принудительно сливаются с ближайшим соседом (не дают мелких
        нечитаемых пятен)
    slic_segments: число стартовых суперпикселей до слияния (чем больше —
        тем точнее исходные границы, но дольше считается)
    compactness: баланс между "следовать цвету" и "держать форму
        компактной" у SLIC; выше — более округлые регионы
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, "ожидается HxWx3 RGB"

    image_float = image_rgb.astype(np.float64) / 255.0

    # Шаг 1: SLIC-суперпиксели. start_label=1, т.к. 0 зарезервирован под фон
    # в графовых операциях skimage.
    initial_labels = slic(
        image_float,
        n_segments=slic_segments,
        compactness=compactness,
        start_label=1,
        channel_axis=-1,
        convert2lab=True,
    )

    # Шаг 2: RAG по среднему цвету + иерархическое слияние до n_colors регионов.
    rag = graph.rag_mean_color(image_rgb, initial_labels)
    merged_labels = _merge_to_target_count(rag, initial_labels, image_rgb, n_colors)

    # Шаг 3: принудительно добить мелкие регионы (ниже порога площади) —
    # RAG-слияние по порогу цвета не гарантирует отсутствие мелких пятен,
    # если они резко отличаются по цвету от всех соседей.
    merged_labels = _merge_small_regions(merged_labels, image_rgb, min_region_fraction)

    return _build_result(merged_labels, image_rgb)


def _merge_to_target_count(
    rag: graph.RAG, labels: np.ndarray, image_rgb: np.ndarray, n_colors: int
) -> np.ndarray:
    """Иерархически сливать регионы с наиболее близким цветом, пока их не
    останется n_colors. Использует встроенный merge_hierarchical skimage
    с порогом, который подбирается бинарным поиском под целевое число
    регионов (сам skimage принимает порог ΔE, а не "число регионов")."""

    def weight_func(graph_, src, dst, n):
        diff = graph_.nodes[dst]["mean color"] - graph_.nodes[n]["mean color"]
        return {"weight": float(np.linalg.norm(diff))}

    def merge_func(graph_, src, dst):
        total = graph_.nodes[dst]["pixel count"] + graph_.nodes[src]["pixel count"]
        graph_.nodes[dst]["mean color"] = (
            graph_.nodes[dst]["mean color"] * graph_.nodes[dst]["pixel count"]
            + graph_.nodes[src]["mean color"] * graph_.nodes[src]["pixel count"]
        ) / total
        graph_.nodes[dst]["pixel count"] = total

    lo, hi = 0.0, 150.0
    best_labels = labels
    for _ in range(14):  # 14 итераций бинарного поиска достаточно для точности ~0.01
        mid = (lo + hi) / 2.0
        rag_copy = rag.copy()
        candidate = graph.merge_hierarchical(
            labels,
            rag_copy,
            thresh=mid,
            rag_copy=False,
            in_place_merge=True,
            merge_func=merge_func,
            weight_func=weight_func,
        )
        count = len(np.unique(candidate))
        best_labels = candidate
        if count > n_colors:
            lo = mid
        elif count < n_colors:
            hi = mid
        else:
            break
    return best_labels


def _merge_small_regions(
    labels: np.ndarray, image_rgb: np.ndarray, min_fraction: float
) -> np.ndarray:
    """Слить регионы мельче min_fraction площади кадра с ближайшим по
    среднему цвету соседним регионом. Повторяется, т.к. слияние может
    создать новый мелкий регион на стыке (редко, но нужна защита)."""
    total_pixels = labels.shape[0] * labels.shape[1]
    min_pixels = max(1, int(total_pixels * min_fraction))

    labels = labels.copy()
    for _ in range(5):  # защита от бесконечного цикла в вырожденных случаях
        if len(np.unique(labels)) <= 1:
            # Вырожденный случай: всё уже слилось в один регион (например,
            # n_colors было слишком мало). rag_mean_color падает на
            # изображении без единой границы между регионами (RAG строит
            # узлы только по границам) — дальше сливать нечего, выходим.
            break
        rag = graph.rag_mean_color(image_rgb, labels)
        areas = {n: rag.nodes[n]["pixel count"] for n in rag.nodes}
        small = [n for n, a in areas.items() if a < min_pixels]
        if not small:
            break
        for n in small:
            if n not in rag.nodes:
                continue
            neighbors = list(rag.neighbors(n))
            if not neighbors:
                continue
            my_color = rag.nodes[n]["mean color"]
            closest = min(
                neighbors,
                key=lambda nb: np.linalg.norm(rag.nodes[nb]["mean color"] - my_color),
            )
            labels[labels == n] = closest
            total = rag.nodes[closest]["pixel count"] + rag.nodes[n]["pixel count"]
            rag.nodes[closest]["mean color"] = (
                rag.nodes[closest]["mean color"] * rag.nodes[closest]["pixel count"]
                + rag.nodes[n]["mean color"] * rag.nodes[n]["pixel count"]
            ) / total
            rag.nodes[closest]["pixel count"] = total
            rag.remove_node(n)
    return labels


def _build_result(labels: np.ndarray, image_rgb: np.ndarray) -> SegmentationResult:
    region_mean_rgb = {}
    region_area = {}
    for region_id in np.unique(labels):
        mask = labels == region_id
        mean_color = image_rgb[mask].mean(axis=0)
        region_mean_rgb[int(region_id)] = tuple(int(c) for c in np.round(mean_color))
        region_area[int(region_id)] = int(mask.sum())
    return SegmentationResult(labels=labels, region_mean_rgb=region_mean_rgb, region_area=region_area)
