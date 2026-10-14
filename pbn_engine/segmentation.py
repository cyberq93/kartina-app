"""
Сегментация фото на регионы для картины по номерам.

Схема (заменила первую версию на SLIC + слияние по RGB, которая на
настоящем портрете сливала кожу со стеной и теряла лицо):

1. Сглаживание шума с сохранением границ (билатеральный фильтр).
2. Кластеризация цветов k-means в пространстве Lab. Lab ближе к тому, как
   глаз различает цвета, поэтому кожа и фон остаются разными кластерами,
   даже когда в RGB они близки. Число кластеров = число цветов картины.
3. Сглаживание границ между кластерами (размытие маски каждого кластера и
   выбор сильнейшего): убирает «лесенку» и одиночные пиксели.
4. Слияние мелких связных пятен (меньше min_region_fraction кадра) с
   соседом, ближайшим по цвету в Lab. Так управляется сложность: чем
   больше порог, тем крупнее и проще пятна.
5. Каждое связное пятно становится отдельным регионом. Все регионы одного
   кластера получают один и тот же цвет, а значит одну краску и один номер.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

MAX_KMEANS_SAMPLES = 150_000  # k-means считаем по выборке пикселей, это быстро и точно
DUPLICATE_CENTER_DISTANCE = 2.0  # кластеры ближе этого в Lab считаются одним цветом


@dataclass
class SegmentationResult:
    labels: np.ndarray  # 2D int-массив: уникальный id на каждое связное пятно
    region_mean_rgb: dict[int, tuple[int, int, int]]  # id пятна -> цвет его кластера (0-255)
    region_area: dict[int, int]  # id пятна -> площадь в пикселях


def _to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> настоящий Lab (L 0-100, a и b примерно ±128), float32."""
    return cv2.cvtColor(image_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _cluster_colors(lab: np.ndarray, n_colors: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means по Lab. Возвращает (карта кластеров HxW, центры Kx3)."""
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3)

    rng = np.random.default_rng(0)  # фиксированное зерно: один и тот же результат на одном фото
    if len(pixels) > MAX_KMEANS_SAMPLES:
        sample = pixels[rng.choice(len(pixels), MAX_KMEANS_SAMPLES, replace=False)]
    else:
        sample = pixels

    k = max(1, min(n_colors, len(np.unique(sample, axis=0))))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3)
    _, _, centers = cv2.kmeans(
        np.ascontiguousarray(sample), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    labels = cKDTree(centers).query(pixels)[1].reshape(h, w)
    return _merge_duplicate_clusters(labels, centers)


def _merge_duplicate_clusters(labels: np.ndarray, centers: np.ndarray):
    """Склеить кластеры с почти одинаковым цветом и выкинуть пустые
    (бывает, когда n_colors больше, чем реально различимых цветов)."""
    k = len(centers)
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            if np.linalg.norm(centers[i] - centers[j]) < DUPLICATE_CENTER_DISTANCE:
                parent[find(j)] = find(i)

    roots = sorted({find(i) for i in range(k)})
    present = set(np.unique(labels).tolist())
    keep = [r for r in roots if any(find(i) == r and i in present for i in range(k))]
    remap = {}
    for new_id, r in enumerate(keep):
        for i in range(k):
            if find(i) == r:
                remap[i] = new_id
    lut = np.zeros(k, dtype=np.int32)
    for old, new in remap.items():
        lut[old] = new
    new_centers = np.array(
        [centers[[i for i in range(k) if remap.get(i) == new_id and i in present]].mean(axis=0)
         for new_id in range(len(keep))],
        dtype=np.float32,
    )
    return lut[labels], new_centers


def _smooth_labels(labels: np.ndarray, n_clusters: int, sigma: float) -> np.ndarray:
    """Сгладить границы: размыть маску каждого кластера и в каждом пикселе
    выбрать кластер с наибольшим значением. Убирает лесенку и шум."""
    best = np.full(labels.shape, -1.0, dtype=np.float32)
    out = labels.copy()
    for c in range(n_clusters):
        mask = (labels == c).astype(np.float32)
        if not mask.any():
            continue
        blurred = cv2.GaussianBlur(mask, (0, 0), sigma)
        better = blurred > best
        best[better] = blurred[better]
        out[better] = c
    return out


def _merge_small_regions(
    labels: np.ndarray, centers: np.ndarray, min_pixels: int, max_passes: int = 8
) -> np.ndarray:
    """Каждое связное пятно меньше min_pixels отдать соседнему кластеру,
    ближайшему по цвету (Lab). Повторять, пока мелких не останется."""
    labels = labels.copy()
    h, w = labels.shape
    for _ in range(max_passes):
        small = []
        for c in np.unique(labels):
            mask = (labels == c).astype(np.uint8)
            n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_pixels:
                    x, y, bw, bh = (int(stats[i, j]) for j in (
                        cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                    small.append((area, int(c), i, x, y, bw, bh, comp))
        if not small:
            break

        changed = False
        for area, c, comp_id, x, y, bw, bh, comp in sorted(small, key=lambda s: s[0]):
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
            region = (comp[y0:y1, x0:x1] == comp_id) & (labels[y0:y1, x0:x1] == c)
            if not region.any():
                continue
            ring = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~region
            neighbours = labels[y0:y1, x0:x1][ring]
            neighbours = neighbours[neighbours != c]
            if neighbours.size == 0:
                continue
            candidates, counts = np.unique(neighbours, return_counts=True)
            dist = np.linalg.norm(centers[candidates] - centers[c], axis=1)
            target = int(candidates[np.argmin(dist)])
            labels[y0:y1, x0:x1][region] = target
            changed = True
        if not changed:
            break
    return labels


def segment_image(
    image_rgb: np.ndarray,
    n_colors: int = 24,
    min_region_fraction: float = 0.002,
) -> SegmentationResult:
    """
    image_rgb: HxWx3 uint8 (0-255)
    n_colors: сколько цветов (красок/номеров) в картине
    min_region_fraction: пятна меньше этой доли кадра приклеиваются к соседям
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, "ожидается HxWx3 RGB"
    h, w = image_rgb.shape[:2]
    side = max(h, w)

    smoothed = cv2.bilateralFilter(image_rgb, 9, 35, max(5.0, side / 60.0))
    lab = _to_lab(smoothed)

    clusters, centers = _cluster_colors(lab, n_colors)
    k = len(centers)
    clusters = _smooth_labels(clusters, k, sigma=max(1.2, side / 400.0))
    min_pixels = max(1, int(h * w * min_region_fraction))
    clusters = _merge_small_regions(clusters, centers, min_pixels)

    return _build_result(clusters, image_rgb)


def _build_result(clusters: np.ndarray, image_rgb: np.ndarray) -> SegmentationResult:
    """Разбить кластеры на связные пятна. Цвет пятна = средний цвет
    всего его кластера по исходному фото, так что пятна одного кластера
    получают одну краску и один номер."""
    labels = np.zeros(clusters.shape, dtype=np.int32)
    region_mean_rgb: dict[int, tuple[int, int, int]] = {}
    region_area: dict[int, int] = {}
    next_id = 1
    for c in np.unique(clusters):
        mask = clusters == c
        color = tuple(int(v) for v in np.round(image_rgb[mask].mean(axis=0)))
        n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            region = comp == i
            labels[region] = next_id
            region_mean_rgb[next_id] = color
            region_area[next_id] = int(region.sum())
            next_id += 1
    return SegmentationResult(labels=labels, region_mean_rgb=region_mean_rgb, region_area=region_area)
