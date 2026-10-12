"""
Движок смешивания красок.

Модель: упрощённая (одноконстантная, покан ально) модель Kubelka-Munk.

Почему не прямое RGB-усреднение / не nearest-match (как в "Раскраска 2.0"):
- RGB-усреднение аддитивно (как смешение света) — реальные краски смешиваются
  субтрактивно (поглощение пигментом), поэтому "в лоб" усреднённый цвет
  получается грязнее/светлее, чем настоящая смесь красок.
- nearest-match по готовому списку красок вообще не создаёт новый цвет —
  просто берёт ближайший существующий, из-за чего многие целевые цвета
  (прежде всего — оттенки кожи) не достижимы, если их нет в списке готовым.

Формула Kubelka-Munk для одного компонента (K — поглощение, S — рассеяние):
    K/S = (1 - R)^2 / (2R),  где R — отражение (0..1]

ВАЖНАЯ ПРАВКА (найдена тестами, не теоретически): первая версия этого модуля
смешивала (K/S) с равными весами, предполагая одинаковое рассеяние S=1 у
всех пигментов. Это физически неверно и ломалось на практике: титановые
белила в реальности рассеивают свет в десятки раз сильнее большинства
других пигментов (поэтому они и используются как основа для осветления —
"кроющая способность") — а чёрные пигменты рассеивают слабо, в основном
поглощая. При равном S чёрный математически "побеждал" смесь почти при
любой доле (белый+чёрный 50/50 давал почти чистый чёрный, что неверно).

Исправление: каждому пигменту присваивается коэффициент рассеяния S,
оцениваемый по его светлоте (Lab L*) — чем светлее пигмент, тем выше S
(гладкая кривая, без реальных измерений это лучшее доступное приближение).
Абсолютный коэффициент поглощения считается как K_i = (K/S)_i * S_i, и смесь
считается по корректной двухконстантной формуле Kubelka-Munk:
    (K/S)_mix = sum(c_i * K_i) / sum(c_i * S_i)
Обратное преобразование в отражение — та же формула, что и раньше:
    R_mix = 1 + (K/S)_mix - sqrt((K/S)_mix^2 + 2*(K/S)_mix)

Применяется независимо к каждому каналу R, G, B (скаляр S — общий для
пигмента, одинаковый для всех трёх каналов, т.к. отдельных по-канальных
данных о рассеянии у нас нет).

ИЗВЕСТНОЕ ОГРАНИЧЕНИЕ этой упрощённой модели (зафиксировано сознательно):
поканальная версия не содержит информацию о спектральном "перекрытии" между
каналами, поэтому классический пример синий+жёлтый=зелёный воспроизводится
слабее, чем в полной спектральной модели (как в Mixbox). На оттенках кожи,
земляных и телесных тонах (наш приоритет №1 по жалобе на "Раскраска 2.0")
это ограничение не проявляется — там как раз работает корректно, потому что
там не нужен спектральный "перескок" через зелёный.
Если понадобится точность на синий+жёлтый — следующий шаг развития: заменить
только функцию _km_mix_channels на полноценную спектральную версию
(Jakob-Hanika sigmoid upsampling + K/S по длинам волн), остальной код
(подбор смеси, пайплайн) трогать не придётся — интерфейс тот же.
"""

from dataclasses import dataclass
from itertools import combinations
from functools import lru_cache

import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

from pbn_engine.palette import PigmentColor

_EPS = 0.02  # минимальное отражение; у реальной краски не бывает буквально 0%

# Кривая "светлота -> коэффициент рассеяния S". Подобрана эмпирически по
# контрольным тестам (белый+чёрный должны давать физически правдоподобный
# серый, а не чёрный); не измерена реальным прибором. При появлении
# реальных спектральных данных по краскам Дениса — заменить на них.
_S_MIN = 0.4   # рассеяние у самых тёмных пигментов (в основном поглощают)
_S_MAX = 16.0  # рассеяние у самых светлых пигментов (сильные "кроющие", как титановые белила)
_S_GAMMA = 2.5  # форма кривой: большинство средних тонов остаются близко к S_MIN


@lru_cache(maxsize=None)
def _scattering_for_rgb(rgb: tuple[int, int, int]) -> float:
    """Оценка коэффициента рассеяния S по светлоте пигмента (Lab L*, 0-100)."""
    lab = rgb2lab(np.array([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]]))
    L = float(np.clip(lab[0, 0, 0], 0.0, 100.0)) / 100.0
    return _S_MIN + (_S_MAX - _S_MIN) * (L**_S_GAMMA)


def _reflectance(rgb: np.ndarray) -> np.ndarray:
    """0-255 -> 0..1 отражение, с защитой от деления на ноль в K/S."""
    r = rgb.astype(np.float64) / 255.0
    return np.clip(r, _EPS, 1.0)


def _ks_from_reflectance(r: np.ndarray) -> np.ndarray:
    return (1.0 - r) ** 2 / (2.0 * r)


def _reflectance_from_ks(ks: np.ndarray) -> np.ndarray:
    ks = np.clip(ks, 0.0, None)
    r = 1.0 + ks - np.sqrt(ks**2 + 2.0 * ks)
    return np.clip(r, 0.0, 1.0)


def km_mix_weighted(rgbs: list[tuple[int, int, int]], weights: list[float]) -> tuple[int, int, int]:
    """Смешать N цветов (0-255 RGB) с заданными весами (сумма = 1) по
    двухконстантной модели Kubelka-Munk (K и S по отдельности, см. докстринг
    модуля). Возвращает RGB 0-255."""
    weights = np.asarray(weights, dtype=np.float64)
    assert abs(weights.sum() - 1.0) < 1e-6, "веса должны суммироваться в 1"

    k_sum = np.zeros(3, dtype=np.float64)
    s_sum = 0.0
    for rgb, w in zip(rgbs, weights):
        rgb_t = tuple(int(c) for c in rgb)
        r = _reflectance(np.asarray(rgb_t, dtype=np.float64))
        ks_ratio = _ks_from_reflectance(r)  # (K/S)_i, не зависит от выбора S_i
        s_i = _scattering_for_rgb(rgb_t)
        k_sum += w * (ks_ratio * s_i)  # K_i = (K/S)_i * S_i
        s_sum += w * s_i

    ks_mix = k_sum / max(s_sum, 1e-9)
    r_mix = _reflectance_from_ks(ks_mix)
    return tuple(np.round(r_mix * 255.0).astype(int).tolist())


def km_mix_two(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Смешать два цвета, t — доля rgb2 (0..1)."""
    return km_mix_weighted([rgb1, rgb2], [1.0 - t, t])


def _rgb_to_lab1(rgb: tuple[int, int, int]) -> np.ndarray:
    arr = np.array([[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]])
    return rgb2lab(arr)[0, 0]


def _delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(deltaE_ciede2000(lab1.reshape(1, 1, 3), lab2.reshape(1, 1, 3))[0, 0])


@dataclass
class MixResult:
    components: list[tuple[PigmentColor, float]]  # (краска, доля 0..1), по убыванию доли
    result_rgb: tuple[int, int, int]
    delta_e: float  # отличие от целевого цвета (0 = идеально, <2 — не отличимо глазом, >6 — заметно)

    def recipe_text(self) -> str:
        parts = []
        for pigment, frac in self.components:
            if frac < 0.02:
                continue
            parts.append(f"{pigment.name} — {round(frac * 100)}%")
        return " + ".join(parts) if parts else "—"


def find_best_mix(
    target_rgb: tuple[int, int, int],
    palette: list[PigmentColor],
    max_components: int = 2,
    coarse_step: float = 0.1,
    refine: bool = True,
) -> MixResult:
    """Перебором найти лучшую смесь из палитры под целевой цвет.

    Для max_components=2 перебираются все пары красок с шагом coarse_step
    по пропорции, затем (если refine=True) уточняются вокруг найденного
    оптимума более мелким шагом. Для max_components=1 просто берётся
    ближайшая одиночная краска (без смешивания).
    """
    target_lab = _rgb_to_lab1(target_rgb)

    best: MixResult | None = None

    # Базовый случай: одна краска без смешивания (нужен как "пол" сравнения,
    # а иногда и правда оптимален, если целевой цвет совпадает с готовой краской).
    for p in palette:
        de = _delta_e(target_lab, _rgb_to_lab1(p.rgb))
        if best is None or de < best.delta_e:
            best = MixResult([(p, 1.0)], p.rgb, de)

    if max_components >= 2:
        steps = np.arange(coarse_step, 1.0, coarse_step)
        for p1, p2 in combinations(palette, 2):
            for t in steps:
                mixed = km_mix_two(p1.rgb, p2.rgb, float(t))
                de = _delta_e(target_lab, _rgb_to_lab1(mixed))
                if de < best.delta_e:
                    best = MixResult([(p1, 1.0 - float(t)), (p2, float(t))], mixed, de)

        if refine and best is not None and len(best.components) == 2:
            (p1, t1), (p2, t2) = best.components
            center = t2
            fine_steps = np.linspace(
                max(0.01, center - coarse_step), min(0.99, center + coarse_step), 9
            )
            for t in fine_steps:
                mixed = km_mix_two(p1.rgb, p2.rgb, float(t))
                de = _delta_e(target_lab, _rgb_to_lab1(mixed))
                if de < best.delta_e:
                    best = MixResult([(p1, 1.0 - float(t)), (p2, float(t))], mixed, de)

    # Красивая сортировка компонентов по убыванию доли для текста рецепта.
    best.components.sort(key=lambda c: -c[1])
    return best
