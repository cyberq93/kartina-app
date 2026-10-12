"""
Контрольные тесты движка смешивания (pbn_engine/mixing.py).

Запуск: python3 -m pytest pbn_engine/tests/test_mixing.py -v
(или напрямую: python3 pbn_engine/tests/test_mixing.py)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pbn_engine.mixing import km_mix_two, km_mix_weighted, find_best_mix
from pbn_engine.palette import PigmentColor, PLACEHOLDER_PALETTE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def test_white_plus_black_is_gray_not_collapsed_to_black():
    """Белый+чёрный пополам должен дать физически правдоподобный серый —
    не чистый чёрный (это был реальный баг первой версии модели: без учёта
    силы рассеяния белил чёрный "побеждал" почти при любой доле) и не
    настолько светлый, будто чёрного вообще нет."""
    mixed = km_mix_two(WHITE, BLACK, 0.5)
    assert mixed[0] > 60, f"смесь схлопнулась в слишком тёмный тон: {mixed} (баг с доминированием чёрного)"
    assert mixed[0] < 220, f"смесь слишком светлая, чёрный почти не повлиял: {mixed}"


def test_small_amount_of_black_does_not_erase_white():
    """10% чёрного в смеси не должны полностью перекрашивать её в чёрный —
    эффект должен быть заметным, но пропорциональным (регрессионный тест
    на конкретный баг: K/S чёрного по эпсилон-отражению улетал в бесконечность
    и забивал любую долю белого)."""
    mixed = km_mix_two(WHITE, BLACK, 0.1)
    assert mixed[0] > 120, f"10% чёрного не должны практически стирать белый: {mixed}"


def test_subtractive_darkening_is_stronger_than_linear_average():
    """Ключевой тест субтрактивности: немного чёрного должно затемнять
    сильнее, чем линейное RGB-усреднение. Линейное среднее белого (255) и
    чёрного (0) с t=0.2 даёт 204. В реальной краске чуть чёрного темнит
    куда агрессивнее (как и в K-M)."""
    mixed = km_mix_two(WHITE, BLACK, 0.2)
    linear_average = tuple(round(255 * (1 - 0.2) + 0 * 0.2) for _ in range(3))
    assert mixed[0] < linear_average[0], (
        f"K-M смесь ({mixed}) должна быть темнее линейного среднего ({linear_average})"
    )


def test_mixing_is_order_independent():
    """Смесь A+B с долей t должна совпадать со смесью B+A с долей (1-t)."""
    a = (200, 50, 50)
    b = (50, 50, 200)
    m1 = km_mix_two(a, b, 0.3)
    m2 = km_mix_two(b, a, 0.7)
    assert all(abs(x - y) <= 1 for x, y in zip(m1, m2)), f"{m1} != {m2}"


def test_pure_pigment_returns_itself():
    """Смесь с долей t=0 должна точно вернуть первый цвет (без искажений
    округления больше чем на 1 единицу)."""
    a = (180, 90, 40)
    b = (10, 10, 10)
    mixed = km_mix_two(a, b, 0.0)
    assert all(abs(x - y) <= 1 for x, y in zip(mixed, a))


def test_three_component_weighted_mix_sums_to_valid_rgb():
    colors = [(255, 255, 255), (200, 150, 100), (50, 30, 20)]
    weights = [0.5, 0.3, 0.2]
    mixed = km_mix_weighted(colors, weights)
    assert all(0 <= c <= 255 for c in mixed)


def test_find_best_mix_recovers_known_skin_tone():
    """Главный прикладной тест: для типичного телесного оттенка подбор
    смеси из плацехолдер-палитры должен находить смесь с небольшим ΔE
    (< 6 — разница, которую глаз почти не замечает), а не просто
    ближайшую одиночную краску с большим отклонением."""
    target_skin = (222, 178, 150)  # характерный светлый телесный тон
    result_single = find_best_mix(target_skin, PLACEHOLDER_PALETTE, max_components=1)
    result_mixed = find_best_mix(target_skin, PLACEHOLDER_PALETTE, max_components=2)

    assert result_mixed.delta_e <= result_single.delta_e, (
        "смесь из 2 красок должна приближаться к цели не хуже одиночной краски"
    )
    assert result_mixed.delta_e < 6.0, (
        f"ΔE={result_mixed.delta_e:.2f} — смешивание не нашло приемлемого приближения "
        f"к телесному тону на плацехолдер-палитре"
    )
    print(f"\n  Телесный тон {target_skin}: рецепт = {result_mixed.recipe_text()}, "
          f"результат = {result_mixed.result_rgb}, ΔE = {result_mixed.delta_e:.2f}")


def test_find_best_mix_pure_match_has_zero_delta_e():
    """Если целевой цвет совпадает с цветом краски в палитре — ΔE должен
    быть практически 0."""
    exact = PLACEHOLDER_PALETTE[0].rgb
    result = find_best_mix(exact, PLACEHOLDER_PALETTE, max_components=2)
    assert result.delta_e < 0.5, f"ожидали ΔE≈0 для точного совпадения, получили {result.delta_e}"


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
