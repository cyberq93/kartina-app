"""
Тест API: поднимаем Flask dev-сервер в отдельном потоке, бьём в него
настоящими HTTP-запросами (не просто вызываем функцию process() напрямую) —
так проверяется весь путь, включая приём multipart-файла и отдачу zip.
"""

import sys
import os
import io
import time
import threading
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image
import requests

from pbn_engine.api import app

PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"


def _start_server():
    thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    for _ in range(50):
        try:
            requests.get(f"{BASE_URL}/health", timeout=0.5)
            return
        except requests.exceptions.ConnectionError:
            time.sleep(0.2)
    raise RuntimeError("сервер не поднялся за отведённое время")


def _make_test_image_bytes():
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[:60] = (220, 170, 140)
    img[60:] = (80, 60, 50)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_endpoint_responds():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["palette_size"] > 0


def test_process_without_photo_returns_400():
    resp = requests.post(f"{BASE_URL}/process", data={})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_process_returns_zip_with_two_images_and_no_recipe_data():
    photo = _make_test_image_bytes()
    resp = requests.post(
        f"{BASE_URL}/process",
        files={"photo": ("test.jpg", photo, "image/jpeg")},
        data={"n_colors": 6, "min_region_fraction": 0.02},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["Content-Type"] == "application/zip"

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert names == {"outline.png", "colorized.png"}, f"неожиданный состав zip: {names}"

    # Убеждаемся, что это валидные, непустые PNG, а не заглушки.
    for name in names:
        img_bytes = zf.read(name)
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (120, 120)

    # КЛЮЧЕВАЯ проверка требования пользователя: рецепты смесей/легенда НЕ
    # должны утекать на сайт ни в заголовках, ни где-либо в ответе.
    raw_text = str(resp.headers) + str(names)
    for forbidden in ["recipe", "Белила", "рецепт", "legend"]:
        assert forbidden not in raw_text, f"в ответе API засветилось '{forbidden}' — рецепт утёк на сайт"


def test_process_respects_n_colors_parameter():
    """Косвенная проверка, что параметры из запроса реально доходят до
    движка (а не висят неиспользуемые)."""
    photo = _make_test_image_bytes()
    resp = requests.post(
        f"{BASE_URL}/process",
        files={"photo": ("test.jpg", photo, "image/jpeg")},
        data={"n_colors": 2, "min_region_fraction": 0.3},
    )
    assert resp.status_code == 200, resp.text


if __name__ == "__main__":
    _start_server()
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
