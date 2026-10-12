"""
Тесты API: настоящий HTTP-сервер в отдельном потоке и реальные запросы.
Проверяем и рабочий путь, и защиту: ключ, лимиты, размер, диапазоны.
"""

import io
import os
import sys
import threading
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEST_KEY = "test-secret-key-123"
os.environ["PBN_API_KEY"] = TEST_KEY
os.environ["PBN_RATE_LIMIT_PER_MIN"] = "1000"  # для обычных тестов лимит не мешает
os.environ["PBN_MAX_UPLOAD_MB"] = "1"

import numpy as np
import requests
from PIL import Image

from pbn_engine import api

PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"
AUTH = {"X-API-Key": TEST_KEY}


def _start_server():
    thread = threading.Thread(
        target=lambda: api.app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    for _ in range(50):
        try:
            requests.get(f"{BASE_URL}/health", timeout=0.5)
            return
        except requests.exceptions.ConnectionError:
            time.sleep(0.2)
    raise RuntimeError("сервер не поднялся")


def _photo(size=120):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[: size // 2] = (220, 170, 140)
    img[size // 2:] = (80, 60, 50)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _post(data=None, headers=AUTH, photo=True):
    files = {"photo": ("t.jpg", _photo(), "image/jpeg")} if photo else None
    return requests.post(f"{BASE_URL}/process", files=files, data=data or {}, headers=headers)


def _reset_limits():
    api._rate_hits.clear()


def test_health_is_open_and_minimal():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}, "health не должен раскрывать детали"


def test_process_without_key_is_401():
    assert _post(headers={}).status_code == 401


def test_process_with_wrong_key_is_401():
    assert _post(headers={"X-API-Key": "wrong"}).status_code == 401


def test_process_fails_closed_when_key_not_configured():
    saved = os.environ.pop("PBN_API_KEY")
    try:
        assert _post().status_code == 503
    finally:
        os.environ["PBN_API_KEY"] = saved


def test_process_without_photo_is_400():
    assert _post(photo=False).status_code == 400


def test_process_returns_two_images_and_no_recipes():
    r = _post({"n_colors": 6, "min_region_fraction": 0.02})
    assert r.status_code == 200, r.text
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert set(zf.namelist()) == {"outline.png", "colorized.png"}
    for name in zf.namelist():
        assert Image.open(io.BytesIO(zf.read(name))).size == (120, 120)
    leaked = str(r.headers) + str(zf.namelist())
    for word in ("recipe", "Белила", "рецепт", "legend", "Palette"):
        assert word not in leaked, f"в ответе утекло '{word}'"
    assert r.headers.get("Cache-Control") == "no-store"


def test_out_of_range_parameters_are_400():
    assert _post({"n_colors": 100000}).status_code == 400
    assert _post({"n_colors": 1}).status_code == 400
    assert _post({"min_region_fraction": 0.9}).status_code == 400
    assert _post({"n_colors": "abc"}).status_code == 400


def test_garbage_file_is_400_without_internal_details():
    files = {"photo": ("t.jpg", io.BytesIO(b"not an image"), "image/jpeg")}
    r = requests.post(f"{BASE_URL}/process", files=files, headers=AUTH)
    assert r.status_code == 400
    assert "Traceback" not in r.text and "PIL" not in r.text


def test_oversized_upload_is_413():
    big = io.BytesIO(os.urandom(2 * 1024 * 1024))  # лимит в тестах 1 МБ
    r = requests.post(f"{BASE_URL}/process", files={"photo": ("big.jpg", big)}, headers=AUTH)
    assert r.status_code == 413


def test_rate_limit_returns_429_with_retry_after():
    _reset_limits()
    os.environ["PBN_RATE_LIMIT_PER_MIN"] = "2"
    try:
        codes = [_post({"n_colors": 4, "min_region_fraction": 0.05}).status_code for _ in range(3)]
        assert codes[:2] == [200, 200] and codes[2] == 429, codes
        r = _post({"n_colors": 4})
        assert r.status_code == 429 and "Retry-After" in r.headers
    finally:
        os.environ["PBN_RATE_LIMIT_PER_MIN"] = "1000"
        _reset_limits()


def test_busy_service_returns_503():
    _reset_limits()
    slots = []
    while api._job_slots.acquire(blocking=False):  # занимаем все слоты обработки
        slots.append(1)
    try:
        r = _post()
        assert r.status_code == 503 and "Retry-After" in r.headers
    finally:
        for _ in slots:
            api._job_slots.release()


if __name__ == "__main__":
    _start_server()
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            _reset_limits()
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
