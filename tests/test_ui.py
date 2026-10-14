"""Тесты интерфейса оператора (pbn_engine/ui.py), серверная часть."""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from pbn_engine import ui

client = ui.app.test_client()


def _photo(w=160, h=120):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = (220, 170, 140)
    img[h // 2:] = (80, 60, 50)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _create(**overrides):
    data = {"width_cm": "40", "height_cm": "50", "n_colors": "6", "detail": "2", "fx": "0.5", "fy": "0.5"}
    data.update({k: str(v) for k, v in overrides.items()})
    data["photo"] = (_photo(), "t.jpg")
    return client.post("/api/jobs", data=data, content_type="multipart/form-data")


def test_crop_matches_canvas_ratio_and_position():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, 150:] = 255                                   # правая четверть белая
    left = ui.crop_to_ratio(img, 1.0, 0.0, 0.5)
    right = ui.crop_to_ratio(img, 1.0, 1.0, 0.5)
    assert left.shape[:2] == (100, 100) and right.shape[:2] == (100, 100)
    assert left.max() == 0 and right.max() == 255, "положение кадра не учитывается"
    tall = ui.crop_to_ratio(np.zeros((300, 100, 3), np.uint8), 2.0, 0.5, 0.5)
    assert tall.shape[:2] == (50, 100)


def test_index_page_and_assets_are_served():
    assert client.get("/").status_code == 200
    for path in ("/static/app.js", "/static/app.css", "/static/fonts/Lora-Variable.ttf"):
        assert client.get(path).status_code == 200, path


def test_foreign_host_is_rejected():
    resp = client.get("/api/info", headers={"Host": "evil.example.com"})
    assert resp.status_code == 403


def test_job_returns_legend_and_images_and_pdfs():
    r = _create()
    assert r.status_code == 200, r.get_data(as_text=True)
    data = r.get_json()
    assert data["legend"] and data["legend"][0]["parts"], "нет рецептов в ответе"
    assert data["print_dpi"] > 0 and data["regions"] > 0

    base = f"/api/jobs/{data['id']}/"
    for kind in ("source", "outline", "colorized"):
        resp = client.get(base + kind + ".png")
        assert resp.status_code == 200 and resp.data[:4] == b"\x89PNG", kind
    for kind in ("outline", "palette"):
        resp = client.get(base + kind + ".pdf")
        assert resp.status_code == 200 and resp.data[:4] == b"%PDF", kind


def test_cropped_source_has_canvas_proportions():
    data = _create(width_cm=40, height_cm=60).get_json()
    img = Image.open(io.BytesIO(client.get(f"/api/jobs/{data['id']}/source.png").data))
    ratio = img.size[0] / img.size[1]
    assert abs(ratio - 40 / 60) < 0.02, f"пропорции кадра {ratio:.3f}, нужны {40/60:.3f}"


def test_bad_input_gives_clear_russian_errors():
    for overrides in ({"width_cm": "5"}, {"height_cm": "abc"}, {"n_colors": "1000"}, {"detail": "9"}):
        r = _create(**overrides)
        assert r.status_code == 400, overrides
        assert "должно быть" in r.get_json()["error"], r.get_json()
    r = client.post("/api/jobs", data={"width_cm": "40", "height_cm": "50"})
    assert r.status_code == 400 and "фото" in r.get_json()["error"].lower()
    bad = client.post("/api/jobs", data={"width_cm": "40", "height_cm": "50",
                                         "photo": (io.BytesIO(b"nope"), "x.jpg")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400 and "Traceback" not in bad.get_data(as_text=True)


def test_unknown_job_and_kind_are_404():
    assert client.get("/api/jobs/doesnotexist/outline.png").status_code == 404
    data = _create().get_json()
    assert client.get(f"/api/jobs/{data['id']}/secret.png").status_code == 404
    assert client.get(f"/api/jobs/{data['id']}/other.pdf").status_code == 404


def test_only_last_jobs_are_kept_in_memory():
    first = _create().get_json()["id"]
    for _ in range(ui.MAX_JOBS_IN_MEMORY):
        _create()
    assert client.get(f"/api/jobs/{first}/outline.png").status_code == 404


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
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} тестов прошли")
    sys.exit(1 if failed else 0)
