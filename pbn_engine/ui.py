"""
Интерфейс оператора: страница в браузере для работы с движком.

Это ОТДЕЛЬНОЕ приложение от публичного API (api.py). Оно не должно быть
доступно из интернета: по умолчанию слушает только 127.0.0.1 (сам ПК).
Рецепты смесей здесь показываются полностью, это рабочий инструмент.

Запуск:
    python3 -m pbn_engine.ui
затем открыть http://127.0.0.1:8080

Фото и результаты хранятся только в памяти процесса (последние 5 работ) и
пропадают при остановке: на диск ничего не пишется.

Если нужен доступ с другого ПК в студии, задайте PBN_UI_HOST=0.0.0.0 и
PBN_UI_ALLOWED_HOSTS=имя-или-адрес-этого-пк:8080. Делайте это только в
доверенной сети, у интерфейса нет пароля.
"""

import io
import os
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

from pbn_engine.export_pdf import MAX_OWN_PRODUCTION_WIDTH_CM, export_outline_sheet, export_palette_sheet
from pbn_engine.intake import IntakeError, read_photo
from pbn_engine.palette import PLACEHOLDER_PALETTE, load_palette_from_json
from pbn_engine.pipeline import generate_pbn

STATIC_DIR = os.path.join(os.path.dirname(__file__), "ui_static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

MAX_PROCESS_SIDE = 1600
MAX_JOBS_IN_MEMORY = 5
# Положение ползунка «Детальность» (1..5) -> минимальная доля кадра для пятна.
DETAIL_TO_MIN_REGION = {1: 0.006, 2: 0.003, 3: 0.0015, 4: 0.0007, 5: 0.0004}
CM_RANGE = (10.0, 150.0)
N_COLORS_RANGE = (4, 48)

_jobs: "OrderedDict[str, dict]" = OrderedDict()
_jobs_lock = threading.Lock()
_work_lock = threading.Lock()  # одновременно считаем одну работу


def _allowed_hosts() -> set[str]:
    hosts = {"localhost", "127.0.0.1", "[::1]"}
    extra = os.environ.get("PBN_UI_ALLOWED_HOSTS", "")
    hosts.update(h.strip().lower() for h in extra.split(",") if h.strip())
    return hosts


@app.before_request
def _guard_host():
    """Защита от DNS-rebinding: страница чужого сайта не должна
    дотянуться до интерфейса через браузер оператора."""
    host = request.host.lower()
    bare = host.rsplit(":", 1)[0] if not host.endswith("]") else host
    if host not in _allowed_hosts() and bare not in _allowed_hosts():
        abort(403)


def _palette():
    path = os.environ.get("PBN_PALETTE_PATH", "")
    if path and os.path.exists(path):
        return load_palette_from_json(path), True
    return PLACEHOLDER_PALETTE, False


def crop_to_ratio(img: np.ndarray, ratio: float, fx: float, fy: float) -> np.ndarray:
    """Вырезать из фото кадр с пропорциями холста (ширина/высота = ratio).
    fx, fy в диапазоне 0..1 задают положение кадра по свободной оси, так же,
    как object-position в браузере, поэтому предпросмотр совпадает с расчётом."""
    h, w = img.shape[:2]
    if w / h > ratio:
        new_w = max(1, round(h * ratio))
        x0 = round((w - new_w) * fx)
        return img[:, x0:x0 + new_w]
    new_h = max(1, round(w / ratio))
    y0 = round((h - new_h) * fy)
    return img[y0:y0 + new_h, :]


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def _number(name: str, raw: str, low: float, high: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Поле «{name}» должно быть числом.")
    if not low <= value <= high:
        raise ValueError(f"Поле «{name}» должно быть от {low:g} до {high:g}.")
    return value


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/info")
def info():
    palette, is_real = _palette()
    return jsonify({
        "palette_is_real": is_real,
        "palette_size": len(palette),
        "max_own_width_cm": MAX_OWN_PRODUCTION_WIDTH_CM,
        "n_colors_range": N_COLORS_RANGE,
        "cm_range": CM_RANGE,
    })


@app.route("/api/jobs", methods=["POST"])
def create_job():
    if "photo" not in request.files:
        return jsonify({"error": "Выберите фото."}), 400
    try:
        width_cm = _number("Ширина холста", request.form.get("width_cm"), *CM_RANGE)
        height_cm = _number("Высота холста", request.form.get("height_cm"), *CM_RANGE)
        n_colors = int(_number("Число красок", request.form.get("n_colors", 24), *N_COLORS_RANGE))
        detail = int(_number("Детальность", request.form.get("detail", 3), 1, 5))
        fx = _number("Положение кадра по горизонтали", request.form.get("fx", 0.5), 0, 1)
        fy = _number("Положение кадра по вертикали", request.form.get("fy", 0.5), 0, 1)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        photo = read_photo(request.files["photo"].read())
    except IntakeError as e:
        return jsonify({"error": e.message[0].upper() + e.message[1:] + "."}), e.status

    if not _work_lock.acquire(blocking=False):
        return jsonify({"error": "Предыдущий расчёт ещё идёт. Подождите несколько секунд."}), 409
    try:
        started = time.time()
        img = crop_to_ratio(np.array(photo), width_cm / height_cm, fx, fy)
        h, w = img.shape[:2]
        scale = min(1.0, MAX_PROCESS_SIDE / max(h, w))
        if scale < 1.0:
            img = np.array(Image.fromarray(img).resize((round(w * scale), round(h * scale)), Image.LANCZOS))
        palette, palette_is_real = _palette()
        result = generate_pbn(img, palette, n_colors=n_colors, min_region_fraction=DETAIL_TO_MIN_REGION[detail])
    except Exception:
        app.logger.exception("Ошибка расчёта")
        return jsonify({"error": "Не удалось посчитать эту картинку. Попробуйте другое фото или меньше красок."}), 500
    finally:
        _work_lock.release()

    long_px = max(img.shape[:2])
    long_cm = max(width_cm, height_cm)
    legend = []
    for entry in sorted(result.legend, key=lambda e: e.number):
        legend.append({
            "number": entry.number,
            "color": _hex(entry.recipe.result_rgb),
            "delta_e": round(entry.recipe.delta_e, 1),
            "parts": [
                {"name": p.name, "percent": round(frac * 100), "color": _hex(p.rgb)}
                for p, frac in entry.recipe.components if frac >= 0.02
            ],
        })

    job_id = uuid.uuid4().hex[:12]
    job = {
        "width_cm": width_cm, "height_cm": height_cm, "result": result,
        "source": _png(img), "outline": _png(result.outline_image),
        "colorized": _png(result.colorized_image),
    }
    with _jobs_lock:
        _jobs[job_id] = job
        while len(_jobs) > MAX_JOBS_IN_MEMORY:
            _jobs.popitem(last=False)

    return jsonify({
        "id": job_id,
        "seconds": round(time.time() - started, 1),
        "pixels": [img.shape[1], img.shape[0]],
        "print_dpi": round(long_px / (long_cm / 2.54)),
        "regions": len(result.region_to_number),
        "palette_is_real": palette_is_real,
        "too_wide_for_own_printer": width_cm > MAX_OWN_PRODUCTION_WIDTH_CM,
        "legend": legend,
    })


def _job_or_404(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        abort(404)
    return job


@app.route("/api/jobs/<job_id>/<kind>.png")
def job_image(job_id, kind):
    if kind not in ("source", "outline", "colorized"):
        abort(404)
    data = _job_or_404(job_id)[kind]
    resp = send_file(io.BytesIO(data), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/jobs/<job_id>/<kind>.pdf")
def job_pdf(job_id, kind):
    job = _job_or_404(job_id)
    buf = io.BytesIO()
    if kind == "outline":
        export_outline_sheet(job["result"], job["width_cm"], job["height_cm"], buf)
        name = "razmetka.pdf"
    elif kind == "palette":
        export_palette_sheet(job["result"], buf)
        name = "palitra.pdf"
    else:
        abort(404)
    buf.seek(0)
    resp = send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=name)
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    host = os.environ.get("PBN_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("PBN_UI_PORT", "8080"))
    _, is_real = _palette()
    if not is_real:
        print("ВНИМАНИЕ: используется ПЛАЦЕХОЛДЕР-палитра, а не реальные краски.")
    print(f"Интерфейс: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
