"""
HTTP API для интеграции с сайтом.

POST /process — принимает фото, отдаёт ДВА изображения (разметка и
раскраска) для слайдера до/после. СОЗНАТЕЛЬНО не отдаёт рецепты смесей —
это внутренняя информация для печати, не для сайта.

Безопасность (сервис работает с персональными данными — фото клиентов):
- /process требует секретный ключ в заголовке X-API-Key. Ключ задаётся
  переменной окружения PBN_API_KEY. Если ключ не задан — /process
  отвечает 503 и НЕ работает (fail closed: лучше недоступный сервис, чем
  открытый всем).
- Размер запроса ограничен (PBN_MAX_UPLOAD_MB, по умолчанию 15 МБ).
- Лимит запросов в минуту (PBN_RATE_LIMIT_PER_MIN, по умолчанию 20) и
  лимит одновременных обработок (PBN_MAX_CONCURRENT, по умолчанию 2) —
  чтобы бот или всплеск трафика не положили машину.
- Фото нигде не сохраняется на диск: обрабатывается в памяти и забывается
  после ответа. Это важно для 152-ФЗ (меньше хранения — меньше рисков).
- Клиенту не отдаются тексты внутренних ошибок и трассировки.

Лимит запросов хранится в памяти процесса. Если запускать несколько
рабочих процессов (gunicorn -w N), лимит действует на каждый отдельно.
Для одного студийного ПК и одного сайта этого достаточно; при росте —
переносить в общее хранилище.

Режим — синхронный: сайт ждёт ответа в том же запросе. Если обработка
одного фото регулярно превышает ~15–20 секунд — переводить на асинхронный
режим (задача + отдельный запрос за результатом); переделка локальна для
этого файла.

Запуск для разработки:
    PBN_API_KEY=секрет python3 -m pbn_engine.api
Запрос:
    curl -H "X-API-Key: секрет" -F "photo=@test.jpg" -F "n_colors=24" \
         http://localhost:8000/process -o result.zip
Для боя — gunicorn (см. deploy/), не встроенный сервер Flask.
"""

import hmac
import io
import os
import threading
import time
import traceback
import zipfile
from collections import defaultdict, deque

import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image

from pbn_engine.intake import IntakeError, read_photo
from pbn_engine.palette import PLACEHOLDER_PALETTE, load_palette_from_json
from pbn_engine.pipeline import generate_pbn

app = Flask(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MAX_UPLOAD_MB = _env_int("PBN_MAX_UPLOAD_MB", 15)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

MAX_IMAGE_SIDE = 1600  # крупнее — уменьшаем перед обработкой
REQUEST_TIMEOUT_WARNING_SEC = 15  # дольше — сигнал, что пора на async

# Допустимые диапазоны параметров из запроса: защита от значений,
# которые могут повесить обработку (например n_colors=100000).
N_COLORS_RANGE = (2, 60)
MIN_REGION_FRACTION_RANGE = (0.0002, 0.2)

_rate_lock = threading.Lock()
_rate_hits: dict[str, deque] = defaultdict(deque)
_job_slots = threading.BoundedSemaphore(_env_int("PBN_MAX_CONCURRENT", 2))


def _get_active_palette():
    path = os.environ.get("PBN_PALETTE_PATH", "")
    if path and os.path.exists(path):
        return load_palette_from_json(path), True
    return PLACEHOLDER_PALETTE, False


def _check_api_key():
    """None, если ключ верный; иначе готовый ответ с ошибкой."""
    expected = os.environ.get("PBN_API_KEY", "")
    if not expected:
        return jsonify({"error": "сервис не настроен"}), 503
    given = request.headers.get("X-API-Key", "")
    # compare_digest — сравнение за постоянное время, чтобы ключ нельзя
    # было подобрать по задержке ответа.
    if not hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8")):
        return jsonify({"error": "доступ запрещён"}), 401
    return None


def _check_rate_limit():
    limit = _env_int("PBN_RATE_LIMIT_PER_MIN", 20)
    now = time.monotonic()
    with _rate_lock:
        hits = _rate_hits["global"]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= limit:
            retry = max(1, int(60 - (now - hits[0])))
            resp = jsonify({"error": "слишком много запросов"})
            resp.status_code = 429
            resp.headers["Retry-After"] = str(retry)
            return resp
        hits.append(now)
    return None


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
        return np.array(pil)
    return img


def _to_png_bytes(image_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"файл больше {MAX_UPLOAD_MB} МБ"}), 413


@app.route("/health", methods=["GET"])
def health():
    # Открытый эндпоинт для мониторинга: только факт, что сервис жив.
    # Никаких деталей о палитре и настройках наружу не отдаём.
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    denied = _check_api_key()
    if denied:
        return denied
    limited = _check_rate_limit()
    if limited:
        return limited

    if "photo" not in request.files:
        return jsonify({"error": "нет файла 'photo' в теле запроса"}), 400

    try:
        n_colors = int(request.form.get("n_colors", 24))
        min_region_fraction = float(request.form.get("min_region_fraction", 0.002))
    except ValueError:
        return jsonify({"error": "n_colors и min_region_fraction должны быть числами"}), 400
    if not N_COLORS_RANGE[0] <= n_colors <= N_COLORS_RANGE[1]:
        return jsonify({"error": f"n_colors должно быть от {N_COLORS_RANGE[0]} до {N_COLORS_RANGE[1]}"}), 400
    if not MIN_REGION_FRACTION_RANGE[0] <= min_region_fraction <= MIN_REGION_FRACTION_RANGE[1]:
        return jsonify({"error": "min_region_fraction вне допустимого диапазона"}), 400

    try:
        pil_img = read_photo(request.files["photo"].read())
    except IntakeError as e:
        return jsonify({"error": e.message}), e.status

    if not _job_slots.acquire(blocking=False):
        resp = jsonify({"error": "сервис занят, повторите позже"})
        resp.status_code = 503
        resp.headers["Retry-After"] = "10"
        return resp

    start = time.time()
    try:
        img = _resize_if_needed(np.array(pil_img))
        palette, palette_is_real = _get_active_palette()
        result = generate_pbn(
            img, palette, n_colors=n_colors, min_region_fraction=min_region_fraction
        )
    except Exception:
        # Трассировка — только в серверный лог, не клиенту.
        traceback.print_exc()
        return jsonify({"error": "внутренняя ошибка обработки"}), 500
    finally:
        _job_slots.release()

    elapsed = time.time() - start
    if elapsed > REQUEST_TIMEOUT_WARNING_SEC:
        app.logger.warning(
            "Обработка заняла %.1fс (> %dс) — пора рассмотреть асинхронный режим.",
            elapsed, REQUEST_TIMEOUT_WARNING_SEC,
        )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("outline.png", _to_png_bytes(result.outline_image))
        zf.writestr("colorized.png", _to_png_bytes(result.colorized_image))
    zip_buf.seek(0)

    response = send_file(zip_buf, mimetype="application/zip", download_name="pbn_result.zip")
    response.headers["X-Processing-Time-Sec"] = f"{elapsed:.2f}"
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    if not os.environ.get("PBN_API_KEY"):
        print("ВНИМАНИЕ: PBN_API_KEY не задан — /process будет отвечать 503.")
    _, is_real = _get_active_palette()
    if not is_real:
        print("ВНИМАНИЕ: работаем на ПЛАЦЕХОЛДЕР-палитре (см. palette.py).")
    # Только локальный интерфейс: наружу сервис выставляется через туннель/прокси.
    app.run(host="127.0.0.1", port=8000, debug=False)
