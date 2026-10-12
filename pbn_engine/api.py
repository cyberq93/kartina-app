"""
HTTP API для интеграции с сайтом.

POST /process — принимает фото, отдаёт ДВА изображения (разметка и
раскраска) для слайдера до/после. СОЗНАТЕЛЬНО не отдаёт рецепты смесей —
это внутренняя информация для печати, не для сайта (см. запрос пользователя).

Режим — синхронный (сайт ждёт ответа в том же запросе). Это сознательный
выбор для v1: проще для интеграции с сайтом, не нужна очередь задач.
ЕСЛИ обработка одного фото начнёт регулярно занимать больше ~15-20 секунд
(например, на очень больших фото или при высокой нагрузке) — это место
нужно будет переделать на асинхронный режим (задача + отдельный запрос за
результатом). Передел локален: меняется только этот файл, pipeline.py и
остальной движок трогать не придётся.

Почему Flask, а не FastAPI: FastAPI не удалось установить в этой рабочей
среде — pip к pypi.org здесь заблокирован политикой организации (403 на
прокси), не наша ошибка и не технический сбой, который можно обойти.
Flask и Starlette уже были в системе — используем Flask, он того же
уровня простоты для одного синхронного эндпоинта. Если на бою нужен именно
FastAPI — на вашей машине достаточно "pip install fastapi uvicorn",
собственно pipeline.py от выбора фреймворка не зависит.

Запуск для теста:
    python3 -m pbn_engine.api
Запрос:
    curl -F "photo=@test.jpg" -F "n_colors=24" -F "canvas_width_cm=40" \
         -F "canvas_height_cm=60" http://localhost:8000/process -o result.zip
"""

import io
import os
import time
import zipfile
import traceback

from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np

from pbn_engine.pipeline import generate_pbn
from pbn_engine.palette import PLACEHOLDER_PALETTE, load_palette_from_json

app = Flask(__name__)

MAX_IMAGE_SIDE = 1600  # защита от слишком тяжёлых фото — пересчитать перед обработкой
REQUEST_TIMEOUT_WARNING_SEC = 15  # после этого порога — лог-сигнал, что пора на async

# Путь к реальной палитре Дениса, когда она появится (см. palette.py).
# Если файла нет — используем плацехолдер и явно предупреждаем в ответе.
REAL_PALETTE_PATH = os.environ.get("PBN_PALETTE_PATH", "")


def _get_active_palette():
    if REAL_PALETTE_PATH and os.path.exists(REAL_PALETTE_PATH):
        return load_palette_from_json(REAL_PALETTE_PATH), True
    return PLACEHOLDER_PALETTE, False


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
        return np.array(pil)
    return img


@app.route("/health", methods=["GET"])
def health():
    palette, is_real = _get_active_palette()
    return jsonify({
        "status": "ok",
        "palette_source": "real" if is_real else "PLACEHOLDER (замените через PBN_PALETTE_PATH)",
        "palette_size": len(palette),
    })


@app.route("/process", methods=["POST"])
def process():
    start = time.time()

    if "photo" not in request.files:
        return jsonify({"error": "нет файла 'photo' в теле запроса"}), 400

    file = request.files["photo"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"не удалось прочитать изображение: {e}"}), 400

    img = np.array(pil_img)
    img = _resize_if_needed(img)

    # Параметры с разумными умолчаниями — сайт может их не присылать вовсе.
    try:
        n_colors = int(request.form.get("n_colors", 24))
        min_region_fraction = float(request.form.get("min_region_fraction", 0.002))
    except ValueError:
        return jsonify({"error": "n_colors/min_region_fraction должны быть числами"}), 400

    palette, palette_is_real = _get_active_palette()

    try:
        result = generate_pbn(
            img, palette, n_colors=n_colors, min_region_fraction=min_region_fraction
        )
    except Exception as e:
        # Не отдаём трассировку клиенту (сайту) — только в серверный лог.
        traceback.print_exc()
        return jsonify({"error": "внутренняя ошибка обработки", "detail": str(e)}), 500

    elapsed = time.time() - start
    if elapsed > REQUEST_TIMEOUT_WARNING_SEC:
        app.logger.warning(
            f"Обработка заняла {elapsed:.1f}с (> {REQUEST_TIMEOUT_WARNING_SEC}с) — "
            f"при росте нагрузки этот эндпоинт пора переводить на асинхронный режим."
        )

    # Отдаём ровно то, что просили: 2 изображения, без рецептов/легенды.
    outline_png = _to_png_bytes(result.outline_image)
    colorized_png = _to_png_bytes(result.colorized_image)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("outline.png", outline_png)
        zf.writestr("colorized.png", colorized_png)
    zip_buf.seek(0)

    response = send_file(zip_buf, mimetype="application/zip", download_name="pbn_result.zip")
    response.headers["X-Processing-Time-Sec"] = f"{elapsed:.2f}"
    response.headers["X-Palette-Source"] = "real" if palette_is_real else "placeholder"
    return response


def _to_png_bytes(image_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    palette, is_real = _get_active_palette()
    if not is_real:
        print(
            "ВНИМАНИЕ: работаем на ПЛАЦЕХОЛДЕР-палитре (12 типовых акриловых "
            "цветов). Чтобы подключить реальные краски Дениса — задайте "
            "переменную окружения PBN_PALETTE_PATH на JSON-файл с палитрой "
            "(формат описан в palette.py: load_palette_from_json)."
        )
    app.run(host="0.0.0.0", port=8000, debug=False)
