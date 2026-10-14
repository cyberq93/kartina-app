"use strict";

/* Готовые размеры холста, см (ширина × высота). Список можно менять:
   достаточно отредактировать массив. */
const PRESETS = [[30, 40], [40, 50], [40, 60], [50, 70], [60, 80]];

const $ = (id) => document.getElementById(id);
const el = {
  form: $("setup"), photo: $("photo"), drop: $("drop"), dropText: $("drop-text"),
  presets: $("presets"), width: $("width"), height: $("height"), swap: $("swap"), sizeNote: $("size-note"),
  nColors: $("n-colors"), nOut: $("n-out"), detail: $("detail"), detailOut: $("detail-out"),
  run: $("run"), error: $("error"),
  tabs: $("tabs"), viewport: $("viewport"), frameImg: $("frame-img"), resultImg: $("result-img"),
  compare: $("compare"), cmpUnder: $("cmp-under"), cmpOver: $("cmp-over"), cmpOverImg: $("cmp-over-img"),
  cmpHandle: $("cmp-handle"), compareMode: $("compare-mode"), panHint: $("pan-hint"),
  busy: $("busy"), facts: $("facts"), stale: $("stale"), downloads: $("downloads"),
  chartCount: $("chart-count"), chartEmpty: $("chart-empty"), chartList: $("chart-list"), chartLegend: $("chart-legend"),
  paletteNote: $("palette-note"),
};

const state = {
  file: null, url: null, fx: 0.5, fy: 0.5,
  job: null, view: "frame", compareLeft: "outline", compareAt: 50, maxOwnWidth: 61,
};

/* ---------- Размер холста ---------- */
function canvasSize() {
  const w = parseFloat(el.width.value), h = parseFloat(el.height.value);
  return w > 0 && h > 0 ? [w, h] : null;
}

function applyRatio(ratio) {
  el.viewport.style.aspectRatio = String(ratio);
  el.viewport.style.setProperty("--ratio", String(ratio));
}

function refreshSizeUi() {
  const size = canvasSize();
  [...el.presets.children].forEach((b) => {
    b.setAttribute("aria-pressed", String(!!size && +b.dataset.w === size[0] && +b.dataset.h === size[1]));
  });
  if (size && state.view === "frame") applyRatio(size[0] / size[1]);
  const wide = size && size[0] > state.maxOwnWidth;
  el.sizeNote.hidden = !wide;
  if (wide) {
    el.sizeNote.textContent = `Шире ${state.maxOwnWidth} см: на своём плоттере такое не напечатать, только через подрядчика.`;
    el.sizeNote.className = "note warn";
  }
}

PRESETS.forEach(([w, h]) => {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = `${w}×${h}`; b.dataset.w = w; b.dataset.h = h;
  b.addEventListener("click", () => { el.width.value = w; el.height.value = h; onSettingsChanged(); });
  el.presets.appendChild(b);
});

el.swap.addEventListener("click", () => {
  [el.width.value, el.height.value] = [el.height.value, el.width.value];
  onSettingsChanged();
});

/* ---------- Выбор фото ---------- */
function setFile(file) {
  if (!file) return;
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file; state.url = URL.createObjectURL(file);
  state.fx = 0.5; state.fy = 0.5;
  el.dropText.textContent = file.name;
  el.drop.classList.add("has-file");
  el.frameImg.onload = () => {
    // Если ориентация фото и холста не совпадают, меняем холст местами: так кадр обычно лучше.
    const size = canvasSize();
    const photoWide = el.frameImg.naturalWidth > el.frameImg.naturalHeight;
    if (size && size[0] !== size[1] && (size[0] > size[1]) !== photoWide) {
      [el.width.value, el.height.value] = [el.height.value, el.width.value];
    }
    refreshSizeUi();
  };
  el.frameImg.src = state.url;
  el.frameImg.hidden = false;
  el.viewport.classList.remove("empty");
  el.run.disabled = false;
  hideError();
  showView("frame");
  markStale();
}

el.photo.addEventListener("change", () => setFile(el.photo.files[0]));
["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
el.drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.photo.click(); } });

/* ---------- Выбор кадра перетаскиванием ---------- */
let drag = null;
el.frameImg.addEventListener("pointerdown", (e) => {
  if (state.view !== "frame") return;
  drag = { x: e.clientX, y: e.clientY, fx: state.fx, fy: state.fy };
  el.frameImg.setPointerCapture(e.pointerId);
  el.frameImg.classList.add("dragging");
});
el.frameImg.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const box = el.viewport.getBoundingClientRect();
  const iw = el.frameImg.naturalWidth, ih = el.frameImg.naturalHeight;
  const s = Math.max(box.width / iw, box.height / ih);      // масштаб object-fit: cover
  const extraX = iw * s - box.width, extraY = ih * s - box.height;
  const clamp = (v) => Math.min(1, Math.max(0, v));
  if (extraX > 1) state.fx = clamp(drag.fx - (e.clientX - drag.x) / extraX);
  if (extraY > 1) state.fy = clamp(drag.fy - (e.clientY - drag.y) / extraY);
  el.frameImg.style.objectPosition = `${state.fx * 100}% ${state.fy * 100}%`;
});
const endDrag = () => {
  if (drag) markStale();
  drag = null; el.frameImg.classList.remove("dragging");
};
el.frameImg.addEventListener("pointerup", endDrag);
el.frameImg.addEventListener("pointercancel", endDrag);

/* ---------- Вкладки ---------- */
function showView(view) {
  state.view = view;
  [...el.tabs.children].forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  el.frameImg.hidden = view !== "frame" || !state.file;
  el.resultImg.hidden = !(view === "outline" || view === "colorized");
  el.compare.hidden = view !== "compare";
  el.compareMode.hidden = view !== "compare";
  el.panHint.hidden = view !== "frame" || !state.file;

  if (view === "frame") {
    const size = canvasSize();
    if (size) applyRatio(size[0] / size[1]);
  } else if (state.job) {
    applyRatio(state.job.ratio);
    const base = `/api/jobs/${state.job.id}/`;
    if (view === "outline" || view === "colorized") {
      el.resultImg.src = base + view + ".png";
      el.resultImg.alt = view === "outline" ? "Разметка с номерами" : "Раскрашенный результат";
    } else {
      el.cmpUnder.src = base + "colorized.png"; el.cmpUnder.alt = "Раскраска";
      el.cmpOverImg.src = base + state.compareLeft + ".png";
      el.cmpOverImg.alt = state.compareLeft === "outline" ? "Разметка" : "Фото";
      setCompare(state.compareAt);
    }
  }
}
el.tabs.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-view]");
  if (b && !b.disabled) showView(b.dataset.view);
});

/* ---------- Сравнение ---------- */
function setCompare(p) {
  state.compareAt = Math.min(100, Math.max(0, p));
  el.cmpOver.style.clipPath = `inset(0 ${100 - state.compareAt}% 0 0)`;
  el.cmpHandle.style.left = state.compareAt + "%";
  el.cmpHandle.setAttribute("aria-valuenow", String(Math.round(state.compareAt)));
}
let sliding = false;
const slideTo = (e) => {
  const box = el.compare.getBoundingClientRect();
  setCompare(((e.clientX - box.left) / box.width) * 100);
};
el.compare.addEventListener("pointerdown", (e) => { sliding = true; el.compare.setPointerCapture(e.pointerId); slideTo(e); });
el.compare.addEventListener("pointermove", (e) => { if (sliding) slideTo(e); });
el.compare.addEventListener("pointerup", () => { sliding = false; });
el.cmpHandle.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") setCompare(state.compareAt - 3);
  if (e.key === "ArrowRight") setCompare(state.compareAt + 3);
});
el.compareMode.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-left]");
  if (!b) return;
  state.compareLeft = b.dataset.left;
  [...el.compareMode.querySelectorAll("button")].forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  showView("compare");
});

/* ---------- Расчёт ---------- */
function showError(msg) { el.error.textContent = msg; el.error.hidden = false; }
function hideError() { el.error.hidden = true; el.error.textContent = ""; }
function markStale() { el.stale.hidden = !state.job; }
function onSettingsChanged() { refreshSizeUi(); markStale(); }

el.nColors.addEventListener("input", () => { el.nOut.textContent = el.nColors.value; markStale(); });
el.detail.addEventListener("input", () => { el.detailOut.textContent = el.detail.value; markStale(); });
el.width.addEventListener("input", onSettingsChanged);
el.height.addEventListener("input", onSettingsChanged);

el.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.file || !canvasSize()) { showError("Выберите фото и укажите размер холста."); return; }
  hideError();
  const body = new FormData();
  body.append("photo", state.file);
  body.append("width_cm", el.width.value);
  body.append("height_cm", el.height.value);
  body.append("n_colors", el.nColors.value);
  body.append("detail", el.detail.value);
  body.append("fx", state.fx);
  body.append("fy", state.fy);

  el.run.disabled = true; el.run.textContent = "Считаю…"; el.busy.hidden = false;
  try {
    const resp = await fetch("/api/jobs", { method: "POST", body });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || "Сервис не ответил. Проверьте, что программа запущена.");
    const [w, h] = canvasSize();
    state.job = { ...data, ratio: w / h };
    renderResult();
  } catch (err) {
    showError(err instanceof TypeError ? "Нет связи с программой. Проверьте, что она запущена." : err.message);
  } finally {
    el.run.disabled = false; el.run.textContent = "Рассчитать"; el.busy.hidden = true;
  }
});

/* ---------- Показ результата ---------- */
function inkFor(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? "#1a1d2e" : "#ffffff";
}

function renderResult() {
  const j = state.job, base = `/api/jobs/${j.id}/`;
  [...el.tabs.children].forEach((b) => { b.disabled = false; });
  el.stale.hidden = true;

  el.facts.hidden = false;
  el.facts.replaceChildren();
  const addFact = (label, value, warn) => {
    const s = document.createElement("span");
    if (warn) s.className = "warn";
    s.append(label + " ");
    const b = document.createElement("b"); b.textContent = value; s.append(b);
    el.facts.append(s);
  };
  addFact("Пятен:", String(j.regions));
  addFact("Расчёт:", `${j.pixels[0]}×${j.pixels[1]} px`);
  addFact("Для печати:", `${j.print_dpi} dpi`, j.print_dpi < 100);
  addFact("Время:", `${j.seconds} с`);
  if (j.print_dpi < 100) {
    const p = document.createElement("span"); p.className = "warn";
    p.textContent = "На таком размере линии на печати будут мягкими.";
    el.facts.append(p);
  }

  el.downloads.hidden = false;
  $("dl-outline-pdf").href = base + "outline.pdf";
  $("dl-palette-pdf").href = base + "palette.pdf";
  $("dl-outline-png").href = base + "outline.png";
  $("dl-color-png").href = base + "colorized.png";

  el.chartEmpty.hidden = true; el.chartList.hidden = false; el.chartLegend.hidden = false;
  el.chartCount.textContent = `(${j.legend.length})`;
  el.chartList.replaceChildren(...j.legend.map(legendRow));

  showView("colorized");
}

function legendRow(entry) {
  const li = document.createElement("li");
  const dab = document.createElement("div");
  dab.className = "dab"; dab.textContent = entry.number;
  dab.style.background = entry.color; dab.style.color = inkFor(entry.color);

  const info = document.createElement("div");
  const bar = document.createElement("div"); bar.className = "mix-bar";
  entry.parts.forEach((p) => {
    const s = document.createElement("span");
    s.style.flex = `${p.percent} 1 0`; s.style.background = p.color; s.title = `${p.name} ${p.percent}%`;
    bar.append(s);
  });
  const text = document.createElement("div"); text.className = "mix-text";
  text.textContent = entry.parts.length === 1
    ? entry.parts[0].name
    : entry.parts.map((p) => `${p.name} ${p.percent}%`).join(" + ");
  const off = document.createElement("div");
  off.className = "off" + (entry.delta_e > 6 ? " bad" : "");
  off.textContent = `Отклонение ${String(entry.delta_e).replace(".", ",")}`;
  text.append(off);

  info.append(bar, text);
  li.append(dab, info);
  return li;
}

/* ---------- Запуск ---------- */
fetch("/api/info").then((r) => r.json()).then((info) => {
  state.maxOwnWidth = info.max_own_width_cm;
  el.nColors.min = info.n_colors_range[0]; el.nColors.max = Math.min(info.n_colors_range[1], 40);
  if (!info.palette_is_real) {
    el.paletteNote.hidden = false;
    el.paletteNote.textContent = `Расчёт идёт на временном наборе из ${info.palette_size} красок. Рецепты пока ориентировочные.`;
  }
  refreshSizeUi();
}).catch(() => { refreshSizeUi(); });
refreshSizeUi();
