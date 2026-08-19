const controls = {
  resultJson: document.querySelector("#resultJson"),
  resultJsonSelect: document.querySelector("#resultJsonSelect"),
  jsonOptionsPanel: document.querySelector("#jsonOptionsPanel"),
  jsonOptionsMeta: document.querySelector("#jsonOptionsMeta"),
  jsonOptionsList: document.querySelector("#jsonOptionsList"),
  projectDir: document.querySelector("#projectDir"),
  baseDir: document.querySelector("#baseDir"),
  lightType: document.querySelector("#lightType"),
  modelPredictionFilter: document.querySelector("#modelPredictionFilter"),
  keyword: document.querySelector("#keyword"),
  imageSearch: document.querySelector("#imageSearch"),
  numCol: document.querySelector("#numCol"),
  numRow: document.querySelector("#numRow"),
  scaleRatio: document.querySelector("#scaleRatio"),
  shuffleImages: document.querySelector("#shuffleImages"),
  jsonFirst: document.querySelector("#jsonFirst"),
  resultJsonClearCache: document.querySelector("#resultJsonClearCache"),
  page: document.querySelector("#page"),
  loadBtn: document.querySelector("#loadBtn"),
  importFile: document.querySelector("#importFile"),
  exportAnnotations: document.querySelector("#exportAnnotations"),
  exportImagePaths: document.querySelector("#exportImagePaths"),
  prevPage: document.querySelector("#prevPage"),
  nextPage: document.querySelector("#nextPage"),
  currentPageText: document.querySelector("#currentPageText"),
  remainingPagesText: document.querySelector("#remainingPagesText"),
  bulkDefaultActions: document.querySelector("#bulkDefaultActions"),
  bulkDefaultCorrect: document.querySelector("#bulkDefaultCorrect"),
  undoBulkDefault: document.querySelector("#undoBulkDefault"),
  bulkDefaultHint: document.querySelector("#bulkDefaultHint"),
  modelFilterButtons: document.querySelectorAll("[data-model-filter]"),
};

const STORAGE_KEYS = {
  resultJson: "inspection.resultJson",
  rememberedJsons: "inspection.rememberedJsons",
};

const gallery = document.querySelector("#gallery");
const statusBox = document.querySelector("#status");
const imageZoomModal = document.querySelector("#imageZoomModal");
const imageZoomImg = document.querySelector("#imageZoomImg");
const imageZoomMeta = document.querySelector("#imageZoomMeta");
const waferMapEls = {
  image: document.querySelector("#waferMapImage"),
  canvas: document.querySelector("#waferMapCanvas"),
  status: document.querySelector("#waferMapStatus"),
  toggle: document.querySelector("#waferMapToggle"),
  finalStatus: document.querySelector("#waferFinalStatus"),
  clearChipFilter: document.querySelector("#clearChipFilter"),
  panel: document.querySelector(".wafer-map-panel"),
};
const template = document.querySelector("#cardTemplate");
const statsEls = {
  meta: document.querySelector("#statsMeta"),
  tp: document.querySelector("#statTp"),
  fn: document.querySelector("#statFn"),
  fp: document.querySelector("#statFp"),
  tn: document.querySelector("#statTn"),
  fpr: document.querySelector("#falsePositiveRate"),
  fnr: document.querySelector("#falseNegativeRate"),
  hzFpr: document.querySelector("#hzFalsePositiveRate"),
  hzFnr: document.querySelector("#hzFalseNegativeRate"),
  byLight: document.querySelector("#lightStatsList"),
  refresh: document.querySelector("#refreshStats"),
};
let shuffleSeed = String(Date.now());
let waferMapState = {
  chips: [],
  selectedIndex: null,
  visibleIndices: [],
  sourceKey: "",
  collapsed: false,
};
let chipFilter = {
  mx: "",
  my: "",
};
let pageState = {
  page: 1,
  totalPages: 1,
};
let currentPageCards = [];
let confusionFilter = {
  cell: "",
  light: "",
};
let clearMatrixFilterButton = null;
let activeMatrixFilterText = null;
let pendingBulkUndoEntries = [];
let jsonOptionsFromTxt = [];
let rememberedJsons = [];
let imageRequestId = 0;
let statsRequestId = 0;
let chipNavigationTimer = null;

// One resize listener for the whole page is substantially cheaper than one
// listener per card, especially when a page contains dozens of cards.
window.addEventListener("resize", () => {
  for (const { node, state } of currentPageCards) {
    const img = node.querySelector("img");
    const canvas = node.querySelector("canvas");
    if (!img || !canvas || !img.complete) continue;
    syncCanvasSize(img, canvas);
    drawRegions(canvas, state);
  }
});

function nowIso() {
  return new Date().toISOString();
}

function setStatus(text, isError = false) {
  statusBox.textContent = text;
  statusBox.classList.toggle("error", isError);
}

function updatePager(page, totalPages) {
  pageState = { page, totalPages };
  if (controls.currentPageText) {
    controls.currentPageText.textContent = `${page} / ${totalPages}`;
  }
  if (controls.remainingPagesText) {
    controls.remainingPagesText.textContent = Math.max(0, totalPages - page);
  }
  if (controls.prevPage) {
    controls.prevPage.disabled = page <= 1;
  }
  if (controls.nextPage) {
    controls.nextPage.disabled = page >= totalPages;
  }
}

function updateBulkDefaultAction() {
  if (!controls.bulkDefaultActions || !controls.bulkDefaultCorrect) return;

  const modelFilter = controls.modelPredictionFilter.value;
  const untaggedCount = currentPageCards.filter(({ state }) => !state.savedTagged).length;
  const show = modelFilter === "合格品" || modelFilter === "缺陷品";
  controls.bulkDefaultActions.classList.toggle("hidden", !show);
  controls.bulkDefaultCorrect.disabled = !show || untaggedCount === 0;

  if (modelFilter === "缺陷品") {
    controls.bulkDefaultCorrect.textContent = "真实合格品已经全部找出，将其余图片默认标注为分类正确";
  } else if (modelFilter === "合格品") {
    controls.bulkDefaultCorrect.textContent = "真实缺陷品已经全部找出，将其余图片默认标注为分类正确";
  } else {
    controls.bulkDefaultCorrect.textContent = "";
  }

  if (controls.bulkDefaultHint) {
    controls.bulkDefaultHint.textContent = show
      ? `仅处理当前页未打标图片：${untaggedCount} 张。已打标图片不会覆盖。`
      : "";
  }
  if (controls.undoBulkDefault) {
    controls.undoBulkDefault.disabled = !show || !pendingBulkUndoEntries.length;
  }
}

function cloneAnnotationSnapshot(annotation) {
  if (!annotation) return null;
  return {
    verdict: annotation.verdict || "",
    greenDefect: Boolean(annotation.greenDefect),
    greenDefectRegions: Array.isArray(annotation.greenDefectRegions) ? annotation.greenDefectRegions.map((item) => ({ ...item })) : [],
    inferenceRemovedRegions: Array.isArray(annotation.inferenceRemovedRegions) ? annotation.inferenceRemovedRegions.map((item) => ({ ...item })) : [],
    detectionIssues: Array.isArray(annotation.detectionIssues) ? [...annotation.detectionIssues] : [],
    missRegions: Array.isArray(annotation.missRegions) ? annotation.missRegions.map((item) => ({ ...item })) : [],
    falseRegions: Array.isArray(annotation.falseRegions) ? annotation.falseRegions.map((item) => ({ ...item })) : [],
    note: annotation.note || "",
    imageName: annotation.imageName || "",
    updatedAt: annotation.updatedAt || "",
      inferenceRegions: Array.isArray(annotation.inferenceRegions) ? annotation.inferenceRegions.map((item) => ({ ...item })) : [],
  };
}

function buildSavedSnapshotFromState(state) {
  if (!isTagged(state)) return null;
  return cloneAnnotationSnapshot({
    verdict: state.verdict,
    greenDefect: state.greenDefect,
    greenDefectRegions: state.greenDefectRegions,
    inferenceRemovedRegions: state.inferenceRemovedRegions,
    detectionIssues: state.detectionIssues,
    missRegions: state.missRegions,
    falseRegions: state.falseRegions,
    note: state.note,
    imageName: state.item.name,
    updatedAt: nowIso(),
      inferenceRegions: state.inferenceRegions,
  });
}

async function restoreAnnotations(entries) {
  const response = await fetch("/api/annotations/restore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entries }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "撤销失败。");
  return data;
}

function isTagged(state) {
  return state.verdict === "分类正确" || state.verdict === "分类错误";
}

function formatRate(value) {
  return value == null ? "-" : `${(value * 100).toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function matrixButton(cell, value, light = "") {
  const lightAttr = escapeHtml(light);
  return `
    <button type="button" class="matrix-cell-button" data-confusion-cell="${cell}" data-confusion-light="${lightAttr}" title="筛选 ${light ? `${light} ` : ""}${cell}">
      <span class="matrix-label">${cell}</span>
      <strong>${value}</strong>
    </button>
  `;
}

function matrixFilterLabel() {
  if (!confusionFilter.cell) return "未启用矩阵筛选";
  return `矩阵筛选：${confusionFilter.light || "全部 light"} / ${confusionFilter.cell}`;
}

function updateConfusionFilterUi() {
  if (clearMatrixFilterButton) {
    clearMatrixFilterButton.disabled = !confusionFilter.cell;
  }
  if (activeMatrixFilterText) {
    activeMatrixFilterText.textContent = matrixFilterLabel();
    activeMatrixFilterText.classList.toggle("active", Boolean(confusionFilter.cell));
  }
  document.querySelectorAll("[data-confusion-cell]").forEach((button) => {
    const sameCell = button.dataset.confusionCell === confusionFilter.cell;
    const sameLight = (button.dataset.confusionLight || "") === (confusionFilter.light || "");
    button.classList.toggle("active", Boolean(confusionFilter.cell) && sameCell && sameLight);
  });
}

function applyConfusionFilter(cell, light = "") {
  confusionFilter = {
    cell: (cell || "").toUpperCase(),
    light: light || "",
  };
  controls.page.value = "1";
  updateConfusionFilterUi();
  loadImages();
}

function clearConfusionFilter() {
  confusionFilter = { cell: "", light: "" };
  controls.page.value = "1";
  updateConfusionFilterUi();
  loadImages();
}

function setupMatrixFilterUi() {
  [
    [statsEls.tp, "TP"],
    [statsEls.fn, "FN"],
    [statsEls.fp, "FP"],
    [statsEls.tn, "TN"],
  ].forEach(([valueEl, cell]) => {
    const td = valueEl?.closest("td");
    if (!td || td.querySelector(".matrix-cell-button")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "matrix-cell-button";
    button.dataset.confusionCell = cell;
    button.dataset.confusionLight = "";
    button.title = `筛选全部 light 的 ${cell}`;
    while (td.firstChild) {
      button.appendChild(td.firstChild);
    }
    td.appendChild(button);
  });

  const panel = document.querySelector(".stats-panel");
  const meta = statsEls.meta;
  if (panel && meta && !document.querySelector("#clearMatrixFilter")) {
    const filterBar = document.createElement("div");
    filterBar.className = "matrix-filter-bar";

    activeMatrixFilterText = document.createElement("span");
    activeMatrixFilterText.id = "activeMatrixFilter";

    clearMatrixFilterButton = document.createElement("button");
    clearMatrixFilterButton.id = "clearMatrixFilter";
    clearMatrixFilterButton.type = "button";
    clearMatrixFilterButton.textContent = "清除矩阵筛选";
    clearMatrixFilterButton.addEventListener("click", clearConfusionFilter);

    filterBar.append(activeMatrixFilterText, clearMatrixFilterButton);
    meta.insertAdjacentElement("afterend", filterBar);
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-confusion-cell]");
    if (!button) return;
    applyConfusionFilter(button.dataset.confusionCell, button.dataset.confusionLight || "");
  });

  updateConfusionFilterUi();
}

function statsBlock(label, stats) {
  const safeLabel = escapeHtml(label);
  return `
    <section class="light-stat-card">
      <div class="light-stat-title">${safeLabel}</div>
      <div class="light-stat-grid">
        ${matrixButton("TP", stats.tp, label)}
        ${matrixButton("FN", stats.fn, label)}
        ${matrixButton("FP", stats.fp, label)}
        ${matrixButton("TN", stats.tn, label)}
      </div>
      <div class="light-rate-row">
        <span>错检率 <strong>${formatRate(stats.falsePositiveRate)}</strong></span>
        <span>漏检率 <strong>${formatRate(stats.falseNegativeRate)}</strong></span>
      </div>
      <div class="light-rate-row">
        <span>HZ_错检率 <strong>${formatRate(stats.hzFalsePositiveRate)}</strong></span>
        <span>HZ_漏检率 <strong>${formatRate(stats.hzFalseNegativeRate)}</strong></span>
      </div>
      <div class="light-stat-meta">统计 ${stats.total} 条，跳过 ${stats.skipped} 条</div>
    </section>
  `;
}

async function loadStats() {
  if (!statsEls.meta) return;
  const requestId = ++statsRequestId;
  try {
    const response = await fetch("/api/stats");
    const data = await response.json();
    if (requestId !== statsRequestId) return;
    if (!response.ok) throw new Error(data.error || "统计加载失败。");

    statsEls.tp.textContent = data.tp;
    statsEls.fn.textContent = data.fn;
    statsEls.fp.textContent = data.fp;
    statsEls.tn.textContent = data.tn;
    statsEls.fpr.textContent = formatRate(data.falsePositiveRate);
    statsEls.fnr.textContent = formatRate(data.falseNegativeRate);
    statsEls.hzFpr.textContent = formatRate(data.hzFalsePositiveRate);
    statsEls.hzFnr.textContent = formatRate(data.hzFalseNegativeRate);
    statsEls.meta.textContent = `已统计 ${data.total} 条保存评价，跳过 ${data.skipped} 条未完成或无法解析记录。`;
    if (statsEls.byLight) {
      const entries = Object.entries(data.byLight || {});
      statsEls.byLight.innerHTML = entries.length
        ? entries.map(([lightType, stats]) => statsBlock(lightType, stats)).join("")
        : "暂无 light 统计。";
    }
    updateConfusionFilterUi();
  } catch (error) {
    statsEls.meta.textContent = error.message;
  }
}

function normalizeAnnotation(annotation = {}) {
  const verdictMap = {
    OK: "分类正确",
    NG: "分类错误",
    "分类正确": "分类正确",
    "分类错误": "分类错误",
  };
  const issues = Array.isArray(annotation.detectionIssues)
    ? annotation.detectionIssues
    : (annotation.defectType ? [annotation.defectType] : []);
  const missRegions = Array.isArray(annotation.missRegions) ? annotation.missRegions : [];
  const falseRegions = Array.isArray(annotation.falseRegions) ? annotation.falseRegions : [];
  const inferenceRegions = Array.isArray(annotation.inferenceRegions) ? annotation.inferenceRegions : [];
  const inferenceRemovedRegions = Array.isArray(annotation.inferenceRemovedRegions) ? annotation.inferenceRemovedRegions : [];
  if (missRegions.length && !issues.includes("漏检")) issues.push("漏检");
  if (falseRegions.length && !issues.includes("错检")) issues.push("错检");
  const verdict = verdictMap[annotation.verdict] || annotation.verdict || "";
  const greenDefectAllowed = verdict === "分类错误";
  return {
    verdict,
    greenDefect: greenDefectAllowed && Boolean(annotation.greenDefect),
    greenDefectRegions: greenDefectAllowed && Array.isArray(annotation.greenDefectRegions) ? annotation.greenDefectRegions : [],
    inferenceRegions,
    inferenceRemovedRegions,
    detectionIssues: issues.filter((issue, index, arr) => ["漏检", "错检"].includes(issue) && arr.indexOf(issue) === index),
    missRegions,
    falseRegions,
    note: annotation.note || "",
  };
}

function toPixelInferenceRegion(region, item) {
  if (!region || typeof region !== "object") return null;
  const rawX = Number(region.x);
  const rawY = Number(region.y);
  const rawW = Number(region.w);
  const rawH = Number(region.h);
  if (![rawX, rawY, rawW, rawH].every((value) => Number.isFinite(value))) return null;

  const looksNormalized = rawX >= 0 && rawY >= 0 && rawW >= 0 && rawH >= 0
    && rawX <= 1.000001 && rawY <= 1.000001 && rawW <= 1.000001 && rawH <= 1.000001;

  const width = Number(item?.width || 0);
  const height = Number(item?.height || 0);
  const x = looksNormalized ? rawX * width : rawX;
  const y = looksNormalized ? rawY * height : rawY;
  const w = looksNormalized ? rawW * width : rawW;
  const h = looksNormalized ? rawH * height : rawH;
  if (!(w > 0 && h > 0)) return null;

  return {
    ...region,
    x,
    y,
    w,
    h,
    label: String(region.label || region.note || region.className || "推理框"),
  };
}

function clampGalleryPageSize(value, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(12, Math.max(1, parsed));
}

function getParams() {
  const safeNumCol = clampGalleryPageSize(controls.numCol.value, 4);
  const safeNumRow = clampGalleryPageSize(controls.numRow.value, 10);
  controls.numCol.value = String(safeNumCol);
  controls.numRow.value = String(safeNumRow);

  return new URLSearchParams({
    result_json: controls.resultJson?.value.trim() || "",
    project_dir: controls.projectDir?.value.trim() || "",
    base_dir: controls.baseDir.value.trim(),
    light_type: controls.lightType.value,
    model_prediction: controls.modelPredictionFilter.value,
    keyword: controls.keyword.value.trim(),
    image_search: controls.imageSearch.value.trim(),
    num_col: String(safeNumCol),
    num_row: String(safeNumRow),
    scale_ratio: controls.scaleRatio.value,
    shuffle: controls.shuffleImages.checked ? "true" : "false",
    shuffle_seed: shuffleSeed,
    json_first: controls.jsonFirst.checked ? "true" : "false",
    confusion_cell: confusionFilter.cell,
    confusion_light: confusionFilter.light,
    final_status: waferMapEls.finalStatus?.value || "All",
    chip_mx: chipFilter.mx,
    chip_my: chipFilter.my,
    page: controls.page.value || "1",
  });
}

function updateImagePathExportLink() {
  if (!controls.exportImagePaths) return;
  const params = getParams();
  // Export is intentionally independent of pagination, shuffle, and ordering.
  params.delete("page");
  params.delete("shuffle");
  params.delete("shuffle_seed");
  params.delete("json_first");
  controls.exportImagePaths.href = `/api/export-image-paths?${params.toString()}`;
}

function updateAnnotationsExportLink() {
  if (!controls.exportAnnotations) return;
  const params = new URLSearchParams({
    result_json: controls.resultJson?.value.trim() || "",
    project_dir: controls.projectDir?.value.trim() || "",
  });
  controls.exportAnnotations.href = `/api/annotations?${params.toString()}`;
}

function uniqueNonEmptyStrings(values) {
  const seen = new Set();
  const result = [];
  for (const value of values || []) {
    const text = String(value || "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    result.push(text);
  }
  return result;
}

function loadRememberedJsons() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEYS.rememberedJsons) || "[]");
    rememberedJsons = uniqueNonEmptyStrings(parsed).slice(0, 50);
  } catch {
    rememberedJsons = [];
  }
}

function saveRememberedJsons() {
  localStorage.setItem(STORAGE_KEYS.rememberedJsons, JSON.stringify(rememberedJsons.slice(0, 50)));
}

function rememberJsonPath(path) {
  const text = String(path || "").trim();
  if (!text) return;
  rememberedJsons = [text, ...rememberedJsons.filter((item) => item !== text)].slice(0, 50);
  saveRememberedJsons();
  localStorage.setItem(STORAGE_KEYS.resultJson, text);
}

function mergeTxtAndRememberedOptions(txtItems = []) {
  const byPath = new Map();
  for (const item of txtItems) {
    if (!item?.path) continue;
    byPath.set(item.path, {
      path: item.path,
      name: item.name || "",
      exists: Boolean(item.exists),
      fromRemembered: false,
    });
  }

  for (const rememberedPath of rememberedJsons) {
    if (!byPath.has(rememberedPath)) {
      const chunks = rememberedPath.split("/");
      byPath.set(rememberedPath, {
        path: rememberedPath,
        name: chunks[chunks.length - 1] || rememberedPath,
        exists: true,
        fromRemembered: true,
      });
    }
  }

  const merged = Array.from(byPath.values());
  merged.sort((a, b) => {
    if (a.fromRemembered !== b.fromRemembered) return a.fromRemembered ? -1 : 1;
    if (a.exists !== b.exists) return a.exists ? -1 : 1;
    return a.path.localeCompare(b.path);
  });
  return merged;
}

function refreshResultJsonSelect() {
  if (!controls.resultJsonSelect) return;

  const currentPath = controls.resultJson?.value.trim() || "";
  const items = getResultJsonSelectItems();
  const seen = new Set();
  const fragment = document.createDocumentFragment();

  const manualOption = document.createElement("option");
  manualOption.value = "";
  manualOption.textContent = "手动输入";
  fragment.appendChild(manualOption);

  items.forEach((item) => {
    const path = item.path || "";
    if (!path || seen.has(path)) return;
    seen.add(path);

    const option = document.createElement("option");
    option.value = path;
    option.textContent = item.name || path;
    if (path === currentPath) {
      option.selected = true;
    }
    fragment.appendChild(option);
  });

  controls.resultJsonSelect.innerHTML = "";
  controls.resultJsonSelect.appendChild(fragment);
  controls.resultJsonSelect.value = currentPath && seen.has(currentPath) ? currentPath : "";
}

function getResultJsonSelectItems() {
  const currentPath = controls.resultJson?.value.trim() || "";
  const items = mergeTxtAndRememberedOptions(jsonOptionsFromTxt);
  if (currentPath && !items.some((item) => item.path === currentPath)) {
    const chunks = currentPath.split("/");
    items.unshift({
      path: currentPath,
      name: chunks[chunks.length - 1] || currentPath,
      exists: true,
      fromRemembered: false,
    });
  }
  return items;
}

function renderJsonOptions(items = []) {
  if (!controls.jsonOptionsPanel || !controls.jsonOptionsList || !controls.jsonOptionsMeta) return;

  controls.jsonOptionsList.innerHTML = "";
  if (!items.length) {
    controls.jsonOptionsPanel.hidden = true;
    return;
  }

  const fragment = document.createDocumentFragment();
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "json-option-btn";
    if (!item.exists) {
      button.classList.add("missing");
    }
    button.innerHTML = `<strong>${escapeHtml(item.name || "(unknown)")}</strong><span>${escapeHtml(item.path || "")}</span>`;
    button.disabled = !item.exists;
    button.addEventListener("click", () => {
      controls.resultJson.value = item.path || "";
      controls.page.value = "1";
      shuffleSeed = String(Date.now());
      rememberJsonPath(item.path || "");
      updateImagePathExportLink();
      updateAnnotationsExportLink();
      loadImages();
    });
    fragment.appendChild(button);
  });
  controls.jsonOptionsList.appendChild(fragment);

  const rememberedCount = items.filter((item) => item.fromRemembered).length;
  controls.jsonOptionsMeta.textContent = `候选 JSON：${items.length} 条（记忆 ${rememberedCount} 条）`;
  controls.jsonOptionsPanel.hidden = false;
}

function refreshJsonOptionsPanel() {
  renderJsonOptions(mergeTxtAndRememberedOptions(jsonOptionsFromTxt));
  refreshResultJsonSelect();
}

async function loadJsonOptionsFromTxt({ silentIfEmpty = false } = {}) {

  try {
    if (!silentIfEmpty) {
      setStatus("正在自动读取 JSON 候选列表...");
    }
    const params = new URLSearchParams({
      project_dir: controls.projectDir?.value.trim() || "",
    });
    const response = await fetch(`/api/json-options?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "读取 JSON 候选失败。");
    }
    jsonOptionsFromTxt = data.items || [];
    refreshJsonOptionsPanel();
    if (data.warning && !silentIfEmpty) {
      setStatus(data.warning, true);
      return;
    }
    if (!silentIfEmpty) {
      setStatus(`已自动加载 JSON 候选：${data.count || 0} 条。`);
    }
  } catch (error) {
    jsonOptionsFromTxt = [];
    refreshJsonOptionsPanel();
    if (!silentIfEmpty) {
      setStatus(error.message, true);
    }
  }
}

function restoreJsonPreference() {
  loadRememberedJsons();
  const rememberedResultJson = localStorage.getItem(STORAGE_KEYS.resultJson) || "";
  if (controls.resultJson && rememberedResultJson && !controls.resultJson.value.trim()) {
    controls.resultJson.value = rememberedResultJson;
  }
}

function hasChipFilter() {
  return chipFilter.mx !== "" && chipFilter.my !== "";
}

function updateChipFilterUi() {
  if (!waferMapEls.clearChipFilter) return;
  waferMapEls.clearChipFilter.disabled = !hasChipFilter();
  waferMapEls.clearChipFilter.textContent = hasChipFilter()
    ? `清除 chip 定位 (${chipFilter.mx}, ${chipFilter.my})`
    : "清除 chip 定位";
}

function selectWaferChip(index, { debounceGallery = false } = {}) {
  const chip = waferMapState.chips[index];
  if (!chip) return;

  chipFilter = { mx: String(chip.mx), my: String(chip.my) };
  waferMapState.selectedIndex = index;
  controls.page.value = "1";
  updateChipFilterUi();
  drawWaferMapPoints();

  if (chipNavigationTimer) {
    clearTimeout(chipNavigationTimer);
    chipNavigationTimer = null;
  }
  if (debounceGallery) {
    chipNavigationTimer = window.setTimeout(() => {
      chipNavigationTimer = null;
      loadImages();
    }, 120);
  } else {
    loadImages();
  }
}

function findAdjacentChip(direction) {
  const currentIndex = waferMapState.selectedIndex;
  const current = waferMapState.chips[currentIndex];
  if (!current || !Number.isFinite(current.x) || !Number.isFinite(current.y)) return null;

  let bestIndex = null;
  let bestScore = Number.POSITIVE_INFINITY;
  waferMapState.visibleIndices.forEach((index) => {
    if (index === currentIndex) return;
    const candidate = waferMapState.chips[index];
    if (!candidate || !Number.isFinite(candidate.x) || !Number.isFinite(candidate.y)) return;

    const deltaX = candidate.x - current.x;
    const deltaY = candidate.y - current.y;
    let forwardDistance;
    let perpendicularDistance;
    if (direction === "ArrowLeft") {
      if (deltaX >= 0) return;
      forwardDistance = -deltaX;
      perpendicularDistance = Math.abs(deltaY);
    } else if (direction === "ArrowRight") {
      if (deltaX <= 0) return;
      forwardDistance = deltaX;
      perpendicularDistance = Math.abs(deltaY);
    } else if (direction === "ArrowUp") {
      if (deltaY >= 0) return;
      forwardDistance = -deltaY;
      perpendicularDistance = Math.abs(deltaX);
    } else if (direction === "ArrowDown") {
      if (deltaY <= 0) return;
      forwardDistance = deltaY;
      perpendicularDistance = Math.abs(deltaX);
    } else {
      return;
    }

    // Prioritize the same row/column; when the wafer edge has no exact grid
    // neighbor, fall back to the closest chip in the requested direction.
    const score = perpendicularDistance * 10000 + forwardDistance;
    if (score < bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function drawWaferMapPoints() {
  if (!waferMapEls.canvas) return;
  const canvas = waferMapEls.canvas;
  const parent = canvas.parentElement;
  if (!parent) return;

  const rect = parent.getBoundingClientRect();
  const width = Math.max(280, Math.floor(rect.width || parent.clientWidth || 640));
  const height = Math.max(280, Math.floor(rect.height || parent.clientHeight || 640));
  const ratio = window.devicePixelRatio || 1;

  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const chips = waferMapState.chips || [];
  const finalStatus = (waferMapEls.finalStatus?.value || "All").toUpperCase();
  const visibleIndices = [];
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.42;

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
  ctx.strokeStyle = "#dfe5ee";
  ctx.lineWidth = 2;
  ctx.stroke();

  if (!chips.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("等待加载 chip 坐标…", centerX, centerY);
    return;
  }

  const validChips = chips.filter((chip) => (
    Number.isFinite(chip.x)
    && Number.isFinite(chip.y)
    && (finalStatus === "ALL" || String(chip.status || "").toUpperCase() === finalStatus)
  ));
  const validX = validChips.map((chip) => chip.x);
  const validY = validChips.map((chip) => chip.y);
  if (!validX.length || !validY.length) {
    ctx.fillStyle = "#667085";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("当前结果中没有可用的 chip 坐标", centerX, centerY);
    return;
  }

  const minX = Math.min(...validX); const maxX = Math.max(...validX);
  const minY = Math.min(...validY); const maxY = Math.max(...validY);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  // 48AMA chips are about five times wider than tall. X/Y are grid counts,
  // so apply this physical chip aspect before fitting the complete map. The
  // source CSV spans X=81 and Y=451; without this adjustment it becomes an
  // incorrectly narrow vertical strip in the otherwise square wafer view.
  const chipAspect = 5;
  const mapScale = Math.min(
    (radius * 1.72) / (rangeX * chipAspect),
    (radius * 1.72) / rangeY,
  );
  const pointPaths = {
    NG: new Path2D(),
    OK: new Path2D(),
    UNKNOWN: new Path2D(),
  };
  let selectedPoint = null;
  // Draw every chip as its physical rectangular cell. Fixed-size circles
  // overlapped adjacent rows, and scan-order sampling formed false triangles.
  const cellWidth = Math.max(1, chipAspect * mapScale * 0.86);
  const cellHeight = Math.max(1, mapScale * 0.86);

  chips.forEach((chip, index) => {
    if (!Number.isFinite(chip.x) || !Number.isFinite(chip.y)) return;
    if (finalStatus !== "ALL" && String(chip.status || "").toUpperCase() !== finalStatus) return;
    const x = centerX + (chip.x - (minX + maxX) / 2) * chipAspect * mapScale;
    const y = centerY + (chip.y - (minY + maxY) / 2) * mapScale;
    chip.pixelX = x;
    chip.pixelY = y;
    visibleIndices.push(index);

    const chipStatus = String(chip.status || "").toUpperCase();
    if (index === waferMapState.selectedIndex) {
      selectedPoint = { x, y, status: chipStatus };
      return;
    }
    const path = pointPaths[chipStatus] || pointPaths.UNKNOWN;
    path.rect(x - cellWidth / 2, y - cellHeight / 2, cellWidth, cellHeight);
  });

  ctx.fillStyle = "#ef4444";
  ctx.fill(pointPaths.NG);
  ctx.fillStyle = "#22c55e";
  ctx.fill(pointPaths.OK);
  ctx.fillStyle = "#f59e0b";
  ctx.fill(pointPaths.UNKNOWN);
  if (selectedPoint) {
    ctx.fillStyle = selectedPoint.status === "NG" ? "#ef4444" : selectedPoint.status === "OK" ? "#22c55e" : "#f59e0b";
    ctx.beginPath();
    ctx.arc(selectedPoint.x, selectedPoint.y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#111827";
    ctx.beginPath();
    ctx.arc(selectedPoint.x, selectedPoint.y, 11, 0, Math.PI * 2);
    ctx.stroke();
  }
  waferMapState.visibleIndices = visibleIndices;
}

async function refreshWaferMapToggle() {
  if (!waferMapEls.panel || !waferMapEls.toggle) return;
  waferMapEls.panel.classList.toggle("collapsed", Boolean(waferMapState.collapsed));
  waferMapEls.toggle.textContent = waferMapState.collapsed ? "展开" : "收起";
  waferMapEls.toggle.setAttribute("aria-expanded", String(!waferMapState.collapsed));
}

function closeImageZoom() {
  if (imageZoomModal) imageZoomModal.classList.add("hidden");
  if (imageZoomImg) imageZoomImg.removeAttribute("src");
  if (imageZoomMeta) imageZoomMeta.textContent = "";
}

function toggleWaferMapPanel() {
  waferMapState.collapsed = !waferMapState.collapsed;
  refreshWaferMapToggle();
}

function waferMapSourceKey() {
  return [
    controls.resultJson?.value.trim() || "",
    controls.projectDir?.value.trim() || "",
  ].join("\u0000");
}

function renderWaferMapData(data) {
  waferMapState.chips = Array.isArray(data.chips) ? data.chips : [];
  waferMapState.selectedIndex = waferMapState.chips.findIndex((chip) => (
    hasChipFilter() && String(chip.mx) === chipFilter.mx && String(chip.my) === chipFilter.my
  ));
  if (waferMapEls.image) {
    waferMapEls.image.removeAttribute("src");
    waferMapEls.image.style.display = "none";
    waferMapEls.image.alt = data.productName ? `${data.productName} wafer map` : "wafer map";
  }
  if (waferMapEls.status) {
    if (data.chipCount) {
      const mapStatus = waferMapEls.finalStatus?.value || "All";
      const visibleCount = mapStatus === "All"
        ? data.chipCount
        : waferMapState.chips.filter((chip) => chip.status === mapStatus).length;
      const chipHint = hasChipFilter() ? ` | 已定位 chip (${chipFilter.mx}, ${chipFilter.my})` : "";
      waferMapEls.status.textContent = `显示 ${visibleCount} / ${data.chipCount || 0} 个 chip${chipHint}`;
    } else {
      waferMapEls.status.textContent = "当前结果 JSON 未生成可用 chip 坐标";
    }
  }
  updateChipFilterUi();
  drawWaferMapPoints();
}

async function loadWaferMap() {
  const currentResultJson = controls.resultJson?.value.trim() || "";
  if (!currentResultJson) {
    waferMapState.chips = [];
    waferMapState.selectedIndex = null;
    waferMapState.sourceKey = "";
    if (waferMapEls.image) waferMapEls.image.removeAttribute("src");
    if (waferMapEls.status) waferMapEls.status.textContent = "等待加载结果 JSON。";
    updateChipFilterUi();
    if (waferMapEls.canvas) drawWaferMapPoints();
    return;
  }

  const sourceKey = waferMapSourceKey();
  if (waferMapState.sourceKey === sourceKey && waferMapState.chips.length) {
    renderWaferMapData({
      productName: currentResultJson.split("/").pop() || "",
      chipCount: waferMapState.chips.length,
      chips: waferMapState.chips,
    });
    return;
  }

  try {
    const params = new URLSearchParams({
      result_json: currentResultJson,
      project_dir: controls.projectDir?.value.trim() || "",
    });
    const response = await fetch(`/api/wafer-map?${params.toString()}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "wafer map 加载失败。");
    }

    waferMapState.sourceKey = sourceKey;
    renderWaferMapData(data);
  } catch (error) {
    if (waferMapEls.status) {
      waferMapEls.status.textContent = error.message;
    }
  }
}

if (waferMapEls.toggle) {
  waferMapEls.toggle.addEventListener("click", toggleWaferMapPanel);
}

if (waferMapEls.finalStatus) {
  waferMapEls.finalStatus.addEventListener("change", () => {
    chipFilter = { mx: "", my: "" };
    controls.page.value = "1";
    updateChipFilterUi();
    // Redraw first so OK/green points disappear immediately; the following
    // gallery request then applies the same final-result constraint to images.
    drawWaferMapPoints();
    loadImages();
  });
}

if (waferMapEls.clearChipFilter) {
  waferMapEls.clearChipFilter.addEventListener("click", () => {
    chipFilter = { mx: "", my: "" };
    controls.page.value = "1";
    updateChipFilterUi();
    loadImages();
  });
}

if (imageZoomModal) {
  imageZoomModal.addEventListener("click", (event) => {
    if (event.target.matches("[data-close-zoom]")) {
      closeImageZoom();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeImageZoom();
    }
  });
}

if (waferMapEls.canvas) {
  waferMapEls.canvas.addEventListener("click", (event) => {
    if (waferMapState.collapsed || !waferMapState.chips.length) return;
    waferMapEls.canvas.focus({ preventScroll: true });
    const rect = waferMapEls.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let nearestIndex = null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    waferMapState.visibleIndices.forEach((index) => {
      const chip = waferMapState.chips[index];
      if (!Number.isFinite(chip.pixelX) || !Number.isFinite(chip.pixelY)) return;
      const dx = chip.pixelX - x;
      const dy = chip.pixelY - y;
      const dist = dx * dx + dy * dy;
      if (dist < nearestDistance) {
        nearestIndex = index;
        nearestDistance = dist;
      }
    });
    if (nearestIndex == null) return;
    selectWaferChip(nearestIndex);
  });

  waferMapEls.canvas.addEventListener("keydown", (event) => {
    if (![
      "ArrowLeft",
      "ArrowRight",
      "ArrowUp",
      "ArrowDown",
    ].includes(event.key)) return;
    if (waferMapState.collapsed || waferMapState.selectedIndex == null) return;

    const nextIndex = findAdjacentChip(event.key);
    if (nextIndex == null) return;
    event.preventDefault();
    selectWaferChip(nextIndex, { debounceGallery: true });
  });
}

async function loadImages() {
  const requestId = ++imageRequestId;
  const currentResultJson = controls.resultJson?.value.trim() || "";
  if (currentResultJson) {
    rememberJsonPath(currentResultJson);
  }

  setStatus("正在加载图片...");
  gallery.innerHTML = "";
  currentPageCards = [];
  gallery.style.setProperty("--cols", controls.numCol.value || "2");

  try {
    const response = await fetch(`/api/images?${getParams().toString()}`);
    const data = await response.json();
    if (requestId !== imageRequestId) return;
    if (!response.ok) {
      throw new Error(data.error || "图片加载失败。");
    }

    controls.page.value = data.page;
    updatePager(data.page, data.totalPages);
    const scale = Number(controls.scaleRatio.value || 1);
    const preferredCols = clampGalleryPageSize(controls.numCol.value, 4);
    if (preferredCols >= 8) {
      gallery.style.gridTemplateColumns = `repeat(auto-fit, minmax(${Math.max(220, Math.round(data.batchWidth * scale))}px, 1fr))`;
    } else {
      gallery.style.gridTemplateColumns = `repeat(${Math.max(1, preferredCols)}, minmax(280px, ${Math.round(data.batchWidth * scale)}px))`;
    }
    const shuffleText = data.shuffle ? ` | 当前页 Shuffle 开启` : " | 当前页 Shuffle 关闭";
    const jsonFirstText = data.jsonFirst ? " | 当前页 JSON优先" : "";
    const searchText = data.imageSearch ? ` | 查找：${data.imageSearch}` : "";
    const modelFilterText = data.modelPredictionFilter && data.modelPredictionFilter !== "All" ? ` | 模型判定：${data.modelPredictionFilter}` : "";
    const matrixFilterText = data.confusionCell ? ` | 矩阵筛选：${data.confusionLight || "全部 light"} / ${data.confusionCell}` : "";
    const cacheText = data.cacheCleared ? " | 已清理推理缓存" : "";
    setStatus(`状态：总计 ${data.total} 张 | 当前第 ${data.page} / ${data.totalPages} 页 | 当前 Batch 分辨率 ${data.batchWidth}x${data.batchHeight}${shuffleText}${jsonFirstText}${modelFilterText}${searchText}${matrixFilterText}${cacheText} | 数据源 ${data.baseDir || "-"}`);
    // Build the whole page off-DOM, then attach it once to avoid repeated
    // layout/repaint work when a page contains many cards.
    const fragment = document.createDocumentFragment();
    data.items.forEach((item) => renderCard(item, fragment));
    gallery.appendChild(fragment);
    updateBulkDefaultAction();
    await loadWaferMap();
  } catch (error) {
    if (requestId !== imageRequestId) return;
    setStatus(error.message, true);
    updateBulkDefaultAction();
  }
}

async function clearSavedEvaluationsAndReload() {
  if (!controls.resultJsonClearCache) return;
  controls.resultJsonClearCache.disabled = true;
  controls.loadBtn.disabled = true;
  setStatus("正在清空历史评价...");

  // Invalidate any image/stat responses that started before the clear.
  imageRequestId += 1;
  statsRequestId += 1;
  try {
    const response = await fetch("/api/annotations/clear", { method: "POST" });
    const data = await response.json();
    if (!response.ok || data.total !== 0) {
      throw new Error(data.error || "历史评价清理失败。");
    }
    controls.page.value = "1";
    controls.resultJsonClearCache.checked = false;
    await Promise.all([loadImages(), loadStats()]);
    setStatus(`已清空历史评价并重新加载。评价文件：${data.annotationsFile || "-"}`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    controls.resultJsonClearCache.disabled = false;
    controls.loadBtn.disabled = false;
  }
}

function renderCard(item, container = gallery) {
  const node = template.content.firstElementChild.cloneNode(true);
  const img = node.querySelector("img");
  const canvas = node.querySelector("canvas");
  const wrap = node.querySelector(".image-wrap");
  const stateLabel = node.querySelector(".save-state");
  const noteInput = node.querySelector("textarea");

  const annotation = normalizeAnnotation(item.annotation);
  const annotationInferenceBase = Array.isArray(annotation.inferenceRegions)
    ? annotation.inferenceRegions
      .map((region) => toPixelInferenceRegion(region, item))
      .filter(Boolean)
    : [];
  const state = {
    item,
    verdict: annotation.verdict,
    savedTagged: annotation.verdict === "分类正确" || annotation.verdict === "分类错误",
    greenDefect: annotation.greenDefect,
    greenDefectRegions: annotation.greenDefectRegions,
    baseInferenceRegions: annotationInferenceBase.length
      ? annotationInferenceBase
      : (Array.isArray(item?.predictionOverlay?.detection) ? item.predictionOverlay.detection.map((region) => ({ ...region })) : []),
    inferenceRemovedRegions: Array.isArray(annotation.inferenceRemovedRegions) ? annotation.inferenceRemovedRegions.map((region) => ({ ...region })) : [],
    inferenceRegions: [],
    detectionIssues: annotation.detectionIssues,
    missRegions: annotation.missRegions,
    falseRegions: annotation.falseRegions,
    activeIssue: annotation.greenDefect ? "绿色缺陷" : (annotation.detectionIssues[0] || ""),
    note: annotation.note,
    drawing: false,
    start: null,
    draft: null,
    imageQuality: "thumbnail",
    serverSnapshot: isTagged({ verdict: annotation.verdict }) ? cloneAnnotationSnapshot(annotation) : null,
    undoSnapshot: null,
  };
  syncInferenceRegionsFromState(state);
  updateTaggedState(node, state, stateLabel);

  node.querySelector(".image-name").textContent = item.name;
  const modelPrediction = node.querySelector("[data-model-prediction]");
  const modelPredictionBar = node.querySelector(".model-prediction");
  if (modelPrediction) {
    modelPrediction.textContent = item.modelPrediction || "未知";
  }
  if (modelPredictionBar) {
    modelPredictionBar.classList.remove("qualified", "defective");
    if (item.modelPrediction === "合格品") {
      modelPredictionBar.classList.add("qualified");
    } else if (item.modelPrediction === "缺陷品") {
      modelPredictionBar.classList.add("defective");
    }
  }
  // Regions are always stored in source-image pixels. Force the preview box
  // to the source aspect ratio so thumbnail resize rounding cannot introduce
  // even a sub-pixel vertical offset in overlays or pointer coordinates.
  if (item.width > 0 && item.height > 0) {
    wrap.style.aspectRatio = `${item.width} / ${item.height}`;
    img.style.height = "100%";
    img.style.objectFit = "fill";
  }
  const qualityState = node.querySelector(".image-quality-state");
  const loadOriginalButton = node.querySelector(".load-original-btn");
  const thumbnailUrl = item.thumbnailUrl || item.imageUrl;
  img.loading = "lazy";
  img.decoding = "async";
  img.src = thumbnailUrl;
  img.alt = item.name;
  noteInput.value = state.note;
  const undoSaveButton = node.querySelector('[data-action="undo-save"]');

  img.addEventListener("load", () => {
    syncCanvasSize(img, canvas);
    drawRegions(canvas, state);
    if (state.imageQuality === "original") {
      if (qualityState) qualityState.textContent = "原图";
      if (loadOriginalButton) {
        loadOriginalButton.textContent = "已加载原图";
        loadOriginalButton.disabled = true;
      }
    }
  });
  img.addEventListener("error", () => {
    if (state.imageQuality !== "thumbnail" || !item.imageUrl) return;
    state.imageQuality = "original";
    if (qualityState) qualityState.textContent = "原图（缩略图加载失败）";
    if (loadOriginalButton) {
      loadOriginalButton.textContent = "已加载原图";
      loadOriginalButton.disabled = true;
    }
    img.src = item.imageUrl;
  });
  if (loadOriginalButton) {
    loadOriginalButton.addEventListener("click", () => {
      if (state.imageQuality === "original") return;
      state.imageQuality = "original";
      loadOriginalButton.disabled = true;
      loadOriginalButton.textContent = "正在加载原图...";
      if (qualityState) qualityState.textContent = "正在切换原图";
      img.src = item.imageUrl;
    });
  }
  node.querySelectorAll("[data-verdict]").forEach((button) => {
    button.addEventListener("click", () => {
      state.verdict = button.dataset.verdict;
      if (state.verdict !== "分类错误") {
        state.greenDefect = false;
        state.greenDefectRegions = [];
        if (state.activeIssue === "绿色缺陷") {
          state.activeIssue = state.detectionIssues[0] || "";
        }
      }
      updateButtons(node, state);
      clearVerdictWarning(node, state, stateLabel);
      updateTaggedState(node, state, stateLabel);
      drawRegions(canvas, state);
    });
  });

  node.querySelectorAll("[data-green-defect]").forEach((button) => {
    button.addEventListener("click", () => {
      state.greenDefect = !state.greenDefect;
      state.activeIssue = state.greenDefect ? "绿色缺陷" : (state.detectionIssues[0] || "");
      updateButtons(node, state);
      drawRegions(canvas, state);
    });
  });

  node.querySelectorAll("[data-issue]").forEach((button) => {
    button.addEventListener("click", () => {
      const issue = button.dataset.issue;
      if (state.detectionIssues.includes(issue) && state.activeIssue === issue) {
        state.detectionIssues = state.detectionIssues.filter((item) => item !== issue);
        state.activeIssue = state.detectionIssues[0] || "";
      } else if (state.detectionIssues.includes(issue)) {
        state.activeIssue = issue;
      } else {
        state.detectionIssues.push(issue);
        state.activeIssue = issue;
      }
      updateButtons(node, state);
      drawRegions(canvas, state);
    });
  });

  node.querySelector('[data-action="undo"]').addEventListener("click", () => {
    const popped = getActiveRegions(state).pop();
    if (state.activeIssue === "错检" && popped?.source === "converted_from_inference") {
      const targetSig = regionSignature({
        x: popped.x,
        y: popped.y,
        w: popped.w,
        h: popped.h,
        label: popped.note || "",
      });
      const index = state.inferenceRemovedRegions.findIndex((region) => regionSignature(region) === targetSig);
      if (index >= 0) {
        state.inferenceRemovedRegions.splice(index, 1);
        syncInferenceRegionsFromState(state);
      }
    }
    drawRegions(canvas, state);
  });

  node.querySelector('[data-action="clear"]').addEventListener("click", () => {
    if (state.activeIssue === "漏检") {
      state.missRegions = [];
    } else if (state.activeIssue === "错检") {
      state.falseRegions = [];
      state.inferenceRemovedRegions = [];
      syncInferenceRegionsFromState(state);
    } else if (state.activeIssue === "绿色缺陷") {
      state.greenDefectRegions = [];
    }
    drawRegions(canvas, state);
  });

  canvas.addEventListener("pointerdown", (event) => {
    if (state.activeIssue === "错检") {
      const converted = convertInferenceRegionToFalse(event, canvas, state);
      if (converted) {
        drawRegions(canvas, state);
        return;
      }
    }
    if (!canDrawActiveIssue(state)) return;
    canvas.setPointerCapture(event.pointerId);
    state.drawing = true;
    state.start = eventToImagePoint(event, canvas, item);
    state.draft = null;
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.drawing || !canDrawActiveIssue(state)) return;
    const current = eventToImagePoint(event, canvas, item);
    state.draft = rectFromPoints(state.start, current, item);
    drawRegions(canvas, state);
  });

  canvas.addEventListener("pointerup", () => {
    if (!state.drawing) return;
    state.drawing = false;
    if (state.draft && state.draft.w >= 3 && state.draft.h >= 3) {
      getActiveRegions(state).push(state.draft);
    }
    state.draft = null;
    drawRegions(canvas, state);
  });

  node.querySelector(".save-btn").addEventListener("click", async () => {
    state.note = noteInput.value.trim();
    if (!isTagged(state)) {
      stateLabel.textContent = "请先选择分类正确或分类错误";
      stateLabel.classList.remove("saved");
      stateLabel.classList.add("error");
      node.classList.add("needs-verdict");
      return;
    }

    stateLabel.textContent = "保存中...";
    stateLabel.classList.remove("saved");
    stateLabel.classList.remove("error");
    node.classList.remove("needs-verdict");
    if (undoSaveButton) {
      undoSaveButton.disabled = true;
    }
    try {
      const previousSnapshot = cloneAnnotationSnapshot(state.serverSnapshot);
      const response = await fetch("/api/annotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          originalPath: item.originalPath,
          modelPrediction: item.modelPrediction || "",
          verdict: state.verdict,
          greenDefect: state.greenDefect,
          greenDefectRegions: state.greenDefectRegions,
          inferenceRegions: state.inferenceRegions,
          inferenceRemovedRegions: state.inferenceRemovedRegions,
          detectionIssues: state.detectionIssues,
          missRegions: state.missRegions,
          falseRegions: state.falseRegions,
          note: state.note,
          updatedAt: nowIso(),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "保存失败。");
      state.savedTagged = true;
      state.serverSnapshot = cloneAnnotationSnapshot(data.annotation);
      state.undoSnapshot = previousSnapshot;
      stateLabel.textContent = "已保存";
      stateLabel.classList.add("saved");
      if (undoSaveButton) {
        undoSaveButton.disabled = false;
      }
      updateTaggedState(node, state, stateLabel);
      updateBulkDefaultAction();
      await loadStats();
    } catch (error) {
      stateLabel.textContent = error.message;
      stateLabel.classList.remove("saved");
      stateLabel.classList.add("error");
      if (undoSaveButton) {
        undoSaveButton.disabled = state.undoSnapshot === null || state.undoSnapshot === undefined;
      }
    }
  });

  if (undoSaveButton) {
    undoSaveButton.addEventListener("click", async () => {
      if (undoSaveButton.disabled || state.undoSnapshot === undefined) return;
      const restoreTarget = state.undoSnapshot === null ? null : cloneAnnotationSnapshot(state.undoSnapshot);
      stateLabel.textContent = "撤销中...";
      stateLabel.classList.remove("error");
      undoSaveButton.disabled = true;
      try {
        await restoreAnnotations([{ originalPath: item.originalPath, annotation: restoreTarget }]);

        const restored = normalizeAnnotation(restoreTarget || {});
        state.verdict = restored.verdict;
        state.greenDefect = restored.greenDefect;
        state.greenDefectRegions = restored.greenDefectRegions;
        if (Array.isArray(restored.inferenceRegions) && restored.inferenceRegions.length) {
          state.baseInferenceRegions = restored.inferenceRegions
            .map((region) => toPixelInferenceRegion(region, state.item))
            .filter(Boolean);
        }
        state.inferenceRemovedRegions = restored.inferenceRemovedRegions;
        syncInferenceRegionsFromState(state);
        state.detectionIssues = restored.detectionIssues;
        state.missRegions = restored.missRegions;
        state.falseRegions = restored.falseRegions;
        state.activeIssue = state.greenDefect ? "绿色缺陷" : (state.detectionIssues[0] || "");
        state.note = restored.note;
        noteInput.value = state.note;
        state.savedTagged = isTagged(state);
        state.serverSnapshot = cloneAnnotationSnapshot(restoreTarget);
        state.undoSnapshot = null;

        updateButtons(node, state);
        updateTaggedState(node, state, stateLabel);
        drawRegions(canvas, state);
        stateLabel.textContent = state.savedTagged ? "已恢复到保存前" : "已撤销本次保存";
        if (state.savedTagged) {
          stateLabel.classList.add("saved");
        } else {
          stateLabel.classList.remove("saved");
        }
        await loadStats();
        updateBulkDefaultAction();
      } catch (error) {
        stateLabel.textContent = error.message;
        stateLabel.classList.add("error");
        undoSaveButton.disabled = false;
      }
    });
  }

  updateButtons(node, state);
  container.appendChild(node);
  currentPageCards.push({ node, state, stateLabel });
}

function updateTaggedState(node, state, stateLabel) {
  const tagged = state.savedTagged;
  node.classList.toggle("tagged", tagged);
  node.classList.toggle("untagged", !tagged);
  if (tagged && stateLabel.textContent === "未保存") {
    stateLabel.textContent = "已打标";
    stateLabel.classList.add("saved");
  }
  if (tagged) {
    stateLabel.classList.remove("error");
    node.classList.remove("needs-verdict");
  }
}

function clearVerdictWarning(node, state, stateLabel) {
  if (!isTagged(state)) return;
  node.classList.remove("needs-verdict");
  stateLabel.classList.remove("error");
  if (!state.savedTagged && stateLabel.textContent === "请先选择分类正确或分类错误") {
    stateLabel.textContent = "未保存";
  }
}

function updateButtons(node, state) {
  node.querySelectorAll("[data-verdict]").forEach((button) => {
    button.classList.toggle("active", button.dataset.verdict === state.verdict);
  });
  const logicSection = node.querySelector(".logic-section");
  if (logicSection) {
    logicSection.classList.toggle("hidden", state.verdict !== "分类错误");
  }
  node.querySelectorAll("[data-green-defect]").forEach((button) => {
    button.classList.toggle("active", state.greenDefect);
    button.classList.toggle("drawing", state.activeIssue === "绿色缺陷");
  });
  node.querySelectorAll("[data-issue]").forEach((button) => {
    button.classList.toggle("active", state.detectionIssues.includes(button.dataset.issue));
    button.classList.toggle("drawing", state.activeIssue === button.dataset.issue);
  });
  node.querySelector(".image-wrap").classList.toggle("disabled", !canDrawActiveIssue(state));
  const hint = node.querySelector(".active-draw-hint");
  if (hint) {
    if (state.activeIssue === "错检") {
      hint.textContent = "当前模式：错检。可直接点击原始推理框转为错检框，或手动拖拽画框。";
    } else {
      hint.textContent = state.activeIssue ? `当前画框：${state.activeIssue}` : "选择漏检、错检或绿色缺陷后，在图片上拖拽画框。";
    }
  }
}

function canDrawActiveIssue(state) {
  if (state.activeIssue === "绿色缺陷") {
    return state.verdict === "分类错误" && state.greenDefect;
  }
  return Boolean(state.activeIssue && state.detectionIssues.includes(state.activeIssue));
}

function getActiveRegions(state) {
  if (state.activeIssue === "错检") return state.falseRegions;
  if (state.activeIssue === "绿色缺陷") return state.greenDefectRegions;
  return state.missRegions;
}

function regionSignature(region = {}) {
  const x = Number(region.x || 0).toFixed(6);
  const y = Number(region.y || 0).toFixed(6);
  const w = Number(region.w || 0).toFixed(6);
  const h = Number(region.h || 0).toFixed(6);
  const label = String(region.label || "");
  return `${x}|${y}|${w}|${h}|${label}`;
}

function syncInferenceRegionsFromState(state) {
  const removed = new Set((state.inferenceRemovedRegions || []).map((region) => regionSignature(region)));
  state.inferenceRegions = (state.baseInferenceRegions || [])
    .filter((region) => !removed.has(regionSignature(region)))
    .map((region) => ({ ...region }));
}

function findInferenceRegionAtPoint(state, point, canvas) {
  const rect = canvas.getBoundingClientRect();
  const tolerancePx = 8;
  const toleranceX = rect.width > 0 ? (tolerancePx / rect.width) * state.item.width : 0;
  const toleranceY = rect.height > 0 ? (tolerancePx / rect.height) * state.item.height : 0;
  for (let index = (state.inferenceRegions?.length || 0) - 1; index >= 0; index -= 1) {
    const region = state.inferenceRegions[index];
    if (!region) continue;
    if (
      point.x >= region.x - toleranceX
      && point.x <= region.x + region.w + toleranceX
      && point.y >= region.y - toleranceY
      && point.y <= region.y + region.h + toleranceY
    ) {
      return { index, region };
    }
  }
  return null;
}

function convertInferenceRegionToFalse(event, canvas, state) {
  if (state.activeIssue !== "错检") return false;
  if (!state.detectionIssues.includes("错检")) return false;

  const point = eventToImagePoint(event, canvas, state.item);
  const hit = findInferenceRegionAtPoint(state, point, canvas);
  if (!hit) return false;

  const hitRegion = hit.region;
  state.falseRegions.push({
    x: hitRegion.x,
    y: hitRegion.y,
    w: hitRegion.w,
    h: hitRegion.h,
    className: "错检",
    note: String(hitRegion.label || ""),
    source: "converted_from_inference",
  });
  state.inferenceRemovedRegions.push({
    x: hitRegion.x,
    y: hitRegion.y,
    w: hitRegion.w,
    h: hitRegion.h,
    label: String(hitRegion.label || ""),
  });
  syncInferenceRegionsFromState(state);
  return true;
}

function syncCanvasSize(img, canvas) {
  const rect = img.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function drawRegions(canvas, state) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  drawPredictionOverlay(ctx, canvas, state);

  drawRegionGroup(ctx, canvas, state, state.missRegions, "#b42318", "rgba(180, 35, 24, 0.12)");
  drawRegionGroup(ctx, canvas, state, state.falseRegions, "#1f7a8c", "rgba(31, 122, 140, 0.12)");
  drawRegionGroup(ctx, canvas, state, state.greenDefectRegions, "#20744a", "rgba(32, 116, 74, 0.14)");
  if (state.draft) {
    const draftColor = state.activeIssue === "绿色缺陷" ? "#20744a" : (state.activeIssue === "错检" ? "#1f7a8c" : "#b42318");
    const draftFill = state.activeIssue === "绿色缺陷" ? "rgba(32, 116, 74, 0.18)" : (state.activeIssue === "错检" ? "rgba(31, 122, 140, 0.16)" : "rgba(180, 35, 24, 0.16)");
    drawRegionGroup(ctx, canvas, state, [state.draft], draftColor, draftFill);
  }
  ctx.restore();
}

function drawPredictionOverlay(ctx, canvas, state) {
  const overlay = state.inferenceRegions;
  if (!Array.isArray(overlay) || !overlay.length) return;

  overlay.filter(Boolean).forEach((region) => {
    const display = imageRectToDisplay(region, canvas, state.item);
    const stroke = "#e59f17";
    const label = region.label || "";

    ctx.lineWidth = 2;
    ctx.strokeStyle = stroke;
    ctx.setLineDash([7, 4]);
    if (display.w > 0 && display.h > 0) {
      ctx.strokeRect(display.x, display.y, display.w, display.h);
    }
    ctx.setLineDash([]);

    if (label) {
      ctx.font = "12px 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif";
      ctx.textBaseline = "top";
      const textPadX = 6;
      const textPadY = 3;
      const textW = ctx.measureText(label).width;
      const boxW = textW + textPadX * 2;
      const boxH = 18;
      const tx = Math.max(0, display.x);
      const ty = Math.max(0, display.y - boxH);
      ctx.fillStyle = stroke;
      ctx.fillRect(tx, ty, boxW, boxH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, tx + textPadX, ty + textPadY);
    }
  });
}

function drawRegionGroup(ctx, canvas, state, regions, strokeStyle, fillStyle) {
  regions.filter(Boolean).forEach((region) => {
    const display = imageRectToDisplay(region, canvas, state.item);
    ctx.lineWidth = 2;
    ctx.strokeStyle = strokeStyle;
    ctx.fillStyle = fillStyle;
    ctx.fillRect(display.x, display.y, display.w, display.h);
    ctx.strokeRect(display.x, display.y, display.w, display.h);
  });
}

function eventToImagePoint(event, canvas, item) {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * item.width;
  const y = ((event.clientY - rect.top) / rect.height) * item.height;
  return {
    x: clamp(Math.round(x), 0, item.width),
    y: clamp(Math.round(y), 0, item.height),
  };
}

function rectFromPoints(a, b, item) {
  const x = clamp(Math.min(a.x, b.x), 0, item.width);
  const y = clamp(Math.min(a.y, b.y), 0, item.height);
  const w = clamp(Math.abs(a.x - b.x), 0, item.width - x);
  const h = clamp(Math.abs(a.y - b.y), 0, item.height - y);
  return { x, y, w, h };
}

function imageRectToDisplay(region, canvas, item) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (region.x / item.width) * rect.width,
    y: (region.y / item.height) * rect.height,
    w: (region.w / item.width) * rect.width,
    h: (region.h / item.height) * rect.height,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

async function bulkDefaultCorrectCurrentPage() {
  const paths = currentPageCards
    .filter(({ state }) => !state.savedTagged)
    .map(({ state }) => state.item.originalPath);

  if (!paths.length) {
    setStatus("当前页没有需要默认标注的未打标图片。");
    updateBulkDefaultAction();
    return;
  }

  controls.bulkDefaultCorrect.disabled = true;
  setStatus(`正在批量保存当前页 ${paths.length} 张未打标图片...`);
  const previousByPath = new Map(
    currentPageCards.map(({ state }) => [state.item.originalPath, cloneAnnotationSnapshot(state.serverSnapshot)])
  );

  try {
    const response = await fetch("/api/annotations/bulk-default-correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        paths,
        note: "当前页批量默认标注为分类正确",
        updatedAt: nowIso(),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "批量保存失败。");
    pendingBulkUndoEntries = (data.updatedPaths || []).map((path) => ({
      originalPath: path,
      annotation: previousByPath.get(path) || null,
    }));
    setStatus(`批量保存完成：新增默认标注 ${data.updated} 张，跳过已打标 ${data.skipped} 张。`);
    await loadStats();
    await loadImages();
    updateBulkDefaultAction();
  } catch (error) {
    setStatus(error.message, true);
    updateBulkDefaultAction();
  }
}

async function undoBulkDefaultCurrentPage() {
  if (!pendingBulkUndoEntries.length) {
    setStatus("没有可撤销的批量默认标注记录。");
    return;
  }

  if (controls.undoBulkDefault) {
    controls.undoBulkDefault.disabled = true;
  }
  setStatus(`正在撤销上次批量默认标注（${pendingBulkUndoEntries.length} 张）...`);

  try {
    await restoreAnnotations(pendingBulkUndoEntries);
    setStatus(`已撤销上次批量默认标注：恢复 ${pendingBulkUndoEntries.length} 张。`);
    pendingBulkUndoEntries = [];
    await loadStats();
    await loadImages();
    updateBulkDefaultAction();
  } catch (error) {
    setStatus(error.message, true);
    updateBulkDefaultAction();
  }
}

controls.loadBtn.addEventListener("click", async () => {
  shuffleSeed = String(Date.now());
  if (controls.resultJsonClearCache?.checked) {
    await clearSavedEvaluationsAndReload();
  } else {
    loadImages();
  }
});
if (controls.resultJsonClearCache) {
  controls.resultJsonClearCache.addEventListener("change", async () => {
    if (!controls.resultJsonClearCache.checked) return;
    await clearSavedEvaluationsAndReload();
  });
}
if (controls.prevPage) {
  controls.prevPage.addEventListener("click", () => {
    if (pageState.page <= 1) return;
    controls.page.value = pageState.page - 1;
    loadImages();
  });
}
if (controls.nextPage) {
  controls.nextPage.addEventListener("click", () => {
    if (pageState.page >= pageState.totalPages) return;
    controls.page.value = pageState.page + 1;
    loadImages();
  });
}
if (controls.bulkDefaultCorrect) {
  controls.bulkDefaultCorrect.addEventListener("click", bulkDefaultCorrectCurrentPage);
}
if (controls.undoBulkDefault) {
  controls.undoBulkDefault.addEventListener("click", undoBulkDefaultCurrentPage);
}
controls.modelFilterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    controls.modelPredictionFilter.value = button.dataset.modelFilter;
    controls.modelFilterButtons.forEach((item) => {
      item.classList.toggle("active", item === button);
    });
    shuffleSeed = String(Date.now());
    controls.page.value = "1";
    loadImages();
  });
});
controls.importFile.addEventListener("change", importAnnotations);
if (controls.exportImagePaths) {
  controls.exportImagePaths.addEventListener("click", updateImagePathExportLink);
  updateImagePathExportLink();
}
if (controls.exportAnnotations) {
  controls.exportAnnotations.addEventListener("click", updateAnnotationsExportLink);
  updateAnnotationsExportLink();
}
if (statsEls.refresh) {
  statsEls.refresh.addEventListener("click", loadStats);
}

[
  controls.resultJson,
  controls.projectDir,
  controls.baseDir,
  controls.lightType,
  controls.keyword,
  controls.imageSearch,
  controls.numCol,
  controls.numRow,
  controls.scaleRatio,
  controls.shuffleImages,
  controls.jsonFirst,
].forEach((control) => {
  control.addEventListener("change", () => {
    controls.page.value = "1";
    updateImagePathExportLink();
    updateAnnotationsExportLink();
  });
});

if (controls.resultJson) {
  controls.resultJson.addEventListener("input", () => {
    updateImagePathExportLink();
    updateAnnotationsExportLink();
    rememberJsonPath(controls.resultJson.value);
    refreshJsonOptionsPanel();
  });
}

if (controls.resultJsonSelect) {
  controls.resultJsonSelect.addEventListener("change", () => {
    const selectedValue = controls.resultJsonSelect.value || "";
    if (controls.resultJson) {
      controls.resultJson.value = selectedValue;
    }
    if (selectedValue) {
      rememberJsonPath(selectedValue);
    }
    updateImagePathExportLink();
    updateAnnotationsExportLink();
    refreshJsonOptionsPanel();
  });
}

if (controls.projectDir) {
  controls.projectDir.addEventListener("change", () => {
    loadJsonOptionsFromTxt({ silentIfEmpty: true });
  });
}

controls.keyword.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    shuffleSeed = String(Date.now());
    controls.page.value = "1";
    loadImages();
  }
});
controls.imageSearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    shuffleSeed = String(Date.now());
    controls.page.value = "1";
    loadImages();
  }
});

async function importAnnotations(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    const text = await file.text();
    const payload = JSON.parse(text);
    const response = await fetch("/api/annotations/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "导入失败。");
    setStatus(`已导入 ${data.imported} 条评价记录，当前共 ${data.total} 条。`);
    controls.page.value = "1";
    await loadStats();
    await loadImages();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    event.target.value = "";
  }
}

setupMatrixFilterUi();
restoreJsonPreference();
loadJsonOptionsFromTxt({ silentIfEmpty: true });
loadImages();
loadStats();
