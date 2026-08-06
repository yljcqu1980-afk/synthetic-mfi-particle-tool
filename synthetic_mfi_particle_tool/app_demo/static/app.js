const colors = {
  0: "#248bc1",
  1: "#50a878",
  2: "#b87333",
  3: "#9aa3ad",
  4: "#111827",
  5: "#8f46a8",
  6: "#d97706",
};

const classNames = {
  0: "气泡",
  1: "短纤维素",
  2: "铜颗粒",
  3: "浅色碳颗粒",
  4: "深色碳颗粒",
  5: "长纤维素",
  6: "其他/待复核",
};

let samples = [];
let filtered = [];
let currentIndex = 0;
let currentSample = null;
let currentImage = null;
let selectedParticleId = null;
let appStatus = null;
let showInternalSets = false;
let expandedInternalSplits = new Set(["test"]);
let selectionScope = "image";
let oilDetailRows = [];
let oilSummaryToken = 0;
let lastOilSummary = null;
let oilSummaryCache = new Map();
let uploadProgress = null;
let operationProgress = null;
let summaryProgress = null;
let oilClassReview = null;
let manualDrawingMode = null;
let manualDrawStart = null;
let manualChanges = [];
const DEFAULT_OIL_GROUP = "未归入油样";
let activeOilGroup = DEFAULT_OIL_GROUP;
let expandedOilGroups = new Set([DEFAULT_OIL_GROUP]);
let oilGroups = new Set([DEFAULT_OIL_GROUP]);
let projects = new Set();
let oilGroupProjects = {};
let expandedProjects = new Set();
let deletedOilGroups = new Set();
let resultMode = "oil";
let riskView = "composite";
const riskZones = [
  { id: "groove", name: "坡口邻近低流速区", release: ["groove", "top", "bottom", "interface"], dwell: 0.92, density: 0.86, field: 0.94, flow: 0.74, structure: 1.0, note: "低流速与高场梯度叠加，优先核查颗粒滞留和界面沉积。" },
  { id: "edge", name: "电极边缘/圆角过渡区", release: ["top", "bottom", "groove"], dwell: 0.72, density: 0.68, field: 1.0, flow: 0.66, structure: 0.94, note: "几何突变导致场强集中，聚集后可能放大局部电气应力。" },
  { id: "interface", name: "油纸界面窄油隙", release: ["interface", "groove", "bottom"], dwell: 0.81, density: 0.76, field: 0.72, flow: 0.48, structure: 0.9, note: "边界捕获与低速流动并存，适合观察纤维类颗粒。" },
  { id: "channel", name: "主油道输运区", release: ["top", "bottom", "interface"], dwell: 0.35, density: 0.52, field: 0.42, flow: 1.0, structure: 0.55, note: "累计经过量较高，但停留量较低，不宜单独判为聚集危险区。" },
  { id: "outlet", name: "出口/高流速区", release: ["top", "bottom"], dwell: 0.18, density: 0.22, field: 0.3, flow: 0.86, structure: 0.4, note: "颗粒更易被带离，通常为低聚集风险区。" },
];
let recognitionQueue = [];
let recognitionRunning = false;
let recognitionTaskId = 0;
const PROJECT_STATE_KEY = "mfiParticleOilProjectsV1";
const STANDARD_VOLUME_ML = 100;
const STANDARD_PARTICLE_SIZE_UM = 5;
const STANDARD_5UM_LIMIT_PER_100ML = 3000;
const DEMO_VOLUME_UL_PER_IMAGE = 100;
const OIL_SUMMARY_CACHE_VERSION = "roi-review-v2";
let inferSettings = {
  conf: 0.25,
  other_conf: 0.45,
  iou: 0.45,
  imgsz: 1024,
  max_det: 300,
  device: "",
  agnostic_nms: true,
  pixel_size_um: 0.7,
};

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 1) => Number(value || 0).toFixed(digits);
const fmtInt = (value) => Math.round(Number(value || 0)).toLocaleString();
const esc = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

function progressBarHtml({ title, current = 0, total = 0, detail = "", compact = false } = {}) {
  const safeTotal = Math.max(0, Number(total || 0));
  const safeCurrent = Math.min(safeTotal || Number(current || 0), Math.max(0, Number(current || 0)));
  const percent = safeTotal > 0 ? Math.round((safeCurrent / safeTotal) * 100) : 0;
  return `
    <div class="task-progress ${compact ? "compact" : ""}">
      <div class="task-progress-head">
        <strong>${esc(title || "处理中")}</strong>
        <span>${safeTotal ? `${safeCurrent}/${safeTotal}` : `${percent}%`}</span>
      </div>
      <div class="task-progress-track"><i style="width:${percent}%"></i></div>
      ${detail ? `<div class="task-progress-detail">${esc(detail)}</div>` : ""}
    </div>
  `;
}

function setAppLoading(percent, title, detail, { indeterminate = false } = {}) {
  const overlay = $("appLoading");
  if (!overlay) return;
  overlay.hidden = false;
  overlay.classList.toggle("is-indeterminate", indeterminate);
  const safePercent = Math.max(0, Math.min(100, Number(percent || 0)));
  $("appLoadingTitle").textContent = title || "正在载入";
  $("appLoadingDetail").textContent = detail || "请稍候...";
  $("appLoadingBar").style.width = `${safePercent}%`;
  $("appLoadingPercent").textContent = indeterminate ? "处理中" : `${Math.round(safePercent)}%`;
}

function finishAppLoading() {
  setAppLoading(100, "加载完成", "界面已经准备好");
  window.setTimeout(() => {
    const overlay = $("appLoading");
    if (overlay) overlay.hidden = true;
  }, 260);
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function hasRealConfidence() {
  return currentSample?.inference_mode === "real_yolo";
}

function confidenceText(det) {
  return det.confidence == null ? "N/A" : fmt(det.confidence, 2);
}

function annotationId(det) {
  if (!det.annotation_id) det.annotation_id = crypto.randomUUID();
  return det.annotation_id;
}

function manualAnnotation(det) {
  return {
    annotation_id: annotationId(det),
    class_id: Number(det.class_id),
    bbox: [...det.bbox],
    origin: det.manual_label ? "manual" : "model",
    operation: det.manual_operation || (det.manual_label ? "modified" : "confirmed"),
    model_class_id: det.model_class_id ?? det.predicted_class_id ?? det.class_id,
    model_confidence: det.model_confidence ?? det.confidence ?? null,
    model_bbox: det.model_bbox || [...det.bbox],
  };
}

function syncManualReviewPanel() {
  const panel = $("manualReviewPanel");
  if (!panel) return;
  const available = currentSample?.split === "uploaded";
  panel.hidden = !available;
  if (!available) return;
  const review = currentSample.manual_review;
  $("manualReviewState").textContent = review?.review_scope === "complete" ? "整图已审核" : review ? "局部修正" : "未复核";
  $("completeImageReview").checked = review?.review_scope === "complete";
  $("manualReviewNote").value = review?.note || "";
  const det = selectedDetection();
  $("manualClassSelect").value = String(det?.class_id ?? 6);
  ["confirmManualClassBtn", "redrawBoxBtn", "removeManualBoxBtn"].forEach((id) => {
    $(id).disabled = !det;
  });
}

function setManualHint(text, isError = false) {
  const hint = $("manualReviewHint");
  if (!hint) return;
  hint.textContent = text;
  hint.style.color = isError ? "#b42318" : "#536770";
}

function setManualDrawingMode(mode) {
  if (!currentSample || currentSample.split !== "uploaded") return;
  if (mode === "redraw" && !selectedDetection()) return;
  manualDrawingMode = mode;
  manualDrawStart = null;
  document.querySelector(".canvas-stage")?.classList.toggle("review-drawing", Boolean(mode));
  setManualHint(mode === "redraw" ? "请在主图上拖动鼠标，重新绘制选中目标的框。" : "请在主图上拖动鼠标，为漏检目标绘制新框。");
}

function canvasImagePoint(event) {
  const canvas = $("imageCanvas");
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.round(((event.clientX - rect.left) / rect.width) * currentSample.width),
    y: Math.round(((event.clientY - rect.top) / rect.height) * currentSample.height),
  };
}

function finishManualBox(event) {
  if (!manualDrawingMode || !manualDrawStart) return;
  const end = canvasImagePoint(event);
  const bbox = [Math.min(manualDrawStart.x, end.x), Math.min(manualDrawStart.y, end.y), Math.max(manualDrawStart.x, end.x), Math.max(manualDrawStart.y, end.y)];
  manualDrawStart = null;
  if (bbox[2] - bbox[0] < 3 || bbox[3] - bbox[1] < 3) {
    setManualHint("框太小，请重新拖动绘制。", true);
    return;
  }
  if (manualDrawingMode === "redraw") {
    const det = selectedDetection();
    if (det) {
      manualChanges.push({ operation: "adjust_bbox", annotation_id: annotationId(det), before: [...det.bbox], after: bbox });
      det.model_bbox ||= [...det.bbox];
      det.bbox = bbox;
      det.manual_label = true;
      det.manual_operation = "adjust_bbox";
    }
  } else {
    const classId = Number($("manualClassSelect").value || 6);
    const id = crypto.randomUUID();
    const det = {
      annotation_id: id,
      particle_id: currentSample.detections.length,
      class_id: classId,
      class_name: "manual",
      class_cn: classNames[classId],
      confidence: null,
      bbox,
      morphology: {},
      eq_diameter_um: 0,
      manual_label: true,
      manual_operation: "added",
      source: "manual_draft",
    };
    currentSample.detections.push(det);
    selectedParticleId = det.particle_id;
    manualChanges.push({ operation: "add", annotation_id: id, class_id: classId, bbox });
  }
  manualDrawingMode = null;
  document.querySelector(".canvas-stage")?.classList.remove("review-drawing");
  setManualHint("人工框已更新。点击“保存人工标签”写入标签库。");
  drawCanvas();
  renderParticleInspector(selectedDetection());
  renderSelection();
  renderRows(activeDetections());
  syncManualReviewPanel();
}

async function saveManualReview() {
  if (!currentSample || currentSample.split !== "uploaded") return;
  const payload = {
    review_scope: $("completeImageReview").checked ? "complete" : "partial",
    reviewer: "人工复核",
    note: $("manualReviewNote").value,
    annotations: currentSample.detections.map(manualAnnotation),
    changes: manualChanges,
  };
  setManualHint("正在保存人工标签...");
  await fetchJson(`/api/manual-labels/${encodeURIComponent(currentSample.image_name)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  manualChanges = [];
  const item = filtered[currentIndex];
  currentSample = await fetchJson(`/api/sample/${encodeURIComponent(item.split)}/${encodeURIComponent(item.image_name)}?${settingsQuery()}`);
  selectedParticleId = null;
  updatePanels();
  drawCanvas();
  setManualHint(payload.review_scope === "complete" ? "已保存：整图审核完成，可进入训练标签导出。" : "已保存为局部修正；暂不进入训练标签导出。 ");
}

function bboxSizeUm(det) {
  const pixel = Number(currentSample?.pixel_size_um || inferSettings.pixel_size_um || 0.7);
  const [xmin, ymin, xmax, ymax] = det.bbox || [0, 0, 0, 0];
  const width = Math.max(0, xmax - xmin) * pixel;
  const height = Math.max(0, ymax - ymin) * pixel;
  return { width, height, maxDim: Math.max(width, height), minDim: Math.min(width, height) };
}

function particleSizeLabel(det) {
  const size = bboxSizeUm(det);
  const isFiber = Number(det.class_id) === 0 || (det.morphology?.aspect_ratio || 0) >= 3;
  if (isFiber) return `长 ${fmt(size.maxDim, 1)} μm`;
  return `径 ${fmt(det.eq_diameter_um, 1)} μm`;
}

function concentrationPerMl(count, volumeUl) {
  const ml = Number(volumeUl || 0) / 1000;
  return ml > 0 ? Number(count || 0) / ml : 0;
}

function concentrationHtml(perMl) {
  return `<span class="primary-conc">${fmtInt(perMl * STANDARD_VOLUME_ML)} 个/${STANDARD_VOLUME_ML} mL</span><small>${fmtInt(perMl)} 个/mL</small>`;
}

function standardParticleCount(detections = []) {
  return detections.filter((det) => Number(det.eq_diameter_um || 0) > STANDARD_PARTICLE_SIZE_UM).length;
}

function smallParticleCount(detections = []) {
  return detections.filter((det) => {
    const diameter = Number(det.eq_diameter_um || 0);
    return diameter > 0 && diameter < STANDARD_PARTICLE_SIZE_UM;
  }).length;
}

function standardConcentrationPer100ml(count, volumeUl) {
  return concentrationPerMl(count, volumeUl) * STANDARD_VOLUME_ML;
}

function standardConcentrationHtml(count, volumeUl, quantifiable = Number(volumeUl) > 0) {
  if (!quantifiable || !(Number(volumeUl) > 0)) {
    return `<span class="primary-conc">不可定量</span><small>当前图像仅用于识别；需完整 4096×3000 原始帧</small>`;
  }
  const per100ml = standardConcentrationPer100ml(count, volumeUl);
  const level = per100ml > STANDARD_5UM_LIMIT_PER_100ML ? "超出参考限值" : "未超参考限值";
  const cls = per100ml > STANDARD_5UM_LIMIT_PER_100ML ? "limit-bad" : "limit-ok";
  return `<span class="primary-conc">${fmtInt(per100ml)} 个/${STANDARD_VOLUME_ML} mL</span><small>&gt;${STANDARD_PARTICLE_SIZE_UM} μm：${count} 个；检测体积 ${fmt(Number(volumeUl) / 1000, 4)} mL；参考限值 ${fmtInt(STANDARD_5UM_LIMIT_PER_100ML)} 个/${STANDARD_VOLUME_ML} mL</small><small class="${cls}">${level}</small>`;
}

function settingsQuery() {
  const params = new URLSearchParams({
    conf: inferSettings.conf,
    other_conf: inferSettings.other_conf,
    iou: inferSettings.iou,
    imgsz: inferSettings.imgsz,
    max_det: inferSettings.max_det,
    device: inferSettings.device,
    agnostic_nms: inferSettings.agnostic_nms ? "1" : "0",
    pixel_size_um: inferSettings.pixel_size_um,
  });
  return params.toString();
}

function applyProjectState(saved = {}) {
  projects = new Set((Array.isArray(saved.projects) ? saved.projects : []).map(recoverProjectName).filter(Boolean));
  oilGroupProjects = {};
  Object.entries(saved.oilGroupProjects && typeof saved.oilGroupProjects === "object" ? saved.oilGroupProjects : {}).forEach(([group, project]) => {
    const projectName = recoverProjectName(project);
    if (projectName) oilGroupProjects[normalizeOilGroupName(group)] = projectName;
  });
  expandedProjects = new Set((Array.isArray(saved.expandedProjects) ? saved.expandedProjects : []).map(recoverProjectName).filter(Boolean));
  deletedOilGroups = new Set((Array.isArray(saved.deletedOilGroups) ? saved.deletedOilGroups : []).map(normalizeOilGroupName).filter(Boolean));
}

function loadProjectState() {
  try {
    const saved = JSON.parse(localStorage.getItem(PROJECT_STATE_KEY) || "{}");
    applyProjectState(saved);
  } catch {
    projects = new Set();
    oilGroupProjects = {};
    expandedProjects = new Set();
    deletedOilGroups = new Set();
  }
}

async function loadServerProjectState() {
  try {
    const saved = await fetchJson("/api/projects");
    applyProjectState(saved);
    localStorage.setItem(PROJECT_STATE_KEY, JSON.stringify(saved));
  } catch {
    loadProjectState();
  }
}

function normalizeOilGroupName(group) {
  const value = String(group || "").trim();
  return !value || value === "未分组" ? DEFAULT_OIL_GROUP : value;
}

function normalizeProjectName(project) {
  if (project && typeof project === "object") return "";
  const value = String(project || "").trim();
  return value && !value.startsWith("[object ") ? value : "";
}

function recoverProjectName(project) {
  return normalizeProjectName(project) || "未命名项目";
}

function compareNames(a, b) {
  return String(a || "").localeCompare(String(b || ""), "zh-Hans-CN");
}

function saveProjectState() {
  projects = new Set([...projects].map(normalizeProjectName).filter(Boolean));
  deletedOilGroups = new Set([...deletedOilGroups].map(normalizeOilGroupName).filter((group) => group !== DEFAULT_OIL_GROUP));
  deletedOilGroups.forEach((group) => {
    delete oilGroupProjects[group];
  });
  Object.values(oilGroupProjects).forEach((project) => {
    const name = normalizeProjectName(project);
    if (name) projects.add(name);
  });
  const state = {
    projects: [...projects],
    oilGroupProjects,
    expandedProjects: [...expandedProjects].map(normalizeProjectName).filter((project) => projects.has(project)),
    deletedOilGroups: [...deletedOilGroups],
  };
  localStorage.setItem(PROJECT_STATE_KEY, JSON.stringify(state));
  fetch("/api/projects", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  }).catch(() => {});
}

function oilSummaryCacheKey(group = activeOilGroup) {
  const members = samples
    .filter((item) => item.split === "uploaded" && uploadedGroup(item) === group)
    .map((item) => `${item.image_name}:${item.particles || 0}`)
    .sort()
    .join("|");
  return `${OIL_SUMMARY_CACHE_VERSION}::${group}::${settingsQuery()}::${members}`;
}

function invalidateOilSummaryCache() {
  oilSummaryCache.clear();
  lastOilSummary = null;
}

function activeDetections() {
  if (!currentSample) return [];
  const diamMin = Number($("diamSlider").value);
  const confMin = Number($("confSlider").value) / 100;
  const cls = $("classFilter").value;
  return currentSample.detections.filter((det) => {
    const classOk = cls === "all" || String(det.class_id) === cls;
    const diamOk = det.eq_diameter_um >= diamMin;
    const confOk = !hasRealConfidence() || (det.confidence ?? 0) >= confMin;
    return classOk && diamOk && confOk;
  });
}

function renderLegend() {
  $("legend").innerHTML = Object.entries(classNames)
    .map(([cid, name]) => `<span class="legend-item"><i class="swatch" style="background:${colors[cid]}"></i>${name}</span>`)
    .join("");
  $("classFilter").innerHTML =
    `<option value="all">全部类别</option>` +
    Object.entries(classNames).map(([cid, name]) => `<option value="${cid}">${name}</option>`).join("");
}

function renderModelStatus() {
  const status = currentSample || appStatus;
  if (!status) return;
  const isReal = status.inference_mode === "real_yolo";
  const isUploadedWaiting = status.inference_mode === "awaiting_real_yolo_weights";
  const modeLabel = isReal ? "真实 MFI YOLO-P2" : isUploadedWaiting ? "等待权重" : "真实数据样本";
  $("modeBanner").className = `mode-banner ${isReal ? "real" : "demo"}`;
  $("modeBanner").textContent = isReal
    ? "陈匡亚真实 MFI 数据模型：完整原始帧自动按 4×3 切片识别，裁剪图仅识别不定量。"
    : isUploadedWaiting
      ? "原始图像已导入：请放置真实 YOLO 权重后运行检测，当前仅显示图像与参数设置。"
      : "已标注样本工作流：当前样图按微流成像识别流程展示检测、分类和参数提取。";
  $("modelStatus").innerHTML = `
    <details>
      <summary>模型状态</summary>
      <div class="status-grid">
        <span>推理模式</span><strong>${modeLabel}</strong>
        <span>模型加载</span><strong>${status.model_loaded ? "true" : "false"}</strong>
        <span>模型路径</span><code>${status.model_path || "-"}</code>
        <span>耗时</span><strong>${status.inference_time_ms == null ? "-" : fmt(status.inference_time_ms, 1) + " ms"}</strong>
      </div>
      <p class="status-note">${status.model_message || status.message || ""}<br>${status.quantification_reason || status.quantification_policy || ""}</p>
    </details>
  `;
  $("confSlider").disabled = !isReal;
  $("confValue").textContent = isReal ? (Number($("confSlider").value) / 100).toFixed(2) : "N/A";
}

function configureSampleControls() {
  const splitFilter = $("splitFilter");
  const uploadInput = $("uploadInput");
  if (uploadInput) {
    uploadInput.multiple = true;
  }
  const uploadBtn = $("uploadBtn");
  if (uploadBtn) uploadBtn.remove();
  $("analyzeUploadedBtn")?.remove();
  const uploadHint = $("uploadHint");
  if (uploadHint) uploadHint.textContent = "先建立或选择油样批次，再批量导入该油样文件夹中的 MFI 图像。";
  document.querySelector(".workflow-details")?.parentElement?.classList.add("workflow-panel");
  splitFilter.innerHTML = showInternalSets
    ? `
      <option value="uploaded">检测图片</option>
      <option value="all">全部样本</option>
      <option value="train">训练集</option>
      <option value="val">验证集</option>
      <option value="test">测试集</option>
    `
    : `<option value="uploaded">检测图片</option>`;
  splitFilter.value = showInternalSets ? "all" : "uploaded";

  if (!$("sideNavBuilt")) {
    buildSideNavigation();
  }

  if ($("internalSetToggle")) return;
  const label = document.createElement("label");
  label.className = "internal-toggle subtle-toggle";
  label.innerHTML = `<input id="internalSetToggle" type="checkbox" /> 显示内置训练/验证/测试样本`;
  const mount = $("navProjectMount") || document.querySelector(".toolbar");
  mount?.insertAdjacentElement("afterend", label);
  $("internalSetToggle").addEventListener("change", () => {
    showInternalSets = $("internalSetToggle").checked;
    configureSampleControls();
    $("splitFilter").value = showInternalSets ? "all" : "uploaded";
    currentIndex = 0;
    applyFilters();
    $("sampleList").scrollTop = 0;
    loadByIndex(0);
  });

  if ($("uploadManager")) return;
  const panel = document.createElement("details");
  panel.id = "uploadManager";
  panel.className = "upload-manager";
  panel.hidden = true;
  panel.innerHTML = `
    <summary>
      <span>管理当前图片</span>
      <small id="managerSummary">名称、油样批次、备注、删除</small>
    </summary>
    <div class="manager-body">
      <label>名称<input id="manageName" type="text" maxlength="120" /></label>
      <label>油样批次<input id="manageGroup" type="text" maxlength="80" placeholder="例如：油样A-20260712 / 复核批次1" /></label>
      <label>备注<textarea id="manageNote" rows="3" maxlength="500" placeholder="记录来源、批次、取样说明或复核意见"></textarea></label>
      <div class="manage-actions">
        <button id="saveSampleMetaBtn" type="button">保存信息</button>
        <button id="deleteUploadedBtn" class="danger-action" type="button">永久删除</button>
      </div>
    </div>
  `;
  document.querySelector(".upload-box")?.insertAdjacentElement("afterend", panel);

  if (!$("oilSamplePanel")) {
    const oilPanel = document.createElement("section");
    oilPanel.id = "oilSamplePanel";
    oilPanel.className = "oil-sample-panel";
    oilPanel.innerHTML = `
      <div class="section-head">
        <h3>油样级汇总</h3>
        <button id="exportOilReportBtn" type="button">导出油样报告</button>
      </div>
      <div class="result-tabs">
        <button id="oilResultTab" class="active" type="button">油样汇总</button>
        <button id="imageResultTab" type="button">当前图像</button>
      </div>
      <div id="oilSampleSummary" class="oil-summary muted-box">导入多张同一油样批次图像后自动汇总。</div>
    `;
    document.querySelector(".analysis-panel")?.insertAdjacentElement("afterbegin", oilPanel);
  }
  const analysisPanel = document.querySelector(".analysis-panel");
  const modelStatus = $("modelStatus");
  const advancedPanel = document.querySelector(".advanced-panel");
  if (analysisPanel && modelStatus && advancedPanel) {
    analysisPanel.appendChild(modelStatus);
    analysisPanel.appendChild(advancedPanel);
  }
}

function buildSideNavigation() {
  const nav = $("sideNav");
  const list = $("sampleList");
  const workflow = $("oilWorkflow");
  const dock = document.querySelector(".action-dock");
  if (!nav || !list || !dock) return;
  nav.innerHTML = `
    <div id="sideNavBuilt" hidden></div>
    <section class="nav-block nav-actions">
      <button id="navNewProjectBtn" class="nav-item" type="button"><span class="nav-icon">□</span><span>新建项目</span></button>
      <button id="navNewOilBtn" class="nav-item" type="button"><span class="nav-icon">＋</span><span>新建油样</span></button>
      <button id="navAggregationRiskBtn" class="nav-item nav-feature" type="button"><span class="nav-icon">◆</span><span>颗粒聚集风险</span></button>
    </section>
    <div id="recognitionStatus" class="recognition-status" hidden>
      <span class="status-dot">◎</span>
      <div>
        <div id="recognitionStatusText">识别中...</div>
        <div id="recognitionQueueText" class="recognition-queue-text"></div>
      </div>
    </div>
    <section class="nav-block nav-projects">
      <h3>绝缘油样本</h3>
      <div id="navProjectMount"></div>
    </section>
  `;
  $("navProjectMount")?.appendChild(list);
  nav.appendChild(dock);
  $("navNewProjectBtn")?.addEventListener("click", openNewProjectDialog);
  $("navNewOilBtn")?.addEventListener("click", () => openNewOilDialog(""));
  $("navAggregationRiskBtn")?.addEventListener("click", openAggregationRiskPage);
}

function riskFilteredZones() {
  const release = $("riskReleaseScenario")?.value || "all";
  return riskZones.filter((z) => release === "all" || z.release.includes(release));
}

function riskScore(zone) {
  const fieldWeight = Number($("riskFieldWeight")?.value || 35) / 100;
  const base = zone.dwell * 0.38 + zone.density * 0.27 + zone.flow * 0.08 + zone.structure * 0.12;
  return Math.min(1, base * (1 - fieldWeight) + (zone.field * 0.15 + base * 0.0) * fieldWeight);
}

function drawAggregationRisk() {
  const canvas = $("aggregationRiskCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const zones = riskFilteredZones();
  const values = zones.map((z) => riskView === "composite" ? riskScore(z) : z[riskView] || 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height); grad.addColorStop(0, "#f7fbfc"); grad.addColorStop(1, "#edf3f5"); ctx.fillStyle = grad; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const cols = 3, rows = Math.ceil(zones.length / cols), cellW = canvas.width / cols, cellH = canvas.height / rows;
  zones.forEach((zone, i) => { const x = (i % cols) * cellW + 24, y = Math.floor(i / cols) * cellH + 24, w = cellW - 48, h = cellH - 48, value = values[i]; const color = `hsl(${Math.max(4, 170 - value * 165)}, 75%, ${Math.max(38, 92 - value * 38)}%)`; ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(x, y, w, h, 16); ctx.fill(); ctx.strokeStyle = "rgba(31,41,51,.16)"; ctx.stroke(); ctx.fillStyle = "#17232d"; ctx.font = "600 20px Microsoft YaHei"; ctx.fillText(zone.name, x + 18, y + 34); ctx.font = "700 36px Segoe UI"; ctx.fillText(`${Math.round(value * 100)}`, x + 18, y + 82); ctx.font = "14px Microsoft YaHei"; ctx.fillText(riskView === "composite" ? "综合风险指数" : `${riskView === "dwell" ? "停留量" : riskView === "density" ? "颗粒浓度" : riskView === "field" ? "电场加权" : "累计经过"}（归一化）`, x + 18, y + 110); });
  const ranked = zones.map((z, i) => ({ ...z, value: values[i], score: riskScore(z) })).sort((a, b) => b.value - a.value);
  $("riskHotspotName").textContent = ranked[0]?.name || "-"; $("riskHotspotScore").textContent = ranked[0] ? `${Math.round(ranked[0].score * 100)}/100` : "-"; $("riskDwellValue").textContent = ranked[0] ? `${Math.round(ranked[0].dwell * 100)}%` : "-"; $("riskFieldValue").textContent = ranked[0] ? `${Math.round(ranked[0].field * 100)}%` : "-";
  $("riskZoneTable").innerHTML = ranked.map((z, i) => `<div class="risk-zone-row"><b>${i + 1}. ${esc(z.name)}</b><strong>${Math.round(z.value * 100)}</strong><small>停留 ${Math.round(z.dwell * 100)}% · 场 ${Math.round(z.field * 100)}%</small></div>`).join("");
  $("riskInterpretation").innerHTML = ranked.slice(0, 3).map((z) => `<p><strong>${esc(z.name)}：</strong>${esc(z.note)}</p>`).join("");
}

function openAggregationRiskPage() { $("aggregationRiskPage").hidden = false; document.querySelector(".app-shell").hidden = true; drawAggregationRisk(); }
function closeAggregationRiskPage() { $("aggregationRiskPage").hidden = true; document.querySelector(".app-shell").hidden = false; }

function uploadedGroup(item) {
  return normalizeOilGroupName(item.group);
}

function currentGroupMembers() {
  return samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === activeOilGroup);
}

function syncOilGroupsFromSamples() {
  samples.filter((item) => item.split === "uploaded").forEach((item) => {
    const group = uploadedGroup(item);
    if (!deletedOilGroups.has(group)) oilGroups.add(group);
  });
  oilGroups.add(activeOilGroup || "未归入油样");
  deletedOilGroups.forEach((group) => {
    oilGroups.delete(group);
    delete oilGroupProjects[group];
    expandedOilGroups.delete(group);
  });
}

function updateOilWorkflowUi() {
  const status = $("recognitionStatus");
  if (status?.hidden) status.classList.remove("done", "error");
}

async function selectOilGroup(group, { loadFirst = false, expand = true } = {}) {
  activeOilGroup = group || "未归入油样";
  oilGroups.add(activeOilGroup);
  if (expand) expandedOilGroups.add(activeOilGroup);
  selectionScope = "oil";
  resultMode = "oil";
  updateOilWorkflowUi();
  renderSampleList();
  await renderOilGroupDetailRows(activeOilGroup);
  await renderOilSampleSummary(activeOilGroup);
  if (loadFirst) {
    const idx = filtered.findIndex((item) => item.split === "uploaded" && uploadedGroup(item) === activeOilGroup);
    if (idx >= 0) await loadByIndex(idx);
  }
}

function applyFilters() {
  const split = $("splitFilter").value;
  const query = "";
  filtered = samples.filter((item) => {
    const inDetectionSet = item.split === "uploaded";
    const splitOk = showInternalSets ? split === "all" || item.split === split : inDetectionSet;
    const searchable = [item.image_name, item.display_name, item.group, item.note].filter(Boolean).join(" ").toLowerCase();
    const queryOk = !query || searchable.includes(query);
    return splitOk && queryOk;
  });
  if (currentIndex >= filtered.length) currentIndex = 0;
  const uploaded = samples.filter((item) => item.split === "uploaded");
  syncOilGroupsFromSamples();
  if (uploaded.length && !uploaded.some((item) => uploadedGroup(item) === activeOilGroup)) {
    const firstAvailable = uploaded.find((item) => !deletedOilGroups.has(uploadedGroup(item)));
    activeOilGroup = firstAvailable ? uploadedGroup(firstAvailable) : DEFAULT_OIL_GROUP;
    expandedOilGroups.add(activeOilGroup);
  }
  const visibleTotal = showInternalSets ? samples.length : samples.filter((item) => item.split === "uploaded").length;
  $("sampleCount").textContent = `${filtered.length} / ${visibleTotal} 张`;
  updateOilWorkflowUi();
  renderSampleList();
}

function renderSampleList() {
  const uploaded = filtered.filter((item) => item.split === "uploaded");
  const internal = filtered.filter((item) => item.split !== "uploaded");
  const groups = new Map();
  uploaded.forEach((item) => {
    const group = uploadedGroup(item);
    if (deletedOilGroups.has(group)) return;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(item);
  });
  syncOilGroupsFromSamples();
  Array.from(oilGroups).forEach((group) => {
    if (!groups.has(group)) groups.set(group, []);
  });
  const sortedGroups = Array.from(groups.entries()).sort((a, b) => {
    if (a[0] === activeOilGroup) return -1;
    if (b[0] === activeOilGroup) return 1;
    return compareNames(a[0], b[0]);
  });
  const renderOilFolder = ([group, items]) => {
    const open = expandedOilGroups.has(group);
    const particleCount = items.reduce((sum, item) => sum + Number(item.particles || 0), 0);
    const active = group === activeOilGroup;
    const folderTaskProgress = uploadProgress?.group === group
      ? progressBarHtml({
          title: `正在导入 ${group}`,
          current: uploadProgress.current,
          total: uploadProgress.total,
          detail: uploadProgress.fileName || "",
          compact: true,
        })
      : operationProgress?.group === group
        ? progressBarHtml({
            title: operationProgress.title || `正在处理 ${group}`,
            current: operationProgress.current,
            total: operationProgress.total,
            detail: operationProgress.detail || "",
            compact: true,
          })
      : "";
    const children = open && items.length
      ? items
          .map((item) => {
            const idx = filtered.indexOf(item);
            return `
              <div class="sample-row child ${idx === currentIndex ? "active" : ""}">
                <button class="sample-item" data-index="${idx}" type="button">
                  <span class="row-main">
                    <span class="sample-name">${esc(item.display_name || item.image_name)}</span>
                    <span class="sample-class">${esc(item.dominant_class || "待检测")}</span>
                    <span class="sample-count">${item.particles}</span>
                  </span>
                  ${item.note ? `<span class="row-sub">有备注</span>` : ""}
                </button>
                <button class="sample-more-btn" data-index="${idx}" title="图片操作" type="button">⋯</button>
              </div>
            `;
          })
          .join("")
      : open
        ? `<div class="empty-folder">空油样。点击油样右侧“…”导入图像。</div>`
        : "";
    return `
      <div class="oil-folder ${active ? "active" : ""}">
        <div class="folder-row oil-folder-row">
          <button class="oil-folder-head ${open ? "open" : ""}" data-group="${esc(group)}" type="button">
            <span class="folder-caret">${open ? "▾" : "▸"}</span>
            <span class="folder-icon" aria-hidden="true"></span>
            <span class="folder-title">${esc(group)}</span>
            <span class="folder-meta">${items.length} 张 · ${particleCount} 个</span>
          </button>
          <div class="folder-actions">
            <button class="folder-action rename-oil-btn" data-group="${esc(group)}" title="重命名油样" type="button">✎</button>
            <button class="folder-action delete-oil-btn" data-group="${esc(group)}" title="删除油样" type="button">×</button>
            <button class="folder-action oil-more-btn" data-group="${esc(group)}" title="更多操作" type="button">⋯</button>
          </div>
        </div>
        ${folderTaskProgress}
        <div class="oil-folder-children">${children}</div>
      </div>
    `;
  };
  const unassigned = [];
  const byProject = new Map();
  sortedGroups.forEach((entry) => {
    const [group] = entry;
    const project = normalizeProjectName(oilGroupProjects[group]);
    if (project) {
      if (!byProject.has(project)) byProject.set(project, []);
      byProject.get(project).push(entry);
    } else {
      unassigned.push(entry);
    }
  });
  Array.from(projects).map(normalizeProjectName).filter(Boolean).forEach((project) => {
    if (!byProject.has(project)) byProject.set(project, []);
  });
  const projectHtml = Array.from(byProject.entries())
    .sort((a, b) => compareNames(a[0], b[0]))
    .map(([project, entries]) => {
      const open = expandedProjects.has(project);
      const imageCount = entries.reduce((sum, [, items]) => sum + items.length, 0);
      const particleCount = entries.reduce((sum, [, items]) => sum + items.reduce((s, item) => s + Number(item.particles || 0), 0), 0);
      return `
        <div class="project-folder">
          <div class="folder-row project-folder-row">
            <button class="project-folder-head ${open ? "open" : ""}" data-project="${esc(project)}" type="button">
              <span class="folder-caret">${open ? "▾" : "▸"}</span>
              <span class="folder-icon" aria-hidden="true"></span>
              <span class="folder-title">${esc(project)}</span>
              <span class="folder-meta">${entries.length} 个油样 · ${imageCount} 张 · ${particleCount} 个</span>
            </button>
            <div class="folder-actions">
              <button class="folder-action rename-project-btn" data-project="${esc(project)}" title="重命名项目" type="button">✎</button>
              <button class="folder-action delete-project-btn" data-project="${esc(project)}" title="删除项目" type="button">×</button>
              <button class="folder-action project-more-btn" data-project="${esc(project)}" title="更多操作" type="button">⋯</button>
            </div>
          </div>
          <div class="project-folder-children">${open ? entries.map(renderOilFolder).join("") || `<div class="empty-folder">空项目。右键油样加入该项目。</div>` : ""}</div>
        </div>
      `;
    })
    .join("");
  const groupHtml = `${projectHtml}${unassigned.map(renderOilFolder).join("")}`;
  const internalNames = { train: "真实训练集", val: "真实验证集", test: "真实测试集" };
  const internalHtml = ["test", "val", "train"]
    .map((split) => {
      const items = internal.filter((item) => item.split === split);
      if (!items.length) return "";
      const open = expandedInternalSplits.has(split);
      const particleCount = items.reduce((sum, item) => sum + Number(item.particles || 0), 0);
      const children = open
        ? items.map((item) => {
            const idx = filtered.indexOf(item);
            return `
              <div class="sample-row child ${idx === currentIndex ? "active" : ""}">
                <button class="sample-item" data-index="${idx}" type="button">
                  <span class="row-main">
                    <span class="sample-name">${esc(item.display_name || item.image_name)}</span>
                    <span class="sample-class">${esc(item.dominant_class || split)}</span>
                    <span class="sample-count">${item.particles}</span>
                  </span>
                </button>
              </div>`;
          }).join("")
        : "";
      return `
        <div class="oil-folder internal-folder">
          <div class="folder-row">
            <button class="internal-folder-head ${open ? "open" : ""}" data-split="${split}" type="button">
              <span class="folder-caret">${open ? "▾" : "▸"}</span>
              <span class="folder-icon" aria-hidden="true"></span>
              <span class="folder-title">${internalNames[split]}</span>
              <span class="folder-meta">${items.length} 张 · ${particleCount} 个标注</span>
            </button>
          </div>
          <div class="oil-folder-children">${children}</div>
        </div>`;
    })
    .join("");
  $("sampleList").innerHTML = `${internalHtml}${groupHtml}`;
  document.querySelectorAll(".internal-folder-head").forEach((btn) => {
    btn.addEventListener("click", () => {
      const split = btn.dataset.split || "test";
      if (expandedInternalSplits.has(split)) expandedInternalSplits.delete(split);
      else expandedInternalSplits.add(split);
      renderSampleList();
    });
  });
  document.querySelectorAll(".project-folder-head").forEach((btn) => {
    btn.addEventListener("click", () => {
      const project = btn.dataset.project || "";
      if (expandedProjects.has(project)) expandedProjects.delete(project);
      else expandedProjects.add(project);
      saveProjectState();
      renderSampleList();
    });
    btn.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      showProjectMenu(btn.dataset.project || "", event.clientX, event.clientY);
    });
  });
  document.querySelectorAll(".oil-folder-head").forEach((btn) => {
    btn.addEventListener("click", () => {
      const group = btn.dataset.group || "未归入油样";
      const shouldClose = activeOilGroup === group && expandedOilGroups.has(group);
      activeOilGroup = group;
      if (shouldClose) {
        expandedOilGroups.delete(group);
      } else {
        expandedOilGroups.add(group);
      }
      selectOilGroup(group, { expand: !shouldClose });
    });
    btn.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      showOilFolderMenu(btn.dataset.group || "未归入油样", event.clientX, event.clientY);
    });
  });
  document.querySelectorAll(".rename-project-btn").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      openRenameProjectDialog(btn.dataset.project || "");
    });
  });
  document.querySelectorAll(".project-more-btn").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      showProjectMenu(btn.dataset.project || "", event.clientX, event.clientY);
    });
  });
  document.querySelectorAll(".delete-project-btn").forEach((btn) => {
    btn.addEventListener("click", async (event) => {
      event.stopPropagation();
      await deleteProject(btn.dataset.project || "");
    });
  });
  document.querySelectorAll(".rename-oil-btn").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      openRenameOilDialog(btn.dataset.group || "未归入油样");
    });
  });
  document.querySelectorAll(".oil-more-btn").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      showOilFolderMenu(btn.dataset.group || "未归入油样", event.clientX, event.clientY);
    });
  });
  document.querySelectorAll(".delete-oil-btn").forEach((btn) => {
    btn.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        await deleteOilGroup(btn.dataset.group || "未归入油样");
      } catch (err) {
        $("uploadHint").textContent = err.message || String(err);
      }
    });
  });
  document.querySelectorAll(".sample-item").forEach((btn) => {
    btn.addEventListener("click", () => loadByIndex(Number(btn.dataset.index)));
  });
  document.querySelectorAll(".sample-more-btn").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      showSampleImageMenu(Number(btn.dataset.index), event.clientX, event.clientY);
    });
  });
}

function showOilFolderMenu(group, x, y) {
  document.querySelector(".context-menu")?.remove();
  const menu = document.createElement("div");
  menu.className = "context-menu";
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  const projectName = oilGroupProjects[group];
  const count = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group).length;
  menu.innerHTML = `
    <button id="importOilGroupBtn" type="button">导入图像</button>
    <button id="batchAnalyzeOilGroupBtn" type="button">批量识别${count ? `（${count}张）` : ""}</button>
    <button id="renameOilGroupBtn" type="button">重命名油样</button>
    <button id="addOilToProjectBtn" type="button">加入项目</button>
    ${projectName ? `<button id="removeOilFromProjectBtn" type="button">移出当前项目</button>` : ""}
    <button id="deleteOilGroupBtn" class="danger-menu-item" type="button">删除油样文件夹${count ? `（${count}张）` : ""}</button>
  `;
  document.body.appendChild(menu);
  const close = () => menu.remove();
  $("importOilGroupBtn").addEventListener("click", () => {
    close();
    activeOilGroup = normalizeOilGroupName(group);
    oilGroups.add(activeOilGroup);
    deletedOilGroups.delete(activeOilGroup);
    expandedOilGroups.add(activeOilGroup);
    selectionScope = "oil";
    resultMode = "oil";
    saveProjectState();
    renderSampleList();
    updateOilWorkflowUi();
    $("uploadHint").textContent = `正在向油样“${activeOilGroup}”导入图像...`;
    $("uploadInput")?.click();
  });
  $("batchAnalyzeOilGroupBtn").addEventListener("click", async () => {
    close();
    queueOilGroupAnalysis(group, { refreshSummary: true });
  });
  $("renameOilGroupBtn").addEventListener("click", () => {
    close();
    openRenameOilDialog(group);
  });
  $("addOilToProjectBtn").addEventListener("click", () => {
    close();
    openAddToProjectDialog(group);
  });
  $("removeOilFromProjectBtn")?.addEventListener("click", () => {
    close();
    removeOilGroupFromProject(group);
  });
  $("deleteOilGroupBtn").addEventListener("click", async () => {
    close();
    try {
      await deleteOilGroup(group);
    } catch (err) {
      $("uploadHint").textContent = err.message || String(err);
    }
  });
  setTimeout(() => {
    document.addEventListener("click", close, { once: true });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") close();
    }, { once: true });
  }, 0);
}

function showSampleImageMenu(index, x, y) {
  const item = filtered[index];
  if (!item) return;
  document.querySelector(".context-menu")?.remove();
  const menu = document.createElement("div");
  menu.className = "context-menu";
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  const isUploaded = item.split === "uploaded";
  menu.innerHTML = `
    <button id="openSampleImageBtn" type="button">打开当前图像</button>
    ${isUploaded ? `<button id="analyzeSampleImageBtn" type="button">识别这张图片</button>` : ""}
    ${isUploaded ? `<button id="editSampleImageBtn" type="button">修改名称和备注</button>` : ""}
    ${isUploaded ? `<button id="deleteSampleImageBtn" class="danger-menu-item" type="button">删除这张图片</button>` : ""}
  `;
  document.body.appendChild(menu);
  const close = () => menu.remove();
  $("openSampleImageBtn").addEventListener("click", async () => {
    close();
    await loadByIndex(index);
  });
  $("analyzeSampleImageBtn")?.addEventListener("click", async () => {
    close();
    queueSampleImageAnalysis(item);
  });
  $("editSampleImageBtn")?.addEventListener("click", async () => {
    close();
    await loadByIndex(index);
    $("uploadManager").hidden = false;
    $("uploadManager").open = true;
  });
  $("deleteSampleImageBtn")?.addEventListener("click", async () => {
    close();
    try {
      await deleteUploadedSampleByName(item.image_name);
    } catch (err) {
      $("uploadHint").textContent = err.message || String(err);
    }
  });
  setTimeout(() => {
    document.addEventListener("click", close, { once: true });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") close();
    }, { once: true });
  }, 0);
}

function showProjectMenu(project, x, y) {
  document.querySelector(".context-menu")?.remove();
  const menu = document.createElement("div");
  menu.className = "context-menu";
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  const count = samples.filter((item) => item.split === "uploaded" && oilGroupProjects[uploadedGroup(item)] === project).length;
  menu.innerHTML = `
    <button id="analyzeProjectBtn" type="button">识别项目全部图像${count ? `（${count}张）` : ""}</button>
    <button id="renameProjectBtn" type="button">重命名项目</button>
    <button id="newOilInProjectBtn" type="button">在项目中新建油样</button>
    <button id="deleteProjectBtn" class="danger-menu-item" type="button">删除项目（保留油样）</button>
  `;
  document.body.appendChild(menu);
  const close = () => menu.remove();
  $("analyzeProjectBtn").addEventListener("click", async () => {
    close();
    queueProjectAnalysis(project);
  });
  $("renameProjectBtn").addEventListener("click", () => {
    close();
    openRenameProjectDialog(project);
  });
  $("newOilInProjectBtn").addEventListener("click", () => {
    close();
    openNewOilDialog(project);
  });
  $("deleteProjectBtn").addEventListener("click", async () => {
    close();
    await deleteProject(project);
  });
  setTimeout(() => {
    document.addEventListener("click", close, { once: true });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") close();
    }, { once: true });
  }, 0);
}

function openRenameOilDialog(group) {
  openOilNameDialog({
    title: "重命名油样",
    initialValue: group,
    confirmText: "保存",
    onConfirm: (name) => renameOilGroup(group, name),
  });
}

function openNewProjectDialog() {
  openOilNameDialog({
    title: "新建项目",
    label: "项目名称",
    initialValue: "",
    confirmText: "创建",
    placeholder: "例如：2026-07-12 某变电站检修",
    onConfirm: (name) => createProject(name),
  });
}

function openRenameProjectDialog(project) {
  openOilNameDialog({
    title: "重命名项目",
    label: "项目名称",
    initialValue: project,
    confirmText: "保存",
    placeholder: "例如：2026-07-12 某变电站检修",
    onConfirm: (name) => renameProject(project, name),
  });
}

function openNewOilDialog(projectName = "") {
  projectName = normalizeProjectName(projectName);
  openOilNameDialog({
    title: projectName ? "在项目中新建油样" : "新建油样",
    label: "油样名称",
    initialValue: "",
    confirmText: "创建",
    placeholder: "例如：1号主变油样-20260712",
    onConfirm: (name) => createOrSwitchOilSample(name, projectName),
  });
}

function openAddToProjectDialog(group) {
  const currentProject = normalizeProjectName(oilGroupProjects[group]);
  const projectList = [...projects].map(normalizeProjectName).filter(Boolean).sort(compareNames);
  if (!projectList.length) {
    openOilNameDialog({
      title: "加入项目",
      label: "项目名称",
      initialValue: currentProject,
      confirmText: "加入",
      placeholder: "例如：2026-07-12 某变电站检修",
      onConfirm: (name) => addOilGroupToProject(group, name),
    });
    return;
  }
  openProjectPickerDialog({
    group,
    projects: projectList,
    currentProject,
    onConfirm: (name) => addOilGroupToProject(group, name),
  });
}

function openProjectPickerDialog({ group, projects: projectList, currentProject = "", onConfirm }) {
  if ($("projectPickerDialog")) return;
  const overlay = document.createElement("div");
  overlay.id = "projectPickerDialog";
  overlay.className = "modal-backdrop";
  overlay.innerHTML = `
    <div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="projectPickerTitle">
      <h3 id="projectPickerTitle">加入项目</h3>
      <div class="modal-note">选择要纳入的项目：${esc(group || "未归入油样")}</div>
      <label>已有项目
        <select id="projectPickerSelect">
          ${projectList.map((project) => `<option value="${esc(project)}" ${project === currentProject ? "selected" : ""}>${esc(project)}</option>`).join("")}
        </select>
      </label>
      <label>或新建项目
        <input id="projectPickerNewName" type="text" maxlength="80" placeholder="例如：2026-07-12 某变电站检修" />
      </label>
      <div class="modal-actions">
        <button id="cancelProjectPickerBtn" type="button">取消</button>
        <button id="confirmProjectPickerBtn" class="primary-action" type="button">加入</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  const confirm = async () => {
    const typed = $("projectPickerNewName")?.value.trim();
    const selected = $("projectPickerSelect")?.value.trim();
    const projectName = typed || selected;
    if (!projectName) return;
    try {
      await onConfirm?.(projectName);
      close();
    } catch (err) {
      $("uploadHint").textContent = err.message || String(err);
    }
  };
  $("cancelProjectPickerBtn").addEventListener("click", close);
  $("confirmProjectPickerBtn").addEventListener("click", confirm);
  $("projectPickerNewName").addEventListener("keydown", (event) => {
    if (event.key === "Enter") confirm();
    if (event.key === "Escape") close();
  });
  $("projectPickerSelect").addEventListener("keydown", (event) => {
    if (event.key === "Enter") confirm();
    if (event.key === "Escape") close();
  });
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });
  setTimeout(() => $("projectPickerSelect")?.focus(), 0);
}

function syncCurrentSampleListItem() {
  if (!currentSample || !filtered.length) return;
  const item = filtered[currentIndex];
  if (!item || item.image_name !== currentSample.image_name || item.split !== currentSample.split) return;
  const detections = currentSample.detections || [];
  const counts = {};
  detections.forEach((det) => {
    counts[det.class_id] = (counts[det.class_id] || 0) + 1;
  });
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  item.particles = detections.length;
  item.dominant_class = dominant ? classNames[dominant[0]] || "-" : "-";
  const sampleItem = samples.find((sample) => sample.image_name === item.image_name && sample.split === item.split);
  if (sampleItem) {
    sampleItem.particles = item.particles;
    sampleItem.dominant_class = item.dominant_class;
    sampleItem.display_name = currentSample.display_name || sampleItem.display_name;
    sampleItem.group = currentSample.group || sampleItem.group;
    sampleItem.note = currentSample.note || sampleItem.note;
  }
  if (currentSample.split === "uploaded") activeOilGroup = currentSample.group || activeOilGroup;
  renderSampleList();
}

function renderUploadManager() {
  const panel = $("uploadManager");
  if (!panel) return;
  if (!currentSample || currentSample.split !== "uploaded") {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("managerSummary").textContent = `${currentSample.group || "未归入油样"} · ${currentSample.note ? "有备注" : "无备注"}`;
  $("manageName").value = currentSample.display_name || currentSample.image_name;
  $("manageGroup").value = currentSample.group || "未归入油样";
  $("manageNote").value = currentSample.note || "";
}

function syncAnalyzeButton() {
  const btn = $("analyzeUploadedBtn");
  if (!btn) return;
  btn.disabled = false;
  btn.textContent = currentSample?.split === "uploaded" ? "识别当前图像" : "重新分析当前样本";
}

function syncDeleteButton() {
  const btn = $("deleteUploadedBtn");
  if (!btn) return;
  btn.disabled = !currentSample || currentSample.split !== "uploaded";
  btn.title = btn.disabled ? "只有导入的检测图片可以删除" : "永久删除当前导入图片";
}

async function loadByIndex(index, { showLoading = true } = {}) {
  if (!filtered.length) return;
  selectionScope = "image";
  currentIndex = Math.max(0, Math.min(index, filtered.length - 1));
  renderSampleList();
  const item = filtered[currentIndex];
  if (showLoading) setAppLoading(15, "正在载入图像", item.display_name || item.image_name, { indeterminate: true });
  try {
    currentSample = await fetchJson(`/api/sample/${encodeURIComponent(item.split)}/${encodeURIComponent(item.image_name)}?${settingsQuery()}`);
    if (showLoading) setAppLoading(72, currentSample.cache_hit ? "正在读取识别缓存" : "YOLO 推理完成", "正在解码原始图像...");
    currentImage = await loadImage(currentSample.image_url);
  } catch (err) {
    if (showLoading) setAppLoading(100, "载入失败", err.message || String(err));
    throw err;
  }
  selectedParticleId = null;
  syncCurrentSampleListItem();
  syncAnalyzeButton();
  syncDeleteButton();
  renderUploadManager();
  updatePanels();
  drawCanvas();
  if (showLoading) finishAppLoading();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    const parsed = new URL(url, window.location.origin);
    const encodedPath = parsed.pathname
      .split("/")
      .map((part) => encodeURIComponent(decodeURIComponent(part)))
      .join("/");
    img.src = `${encodedPath}?t=${Date.now()}`;
  });
}

function drawCanvas() {
  if (!currentSample || !currentImage) return;
  const canvas = $("imageCanvas");
  const ctx = canvas.getContext("2d");
  canvas.width = currentSample.width;
  canvas.height = currentSample.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0);
  if (!$("boxToggle").checked) return;
  const showLabel = $("labelToggle").checked;
  ctx.font = "13px Segoe UI, Arial";
  activeDetections().forEach((det) => {
    const [xmin, ymin, xmax, ymax] = det.bbox;
    const color = colors[det.class_id] || "#e11d48";
    ctx.strokeStyle = color;
    ctx.lineWidth = det.particle_id === selectedParticleId ? 3.2 : 1.35;
    ctx.strokeRect(xmin + 0.5, ymin + 0.5, xmax - xmin, ymax - ymin);
    if (showLabel) {
      const label = `${det.class_cn} ${particleSizeLabel(det)}`;
      const metrics = ctx.measureText(label);
      const lx = xmin;
      const ly = Math.max(0, ymin - 18);
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.fillRect(lx, ly, metrics.width + 8, 17);
      ctx.fillStyle = color;
      ctx.fillText(label, lx + 4, ly + 12);
    }
  });
}

function updatePanels() {
  renderModelStatus();
  const active = activeDetections();
  const summary = currentSample.sample_level_summary || {};
  const groupText = currentSample.split === "uploaded" ? `${currentSample.group || "未归入油样"} / ` : "";
  $("imageTitle").textContent = `${groupText}${currentSample.display_name || currentSample.image_name}`;
  $("imageMeta").textContent = `${currentSample.split} · ${currentSample.width}×${currentSample.height} · 当前显示 ${active.length}/${currentSample.detections.length} 个目标 · ${currentSample.quantifiable ? "可定量原始帧" : "识别图（不可定量）"}`;
  $("totalParticles").textContent = active.length;
  $("smallParticleMetric").innerHTML = `${smallParticleCount(active)} 个<small>本项目关注口径；不用于线性限值判定</small>`;
  $("totalConc").innerHTML = standardConcentrationHtml(
    standardParticleCount(currentSample.detections || []),
    currentSample.volume_ul,
    currentSample.quantifiable,
  );
  $("diamValue").textContent = `${$("diamSlider").value} μm`;
  $("diamQuantiles").textContent = `${fmt(summary.d10, 1)} / ${fmt(summary.d50, 1)} / ${fmt(summary.d90, 1)} μm`;
  $("riskLevel").textContent = summary.overall_risk_level || "-";
  renderClassStats(active);
  renderHistogram(active);
  renderSampleRisk(summary);
  renderSelection();
  renderParticleInspector(selectedDetection());
  renderRows(active);
  renderOilSampleSummary();
  syncResultMode();
}

async function collectOilGroupDetections(group) {
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group);
  const rows = [];
  for (const item of members) {
    const payload =
      currentSample?.split === "uploaded" && currentSample.image_name === item.image_name
        ? currentSample
        : await fetchJson(`/api/sample/uploaded/${encodeURIComponent(item.image_name)}?${settingsQuery()}`);
    (payload.detections || []).forEach((det) => {
      rows.push({
        ...det,
        source_image_name: item.image_name,
        source_display_name: item.display_name || item.image_name,
      });
    });
  }
  return rows;
}

async function renderOilGroupDetailRows(group) {
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group);
  if (!members.length) {
    oilDetailRows = [];
    $("rowCount").textContent = "0 行";
    $("detailRows").innerHTML = "";
    return;
  }
  $("rowCount").textContent = `正在汇总 ${members.length} 张图像...`;
  oilDetailRows = await collectOilGroupDetections(group);
  renderRows(oilDetailRows, { scope: "oil" });
}

function renderClassStats(active) {
  const counts = {};
  Object.keys(classNames).forEach((cid) => (counts[cid] = 0));
  active.forEach((det) => (counts[det.class_id] += 1));
  const maxCount = Math.max(...Object.values(counts), 1);
  const volumeUl = currentSample?.volume_ul;
  $("classStats").innerHTML = Object.entries(classNames)
    .map(([cid, name]) => {
      const count = counts[cid] || 0;
      const rows = active.filter((det) => String(det.class_id) === cid);
      const meanDiam = rows.length ? rows.reduce((sum, det) => sum + det.eq_diameter_um, 0) / rows.length : 0;
      const ratio = active.length ? (count / active.length) * 100 : 0;
      const width = (count / maxCount) * 100;
      const conc = currentSample?.quantifiable ? concentrationPerMl(count, volumeUl) : null;
      return `
        <div class="class-row">
          <button class="class-chip" data-class="${cid}"><i class="swatch" style="background:${colors[cid]}"></i>${name}</button>
          <span class="bar"><span style="width:${width}%; background:${colors[cid]}"></span></span>
          <strong>${count}</strong>
        </div>
        <div class="row-sub">${fmt(ratio, 1)}% · ${conc == null ? "不可定量" : `${fmtInt(conc * STANDARD_VOLUME_ML)} 个/${STANDARD_VOLUME_ML} mL`} · 平均等效直径 ${fmt(meanDiam, 1)} μm</div>
      `;
    })
    .join("");
  document.querySelectorAll(".class-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("classFilter").value = btn.dataset.class;
      updatePanels();
      drawCanvas();
    });
  });
}
function renderHistogram(active) {
  const canvas = $("histCanvas");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(360, Math.floor(rect.width || canvas.clientWidth || 360));
  const h = 220;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.height = `${h}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#f8fafb";
  ctx.fillRect(0, 0, w, h);
  const bins = new Array(12).fill(0);
  active.forEach((det) => {
    const idx = Math.max(0, Math.min(bins.length - 1, Math.floor(det.eq_diameter_um / 8)));
    bins[idx] += 1;
  });
  const max = Math.max(...bins, 1);
  const left = 46;
  const right = 18;
  const top = 22;
  const bottom = 42;
  const plotW = w - left - right;
  const plotH = h - top - bottom;
  const bw = plotW / bins.length;
  const niceMax = Math.max(1, Math.ceil(max / 5) * 5);

  ctx.strokeStyle = "#e0e7eb";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#6b7782";
  ctx.font = "12px Segoe UI, Microsoft YaHei, Arial";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  [0, 0.5, 1].forEach((ratio) => {
    const y = top + plotH * (1 - ratio);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(w - right, y);
    ctx.stroke();
    ctx.fillText(String(Math.round(niceMax * ratio)), left - 8, y);
  });

  ctx.strokeStyle = "#9fb0bb";
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, top + plotH);
  ctx.lineTo(w - right, top + plotH);
  ctx.stroke();

  bins.forEach((v, i) => {
    const bh = (v / niceMax) * plotH;
    const x = left + i * bw + 5;
    const y = top + plotH - bh;
    const barW = Math.max(8, bw - 10);
    ctx.fillStyle = i < 2 ? "#9aa3ad" : i < 6 ? "#2b7898" : "#8d4ca6";
    ctx.fillRect(x, y, barW, bh);
    if (v > 0) {
      ctx.fillStyle = "#26323b";
      ctx.font = "11px Segoe UI, Microsoft YaHei, Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(String(v), x + barW / 2, y - 4);
    }
  });

  ctx.fillStyle = "#53616c";
  ctx.font = "12px Segoe UI, Microsoft YaHei, Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  [0, 24, 48, 72, 96].forEach((tick) => {
    const x = left + (tick / 96) * plotW;
    ctx.strokeStyle = "#9fb0bb";
    ctx.beginPath();
    ctx.moveTo(x, top + plotH);
    ctx.lineTo(x, top + plotH + 5);
    ctx.stroke();
    ctx.fillText(tick === 96 ? "96 μm" : String(tick), x, top + plotH + 10);
  });
  ctx.textAlign = "left";
  ctx.font = "12px Segoe UI, Microsoft YaHei, Arial";
  ctx.fillText("粒径 / μm", left, h - 18);
}

function renderSampleRisk(summary) {
  const actions = (summary.recommended_actions || []).map((x) => `<li>${x}</li>`).join("");
  $("sampleRisk").innerHTML = `
    <p><strong>主导类型：</strong>${summary.dominant_particle_type || "-"}</p>
    <p><strong>高风险颗粒数：</strong>${summary.high_risk_particle_count ?? 0}</p>
    <p>${summary.risk_summary || "暂无样品级风险提示。"}</p>
    <ul>${actions}</ul>
  `;
}
function renderSelection() {
  const det = currentSample?.detections.find((item) => item.particle_id === selectedParticleId);
  if (!det) {
    $("selectionInfo").textContent = "点击图像框或表格行查看单颗粒参数、解释依据和风险提示。";
    return;
  }
  const m = det.morphology || {};
  const size = bboxSizeUm(det);
  $("selectionInfo").innerHTML = `
    <strong>${det.class_cn}</strong>
    <span>ID ${det.particle_id} · 置信度 ${confidenceText(det)} · bbox ${det.bbox.join(",")}</span>
    <span>等效直径 ${fmt(det.eq_diameter_um, 2)} μm · 框尺寸 ${fmt(size.width, 1)}×${fmt(size.height, 1)} μm · 最大径 ${fmt(size.maxDim, 1)} μm</span>
    <span>长宽比 ${fmt(m.aspect_ratio, 2)} · 圆度 ${fmt(m.circularity, 2)}</span>
    <span>实心度 ${fmt(m.solidity, 2)} · 边缘清晰度 ${fmt(m.edge_sharpness, 1)} · 填充率 ${fmt(m.bbox_fill_ratio, 2)}</span>
    <p><strong>解释依据：</strong>${det.explanation}</p>
    <p><strong>风险提示：</strong>${det.risk_hint}</p>
  `;
}
function selectedDetection() {
  return currentSample?.detections.find((item) => item.particle_id === selectedParticleId) || null;
}

function renderParticleInspector(det = selectedDetection()) {
  const canvas = $("roiCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f8fafb";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!det || !currentImage || !currentSample) {
    $("roiTitle").textContent = "未选择颗粒";
    $("roiClassBadge").textContent = "-";
    $("roiClassBadge").style.background = "#edf3f6";
    $("roiClassBadge").style.color = "#32414c";
    $("roiMetrics").className = "roi-metrics empty";
    $("roiMetrics").textContent = "点击左侧检测框，或点击下方表格中的 ID，查看该颗粒的高清放大图和形态参数标识。";
    ctx.fillStyle = "#8a98a5";
    ctx.font = "14px Segoe UI, Microsoft YaHei, Arial";
    ctx.textAlign = "center";
    ctx.fillText("选择一个颗粒查看 ROI", canvas.width / 2, canvas.height / 2);
    syncManualReviewPanel();
    return;
  }

  const [xmin, ymin, xmax, ymax] = det.bbox;
  const bw = Math.max(1, xmax - xmin);
  const bh = Math.max(1, ymax - ymin);
  const pad = Math.max(10, Math.ceil(Math.max(bw, bh) * 1.8));
  const sx = Math.max(0, xmin - pad);
  const sy = Math.max(0, ymin - pad);
  const ex = Math.min(currentSample.width, xmax + pad);
  const ey = Math.min(currentSample.height, ymax + pad);
  const sw = Math.max(1, ex - sx);
  const sh = Math.max(1, ey - sy);
  const scale = Math.min(canvas.width / sw, canvas.height / sh);
  const dw = sw * scale;
  const dh = sh * scale;
  const dx = (canvas.width - dw) / 2;
  const dy = (canvas.height - dh) / 2;
  const color = colors[det.class_id] || "#e11d48";

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(currentImage, sx, sy, sw, sh, dx, dy, dw, dh);
  ctx.imageSmoothingEnabled = true;

  const rx = dx + (xmin - sx) * scale;
  const ry = dy + (ymin - sy) * scale;
  const rw = bw * scale;
  const rh = bh * scale;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.strokeRect(rx + 0.5, ry + 0.5, rw, rh);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "rgba(40, 111, 143, 0.7)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(rx + rw / 2, dy);
  ctx.lineTo(rx + rw / 2, dy + dh);
  ctx.moveTo(dx, ry + rh / 2);
  ctx.lineTo(dx + dw, ry + rh / 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillRect(8, 8, 128, 26);
  ctx.fillStyle = color;
  ctx.font = "12px Segoe UI, Microsoft YaHei, Arial";
  ctx.textAlign = "left";
  ctx.fillText(`ID ${det.particle_id} · ${particleSizeLabel(det)}`, 14, 25);

  const m = det.morphology || {};
  const size = bboxSizeUm(det);
  $("roiTitle").textContent = `ID ${det.particle_id}`;
  $("roiClassBadge").textContent = det.class_cn;
  $("roiClassBadge").style.background = `${color}1f`;
  $("roiClassBadge").style.color = color;
  $("roiMetrics").className = "roi-metrics";
  $("roiMetrics").innerHTML = `
    <div class="roi-metric"><span>类别</span><strong>${det.class_cn}</strong></div>
    <div class="roi-metric"><span>置信度</span><strong>${confidenceText(det)}</strong></div>
    <div class="roi-metric"><span>等效直径</span><strong>${fmt(det.eq_diameter_um, 2)} μm</strong></div>
    <div class="roi-metric"><span>框尺寸</span><strong>${fmt(size.width, 1)} × ${fmt(size.height, 1)} μm</strong></div>
    <div class="roi-metric"><span>最大径</span><strong>${fmt(size.maxDim, 1)} μm</strong></div>
    <div class="roi-metric"><span>bbox</span><strong>${det.bbox.join(", ")}</strong></div>
    <div class="roi-metric"><span>长宽比</span><strong>${fmt(m.aspect_ratio, 2)}</strong></div>
    <div class="roi-metric"><span>主轴比</span><strong>${fmt(m.principal_axis_ratio, 2)}</strong></div>
    <div class="roi-metric"><span>圆度</span><strong>${fmt(m.circularity, 2)}</strong></div>
    <div class="roi-metric"><span>实心度</span><strong>${fmt(m.solidity, 2)}</strong></div>
    <div class="roi-metric"><span>边缘清晰度</span><strong>${fmt(m.edge_sharpness, 1)}</strong></div>
    <div class="roi-metric"><span>填充率</span><strong>${fmt(m.bbox_fill_ratio, 2)}</strong></div>
    <div class="roi-metric"><span>灰度均值</span><strong>${fmt(m.gray_mean, 1)}</strong></div>
    ${det.shape_review_reason ? `<div class="roi-note"><strong>形态审核：</strong>${esc(det.shape_review_reason)}</div>` : ""}
    ${det.classification_rescued ? `<div class="roi-note"><strong>处理说明：</strong>低置信度纤维候选已由形态特征救回，建议人工复核。</div>` : ""}
    <div class="roi-note"><strong>解释依据：</strong>${det.explanation || "-"}</div>
    <div class="roi-note"><strong>风险提示：</strong>${det.risk_hint || "-"}</div>
  `;
}
function renderRows(active = activeDetections(), options = {}) {
  const scope = options.scope || selectionScope;
  renderDetailHeader(scope);
  const rows = active.slice(0, 650);
  const label = scope === "oil" ? `${activeOilGroup} · 油样级` : "当前图像";
  $("rowCount").textContent = `${label} · ${rows.length} 行${active.length > rows.length ? "（截取显示）" : ""}`;
  $("detailRows").innerHTML = rows
    .map((det, idx) => {
      const m = det.morphology || {};
      const selected = det.particle_id === selectedParticleId && (!det.source_image_name || det.source_image_name === currentSample?.image_name);
      return `
      <tr class="${selected ? "selected" : ""}">
        ${scope === "oil" ? `<td>${idx + 1}</td><td>${esc(det.source_display_name || det.source_image_name || "-")}</td>` : ""}
        <td><button class="id-button" data-row-index="${idx}" title="查看第 ${idx + 1} 行颗粒的 ROI 放大图">${det.particle_id}</button></td>
        <td><span class="legend-item"><i class="swatch" style="background:${colors[det.class_id]}"></i>${det.class_cn}</span></td>
        <td>${confidenceText(det)}</td>
        <td>${det.bbox.join(",")}</td>
        <td>${fmt(det.eq_diameter_um, 2)}</td>
        <td>${fmt(m.aspect_ratio, 2)}</td>
        <td>${fmt(m.circularity, 2)}</td>
        <td>${fmt(m.solidity, 2)}</td>
        <td>${fmt(m.edge_sharpness, 1)}</td>
        <td>${fmt(m.bbox_fill_ratio, 2)}</td>
        <td>${fmt(m.gray_mean, 1)}</td>
      </tr>`;
    })
    .join("");
  document.querySelectorAll("#detailRows tr").forEach((tr, idx) => {
    tr.addEventListener("click", async () => {
      const row = rows[idx];
      if (scope === "oil" && row.source_image_name) {
        await openOilDetection(row);
        return;
      }
      selectedParticleId = row.particle_id;
      renderSelection();
      renderParticleInspector(selectedDetection());
      renderRows(scope === "oil" ? oilDetailRows : activeDetections(), { scope });
      drawCanvas();
    });
  });
  document.querySelectorAll(".id-button").forEach((btn) => {
    btn.addEventListener("click", async (event) => {
      event.stopPropagation();
      const row = rows[Number(btn.dataset.rowIndex)];
      if (scope === "oil" && row?.source_image_name) {
        await openOilDetection(row);
        return;
      }
      selectedParticleId = row?.particle_id ?? null;
      renderSelection();
      renderParticleInspector(selectedDetection());
      renderRows(scope === "oil" ? oilDetailRows : activeDetections(), { scope });
      drawCanvas();
    });
  });
}

async function openOilDetection(det, { preserveReview = false } = {}) {
  if (!det?.source_image_name) return;
  const keepReview = preserveReview && oilClassReview
    ? {
        group: oilClassReview.group,
        detections: oilClassReview.detections || [],
        activeClassId: oilClassReview.activeClassId,
        scrollTop: oilClassReview.scrollTop || 0,
      }
    : null;
  const imageIdx = filtered.findIndex((item) => item.split === "uploaded" && item.image_name === det.source_image_name);
  if (imageIdx >= 0) {
    await loadByIndex(imageIdx);
  }
  selectedParticleId = det.particle_id ?? null;
  renderSelection();
  renderParticleInspector(selectedDetection());
  renderRows(activeDetections(), { scope: "image" });
  drawCanvas();
  if (keepReview) {
    oilClassReview = keepReview;
    resultMode = "oil";
    syncResultMode();
    await renderOilSampleSummary(keepReview.group);
  }
}

function renderDetailHeader(scope = selectionScope) {
  const header = $("detailHeaderRow");
  if (!header) return;
  header.innerHTML = `
    ${scope === "oil" ? "<th>序号</th><th>来源图像</th>" : ""}
    <th>图内编号</th>
    <th>类别</th>
    <th>置信度</th>
    <th>bbox</th>
    <th>等效直径/μm</th>
    <th>长宽比</th>
    <th>圆度</th>
    <th>实心度</th>
    <th>边缘清晰度</th>
    <th>填充率</th>
    <th>灰度均值</th>
  `;
  syncManualReviewPanel();
}

function exportCurrentCsv() {
  if (!currentSample) return;
  const headers = ["particle_id", "class_id", "class_name", "confidence", "bbox", "eq_diameter_um", "aspect_ratio", "circularity", "solidity", "edge_sharpness", "bbox_fill_ratio", "explanation", "risk_hint"];
  const lines = [headers.join(",")];
  activeDetections().forEach((det) => {
    const m = det.morphology || {};
    const row = {
      particle_id: det.particle_id,
      class_id: det.class_id,
      class_name: det.class_name,
      confidence: det.confidence ?? "",
      bbox: det.bbox.join(" "),
      eq_diameter_um: det.eq_diameter_um,
      aspect_ratio: m.aspect_ratio,
      circularity: m.circularity,
      solidity: m.solidity,
      edge_sharpness: m.edge_sharpness,
      bbox_fill_ratio: m.bbox_fill_ratio,
      explanation: det.explanation,
      risk_hint: det.risk_hint,
    };
    lines.push(headers.map((key) => `"${String(row[key] ?? "").replaceAll('"', '""')}"`).join(","));
  });
  downloadBlob(lines.join("\n"), currentSample.image_name.replace(".png", "_detections.csv"), "text/csv;charset=utf-8");
}

async function exportReport() {
  if (!currentSample) return;
  const report = await fetchJson(`/api/report/${currentSample.split}/${currentSample.image_name}?${settingsQuery()}`);
  downloadBlob(JSON.stringify(report, null, 2), currentSample.image_name.replace(".png", "_report.json"), "application/json;charset=utf-8");
}

function downloadBlob(text, filename, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function confirmDialog({ title = "确认操作", message = "", confirmText = "确定", danger = false } = {}) {
  return new Promise((resolve) => {
    document.querySelector(".context-menu")?.remove();
    $("confirmDialog")?.remove();
    const overlay = document.createElement("div");
    overlay.id = "confirmDialog";
    overlay.className = "modal-backdrop";
    overlay.innerHTML = `
      <div class="modal-panel confirm-panel" role="dialog" aria-modal="true" aria-labelledby="confirmDialogTitle">
        <h3 id="confirmDialogTitle">${esc(title)}</h3>
        <p class="confirm-message">${esc(message)}</p>
        <div class="modal-actions">
          <button id="cancelConfirmBtn" type="button">取消</button>
          <button id="okConfirmBtn" class="${danger ? "danger-action" : "primary-action"}" type="button">${esc(confirmText)}</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    const finish = (ok) => {
      overlay.remove();
      resolve(ok);
    };
    $("cancelConfirmBtn").addEventListener("click", () => finish(false));
    $("okConfirmBtn").addEventListener("click", () => finish(true));
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) finish(false);
    });
    overlay.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish(false);
      if (event.key === "Enter") finish(true);
    });
    setTimeout(() => $("okConfirmBtn")?.focus(), 0);
  });
}

function syncSettingsUi() {
  $("settingConf").value = inferSettings.conf;
  $("settingOtherConf").value = inferSettings.other_conf;
  $("settingIou").value = inferSettings.iou;
  $("settingImgsz").value = inferSettings.imgsz;
  $("settingMaxDet").value = inferSettings.max_det;
  $("settingDevice").value = inferSettings.device;
  $("settingPixelSize").value = inferSettings.pixel_size_um;
  $("settingAgnostic").checked = inferSettings.agnostic_nms;
  $("settingConfValue").textContent = Number(inferSettings.conf).toFixed(2);
  $("settingOtherConfValue").textContent = Number(inferSettings.other_conf).toFixed(2);
  $("settingIouValue").textContent = Number(inferSettings.iou).toFixed(2);
}

function readSettingsUi() {
  inferSettings = {
    conf: Number($("settingConf").value),
    other_conf: Math.max(Number($("settingConf").value), Number($("settingOtherConf").value)),
    iou: Number($("settingIou").value),
    imgsz: Number($("settingImgsz").value),
    max_det: Number($("settingMaxDet").value),
    device: $("settingDevice").value.trim(),
    agnostic_nms: $("settingAgnostic").checked,
    pixel_size_um: Number($("settingPixelSize").value),
  };
  syncSettingsUi();
}

async function reloadCurrentSample() {
  if (!filtered.length || !currentSample) {
    $("uploadHint").textContent = "请先上传或选择一张图像，再点击识别。";
    return;
  }
  const keepId = selectedParticleId;
  const item = filtered[currentIndex];
  try {
    setRecognitionStatus(`识别中... 当前图像 · ${item.display_name || item.image_name}`);
    currentSample = await fetchJson(`/api/sample/${encodeURIComponent(item.split)}/${encodeURIComponent(item.image_name)}?${settingsQuery()}`);
    currentImage = await loadImage(currentSample.image_url);
    selectedParticleId = keepId;
    if (!selectedDetection()) selectedParticleId = null;
    syncCurrentSampleListItem();
    invalidateOilSummaryCache();
    updateOilWorkflowUi();
    renderUploadManager();
    updatePanels();
    drawCanvas();
  } finally {
    setRecognitionStatus("识别完成：当前图像", "done");
    syncAnalyzeButton();
  }
}

async function uploadImage(file) {
  if (!file) return;
  $("uploadHint").textContent = `正在导入并识别：${file.name}`;
  const body = new FormData();
  body.append("image", file);
  body.append("display_name", file.name);
  body.append("group", activeOilGroup || DEFAULT_OIL_GROUP);
  body.append("note", "");
  const res = await fetch("/api/upload", { method: "POST", body });
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || "上传失败");
  samples = await fetchJson("/api/samples");
  $("splitFilter").value = "uploaded";
  applyFilters();
  const idx = filtered.findIndex((item) => item.split === payload.split && item.image_name === payload.image_name);
  $("uploadHint").textContent = "图像已导入，正在运行 YOLO 推理...";
  await loadByIndex(Math.max(0, idx));
  $("uploadHint").textContent = currentSample?.inference_time_ms == null
    ? "识别完成。"
    : `识别完成：YOLO 推理耗时 ${fmt(currentSample.inference_time_ms, 1)} ms。`;
}

async function uploadImages(files) {
  const queue = Array.from(files || []).filter(Boolean);
  if (!queue.length) return;
  const uploaded = [];
  activeOilGroup = activeOilGroup || "未归入油样";
  deletedOilGroups.delete(activeOilGroup);
  expandedOilGroups.add(activeOilGroup);
  saveProjectState();
  uploadProgress = { group: activeOilGroup, current: 0, total: queue.length, fileName: "" };
  renderSampleList();
  try {
    for (let i = 0; i < queue.length; i += 1) {
      const file = queue[i];
      uploadProgress = { group: activeOilGroup, current: i, total: queue.length, fileName: file.name };
      renderSampleList();
      $("uploadHint").textContent = `正在导入 ${activeOilGroup} 图像 ${i + 1}/${queue.length}：${file.name}`;
      const body = new FormData();
      body.append("image", file);
      body.append("display_name", file.name);
      body.append("group", activeOilGroup);
      body.append("note", "");
      const res = await fetch("/api/upload", { method: "POST", body });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || "上传失败");
      const metaPayload = {
        display_name: file.name,
        group: activeOilGroup,
        note: "",
      };
      await fetchJson(`/api/uploaded/${encodeURIComponent(payload.image_name)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(metaPayload),
      });
      uploaded.push(payload);
      uploadProgress = { group: activeOilGroup, current: i + 1, total: queue.length, fileName: file.name };
      renderSampleList();
    }
  } finally {
    uploadProgress = null;
    renderSampleList();
  }
  samples = await fetchJson("/api/samples");
  invalidateOilSummaryCache();
  $("splitFilter").value = "uploaded";
  applyFilters();
  const last = uploaded[uploaded.length - 1];
  const idx = filtered.findIndex((item) => item.split === last.split && item.image_name === last.image_name);
  updateOilWorkflowUi();
  $("uploadHint").textContent = `已导入 ${activeOilGroup} 的 ${uploaded.length} 张图像，正在显示最后一张并汇总油样批次...`;
  await loadByIndex(Math.max(0, idx));
  $("uploadHint").textContent = `已导入 ${uploaded.length} 张图像到油样批次：${activeOilGroup}。`;
}

async function saveCurrentSampleMeta() {
  if (!currentSample || currentSample.split !== "uploaded") return;
  const payload = {
    display_name: $("manageName").value.trim() || currentSample.image_name,
    group: $("manageGroup").value.trim() || "未归入油样",
    note: $("manageNote").value.trim(),
  };
  const saved = await fetchJson(`/api/uploaded/${encodeURIComponent(currentSample.image_name)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  Object.assign(currentSample, saved);
  invalidateOilSummaryCache();
  const sample = samples.find((item) => item.split === "uploaded" && item.image_name === currentSample.image_name);
  if (sample) Object.assign(sample, saved);
  const item = filtered.find((entry) => entry.split === "uploaded" && entry.image_name === currentSample.image_name);
  if (item) Object.assign(item, saved);
  activeOilGroup = saved.group || activeOilGroup;
  deletedOilGroups.delete(normalizeOilGroupName(activeOilGroup));
  expandedOilGroups.add(activeOilGroup);
  saveProjectState();
  renderSampleList();
  updateOilWorkflowUi();
  renderUploadManager();
  $("uploadManager").open = false;
  $("uploadHint").textContent = "样本名称、油样批次和备注已保存。";
}

async function deleteCurrentUploadedSample() {
  if (!currentSample || currentSample.split !== "uploaded") return;
  await deleteUploadedSampleByName(currentSample.image_name, currentSample.display_name || currentSample.image_name);
}

async function deleteUploadedSampleByName(imageName, label = imageName) {
  const ok = await confirmDialog({
    title: "删除检测图片",
    message: `永久删除这张检测图片？${label ? `\n${label}` : ""}`,
    confirmText: "删除图片",
    danger: true,
  });
  if (!ok) return;
  try {
    setOperationProgress({
      group: activeOilGroup,
      title: "正在删除检测图片",
      label: "删除图片",
      current: 0,
      total: 1,
      detail: label,
    });
    await fetchJson(`/api/uploaded/${encodeURIComponent(imageName)}`, { method: "DELETE" });
    setOperationProgress({
      group: activeOilGroup,
      title: "正在删除检测图片",
      label: "删除图片",
      current: 1,
      total: 1,
      detail: label,
    });
  } finally {
    setOperationProgress(null);
  }
  samples = await fetchJson("/api/samples");
  invalidateOilSummaryCache();
  currentIndex = Math.max(0, currentIndex - 1);
  if (currentSample?.image_name === imageName && currentSample?.split === "uploaded") {
    currentSample = null;
    currentImage = null;
    selectedParticleId = null;
  }
  applyFilters();
  if (filtered.length) {
    await loadByIndex(currentIndex);
  } else {
    renderUploadManager();
    $("imageTitle").textContent = "-";
    $("imageMeta").textContent = "请导入一张 MFI 原始图像。";
    $("uploadHint").textContent = "已永久删除。当前没有检测图片。";
    syncDeleteButton();
  }
}

async function deleteOilGroup(group) {
  group = normalizeOilGroupName(group);
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group);
  if (!members.length) {
    if (group !== DEFAULT_OIL_GROUP) deletedOilGroups.add(group);
    oilGroups.delete(group);
    delete oilGroupProjects[group];
    expandedOilGroups.delete(group);
    if (activeOilGroup === group) activeOilGroup = DEFAULT_OIL_GROUP;
    saveProjectState();
    applyFilters();
    $("uploadHint").textContent = `已删除空油样文件夹：${group}`;
    return;
  }
  const ok = await confirmDialog({
    title: "删除油样文件夹",
    message: `永久删除油样文件夹“${group}”及其中 ${members.length} 张检测图片？`,
    confirmText: "删除油样",
    danger: true,
  });
  if (!ok) return;
  try {
    expandedOilGroups.add(group);
    for (let i = 0; i < members.length; i += 1) {
      const item = members[i];
      setOperationProgress({
        group,
        title: `正在删除油样“${group}”`,
        label: `删除 ${group}`,
        current: i,
        total: members.length,
        detail: item.display_name || item.image_name,
      });
      await fetchJson(`/api/uploaded/${encodeURIComponent(item.image_name)}`, { method: "DELETE" });
      setOperationProgress({
        group,
        title: `正在删除油样“${group}”`,
        label: `删除 ${group}`,
        current: i + 1,
        total: members.length,
        detail: item.display_name || item.image_name,
      });
    }
  } finally {
    setOperationProgress(null);
  }
  samples = await fetchJson("/api/samples");
  if (group !== DEFAULT_OIL_GROUP) deletedOilGroups.add(group);
  oilGroups.delete(group);
  delete oilGroupProjects[group];
  expandedOilGroups.delete(group);
  if (activeOilGroup === group) activeOilGroup = DEFAULT_OIL_GROUP;
  currentSample = null;
  currentImage = null;
  selectedParticleId = null;
  invalidateOilSummaryCache();
  saveProjectState();
  applyFilters();
  if (filtered.length) {
    await loadByIndex(0);
  } else {
    renderUploadManager();
    $("imageTitle").textContent = "-";
    $("imageMeta").textContent = "请导入一张 MFI 原始图像。";
    syncDeleteButton();
  }
  $("uploadHint").textContent = `已删除油样文件夹“${group}”及其中 ${members.length} 张检测图片。`;
}

function createOrSwitchOilSample(name, projectName = "") {
  name = normalizeOilGroupName(name);
  projectName = normalizeProjectName(projectName);
  deletedOilGroups.delete(name);
  oilGroups.add(name);
  if (projectName) {
    projects.add(projectName);
    expandedProjects.add(projectName);
    oilGroupProjects[name] = projectName;
  }
  saveProjectState();
  selectOilGroup(name);
  $("uploadHint").textContent = projectName
    ? `已在项目“${projectName}”中新建油样：${name}。现在可以导入该油样的 MFI 图像。`
    : `已选中油样：${name}。现在可以导入该油样文件夹中的 MFI 图像。`;
}

function openOilNameDialog({ title, label = "油样名称", initialValue = "", confirmText = "确定", placeholder = "例如：1号主变油样-20260712", onConfirm }) {
  if ($("newOilDialog")) return;
  const overlay = document.createElement("div");
  overlay.id = "newOilDialog";
  overlay.className = "modal-backdrop";
  overlay.innerHTML = `
    <div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="newOilTitle">
      <h3 id="newOilTitle">${esc(title)}</h3>
      <label>${esc(label)}<input id="newOilNameInput" type="text" maxlength="80" value="${esc(initialValue)}" placeholder="${esc(placeholder)}" /></label>
      <div class="modal-actions">
        <button id="cancelNewOilBtn" type="button">取消</button>
        <button id="confirmNewOilBtn" class="primary-action" type="button">${esc(confirmText)}</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  const confirm = async () => {
    const name = $("newOilNameInput")?.value.trim();
    if (!name) {
      $("newOilNameInput")?.focus();
      return;
    }
    try {
      await onConfirm?.(name);
      close();
    } catch (err) {
      $("uploadHint").textContent = err.message || String(err);
    }
  };
  $("cancelNewOilBtn").addEventListener("click", close);
  $("confirmNewOilBtn").addEventListener("click", confirm);
  $("newOilNameInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") confirm();
    if (event.key === "Escape") close();
  });
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });
  setTimeout(() => $("newOilNameInput")?.focus(), 0);
}

async function renameOilGroup(oldName, newName) {
  oldName = normalizeOilGroupName(oldName);
  newName = normalizeOilGroupName(newName);
  if (oldName === newName) return;
  deletedOilGroups.delete(newName);
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === oldName);
  for (const item of members) {
    const payload = {
      display_name: item.display_name || item.image_name,
      group: newName,
      note: item.note || "",
    };
    await fetchJson(`/api/uploaded/${encodeURIComponent(item.image_name)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    Object.assign(item, payload);
  }
  oilGroups.delete(oldName);
  oilGroups.add(newName);
  if (oilGroupProjects[oldName]) {
    oilGroupProjects[newName] = oilGroupProjects[oldName];
    delete oilGroupProjects[oldName];
  }
  if (expandedOilGroups.has(oldName)) {
    expandedOilGroups.delete(oldName);
    expandedOilGroups.add(newName);
  }
  if (activeOilGroup === oldName) activeOilGroup = newName;
  if (currentSample?.split === "uploaded" && (currentSample.group || "未归入油样") === oldName) {
    currentSample.group = newName;
  }
  saveProjectState();
  invalidateOilSummaryCache();
  applyFilters();
  renderUploadManager();
  renderOilSampleSummary();
  $("uploadHint").textContent = `油样已重命名：${oldName} → ${newName}`;
}

function addOilGroupToProject(group, projectName) {
  group = group || "未归入油样";
  projectName = recoverProjectName(projectName);
  projects.add(projectName);
  expandedProjects.add(projectName);
  oilGroupProjects[group] = projectName;
  saveProjectState();
  renderSampleList();
  $("uploadHint").textContent = `已将油样“${group}”加入项目“${projectName}”。`;
}

function createProject(projectName) {
  projectName = recoverProjectName(projectName);
  projects.add(projectName);
  expandedProjects.add(projectName);
  saveProjectState();
  renderSampleList();
  $("uploadHint").textContent = `已新建项目“${projectName}”。可以右键油样加入该项目，或在项目菜单中新建油样。`;
}

function renameProject(oldName, newName) {
  oldName = recoverProjectName(oldName);
  newName = recoverProjectName(newName);
  if (oldName === newName) return;
  projects.delete(oldName);
  projects.add(newName);
  Object.keys(oilGroupProjects).forEach((group) => {
    if (oilGroupProjects[group] === oldName) oilGroupProjects[group] = newName;
  });
  if (expandedProjects.has(oldName)) {
    expandedProjects.delete(oldName);
    expandedProjects.add(newName);
  }
  saveProjectState();
  renderSampleList();
  $("uploadHint").textContent = `项目已重命名：${oldName} → ${newName}`;
}

async function deleteProject(projectName) {
  projectName = recoverProjectName(projectName);
  const memberCount = Object.values(oilGroupProjects).filter((project) => project === projectName).length;
  const ok = await confirmDialog({
    title: "删除项目",
    message: `删除项目“${projectName}”？项目中的 ${memberCount} 个油样和图片会保留，只是移出项目。`,
    confirmText: "删除项目",
    danger: true,
  });
  if (!ok) return;
  projects.delete(projectName);
  expandedProjects.delete(projectName);
  Object.keys(oilGroupProjects).forEach((group) => {
    if (oilGroupProjects[group] === projectName) delete oilGroupProjects[group];
  });
  saveProjectState();
  renderSampleList();
  $("uploadHint").textContent = `已删除项目“${projectName}”，其中油样和图片已保留。`;
}

function removeOilGroupFromProject(group) {
  group = group || "未归入油样";
  const projectName = oilGroupProjects[group];
  delete oilGroupProjects[group];
  saveProjectState();
  renderSampleList();
  $("uploadHint").textContent = projectName
    ? `已将油样“${group}”从项目“${projectName}”移出。`
    : `油样“${group}”当前不在项目中。`;
}

function setRecognitionStatus(text, state = "active") {
  const box = $("recognitionStatus");
  const label = $("recognitionStatusText");
  if (!box || !label) return;
  box.hidden = false;
  box.classList.toggle("done", state === "done");
  box.classList.toggle("error", state === "error");
  label.textContent = text;
  renderRecognitionQueue();
}

function setOperationProgress(progress = null) {
  operationProgress = progress;
  const box = $("recognitionStatus");
  const label = $("recognitionQueueText");
  if (progress) {
    setRecognitionStatus(progress.title || "正在处理...");
    if (label) {
      label.innerHTML = progressBarHtml({
        title: progress.label || progress.title || "处理中",
        current: progress.current,
        total: progress.total,
        detail: progress.detail || "",
        compact: true,
      });
    }
  } else if (box && label && !recognitionQueue.length) {
    label.textContent = "";
  }
  renderSampleList();
}

function renderRecognitionQueue() {
  const label = $("recognitionQueueText");
  if (!label) return;
  if (operationProgress) return;
  if (!recognitionQueue.length) {
    label.textContent = "";
    return;
  }
  const preview = recognitionQueue.slice(0, 3).map((task) => task.label).join("；");
  const more = recognitionQueue.length > 3 ? `；另 ${recognitionQueue.length - 3} 个` : "";
  label.textContent = `等待队列：${preview}${more}`;
}

function enqueueRecognitionTask(task) {
  const normalized = {
    id: ++recognitionTaskId,
    label: task.label,
    items: task.items || [],
    finalText: task.finalText || "",
    afterRun: task.afterRun,
  };
  recognitionQueue.push(normalized);
  setRecognitionStatus(
    recognitionRunning
      ? `已加入识别队列：${normalized.label}`
      : `准备识别：${normalized.label}`,
  );
  runRecognitionQueue();
}

async function runRecognitionQueue() {
  if (recognitionRunning) return;
  recognitionRunning = true;
  try {
    while (recognitionQueue.length) {
      const task = recognitionQueue.shift();
      renderRecognitionQueue();
      try {
        await analyzeUploadedItems(task.items, task.label, { finalText: task.finalText });
        await task.afterRun?.();
      } catch (err) {
        setRecognitionStatus(`${task.label} 识别失败：${err.message || String(err)}`, "error");
      }
    }
  } finally {
    recognitionRunning = false;
    renderRecognitionQueue();
  }
}

function queueSampleImageAnalysis(item) {
  if (!item || item.split !== "uploaded") return;
  const itemSnapshot = { ...item };
  enqueueRecognitionTask({
    label: `当前图像 · ${item.display_name || item.image_name}`,
    items: [itemSnapshot],
    finalText: `识别完成：${item.display_name || item.image_name}`,
    afterRun: async () => {
      const idx = filtered.findIndex((entry) => entry.split === itemSnapshot.split && entry.image_name === itemSnapshot.image_name);
      if (idx >= 0) await loadByIndex(idx);
    },
  });
}

function queueOilGroupAnalysis(group, { refreshSummary = false } = {}) {
  group = normalizeOilGroupName(group);
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group).map((item) => ({ ...item }));
  enqueueRecognitionTask({
    label: group,
    items: members,
    finalText: `识别完成：${group} 共 ${members.length} 张图像。`,
    afterRun: async () => {
      if (refreshSummary) {
        activeOilGroup = group;
        resultMode = "oil";
        invalidateOilSummaryCache();
        await renderOilGroupDetailRows(group);
        await renderOilSampleSummary(group);
        renderSampleList();
      }
    },
  });
}

function queueProjectAnalysis(projectName) {
  const members = samples
    .filter((item) => item.split === "uploaded" && oilGroupProjects[uploadedGroup(item)] === projectName)
    .map((item) => ({ ...item }));
  enqueueRecognitionTask({
    label: `项目 ${projectName}`,
    items: members,
    finalText: `识别完成：项目“${projectName}”共 ${members.length} 张图像。`,
  });
}

async function analyzeSampleImage(item) {
  if (!item || item.split !== "uploaded") return;
  readSettingsUi();
  await analyzeUploadedItems([item], `当前图像`, { finalText: `识别完成：${item.display_name || item.image_name}` });
  const idx = filtered.findIndex((entry) => entry.split === item.split && entry.image_name === item.image_name);
  if (idx >= 0) await loadByIndex(idx);
}

async function analyzeOilGroup(group) {
  group = normalizeOilGroupName(group);
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group);
  await analyzeUploadedItems(members, group, { finalText: `识别完成：${group} 共 ${members.length} 张图像。` });
}

async function analyzeProject(projectName) {
  const members = samples.filter((item) => item.split === "uploaded" && oilGroupProjects[uploadedGroup(item)] === projectName);
  await analyzeUploadedItems(members, `项目 ${projectName}`, { finalText: `识别完成：项目“${projectName}”共 ${members.length} 张图像。` });
}

async function batchAnalyzeCurrentOilSample() {
  queueOilGroupAnalysis(activeOilGroup);
}

async function analyzeUploadedItems(members, scopeLabel, { finalText = "" } = {}) {
  if (!members.length) {
    setRecognitionStatus(`${scopeLabel} 没有可识别图像。`, "error");
    return;
  }
  readSettingsUi();
  try {
    let total = 0;
    let cachedCount = 0;
    for (let i = 0; i < members.length; i += 1) {
      const item = members[i];
      setRecognitionStatus(`识别中... ${scopeLabel}：${i + 1}/${members.length} · ${item.display_name || item.image_name}`);
      setOperationProgress({
        group: uploadedGroup(item),
        title: `正在识别：${scopeLabel}`,
        label: scopeLabel,
        current: i,
        total: members.length,
        detail: item.display_name || item.image_name,
      });
      const payload = await fetchJson(`/api/sample/uploaded/${encodeURIComponent(item.image_name)}?${settingsQuery()}`);
      if (payload.cache_hit) cachedCount += 1;
      item.particles = payload.total_particles || 0;
      const counts = payload.class_counts || {};
      const dominant = Object.entries(counts).sort((a, b) => Number(b[1]) - Number(a[1]))[0];
      item.dominant_class = dominant ? classNames[dominant[0]] || "-" : "-";
      item.recognized = true;
      total += item.particles;
      [samples, filtered].forEach((list) => {
        const row = list.find((entry) => entry.split === "uploaded" && entry.image_name === item.image_name);
        if (row) {
          row.particles = item.particles;
          row.dominant_class = item.dominant_class;
          row.recognized = true;
        }
      });
      if (currentSample?.image_name === item.image_name && currentSample?.split === "uploaded") {
        currentSample = payload;
        currentImage = await loadImage(currentSample.image_url);
      }
      setOperationProgress({
        group: uploadedGroup(item),
        title: `正在识别：${scopeLabel}`,
        label: scopeLabel,
        current: i + 1,
        total: members.length,
        detail: item.display_name || item.image_name,
      });
      renderSampleList();
    }
    invalidateOilSummaryCache();
    if (currentSample) {
      updatePanels();
      drawCanvas();
    } else {
      const firstIdx = filtered.findIndex((item) => item.split === "uploaded" && uploadedGroup(item) === activeOilGroup);
      if (firstIdx >= 0) await loadByIndex(firstIdx);
    }
    const cacheNote = cachedCount ? `，其中 ${cachedCount} 张使用已有识别结果` : "";
    setRecognitionStatus(finalText ? `${finalText}${cacheNote}` : `识别完成：${scopeLabel} 共 ${members.length} 张图像，识别颗粒 ${total} 个${cacheNote}。`, "done");
  } finally {
    setOperationProgress(null);
    updateOilWorkflowUi();
  }
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] == null ? sorted[base] : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function diameterBins(diameters) {
  const bins = [
    { label: "0-5 μm", min: 0, max: 5 },
    { label: "5-10 μm", min: 5, max: 10 },
    { label: "10-15 μm", min: 10, max: 15 },
    { label: "15-25 μm", min: 15, max: 25 },
    { label: "25-50 μm", min: 25, max: 50 },
    { label: ">50 μm", min: 50, max: Infinity },
  ].map((bin) => ({ ...bin, count: 0 }));
  diameters.forEach((value) => {
    const hit = bins.find((bin) => value >= bin.min && value < bin.max) || bins[bins.length - 1];
    hit.count += 1;
  });
  return bins;
}

async function buildOilSampleSummary(group = activeOilGroup, onProgress = null) {
  group = normalizeOilGroupName(group);
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group && item.recognized);
  const payloads = [];
  for (let i = 0; i < members.length; i += 1) {
    const item = members[i];
    onProgress?.({ current: i, total: members.length, fileName: item.display_name || item.image_name });
    if (currentSample?.split === "uploaded" && item.image_name === currentSample.image_name) {
      payloads.push(currentSample);
    } else {
      payloads.push(await fetchJson(`/api/cache/uploaded/${encodeURIComponent(item.image_name)}`));
    }
    onProgress?.({ current: i + 1, total: members.length, fileName: item.display_name || item.image_name });
  }
  const detections = payloads.flatMap((payload) =>
    (payload.detections || []).map((det) => ({
      ...det,
      source_image_name: payload.image_name,
      source_display_name: payload.display_name || payload.image_name,
      source_image_url: payload.image_url,
    })),
  );
  const quantifiable = payloads.length > 0 && payloads.every((payload) => payload.quantifiable === true);
  const realVolumeUl = quantifiable ? payloads.reduce((sum, payload) => sum + Number(payload.volume_ul || 0), 0) : 0;
  const demoVolumeUl = payloads.length * DEMO_VOLUME_UL_PER_IMAGE;
  const totalVolumeUl = realVolumeUl > 0 ? realVolumeUl : demoVolumeUl;
  const volumeMode = realVolumeUl > 0 ? "instrument" : "demo";
  const standardCount = standardParticleCount(detections);
  const counts = {};
  Object.keys(classNames).forEach((cid) => (counts[cid] = 0));
  detections.forEach((det) => (counts[det.class_id] += 1));
  const diameters = detections.map((det) => Number(det.eq_diameter_um || 0)).filter((v) => v > 0);
  const smallCount = smallParticleCount(detections);
  const summary = {
    group,
    image_count: payloads.length,
    total_particles: detections.length,
    small_particle_count: smallCount,
    standard_particle_count: standardCount,
    quantifiable: true,
    instrument_quantifiable: quantifiable,
    volume_mode: volumeMode,
    total_volume_ul: totalVolumeUl,
    concentration_per_ml: totalVolumeUl > 0 ? detections.length / (totalVolumeUl / 1000) : 0,
    total_concentration_per_100ml: totalVolumeUl > 0 ? standardConcentrationPer100ml(detections.length, totalVolumeUl) : null,
    standard_concentration_per_100ml: totalVolumeUl > 0 ? standardConcentrationPer100ml(standardCount, totalVolumeUl) : null,
    counts,
    d10: quantile(diameters, 0.1),
    d50: quantile(diameters, 0.5),
    d90: quantile(diameters, 0.9),
    diameter_bins: diameterBins(diameters),
    detections,
    generated_at: new Date().toISOString(),
  };
  summary.fault_diagnosis = diagnoseOilSample(summary);
  return summary;
}

async function renderOilSampleSummary(group = activeOilGroup) {
  group = normalizeOilGroupName(group);
  const panel = $("oilSamplePanel");
  const box = $("oilSampleSummary");
  if (!panel || !box) return;
  const members = samples.filter((item) => item.split === "uploaded" && uploadedGroup(item) === group);
  if (!members.length) {
    panel.hidden = true;
    lastOilSummary = null;
    return;
  }
  panel.hidden = false;
  syncResultMode();
  const recognizedMembers = members.filter((item) => item.recognized);
  const pendingCount = members.length - recognizedMembers.length;
  if (!recognizedMembers.length) {
    lastOilSummary = null;
    box.innerHTML = `
      <div class="oil-pending">
        <strong>${esc(group)}</strong>
        <span>已导入 ${members.length} 张图像，尚未完成识别。</span>
        <p>请在左侧该油样文件夹的“…”菜单中点击“批量识别”，完成后这里会自动显示油样汇总、粒径分布和故障分析。</p>
      </div>
    `;
    return;
  }
  if (pendingCount > 0) {
    lastOilSummary = null;
    box.innerHTML = `
      <div class="oil-pending">
        <strong>${esc(group)}</strong>
        <span>${recognizedMembers.length}/${members.length} 张图像已识别，仍有 ${pendingCount} 张待识别。</span>
        <p>为避免油样统计口径不一致，请先完成该油样的批量识别后再分析。</p>
      </div>
    `;
    return;
  }
  const cacheKey = oilSummaryCacheKey(group);
  const cached = oilSummaryCache.get(cacheKey);
  if (cached) {
    lastOilSummary = cached.summary;
    oilClassReview = {
      group: cached.summary.group,
      detections: cached.summary.detections || [],
      activeClassId: oilClassReview?.group === cached.summary.group ? oilClassReview.activeClassId : null,
      scrollTop: oilClassReview?.group === cached.summary.group ? oilClassReview.scrollTop || 0 : 0,
    };
    box.innerHTML = cached.html;
    bindOilClassReviewControls();
    return;
  }
  const token = ++oilSummaryToken;
  summaryProgress = { group, current: 0, total: recognizedMembers.length, fileName: "" };
  box.innerHTML = progressBarHtml({
    title: `正在汇总油样“${group}”`,
    current: 0,
    total: recognizedMembers.length,
    detail: "正在读取已识别图像结果...",
  });
  try {
    const summary = await buildOilSampleSummary(group, ({ current, total, fileName }) => {
      summaryProgress = { group, current, total, fileName };
      if (token !== oilSummaryToken || group !== activeOilGroup) return;
      box.innerHTML = progressBarHtml({
        title: `正在汇总油样“${group}”`,
        current,
        total,
        detail: fileName ? `正在处理：${fileName}` : "正在读取已识别图像结果...",
      });
    });
    if (token !== oilSummaryToken || group !== activeOilGroup || !summary) return;
    lastOilSummary = summary;
    const html = renderOilSummaryHtml(summary);
    oilSummaryCache.set(cacheKey, { summary, html });
    box.innerHTML = html;
    bindOilClassReviewControls();
  } catch (err) {
    if (token !== oilSummaryToken || group !== activeOilGroup) return;
    box.textContent = err.message || String(err);
  } finally {
    if (token === oilSummaryToken && group === activeOilGroup) summaryProgress = null;
  }
}

function pieGradient(summary) {
  let cursor = 0;
  const segments = Object.keys(classNames).map((cid) => {
    const count = summary.counts[cid] || 0;
    const ratio = summary.total_particles ? count / summary.total_particles : 0;
    const start = cursor;
    cursor += ratio * 360;
    return `${colors[cid]} ${start.toFixed(2)}deg ${cursor.toFixed(2)}deg`;
  });
  return segments.length ? segments.join(", ") : "#e5eaee 0deg 360deg";
}

function oilFaultFeatures(summary) {
  const total = Math.max(1, Number(summary.total_particles || 0));
  const carbon = Number(summary.counts[1] || 0) + Number(summary.counts[2] || 0);
  const fiber = Number(summary.counts[0] || 0);
  const metal = Number(summary.counts[3] || 0);
  const unknown = Number(summary.counts[5] || 0);
  const small = Number(summary.small_particle_count || 0);
  const totalPer100 = Number(summary.total_concentration_per_100ml || 0);
  const standardPer100 = Number(summary.standard_concentration_per_100ml || 0);
  const ratios = {
    fiber: fiber / total,
    carbon: carbon / total,
    metal: metal / total,
    unknown: unknown / total,
    small: small / total,
  };
  return { total, carbon, fiber, metal, unknown, totalPer100, standardPer100, ratios };
}

function diagnoseOilSample(summary) {
  const f = oilFaultFeatures(summary);
  let label = "正常运行区域";
  let level = "normal";
  let reason = "总颗粒数未明显超标，纤维颗粒占主导，碳/金属颗粒未出现异常富集。";
  if (f.ratios.metal >= 0.2 || (f.ratios.metal >= 0.14 && f.totalPer100 > 3000)) {
    label = "机械磨损故障区域";
    level = "wear";
    reason = "金属颗粒占比明显升高，且通常伴随较多大粒径颗粒，提示油泵或机械部件磨损风险。";
  }
  if (f.totalPer100 > 3000 && f.ratios.carbon >= 0.32 && f.ratios.carbon < 0.55) {
    label = "轻微热/电故障关注区";
    level = "warning";
    reason = "总颗粒数超过演示阈值，碳颗粒比例升高，符合轻微过热或局部放电早期特征。";
  }
  if (f.totalPer100 > 6000 && f.ratios.carbon >= 0.5) {
    label = "热/放电故障区域";
    level = "fault";
    reason = "碳颗粒占比超过50%，且总颗粒数显著升高，提示过热、放电或油纸绝缘分解风险。";
  }
  if (f.totalPer100 > 15000 && f.ratios.carbon >= 0.5) {
    label = "严重热电故障区域";
    level = "severe";
    reason = "总颗粒数极高且碳颗粒主导，接近严重放电、击穿或复合劣化样本特征。";
  }
  if (f.ratios.fiber >= 0.45 && f.totalPer100 <= 3000) {
    label = "正常运行区域";
    level = "normal";
    reason = "纤维颗粒占主导且总颗粒数低于3000个/100mL演示阈值，符合正常或轻微纤维污染油样。";
  }
  if (f.ratios.unknown >= 0.18) {
    reason += " 其中未知/其他颗粒比例偏高，建议结合 ROI 复核，不宜直接归入某一种确定故障来源。";
  }
  return { label, level, reason, features: f };
}

function renderFaultSpace(summary) {
  const diag = diagnoseOilSample(summary);
  const ratios = diag.features.ratios;
  const fx = Math.min(1, Math.max(0, ratios.fiber));
  const cy = Math.min(1, Math.max(0, ratios.carbon));
  const mz = Math.min(1, Math.max(0, ratios.metal));
  const origin = { x: 82, y: 262 };
  const vx = { x: 330, y: -42 };
  const vy = { x: 142, y: -126 };
  const vz = { x: 0, y: -178 };
  const project = (x, y, z) => ({
    x: origin.x + vx.x * x + vy.x * y + vz.x * z,
    y: origin.y + vx.y * x + vy.y * y + vz.y * z,
  });
  const point = project(fx, cy, mz);
  const base = project(fx, cy, 0);
  const xDrop = project(fx, 0, 0);
  const yDrop = project(0, cy, 0);
  const corners = {
    o: project(0, 0, 0),
    x: project(1, 0, 0),
    y: project(0, 1, 0),
    z: project(0, 0, 1),
    xy: project(1, 1, 0),
    xz: project(1, 0, 1),
    yz: project(0, 1, 1),
    xyz: project(1, 1, 1),
  };
  const grid = [0.25, 0.5, 0.75].map((t) => {
    const a = project(t, 0, 0);
    const b = project(t, 1, 0);
    const c = project(0, t, 0);
    const d = project(1, t, 0);
    const e = project(0, 0, t);
    const f = project(1, 0, t);
    const g = project(0, 1, t);
    return `
      <line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" class="fault-grid-line" />
      <line x1="${c.x}" y1="${c.y}" x2="${d.x}" y2="${d.y}" class="fault-grid-line" />
      <line x1="${e.x}" y1="${e.y}" x2="${f.x}" y2="${f.y}" class="fault-grid-line faint" />
      <line x1="${e.x}" y1="${e.y}" x2="${g.x}" y2="${g.y}" class="fault-grid-line faint" />
    `;
  }).join("");
  const z = Math.round(ratios.metal * 100);
  const size = Math.min(22, Math.max(10, 8 + Math.sqrt(Math.max(0, diag.features.totalPer100)) / 24));
  return `
    <div class="fault-panel ${diag.level}">
      <div class="fault-head">
        <div>
          <span>油样故障空间分析</span>
          <strong>${diag.label}</strong>
        </div>
        <em>${fmtInt(diag.features.totalPer100)} 个/100 mL</em>
      </div>
      <svg class="fault-space-3d" viewBox="0 0 560 315" role="img" aria-label="基于纤维、碳和金属颗粒占比的三维故障空间">
        <defs>
          <filter id="faultPointShadow" x="-80%" y="-80%" width="260%" height="260%">
            <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#1d2b34" flood-opacity="0.28" />
          </filter>
        </defs>
        <polygon points="${corners.o.x},${corners.o.y} ${corners.x.x},${corners.x.y} ${corners.xy.x},${corners.xy.y} ${corners.y.x},${corners.y.y}" class="fault-plane floor" />
        <polygon points="${corners.o.x},${corners.o.y} ${corners.x.x},${corners.x.y} ${corners.xz.x},${corners.xz.y} ${corners.z.x},${corners.z.y}" class="fault-plane back-x" />
        <polygon points="${corners.o.x},${corners.o.y} ${corners.y.x},${corners.y.y} ${corners.yz.x},${corners.yz.y} ${corners.z.x},${corners.z.y}" class="fault-plane back-y" />
        ${grid}
        <polyline points="${corners.xz.x},${corners.xz.y} ${corners.xyz.x},${corners.xyz.y} ${corners.yz.x},${corners.yz.y}" class="fault-cube-edge soft" />
        <line x1="${corners.x.x}" y1="${corners.x.y}" x2="${corners.xyz.x}" y2="${corners.xyz.y}" class="fault-cube-edge soft" />
        <line x1="${corners.y.x}" y1="${corners.y.y}" x2="${corners.xyz.x}" y2="${corners.xyz.y}" class="fault-cube-edge soft" />
        <line x1="${corners.z.x}" y1="${corners.z.y}" x2="${corners.xyz.x}" y2="${corners.xyz.y}" class="fault-cube-edge soft" />
        <line x1="${corners.o.x}" y1="${corners.o.y}" x2="${corners.x.x}" y2="${corners.x.y}" class="fault-axis-line x" />
        <line x1="${corners.o.x}" y1="${corners.o.y}" x2="${corners.y.x}" y2="${corners.y.y}" class="fault-axis-line y" />
        <line x1="${corners.o.x}" y1="${corners.o.y}" x2="${corners.z.x}" y2="${corners.z.y}" class="fault-axis-line z" />
        <line x1="${base.x}" y1="${base.y}" x2="${point.x}" y2="${point.y}" class="fault-guide" />
        <line x1="${xDrop.x}" y1="${xDrop.y}" x2="${base.x}" y2="${base.y}" class="fault-guide" />
        <line x1="${yDrop.x}" y1="${yDrop.y}" x2="${base.x}" y2="${base.y}" class="fault-guide" />
        <text x="${project(0.18, 0.08, 0.02).x}" y="${project(0.18, 0.08, 0.02).y}" class="fault-zone-label normal">正常运行</text>
        <text x="${project(0.55, 0.08, 0.04).x}" y="${project(0.55, 0.08, 0.04).y}" class="fault-zone-label wear">机械磨损</text>
        <text x="${project(0.34, 0.58, 0.34).x}" y="${project(0.34, 0.58, 0.34).y}" class="fault-zone-label discharge">热/放电</text>
        <text x="${project(0.68, 0.55, 0.18).x}" y="${project(0.68, 0.55, 0.18).y}" class="fault-zone-label mixed">复合劣化</text>
        <circle cx="${point.x}" cy="${point.y}" r="${size / 2}" class="fault-point-3d ${diag.level}" filter="url(#faultPointShadow)" />
        <circle cx="${base.x}" cy="${base.y}" r="3" class="fault-projection-dot" />
        <text x="${corners.x.x + 12}" y="${corners.x.y + 3}" class="fault-axis-title">X 纤维 ${fmt(ratios.fiber * 100, 1)}%</text>
        <text x="${corners.y.x + 8}" y="${corners.y.y - 8}" class="fault-axis-title">Y 碳 ${fmt(ratios.carbon * 100, 1)}%</text>
        <text x="${corners.z.x - 46}" y="${corners.z.y - 8}" class="fault-axis-title">Z 金属 ${z}%</text>
      </svg>
      <div class="fault-metrics">
        <span>纤维 ${fmt(ratios.fiber * 100, 1)}%</span>
        <span>碳 ${fmt(ratios.carbon * 100, 1)}%</span>
        <span>金属 ${fmt(ratios.metal * 100, 1)}%</span>
        <span>未知 ${fmt(ratios.unknown * 100, 1)}%</span>
        <span>&lt;5 μm ${fmt(ratios.small * 100, 1)}%</span>
      </div>
      <details class="compact-note">
        <summary>查看诊断依据</summary>
        <p>${diag.reason}</p>
        <small>演示诊断规则：总颗粒数阈值 3000 个/100 mL；碳颗粒升高指向热/放电；金属颗粒升高指向机械磨损；未知/其他颗粒作为不确定项，不直接归入明确故障来源。当前体积口径为 ${summary.volume_mode === "demo" ? `演示标定 ${DEMO_VOLUME_UL_PER_IMAGE} μL/张` : "仪器原始帧标定"}。</small>
      </details>
    </div>
  `;
}

function renderOilSummaryHtml(summary) {
  const features = oilFaultFeatures(summary);
  const allConc = fmtInt(summary.total_concentration_per_100ml);
  const standardConc = fmtInt(summary.standard_concentration_per_100ml);
  const volumeMl = Number(summary.total_volume_ul || 0) / 1000;
  oilClassReview = {
    group: summary.group,
    detections: summary.detections || [],
    activeClassId: oilClassReview?.group === summary.group ? oilClassReview.activeClassId : null,
    scrollTop: oilClassReview?.group === summary.group ? oilClassReview.scrollTop || 0 : 0,
  };
  const classRows = Object.entries(classNames)
    .map(([cid, name]) => {
      const count = summary.counts[cid] || 0;
      const ratio = summary.total_particles ? (count / summary.total_particles) * 100 : 0;
      return `
        <span class="oil-class-row">
          <i class="swatch" style="background:${colors[cid]}"></i>
          <b>${name}</b>
          <em>${count} (${fmt(ratio, 1)}%)</em>
          <button class="class-review-btn" data-class-id="${cid}" type="button" title="查看该类颗粒图像" ${count ? "" : "disabled"}>◉</button>
        </span>
      `;
    })
    .join("");
  const maxBin = Math.max(...(summary.diameter_bins || []).map((bin) => bin.count), 1);
  const diameterRows = (summary.diameter_bins || [])
    .map((bin) => {
      const ratio = summary.total_particles ? (bin.count / summary.total_particles) * 100 : 0;
      const width = (bin.count / maxBin) * 100;
      return `
        <div class="oil-diameter-row">
          <span>${bin.label}</span>
          <div class="oil-diameter-bar"><i style="width:${width}%"></i></div>
          <strong>${bin.count}</strong>
          <em>${fmt(ratio, 1)}%</em>
        </div>
      `;
    })
    .join("");
  return `
      <div class="oil-summary-section">
        <div class="oil-section-title">样本信息</div>
        <div class="oil-summary-grid">
          <div><span>样本名称</span><strong>${summary.group}</strong></div>
          <div><span>图像数量</span><strong>${summary.image_count} 张</strong></div>
          <div><span>检测油样体积</span><strong>${fmt(summary.total_volume_ul, 2)} μL<small>${fmt(volumeMl, 4)} mL · ${summary.volume_mode === "demo" ? `演示口径 ${DEMO_VOLUME_UL_PER_IMAGE} μL/张` : "仪器标定口径"}</small></strong></div>
        </div>
      </div>

      <div class="oil-summary-section">
        <div class="oil-section-title">检测结果</div>
        <div class="oil-summary-grid">
          <div><span>总颗粒数</span><strong>${summary.total_particles}<small>&gt;${STANDARD_PARTICLE_SIZE_UM} μm：${summary.standard_particle_count}；&lt;${STANDARD_PARTICLE_SIZE_UM} μm：${summary.small_particle_count}</small></strong></div>
          <div><span>总颗粒浓度</span><strong><span class="primary-conc">${standardConc} / ${allConc} 个/${STANDARD_VOLUME_ML} mL</span><small>&gt;${STANDARD_PARTICLE_SIZE_UM} μm标准口径 / 全部颗粒口径</small></strong></div>
          <div><span>D10 / D50 / D90</span><strong>${fmt(summary.d10, 1)} / ${fmt(summary.d50, 1)} / ${fmt(summary.d90, 1)} μm</strong></div>
        </div>
        <div class="oil-diameter-panel">
          <div class="oil-subtitle">粒径分布</div>
          <div class="oil-diameter-chart">${diameterRows || `<div class="empty-folder">暂无粒径数据</div>`}</div>
        </div>
        <div class="oil-pie-wrap">
          <div class="oil-pie" style="background: conic-gradient(${pieGradient(summary)})">
            <span>${summary.total_particles}</span>
          </div>
          <div>
            <div class="oil-subtitle">颗粒种类及数量</div>
            <div class="oil-class-list">${classRows}</div>
          </div>
        </div>
        <div id="oilClassReviewPanel" class="oil-class-review" hidden></div>
        <div class="oil-feature-grid">
          <div><span>纤维占比</span><strong>${fmt(features.ratios.fiber * 100, 1)}%</strong></div>
          <div><span>碳占比</span><strong>${fmt(features.ratios.carbon * 100, 1)}%</strong></div>
          <div><span>金属占比</span><strong>${fmt(features.ratios.metal * 100, 1)}%</strong></div>
          <div><span>未知占比</span><strong>${fmt(features.ratios.unknown * 100, 1)}%</strong></div>
        </div>
        ${renderFaultSpace(summary)}
      </div>
      <details class="compact-note">
        <summary>查看统计口径</summary>
        <p>本项目重点关注 &lt;5 μm 小颗粒；&gt;5 μm 作为线性/规程参照。当前演示版默认将导入图像按 ${DEMO_VOLUME_UL_PER_IMAGE} μL/张换算为 100 mL 等效颗粒数；后续可按真实 MFI 设备单帧体积统一修正。</p>
      </details>
    `;
}

function bindOilClassReviewControls() {
  document.querySelectorAll(".class-review-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cid = Number(btn.dataset.classId);
      renderOilClassReview(cid);
    });
  });
  if (oilClassReview?.activeClassId != null) {
    renderOilClassReview(oilClassReview.activeClassId, { restoreScroll: true });
  }
}

function renderOilClassReview(classId, { restoreScroll = false } = {}) {
  const panel = $("oilClassReviewPanel");
  if (!panel || !oilClassReview) return;
  oilClassReview.activeClassId = Number(classId);
  const detections = (oilClassReview.detections || []).filter((det) => Number(det.class_id) === Number(classId));
  const title = classNames[classId] || `类别 ${classId}`;
  const rows = detections.slice(0, 240);
  const diameters = detections.map((det) => Number(det.eq_diameter_um || 0)).filter((value) => value > 0);
  const bins = diameterBins(diameters);
  const maxBin = Math.max(...bins.map((bin) => bin.count), 1);
  const diameterTotal = Math.max(diameters.length, 1);
  const d10 = quantile(diameters, 0.1);
  const d50 = quantile(diameters, 0.5);
  const d90 = quantile(diameters, 0.9);
  const binRows = bins.map((bin) => {
    const ratio = diameters.length ? (bin.count / diameterTotal) * 100 : 0;
    const width = (bin.count / maxBin) * 100;
    return `
      <div class="review-diameter-row">
        <span>${bin.label}</span>
        <div><i style="width:${width}%"></i></div>
        <strong>${bin.count}</strong>
        <em>${fmt(ratio, 1)}%</em>
      </div>
    `;
  }).join("");
  panel.hidden = false;
  panel.innerHTML = `
    <div class="review-head">
      <div>
        <span>分类查看</span>
        <strong>${esc(title)} · ${detections.length} 个</strong>
      </div>
      <button id="closeOilClassReviewBtn" type="button" title="关闭">×</button>
    </div>
    <div class="review-diameter-panel">
      <div class="review-subtitle">
        <span>该类粒径分布</span>
        <strong>D10 / D50 / D90：${fmt(d10, 1)} / ${fmt(d50, 1)} / ${fmt(d90, 1)} μm</strong>
      </div>
      <div class="review-diameter-chart">${binRows}</div>
    </div>
    <div class="review-grid">
      ${rows.map((det, idx) => `
        <button class="review-thumb ${currentSample?.image_name === det.source_image_name && selectedParticleId === det.particle_id ? "active" : ""}" data-review-index="${idx}" type="button" title="${esc(det.source_display_name || det.source_image_name || "")}">
          <canvas width="88" height="72" data-review-canvas="${idx}"></canvas>
          <span>${esc(det.source_display_name || det.source_image_name || "-")}</span>
          <em>${particleSizeLabelFromDetection(det)}</em>
        </button>
      `).join("")}
    </div>
    ${detections.length > rows.length ? `<div class="review-note">为保证界面流畅，当前显示前 ${rows.length} 个，可在形态参数明细中继续查看。</div>` : ""}
  `;
  $("closeOilClassReviewBtn")?.addEventListener("click", () => {
    oilClassReview.activeClassId = null;
    oilClassReview.scrollTop = 0;
    panel.hidden = true;
  });
  const grid = panel.querySelector(".review-grid");
  if (grid) {
    if (restoreScroll) grid.scrollTop = oilClassReview.scrollTop || 0;
    grid.addEventListener("scroll", () => {
      oilClassReview.scrollTop = grid.scrollTop;
    });
  }
  panel.querySelectorAll(".review-thumb").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const grid = panel.querySelector(".review-grid");
      if (grid) oilClassReview.scrollTop = grid.scrollTop;
      const det = rows[Number(btn.dataset.reviewIndex)];
      await openOilDetection(det, { preserveReview: true });
    });
  });
  rows.forEach((det, idx) => drawReviewThumb(det, panel.querySelector(`canvas[data-review-canvas="${idx}"]`)));
}

function particleSizeLabelFromDetection(det) {
  const isFiber = [1, 5].includes(Number(det.class_id)) || Number(det.morphology?.aspect_ratio || 0) >= 3;
  if (isFiber && det.bbox) {
    const pixel = Number(det.pixel_size_um || inferSettings.pixel_size_um || 0.7);
    const [xmin, ymin, xmax, ymax] = det.bbox;
    return `长 ${fmt(Math.max(xmax - xmin, ymax - ymin) * pixel, 1)} μm`;
  }
  return `径 ${fmt(det.eq_diameter_um, 1)} μm`;
}

async function drawReviewThumb(det, canvas) {
  if (!det || !canvas || !det.source_image_url) return;
  try {
    const img = await loadImage(det.source_image_url);
    const ctx = canvas.getContext("2d");
    const [xmin, ymin, xmax, ymax] = det.bbox || [0, 0, img.width, img.height];
    const pad = Math.max(8, Math.round(Math.max(xmax - xmin, ymax - ymin) * 0.65));
    const sx = Math.max(0, xmin - pad);
    const sy = Math.max(0, ymin - pad);
    const sw = Math.min(img.width - sx, xmax - xmin + pad * 2);
    const sh = Math.min(img.height - sy, ymax - ymin + pad * 2);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f7fafb";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const scale = Math.min(canvas.width / sw, canvas.height / sh);
    const dw = sw * scale;
    const dh = sh * scale;
    const dx = (canvas.width - dw) / 2;
    const dy = (canvas.height - dh) / 2;
    ctx.drawImage(img, sx, sy, sw, sh, dx, dy, dw, dh);
    ctx.strokeStyle = colors[det.class_id] || "#2d7f9f";
    ctx.lineWidth = 2;
    ctx.strokeRect(dx + (xmin - sx) * scale, dy + (ymin - sy) * scale, (xmax - xmin) * scale, (ymax - ymin) * scale);
  } catch {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#eef3f5";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#6b7780";
    ctx.font = "12px Segoe UI";
    ctx.fillText("无法预览", 18, 38);
  }
}

function setResultMode(mode) {
  resultMode = mode === "image" ? "image" : "oil";
  syncResultMode();
  if (resultMode === "oil") renderOilSampleSummary();
}

function syncResultMode() {
  const oilTab = $("oilResultTab");
  const imageTab = $("imageResultTab");
  if (!oilTab || !imageTab) return;
  oilTab.classList.toggle("active", resultMode === "oil");
  imageTab.classList.toggle("active", resultMode === "image");
  const showOil = resultMode === "oil";
  ["oilSampleSummary"].forEach((id) => {
    const el = $(id);
    if (el) el.hidden = !showOil;
  });
  const metricGrid = document.querySelector(".metric-grid");
  if (metricGrid) metricGrid.hidden = showOil;
  ["classStats", "sampleRisk"].forEach((id) => {
    const el = $(id);
    if (el?.parentElement) el.parentElement.hidden = showOil;
  });
  const hist = $("histCanvas");
  if (hist?.parentElement) hist.parentElement.hidden = showOil;
}

function exportOilReport() {
  if (!lastOilSummary) return;
  const report = {
    report_type: "oil_sample_batch_summary",
    note: "油样级结果由同一油样批次内多张 MFI 图像汇总得到；浓度依赖单图体积标定。",
    inference_settings: inferSettings,
    summary: lastOilSummary,
  };
  downloadBlob(JSON.stringify(report, null, 2), `${lastOilSummary.group}_oil_sample_report.json`, "application/json;charset=utf-8");
}

function handleCanvasClick(event) {
  if (!currentSample) return;
  if (manualDrawingMode) return;
  const rect = $("imageCanvas").getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * currentSample.width;
  const y = ((event.clientY - rect.top) / rect.height) * currentSample.height;
  const hit = activeDetections()
    .filter((det) => x >= det.bbox[0] && x <= det.bbox[2] && y >= det.bbox[1] && y <= det.bbox[3])
    .sort((a, b) => {
      const aa = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
      const bb = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
      return aa - bb;
    })[0];
  if (hit) {
    selectedParticleId = hit.particle_id;
    renderSelection();
    renderParticleInspector(selectedDetection());
    renderRows(activeDetections());
    drawCanvas();
  }
}

async function boot() {
  setAppLoading(5, "正在启动应用", "正在初始化界面...");
  const appTitle = "变压器油中典型游离杂质颗粒识别系统YOLO";
  const appVersion = "Version 1.1";
  document.title = appTitle;
  const titlePanel = document.querySelector(".panel-title");
  if (titlePanel) {
    titlePanel.innerHTML = `
      <div class="brand-strip">
        <div class="brand-logo"><img src="/static/assets/cqu-logo-real.png" alt="重庆大学" /><span>重庆大学</span></div>
        <div class="brand-logo"><img src="/static/assets/state-grid-logo-real.png" alt="国家电网公司" /><span>国家电网公司</span></div>
      </div>
      <h1>${appTitle}</h1>
      <div class="title-meta">
        <span id="sampleCount">加载中</span>
        <strong>${appVersion}</strong>
      </div>
    `;
  }
  configureSampleControls();
  renderLegend();
  $("manualClassSelect").innerHTML = Object.entries(classNames)
    .map(([cid, name]) => `<option value="${cid}">${name}</option>`)
    .join("");
  setAppLoading(18, "正在读取项目结构", "载入项目、油样文件夹和界面设置...");
  await loadServerProjectState();
  setAppLoading(35, "正在连接识别后端", "检查 YOLO 模型与推理参数...");
  appStatus = await fetchJson("/api/status");
  inferSettings = { ...inferSettings, ...(appStatus.default_settings || {}) };
  syncSettingsUi();
  renderModelStatus();
  setAppLoading(52, "正在读取图片清单", "图片较多时需要扫描文件与识别缓存...", { indeterminate: true });
  samples = await fetchJson("/api/samples");
  setAppLoading(78, "图片清单已载入", `共读取 ${samples.length} 张图片，正在整理项目列表...`);
  applyFilters();
  if (filtered.length) {
    setAppLoading(86, "正在打开首张图片", filtered[0].display_name || filtered[0].image_name, { indeterminate: true });
    await loadByIndex(0, { showLoading: false });
  }
  $("splitFilter").addEventListener("change", () => {
    applyFilters();
    loadByIndex(0);
  });
  $("classFilter").addEventListener("change", () => {
    updatePanels();
    drawCanvas();
  });
  $("confSlider").addEventListener("input", () => {
    updatePanels();
    drawCanvas();
  });
  $("diamSlider").addEventListener("input", () => {
    updatePanels();
    drawCanvas();
  });
  $("boxToggle").addEventListener("change", drawCanvas);
  $("labelToggle").addEventListener("change", drawCanvas);
  $("prevBtn").addEventListener("click", () => loadByIndex(currentIndex - 1));
  $("nextBtn").addEventListener("click", () => loadByIndex(currentIndex + 1));
  $("exportBtn").addEventListener("click", exportCurrentCsv);
  $("reportBtn").addEventListener("click", exportReport);
  $("imageCanvas").addEventListener("click", handleCanvasClick);
  $("imageCanvas").addEventListener("pointerdown", (event) => {
    if (!manualDrawingMode) return;
    event.preventDefault();
    manualDrawStart = canvasImagePoint(event);
    $("imageCanvas").setPointerCapture?.(event.pointerId);
  });
  $("imageCanvas").addEventListener("pointerup", finishManualBox);
  $("confirmManualClassBtn").addEventListener("click", () => {
    const det = selectedDetection();
    if (!det) return;
    const nextClass = Number($("manualClassSelect").value);
    const previousClass = Number(det.class_id);
    if (nextClass === previousClass) {
      det.manual_label = true;
      det.manual_operation = "confirmed";
      setManualHint("已确认当前类别。点击保存写入标签库。");
    } else {
      manualChanges.push({ operation: "change_class", annotation_id: annotationId(det), before: previousClass, after: nextClass });
      det.model_class_id ??= det.predicted_class_id ?? previousClass;
      det.model_confidence ??= det.confidence ?? null;
      det.class_id = nextClass;
      det.class_cn = classNames[nextClass];
      det.manual_label = true;
      det.manual_operation = "change_class";
      setManualHint(`已将类别改为“${classNames[nextClass]}”。点击保存写入标签库。`);
    }
    updatePanels();
    drawCanvas();
  });
  $("redrawBoxBtn").addEventListener("click", () => setManualDrawingMode("redraw"));
  $("addManualBoxBtn").addEventListener("click", () => setManualDrawingMode("add"));
  $("removeManualBoxBtn").addEventListener("click", () => {
    const det = selectedDetection();
    if (!det) return;
    manualChanges.push({ operation: "remove_false_positive", annotation_id: annotationId(det), class_id: det.class_id, bbox: [...det.bbox] });
    currentSample.detections = currentSample.detections.filter((item) => item !== det);
    currentSample.detections.forEach((item, idx) => { item.particle_id = idx; item.id = idx; });
    selectedParticleId = null;
    setManualHint("已标记为非目标并从人工标签中移除。点击保存写入标签库。");
    updatePanels();
    drawCanvas();
  });
  $("saveManualReviewBtn").addEventListener("click", async () => {
    try { await saveManualReview(); } catch (err) { setManualHint(err.message || String(err), true); }
  });
  $("exportManualLabelsBtn").addEventListener("click", async () => {
    try {
      const payload = await fetchJson("/api/manual-labels/export");
      downloadBlob(JSON.stringify(payload, null, 2), "manual_yolo_label_manifest.json", "application/json;charset=utf-8");
      setManualHint(`已导出 ${payload.eligible_image_count} 张整图审核完成的标签清单。`);
    } catch (err) { setManualHint(err.message || String(err), true); }
  });
  $("saveSampleMetaBtn").addEventListener("click", async () => {
    try {
      await saveCurrentSampleMeta();
    } catch (err) {
      $("uploadHint").textContent = err.message || String(err);
    }
  });
  $("deleteUploadedBtn").addEventListener("click", async () => {
    try {
      await deleteCurrentUploadedSample();
    } catch (err) {
      $("uploadHint").textContent = err.message || String(err);
    }
  });
  $("uploadInput").addEventListener("change", async () => {
    try {
      await uploadImages($("uploadInput").files);
    } catch (err) {
      $("uploadHint").textContent = err.message || String(err);
    } finally {
      $("uploadInput").value = "";
    }
  });
  $("exportOilReportBtn").addEventListener("click", exportOilReport);
  $("closeAggregationRiskBtn")?.addEventListener("click", closeAggregationRiskPage);
  $("riskReleaseScenario")?.addEventListener("change", drawAggregationRisk);
  $("riskParticleType")?.addEventListener("change", drawAggregationRisk);
  $("riskFieldWeight")?.addEventListener("input", () => { $("riskFieldWeightValue").textContent = `${$("riskFieldWeight").value}%`; drawAggregationRisk(); });
  $("riskChartTabs")?.addEventListener("click", (event) => { const btn = event.target.closest("button[data-risk-view]"); if (!btn) return; riskView = btn.dataset.riskView; document.querySelectorAll("#riskChartTabs button").forEach((item) => item.classList.toggle("active", item === btn)); drawAggregationRisk(); });
  $("exportAggregationRiskBtn")?.addEventListener("click", () => { const data = riskFilteredZones().map((z) => ({ zone: z.name, composite_risk: Number(riskScore(z).toFixed(4)), dwell: z.dwell, density: z.density, field_weighted: z.field, cumulative_passage: z.flow })); downloadBlob(JSON.stringify({ report_type: "particle_aggregation_risk", note: "前端原型示例数据；接入真实网格/轨迹后替换。", data }, null, 2), "particle_aggregation_risk.json", "application/json;charset=utf-8"); });
  $("oilResultTab").addEventListener("click", () => setResultMode("oil"));
  $("imageResultTab").addEventListener("click", () => setResultMode("image"));
  $("analyzeUploadedBtn")?.addEventListener("click", async () => {
    try {
      readSettingsUi();
      await reloadCurrentSample();
    } catch (err) {
      setRecognitionStatus(err.message || String(err), "error");
    }
  });
  ["settingConf", "settingOtherConf", "settingIou"].forEach((id) => {
    $(id).addEventListener("input", () => {
      readSettingsUi();
    });
  });
  ["settingImgsz", "settingMaxDet", "settingDevice", "settingPixelSize", "settingAgnostic"].forEach((id) => {
    $(id).addEventListener("change", readSettingsUi);
  });
  $("applySettingsBtn").addEventListener("click", async () => {
    readSettingsUi();
    invalidateOilSummaryCache();
    await reloadCurrentSample();
  });
  $("resetSettingsBtn").addEventListener("click", async () => {
    inferSettings = { ...inferSettings, ...(appStatus.default_settings || {}) };
    syncSettingsUi();
    invalidateOilSummaryCache();
    await reloadCurrentSample();
  });
  finishAppLoading();
}

boot().catch((err) => {
  console.error(err);
  setAppLoading(100, "应用启动失败", err.message || String(err));
  const retried = sessionStorage.getItem("mfiBootRecoveryTried") === "1";
  if (!retried) {
    sessionStorage.setItem("mfiBootRecoveryTried", "1");
    localStorage.removeItem(PROJECT_STATE_KEY);
    window.location.reload();
    return;
  }
  sessionStorage.removeItem("mfiBootRecoveryTried");
  document.body.innerHTML = `<pre style="padding:24px;color:#b91c1c">${err.stack || err}</pre>`;
});





