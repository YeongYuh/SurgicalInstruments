'use strict';

// ── State ─────────────────────────────────────────────────────────────────
let standards   = {};   // {class_name: int}
let unitWeights = {};   // {class_name: float}
let lastCounts  = {};
let cameraActive = false;
let pendingFile  = null;
let stdDebounce  = null;
let uwDebounce   = null;
let donutChart   = null;
let pollTimer            = null;
let weightPollTimer      = null;
let _previewController   = null;   // AbortController for the async preview loop
let _previewBlobUrl      = null;   // last blob URL set on #camera-stream
let cameraSessionId      = 0;      // matches backend camera_session_id
let _lastFrameSeq        = -1;     // last X-Frame-Seq seen; skip display if unchanged

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  initChart();
  await fetchStandards();
  await fetchUnitWeights();
  // Resolve camera state from the server BEFORE starting the poll loop so
  // the first pollStatus() does not see a stale cameraActive=false and
  // accidentally restore an old annotated image via syncCameraUI().
  await initCameraState();
  startPolling();
  startWeightPolling();
});

async function initCameraState() {
  try {
    const res  = await fetch('/camera/status');
    const data = await res.json();
    cameraActive    = data.running === true;
    cameraSessionId = data.session_id || 0;
    syncCameraUI();   // starts preview loop if camera was already running
  } catch (e) { /* server not ready yet — leave defaults */ }
}

// ── Chart.js donut ────────────────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById('donut-chart').getContext('2d');
  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['正常', '缺少', '多出'],
      datasets: [{
        data: [0, 0, 0],
        backgroundColor: ['#4caf50', '#f44336', '#ff9800'],
        borderWidth: 0,
        hoverOffset: 4,
      }]
    },
    options: {
      responsive: false,
      cutout: '60%',
      plugins: { legend: { display: false } },
      animation: { duration: 300 },
    }
  });
}

function updateChart(normal, missing, extra) {
  if (!donutChart) return;
  const total = normal + missing + extra;
  donutChart.data.datasets[0].data = total > 0 ? [normal, missing, extra] : [1, 0, 0];
  donutChart.data.datasets[0].backgroundColor = total > 0
    ? ['#4caf50', '#f44336', '#ff9800']
    : ['#2d3448', '#2d3448', '#2d3448'];
  donutChart.update('none');
  document.getElementById('cnt-normal').textContent  = normal;
  document.getElementById('cnt-missing').textContent = missing;
  document.getElementById('cnt-extra').textContent   = extra;
}

// ── Standards ─────────────────────────────────────────────────────────────
async function fetchStandards() {
  try {
    const res = await fetch('/standards');
    standards = await res.json();
  } catch (e) { console.error('fetchStandards:', e); }
}

function scheduleStandardsSync() {
  clearTimeout(stdDebounce);
  stdDebounce = setTimeout(pushStandards, 300);
}

async function pushStandards() {
  const payload = {};
  document.querySelectorAll('.std-input').forEach(inp => {
    payload[inp.dataset.cls] = parseInt(inp.value) || 0;
  });
  try {
    await fetch('/standards', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    standards = payload;
    rerenderTable(lastCounts);
    refreshWeightVerification();
  } catch (e) { console.error('pushStandards:', e); }
}

// ── Unit weights ──────────────────────────────────────────────────────────
async function fetchUnitWeights() {
  try {
    const res = await fetch('/unit_weights');
    unitWeights = await res.json();
  } catch (e) { console.error('fetchUnitWeights:', e); }
}

function scheduleUnitWeightsSync() {
  clearTimeout(uwDebounce);
  uwDebounce = setTimeout(pushUnitWeights, 300);
}

async function pushUnitWeights() {
  const payload = {};
  document.querySelectorAll('.uw-input').forEach(inp => {
    payload[inp.dataset.cls] = parseFloat(inp.value) || 0;
  });
  try {
    await fetch('/unit_weights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    unitWeights = payload;
    refreshWeightVerification();
  } catch (e) { console.error('pushUnitWeights:', e); }
}

// ── Weight verification display ───────────────────────────────────────────
function renderWeightVerification(wv) {
  if (!wv) return;

  // Row-2 stat boxes in 盤點統計
  const expWtEl  = document.getElementById('stat-exp-wt');
  const actWtEl  = document.getElementById('stat-act-wt');
  const wokEl    = document.getElementById('stat-wt-ok');
  const wokBox   = document.getElementById('stat-wt-ok-box');
  if (expWtEl) expWtEl.textContent = wv.expected != null ? `${wv.expected.toFixed(1)} g` : '—';
  if (actWtEl) actWtEl.textContent = wv.actual   != null ? `${wv.actual.toFixed(1)} g`   : '—';
  if (wokEl && wokBox) {
    // No BOM configured: at least one instrument must have a unit weight > 0,
    // otherwise expected=0 and any actual reading would falsely show 符合.
    const bomConfigured = Object.values(unitWeights).some(v => v > 0);
    if (wv.actual == null) {
      wokEl.textContent = '—';
      wokBox.className  = 'stat-box weight-ok-box';
    } else if (!bomConfigured) {
      wokEl.textContent = '不明';
      wokBox.className  = 'stat-box weight-ok-box';
    } else if (wv.passed) {
      wokEl.textContent = '符合';
      wokBox.className  = 'stat-box weight-ok-box passed';
    } else {
      wokEl.textContent = '不符合';
      wokBox.className  = 'stat-box weight-ok-box failed';
    }
  }
}

// Compute expected weight locally from current standards + unit weights
// and trigger a re-render with the last known actual weight.
function refreshWeightVerification() {
  const lastWV = _lastKnownWV;
  if (!lastWV) return;
  // Recompute expected from current local state
  let expected = 0;
  const allCls = new Set([...Object.keys(standards), ...Object.keys(unitWeights)]);
  allCls.forEach(cls => {
    expected += (standards[cls] || 0) * (unitWeights[cls] || 0);
  });
  const actual = lastWV.actual;
  const tolerance = lastWV.tolerance;
  const diff = actual != null ? Math.abs(actual - expected) : null;
  const passed = diff != null ? diff <= tolerance : false;
  renderWeightVerification({
    passed,
    expected,
    actual,
    difference: diff,
    tolerance,
    message: diff == null ? '無法讀取重量' : (passed ? '重量在容許範圍內' : '重量超出容許範圍'),
  });
}

let _lastKnownWV = null;

// ── Results table ─────────────────────────────────────────────────────────
function rerenderTable(counts) {
  lastCounts = counts || {};
  const tbody = document.getElementById('results-tbody');

  if (Object.keys(lastCounts).length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-row">尚無辨識結果</td></tr>';
    updateChart(0, 0, 0);
    return;
  }

  let normal = 0, missing = 0, extra = 0;
  let rows = '';

  Object.entries(lastCounts).sort().forEach(([cls, cnt]) => {
    const std = standards[cls] !== undefined ? standards[cls] : 0;
    let badge = '';
    if (cnt === std)    { badge = '<span class="badge badge-normal">正常</span>';  normal++;  }
    else if (cnt < std) { badge = '<span class="badge badge-missing">缺少</span>'; missing++; }
    else                { badge = '<span class="badge badge-extra">多出</span>';   extra++;   }

    rows += `<tr>
      <td>${cls}</td>
      <td>${cnt}</td>
      <td><input type="number" min="0" step="1" class="std-input" data-cls="${cls}"
           value="${std}" oninput="scheduleStandardsSync()"></td>
      <td>${badge}</td>
    </tr>`;
  });

  tbody.innerHTML = rows;
  updateChart(normal, missing, extra);
}

// ── Display helpers ───────────────────────────────────────────────────────
function showAnnotatedImage(b64) {
  if (!b64) return;
  const img = document.getElementById('preview-img');
  img.src = 'data:image/jpeg;base64,' + b64;
  img.style.display = 'block';
  document.getElementById('camera-stream').style.display = 'none';
  document.getElementById('placeholder').classList.remove('visible');
}

function showCameraStream() {
  document.getElementById('camera-stream').style.display = 'block';
  document.getElementById('preview-img').style.display = 'none';
  document.getElementById('placeholder').classList.remove('visible');
}

// ── Live preview — AbortController-driven fetch loop ─────────────────────
// One request at a time: only fetch the next frame after the current one
// completes. This prevents request pileup on a slow Jetson server and
// ensures the browser always displays a valid JPEG, never a canceled load.

function _previewDelay(ms, signal) {
  return new Promise(resolve => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(t); resolve(); }, { once: true });
  });
}

async function _previewLoop(signal) {
  const stream = document.getElementById('camera-stream');
  while (!signal.aborted) {
    try {
      const url = '/camera/frame?session=' + cameraSessionId + '&t=' + Date.now();
      const res = await fetch(url, { cache: 'no-store', signal });
      if (signal.aborted) break;
      const ct = res.headers.get('content-type') || '';
      if (res.ok && ct.startsWith('image/jpeg')) {
        const seq = parseInt(res.headers.get('X-Frame-Seq') || '-1', 10);
        const blob = await res.blob();
        if (signal.aborted) break;
        // Only update display when the backend has a genuinely new frame.
        // This avoids redundant repaints when polling faster than ~7.5 FPS camera.
        if (seq < 0 || seq !== _lastFrameSeq) {
          const newUrl = URL.createObjectURL(blob);
          stream.src = newUrl;
          if (_previewBlobUrl) URL.revokeObjectURL(_previewBlobUrl);
          _previewBlobUrl = newUrl;
          _lastFrameSeq = seq;
        }
        // Camera produces ~7.5 FPS → 133 ms/frame; stay close to that rate.
        await _previewDelay(130, signal);
      } else {
        // 204: camera starting or stopped — keep the last good frame displayed.
        await _previewDelay(200, signal);
      }
    } catch (e) {
      if (signal.aborted) break;
      await _previewDelay(500, signal);
    }
  }
}

function startCameraPreview() {
  if (_previewController) return;          // already running
  _previewController = new AbortController();
  _previewLoop(_previewController.signal);
}

function stopCameraPreview() {
  _lastFrameSeq = -1;
  if (_previewController) {
    _previewController.abort();
    _previewController = null;
  }
  if (_previewBlobUrl) {
    URL.revokeObjectURL(_previewBlobUrl);
    _previewBlobUrl = null;
  }
  const stream = document.getElementById('camera-stream');
  if (stream) stream.src = '';
}

function setUpdateTime(ts) {
  document.getElementById('update-time').textContent = ts ? '更新時間：' + ts : '更新時間：—';
}

// ── Polling /status every 2 s ─────────────────────────────────────────────
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollStatus, 2000);
  pollStatus();
}

async function pollStatus() {
  try {
    const res  = await fetch('/status');
    const data = await res.json();

    if (data.camera_active !== cameraActive) {
      cameraActive = data.camera_active;
      syncCameraUI();
    }

    if (data.counts && Object.keys(data.counts).length > 0) {
      rerenderTable(data.counts);
      setUpdateTime(data.timestamp);
    }

    if (data.weight_verification) {
      _lastKnownWV = data.weight_verification;
      renderWeightVerification(data.weight_verification);
    }

    // Annotated images are only shown by explicit user actions (upload / recognize).
    // pollStatus() never touches the image area — avoids stale camera frames
    // appearing on page refresh after the camera is stopped.
  } catch (e) { /* server starting up */ }
}

async function manualRefresh() {
  await fetchStandards();
  await fetchUnitWeights();
  await pollStatus();
}

// ── Live weight polling every 1 s ─────────────────────────────────────────
function startWeightPolling() {
  if (weightPollTimer) clearInterval(weightPollTimer);
  weightPollTimer = setInterval(pollWeight, 1000);
  pollWeight();
}

async function pollWeight() {
  try {
    const res  = await fetch('/api/weight');
    const data = await res.json();
    if (data.weight_verification) {
      _lastKnownWV = data.weight_verification;
      renderWeightVerification(data.weight_verification);
    }
  } catch (e) { /* server starting up */ }
}

// ── Upload image ──────────────────────────────────────────────────────────
function onFileChange(evt) {
  const file = evt.target.files[0];
  if (!file) return;
  pendingFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview-img');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('camera-stream').style.display = 'none';
    document.getElementById('placeholder').classList.remove('visible');
  };
  reader.readAsDataURL(file);
}

function onDragOver(evt) {
  evt.preventDefault();
  document.getElementById('image-area').classList.add('drag-over');
}

function onDragLeave() {
  document.getElementById('image-area').classList.remove('drag-over');
}

function onDrop(evt) {
  evt.preventDefault();
  document.getElementById('image-area').classList.remove('drag-over');
  const file = evt.dataTransfer.files[0];
  if (!file || !file.type.startsWith('image/')) return;
  pendingFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview-img');
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('placeholder').classList.remove('visible');
  };
  reader.readAsDataURL(file);
}

// ── Recognize ─────────────────────────────────────────────────────────────
async function startRecognize() {
  const btn = document.getElementById('btn-recognize');
  btn.disabled = true;
  btn.textContent = '辨識中…';
  try {
    let data;
    if (cameraActive) {
      const res = await fetch('/recognize', { method: 'POST' });
      data = await res.json();
    } else if (pendingFile) {
      const form = new FormData();
      form.append('image', pendingFile);
      const res = await fetch('/upload', { method: 'POST', body: form });
      data = await res.json();
    } else {
      alert('請先上傳圖片或開啟攝影機');
      return;
    }
    if (data.ok) {
      rerenderTable(data.counts);
      setUpdateTime(data.timestamp);
      if (!cameraActive) showAnnotatedImage(data.annotated_image);
      if (data.weight_verification) {
        _lastKnownWV = data.weight_verification;
        renderWeightVerification(data.weight_verification);
      }
    } else {
      alert('辨識失敗：' + (data.error || '未知錯誤'));
    }
  } catch (e) {
    alert('連線錯誤：' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '⚐ 開始辨識';
  }
}

// ── Camera toggle ─────────────────────────────────────────────────────────
async function toggleCamera() {
  const btn = document.getElementById('btn-camera');
  btn.disabled = true;
  try {
    if (cameraActive) {
      stopCameraPreview();                                   // no more /camera/frame before stop
      await fetch('/camera/stop', { method: 'POST' });      // waits for cap.release()
      cameraActive = false;
    } else {
      // /camera/start waits 0.6 s internally to detect open failures
      const res  = await fetch('/camera/start', { method: 'POST' });
      const data = await res.json();
      if (data.ok) {
        cameraActive    = true;
        cameraSessionId = data.session_id || 0;
        _lastFrameSeq   = -1;
      } else {
        alert('無法開啟攝影機：' + (data.error || '未知錯誤'));
        return;
      }
    }
    syncCameraUI();
  } finally {
    btn.disabled = false;
  }
}

function syncCameraUI() {
  const btn         = document.getElementById('btn-camera');
  const stream      = document.getElementById('camera-stream');
  const preview     = document.getElementById('preview-img');
  const placeholder = document.getElementById('placeholder');

  if (cameraActive) {
    btn.textContent = '⏹ 關閉攝影機';
    btn.classList.add('active');
    stream.style.display = 'block';
    preview.style.display = 'none';
    placeholder.classList.remove('visible');
    startCameraPreview();
  } else {
    btn.textContent = '◉ 開啟攝影機';
    btn.classList.remove('active');
    stopCameraPreview();
    stream.style.display = 'none';
    placeholder.classList.add('visible');
    preview.style.display = 'none';
  }
}

// ── History modal ─────────────────────────────────────────────────────────
async function openHistory() {
  document.getElementById('history-modal').style.display = 'flex';
  try {
    const res     = await fetch('/history');
    const records = await res.json();
    const tbody   = document.getElementById('history-tbody');
    if (!records.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#6e7681;padding:16px">尚無歷史記錄</td></tr>';
      return;
    }
    tbody.innerHTML = records.slice().reverse().map(rec => {
      const instruments = Object.entries(rec.counts || {})
        .map(([n, c]) => `${n}×${c}`).join(', ') || '—';
      const w = rec.weight != null ? rec.weight.toFixed(2) + ' g' : '—';
      return `<tr>
        <td>${rec.timestamp}</td>
        <td>${rec.source}</td>
        <td>${instruments}</td>
        <td>${w}</td>
      </tr>`;
    }).join('');
  } catch (e) { console.error('openHistory:', e); }
}

function closeHistory() {
  document.getElementById('history-modal').style.display = 'none';
}

document.addEventListener('click', e => {
  if (e.target.id === 'history-modal') closeHistory();
});

// ── Downloads ─────────────────────────────────────────────────────────────
function downloadReport() { window.location.href = '/report'; }
function downloadBOM()    { window.location.href = '/bom_report'; }
