'use strict';

// ── Instance identity — unique per page load ───────────────────────────────
// Appended to /status, /api/weight, and /camera/stream so Flask logs can
// distinguish requests from this tab vs. another tab or a stale old page.
const CLIENT_ID = Date.now() + '-' + Math.random().toString(16).slice(2);
let appInitialized = false;   // guards DOMContentLoaded against duplicate fires

// ── State ─────────────────────────────────────────────────────────────────
let standards    = {};   // {class_name: int}
let unitWeights  = {};   // {class_name: float}  (output/unit_weights.json — legacy)
let classWeights = {};   // {class_name: float}  (models/class_weight.json — source of truth)
let lastCounts  = {};
let cameraActive      = false;
let recognitionActive = false;
let pendingFile       = null;
let stdDebounce  = null;
let uwDebounce   = null;
let donutChart   = null;
let statusPollTimer          = null;   // the single /status setInterval handle
let weightPollingActive      = false;
let weightPollToken          = 0;
let weightPollAbortController = null;
let _previewActive          = false;  // true once img.src is assigned for the current stream
let previewStarting          = false;  // true between session claim and src assignment
let activeStreamSession      = null;  // session_id the current stream was opened for
let cameraSessionId         = 0;      // matches backend camera_session_id
let recognitionStopPending  = false;  // true while stop is in-flight / unconfirmed by backend
let cameraStopPending       = false;  // true while camera stop is in-flight / unconfirmed
let cameraStartPending      = false;  // true while /camera/start fetch is in-flight

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  if (appInitialized) {
    console.warn('[init] DOMContentLoaded fired again — ignoring duplicate CLIENT_ID=' + CLIENT_ID);
    return;
  }
  appInitialized = true;
  console.debug('[init] app start CLIENT_ID=' + CLIENT_ID);
  initChart();
  await fetchStandards();
  await fetchUnitWeights();
  await fetchClassWeights();
  // Resolve camera state from the server BEFORE starting the poll loop so
  // the first pollStatus() does not see a stale cameraActive=false and
  // accidentally restore an old annotated image via syncCameraUI().
  await initCameraState();
  startStatusPolling();
  // Weight polling is NOT started here — it runs only while recognition is active.
});

async function initCameraState() {
  try {
    const res  = await fetch('/camera/status');
    const data = await res.json();
    // stopping=true means stop() was called but join hasn't completed — treat as inactive
    // so the UI doesn't restore an active-camera state that is about to be torn down.
    const isStopping  = data.stopping === true;
    cameraActive      = data.running === true && !isStopping;
    cameraSessionId   = data.session_id || 0;
    recognitionActive = data.recognition_running === true && !isStopping;
    console.debug('[init] camera=' + cameraActive + ' stopping=' + isStopping
                  + ' session=' + cameraSessionId + ' rec=' + recognitionActive);
    if (recognitionActive) {
      startWeightPolling('page-init-resuming');   // was already running — resume polling
    } else {
      stopWeightPolling('page-init-not-recognizing');
      resetWeightDisplay();   // no active recognition — show 未量測
    }
    syncCameraUI();   // starts preview loop if camera was already running
  } catch (e) {
    resetWeightDisplay();   // server not ready — show 未量測 rather than blank
  }
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

async function fetchClassWeights() {
  try {
    const res = await fetch('/class_weights');
    classWeights = await res.json();
    console.debug('[init] classWeights loaded: ' + Object.keys(classWeights).length + ' classes');
  } catch (e) { console.error('fetchClassWeights:', e); }
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
    // Use backend-computed expected weight to determine if BOM is configured.
    // expected = Σ(standard_count × unit_weight_per_class).
    // If expected == 0 the BOM has no meaningful data (no unit weights set).
    // Explicit null check avoids JS truthiness traps (0.0 is falsy).
    const bomConfigured = wv.expected != null && wv.expected > 0;
    console.debug(
      '[wv] actual=' + wv.actual + ' expected=' + wv.expected
      + ' tolerance=' + wv.tolerance + ' passed=' + wv.passed
      + ' bomConfigured=' + bomConfigured,
    );
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

// Delegates to recalculateExpectedAndRender — kept so pushStandards / pushUnitWeights
// callers do not need to change.
function refreshWeightVerification() {
  recalculateExpectedAndRender('standard-edit');
}

// ── Single source of truth for 標準重量 ────────────────────────────────────
// expected is ALWAYS computed here from the current frontend standards × classWeights.
// Backend weight_verification.expected is never trusted after initial receipt —
// every caller that receives a backend wv must:
//   1. update _lastKnownWV.actual and .tolerance (scale / inference data)
//   2. then call recalculateExpectedAndRender() to recompute expected locally
// This prevents the 2-second pollStatus() cycle from overwriting manual standard edits.
function recalculateExpectedAndRender(source) {
  if (_lastKnownWV == null) return;   // no recognition context yet — keep '未量測' state

  let expected = 0;
  const debugParts = [];
  Object.entries(standards).forEach(([cls, std]) => {
    const qty = Number(std);   // Number("0")=0  Number("")=0  Number(undefined)=NaN
    if (!Number.isFinite(qty) || qty === 0) return;   // 0 contributes nothing; NaN is invalid
    const uw = Number(classWeights[cls]);
    if (!Number.isFinite(uw)) {
      console.warn('[wv] class "' + cls + '" not in classWeights — 0 g');
      return;
    }
    expected += qty * uw;
    debugParts.push(cls + ':' + qty + '×' + uw + '=' + (qty * uw).toFixed(1));
  });

  const actual    = _lastKnownWV.actual;
  const tolerance = _lastKnownWV.tolerance;
  const diff      = (actual != null && tolerance != null) ? Math.abs(actual - expected) : null;
  const passed    = diff != null ? diff <= tolerance : false;

  // Patch _lastKnownWV so any subsequent call sees the fresh expected value.
  _lastKnownWV.expected   = expected;
  _lastKnownWV.difference = diff;
  _lastKnownWV.passed     = passed;

  console.debug('[wv] recalc source=' + source
    + ' expected=' + expected.toFixed(1)
    + ' actual=' + actual
    + ' passed=' + passed
    + (debugParts.length
        ? '  classes=[' + debugParts.slice(0, 6).join(', ')
          + (debugParts.length > 6 ? '…' : '') + ']'
        : '  [no contributing classes]'));

  renderWeightVerification(_lastKnownWV);
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

// ── Live preview — MJPEG stream via img.src ──────────────────────────────
// The browser handles the multipart/x-mixed-replace stream natively —
// no JS fetch loop, no blob URLs, no AbortController needed.
// Reassigning img.src automatically drops the old connection.

function startCameraPreview() {
  console.debug('[preview] startCameraPreview called'
    + ' session=' + cameraSessionId
    + ' active=' + _previewActive
    + ' starting=' + previewStarting
    + ' activeSess=' + activeStreamSession
    + ' camActive=' + cameraActive);

  // Guard 1: camera must be active and session must be valid.
  if (!cameraActive || !cameraSessionId) {
    console.debug('[preview] skip — camera not active or session=0');
    return;
  }
  // Guard 2: live stream already open for this session.
  if (_previewActive && activeStreamSession === cameraSessionId) {
    console.debug('[preview] skip — already streaming session=' + cameraSessionId);
    return;
  }
  // Guard 3: src assignment already in progress for this session
  // (guards the window between "claim" and "stream.src = url").
  if (previewStarting && activeStreamSession === cameraSessionId) {
    console.debug('[preview] skip — already starting session=' + cameraSessionId);
    return;
  }

  // Claim this session — any concurrent call for the same session will hit guard 3.
  previewStarting     = true;
  activeStreamSession = cameraSessionId;

  const stream = document.getElementById('camera-stream');
  if (!stream) { previewStarting = false; return; }

  // Remove the old src before assigning new one so Firefox does not reuse a
  // cached or half-closed connection.  load() is a no-op on <img> but harmless.
  stream.removeAttribute('src');
  stream.load?.();

  const url = '/camera/stream?session=' + cameraSessionId
              + '&overlay=0&client=' + CLIENT_ID + '&t=' + Date.now();
  console.debug('[preview] stream url assigned session=' + cameraSessionId + ' url=' + url);
  stream.onload = () => console.debug('[preview] stream load event session=' + cameraSessionId);
  stream.src    = url;
  _previewActive  = true;
  previewStarting = false;
}

function stopCameraPreview() {
  const stream = document.getElementById('camera-stream');
  if (!stream) return;
  console.debug('[preview] STOP session=' + activeStreamSession);
  stream.onerror = null;           // prevent spurious onerror when removing src
  stream.removeAttribute('src');   // abort MJPEG connection without triggering load of page URL
  _previewActive  = false;
  previewStarting = false;
  activeStreamSession = null;
}

// Called when the MJPEG stream img fires onerror (connection closed by server).
// This can happen when the camera stops or the session ends.
function _onStreamError(e) {
  console.debug('[preview] stream error/close — session=' + activeStreamSession
                + '; will re-check via pollStatus');
  _previewActive  = false;
  previewStarting = false;
  activeStreamSession = null;
}

function setUpdateTime(ts) {
  document.getElementById('update-time').textContent = ts ? '更新時間：' + ts : '更新時間：—';
}

// ── Polling /status every 2 s ─────────────────────────────────────────────
// startStatusPolling() is idempotent: if the timer is already running it
// returns immediately instead of stacking a second interval.
function startStatusPolling() {
  if (statusPollTimer !== null) {
    console.debug('[poll] startStatusPolling — timer already running, skip');
    return;
  }
  statusPollTimer = setInterval(pollStatus, 2000);
  console.debug('[poll] status polling started CLIENT_ID=' + CLIENT_ID);
  pollStatus();
}

function stopStatusPolling() {
  if (statusPollTimer !== null) {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
  }
}

async function pollStatus() {
  try {
    const res  = await fetch('/status?client=' + CLIENT_ID);
    const data = await res.json();

    const newCamActive = data.camera_active === true;
    const newRecActive = data.recognition_running === true;
    // session_id is the authoritative backend value; cameraSessionId must match it.
    const newSid = (data.session_id != null) ? data.session_id : cameraSessionId;

    // Detect session mismatch — happens when:
    //   - app restarts while the browser page stays open
    //   - camera is stopped/started externally
    //   - a failed start incremented the backend counter without the frontend knowing
    if (newSid !== cameraSessionId) {
      console.debug('[poll] session sync', cameraSessionId, '→', newSid);
      cameraSessionId = newSid;
      // Backend serves a new MJPEG stream URL per session.  Restart the img.src
      // so the browser connects to the current session's stream.
      // Skip if /camera/start is in-flight: toggleCamera() will call syncCameraUI()
      // with the definitive session_id when the response arrives, preventing a
      // duplicate stream for the same session.
      if (newCamActive && !cameraStartPending) startCameraPreview();
    }

    // Camera-stop-pending guard: suppress stale camera_running=true while the
    // stop request is in-flight or the backend is racing the status poll.
    if (cameraStopPending) {
      if (newCamActive === false) {
        console.debug('[poll] camera stop confirmed via /status');
        cameraStopPending = false;
        // Fall through — cameraActive is already false, syncCameraUI() not needed.
      } else {
        console.debug('[poll] camera stop pending — ignoring stale camera_running=true');
        return;
      }
    }

    // Stop-pending guard: if the user clicked stop but /status still echoes
    // recognition_running=true (request in-flight or backend race), hold the
    // UI in the stopped state until the server confirms recognition_running=false.
    if (recognitionStopPending && newRecActive === true) {
      console.debug('[poll] stop pending — ignoring stale recognition_running=true');
      // Still handle camera-level changes (e.g. camera itself stopped)
      if (newCamActive !== cameraActive) {
        cameraActive = newCamActive;
        syncCameraUI();
      } else if (newCamActive && !_previewActive && !cameraStartPending) {
        startCameraPreview();
      }
    } else {
      if (recognitionStopPending && newRecActive === false) {
        console.debug('[poll] stop confirmed via /status poll');
        recognitionStopPending = false;
      }
      if (newCamActive !== cameraActive || newRecActive !== recognitionActive) {
        const wasRec = recognitionActive;
        cameraActive      = newCamActive;
        recognitionActive = newRecActive;
        // Stop weight polling when recognition ends externally (camera killed, server
        // restart, etc.).  Do NOT call startWeightPolling() here: a stale
        // recognition_running=true status response arriving after the user clicked
        // 停止辨識 (and recognitionStopPending was already cleared by the stop
        // response) would restart polling.  Explicit starts happen only in
        // startRecognize() and initCameraState().
        if (wasRec && !newRecActive) stopWeightPolling('pollStatus-recognition-ended');
        syncCameraUI();
      } else if (newCamActive && !_previewActive && !cameraStartPending) {
        // Camera active but stream was cleared (e.g. onerror fired) and no start in progress.
        startCameraPreview();
      }
    }

    // Safety: if backend confirms camera is active, the recognize button must be
    // enabled.  Guards against any race or missed syncCameraUI() call.
    if (newCamActive) {
      const btnRec = document.getElementById('btn-recognize');
      if (btnRec && btnRec.disabled) {
        console.warn('[poll] safety: camera active but btnRec disabled — forcing enabled');
        btnRec.disabled = false;
      }
    }

    // In camera mode, freeze the results table when recognition is stopped or stopping.
    // This prevents a late inference result from updating the UI after the user stopped.
    // When camera is off (upload mode), always render.
    const shouldRenderCounts = !cameraActive || (recognitionActive && !recognitionStopPending);
    if (shouldRenderCounts && data.counts && Object.keys(data.counts).length > 0) {
      rerenderTable(data.counts);
      setUpdateTime(data.timestamp);
    }

    // 標準重量: update whenever new recognition results arrive via /status.
    // Only take actual weight and tolerance from the backend response.
    // Expected is always recomputed from current frontend standards × classWeights so
    // that manual standard edits are never overwritten by the 2-second poll cycle.
    if (shouldRenderCounts && data.weight_verification) {
      if (_lastKnownWV == null) {
        _lastKnownWV = { ...data.weight_verification };
      } else {
        _lastKnownWV.actual    = data.weight_verification.actual;
        _lastKnownWV.tolerance = data.weight_verification.tolerance;
      }
      recalculateExpectedAndRender('pollStatus');
    }

    // recalculateExpectedAndRender owns expected; pollWeight() updates actual at 500 ms.

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

// ── Utility ───────────────────────────────────────────────────────────────────
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Weight poll interval while recognition is active.
// 500 ms gives a good balance: fast enough to feel live, slow enough to not
// flood the backend.  The backend background thread refreshes the cache at
// 10 Hz so the UI never lags more than WEIGHT_POLL_INTERVAL_MS from a weight
// change.
const WEIGHT_POLL_INTERVAL_MS = 500;

// ── Live weight polling — async loop, only active while recognition is running ─
//
// Design: startWeightPolling() launches a single async weightPollingLoop().
// The loop runs while weightPollingActive, recognitionActive, and
// token === weightPollToken are all true.  stopWeightPolling() sets
// weightPollingActive=false, increments weightPollToken (causing the loop
// condition to fail on the next iteration), and aborts any in-flight fetch.
// No setInterval is used — the loop awaits sleep(WEIGHT_POLL_INTERVAL_MS) between each fetch.
//
// Callers that may call startWeightPolling():
//   • startRecognize()   — user clicks 開始辨識
//   • initCameraState()  — page load when recognition was already running
//
// pollStatus() must NEVER call startWeightPolling().

function startWeightPolling(reason = '') {
  if (!recognitionActive) {
    console.warn('[weight] startWeightPolling ignored — recognitionActive=false reason=' + reason);
    return;
  }
  stopWeightPolling('restart-before-start');
  weightPollingActive = true;
  weightPollToken++;
  const token = weightPollToken;
  console.debug('[weight] startWeightPolling token=' + token + ' reason=' + reason);
  weightPollingLoop(token, reason);   // fire-and-forget; loop exits when token is invalidated
}

function stopWeightPolling(reason = '') {
  weightPollingActive = false;
  weightPollToken++;
  if (weightPollAbortController) {
    weightPollAbortController.abort();
    weightPollAbortController = null;
  }
  console.debug('[weight] stopWeightPolling — token now=' + weightPollToken + ' reason=' + reason);
}

// Show "未量測" before recognition starts (or after it stops without a final read).
function resetWeightDisplay() {
  const actWtEl = document.getElementById('stat-act-wt');
  const wokEl   = document.getElementById('stat-wt-ok');
  const wokBox  = document.getElementById('stat-wt-ok-box');
  if (actWtEl) actWtEl.textContent = '未量測';
  if (wokEl && wokBox) {
    wokEl.textContent = '不明';
    wokBox.className  = 'stat-box weight-ok-box';
  }
  _lastKnownWV = null;
}

async function weightPollingLoop(token, reason) {
  console.debug('[weight] loop start token=' + token + ' reason=' + reason);
  while (
    weightPollingActive &&
    recognitionActive &&
    !recognitionStopPending &&
    !cameraStopPending &&
    token === weightPollToken
  ) {
    await pollWeight({ final: false, token, reason: 'loop' });
    await sleep(WEIGHT_POLL_INTERVAL_MS);
    // loop condition re-evaluated after each sleep
  }
  console.debug('[weight] loop exit', {
    token,
    current: weightPollToken,
    weightPollingActive,
    recognitionActive,
    recognitionStopPending,
    cameraStopPending,
  });
}

// pollWeight({ final, token, reason })
//
//   final=true  — one-shot read (freeze after stop). No polling guards applied.
//   final=false — loop call. Returns immediately if ANY guard fails:
//     • !weightPollingActive
//     • !recognitionActive
//     • recognitionStopPending
//     • cameraStopPending
//     • token !== weightPollToken
//   An AbortController is stored in weightPollAbortController so stopWeightPolling()
//   can cancel an in-flight fetch instantly.
async function pollWeight({ final: isFinal = false, token = undefined, reason = '' } = {}) {
  if (!isFinal) {
    if (!weightPollingActive) {
      console.debug('[weight] skip — not active reason=' + reason);
      return;
    }
    if (!recognitionActive) {
      console.debug('[weight] skip — recognition inactive reason=' + reason);
      return;
    }
    if (recognitionStopPending) {
      console.debug('[weight] skip — recognitionStopPending reason=' + reason);
      return;
    }
    if (cameraStopPending) {
      console.debug('[weight] skip — cameraStopPending reason=' + reason);
      return;
    }
    if (token !== weightPollToken) {
      console.debug('[weight] skip — stale token ' + token + '/' + weightPollToken + ' reason=' + reason);
      return;
    }
    console.debug('[weight] fetch reason=' + reason + ' token=' + token + ' client=' + CLIENT_ID);
  } else {
    console.debug('[weight] final read reason=' + reason + ' client=' + CLIENT_ID);
  }

  const controller = new AbortController();
  weightPollAbortController = controller;
  try {
    const url = '/api/weight?client=' + CLIENT_ID + '&reason=' + encodeURIComponent(reason);
    const res  = await fetch(url, { signal: controller.signal });
    const data = await res.json();
    // Discard result if state changed while fetch was in-flight (non-final only).
    if (!isFinal && (!weightPollingActive || token !== weightPollToken || !recognitionActive)) {
      console.debug('[weight] result discarded after fetch reason=' + reason);
      return;
    }
    if (data.weight_verification) {
      // Update actual weight from scale (fresh read); recompute expected locally.
      if (_lastKnownWV == null) {
        _lastKnownWV = { ...data.weight_verification };
      } else {
        _lastKnownWV.actual    = data.weight_verification.actual;
        _lastKnownWV.tolerance = data.weight_verification.tolerance;
      }
      recalculateExpectedAndRender('pollWeight');
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      console.debug('[weight] fetch aborted reason=' + reason);
    }
    // Other errors: server starting up — ignore
  } finally {
    if (weightPollAbortController === controller) {
      weightPollAbortController = null;
    }
  }
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
  syncCameraUI();   // updates btnRec.disabled = !pendingFile when camera is off
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
  syncCameraUI();   // updates btnRec.disabled = !pendingFile when camera is off
}

// ── Recognize ─────────────────────────────────────────────────────────────
async function startRecognize() {
  const btn = document.getElementById('btn-recognize');
  console.debug('[recognize] click: cameraActive=' + cameraActive
    + ' recognitionActive=' + recognitionActive
    + ' btn.disabled=' + btn.disabled
    + ' pendingFile=' + !!pendingFile);

  if (cameraActive) {
    if (recognitionActive) {
      // Optimistic stop: update UI immediately; await server confirmation before final weight read.
      console.debug('[recognize] stop clicked — recognitionStopPending=true');
      recognitionStopPending = true;
      recognitionActive = false;
      stopWeightPolling('recognition-stop-click');
      syncCameraUI();
      try {
        const res = await fetch('/camera/recognition/stop', { method: 'POST' });
        const d   = await res.json();
        if (d.recognition_running === false) {
          console.debug('[recognize] stop confirmed by server gen=' + d.recognition_generation);
          recognitionStopPending = false;
        }
        // If recognition_running is still true the guard stays up until the next /status poll.
      } catch (e) {
        // Network error: resync from server to restore consistent state.
        recognitionStopPending = false;
        try {
          const r = await fetch('/camera/status');
          const d = await r.json();
          recognitionActive = d.recognition_running === true;
          syncCameraUI();
        } catch (_) {}
      }
      // One final weight read after the server confirms recognition stopped.
      await pollWeight({ final: true, reason: 'recognition-stop-final' });
    } else {
      // Start recognition: wait for server confirmation before updating UI.
      recognitionStopPending = false;   // user explicitly started — cancel any pending guard
      btn.disabled = true;
      try {
        console.debug('[recognize] POST /camera/recognition/start');
        const res  = await fetch('/camera/recognition/start', { method: 'POST' });
        const data = await res.json();
        console.debug('[recognize] response: ok=' + data.ok + ' status=' + data.status);
        if (data.ok) {
          recognitionActive = true;
          startWeightPolling('recognition-start-click');
        } else {
          alert('無法開始辨識：' + (data.error || '未知錯誤'));
        }
        syncCameraUI();
      } catch (e) {
        alert('連線錯誤：' + e.message);
      } finally {
        btn.disabled = false;
      }
    }
  } else if (pendingFile) {
    // One-shot image upload detection
    btn.disabled = true;
    btn.textContent = '辨識中…';
    try {
      const form = new FormData();
      form.append('image', pendingFile);
      const res  = await fetch('/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (data.ok) {
        rerenderTable(data.counts);
        setUpdateTime(data.timestamp);
        showAnnotatedImage(data.annotated_image);
        if (data.weight_verification) {
          // Seed _lastKnownWV with actual weight and tolerance from backend;
          // recompute expected from current standards × classWeights.
          if (_lastKnownWV == null) {
            _lastKnownWV = { ...data.weight_verification };
          } else {
            _lastKnownWV.actual    = data.weight_verification.actual;
            _lastKnownWV.tolerance = data.weight_verification.tolerance;
          }
          recalculateExpectedAndRender('upload');
        }
      } else {
        alert('辨識失敗：' + (data.error || '未知錯誤'));
      }
    } catch (e) {
      alert('連線錯誤：' + e.message);
    } finally {
      btn.disabled = !pendingFile;
      btn.textContent = '⚐ 開始辨識';
    }
  } else {
    alert('請先上傳圖片或開啟攝影機');
  }
}

// ── Camera toggle ─────────────────────────────────────────────────────────
async function toggleCamera() {
  const btn = document.getElementById('btn-camera');

  if (cameraActive) {
    // STOP PATH — update UI immediately; do NOT await the backend call.
    // The Jetson camera thread join can take ~2 s; the button must change at once.
    btn.disabled           = true;   // prevent double-click during state flip
    cameraStopPending      = true;
    cameraActive           = false;
    recognitionActive      = false;
    recognitionStopPending = false;
    cameraSessionId        = 0;
    stopWeightPolling('camera-stop-click');
    stopCameraPreview();
    syncCameraUI();          // button shows 開啟攝影機 immediately
    btn.disabled = false;    // re-enable immediately — user can start again

    // Fire stop request in background — UI is already updated, don't block.
    fetch('/camera/stop', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        console.debug('[toggle] camera stop confirmed camera_running=' + d.camera_running);
        cameraStopPending = false;
      })
      .catch(e => {
        console.warn('[toggle] /camera/stop error:', e.message);
        cameraStopPending = false;
        // Optimistic state already applied; pollStatus() will resync on next tick.
      });
    return;
  }

  // START PATH — must await to get session_id and detect open failures.
  btn.disabled       = true;
  cameraStartPending = true;   // suppress pollStatus() preview starts during the wait
  console.debug('[toggle] start: cameraStopPending=' + cameraStopPending
                + ' _previewActive=' + _previewActive
                + ' activeStreamSession=' + activeStreamSession
                + ' cameraSessionId=' + cameraSessionId);
  try {
    const res  = await fetch('/camera/start', { method: 'POST' });
    const data = await res.json();
    console.debug('[toggle] /camera/start response:', JSON.stringify(data));
    if (data.ok) {
      const newSessionId = data.session_id || 0;
      // Only tear down the current stream if it belongs to a different (wrong) session.
      // Race: pollStatus() is suppressed by cameraStartPending, but if it fired before
      // the flag was set, it may have started a stream for the correct newSessionId
      // already.  Destroying that stream and opening a second one would cause the
      // duplicate /camera/stream connections we observed.
      if (activeStreamSession !== null && activeStreamSession !== newSessionId) {
        console.debug('[toggle] clearing stale stream session=' + activeStreamSession
                      + ' → new=' + newSessionId);
        stopCameraPreview();
      }
      cameraActive      = true;
      cameraSessionId   = newSessionId;
      cameraStopPending = false;   // start succeeded — any prior stop is resolved
      recognitionActive = false;
      console.debug('[toggle] camera started session=' + cameraSessionId
                    + ' existingStream=' + activeStreamSession);
    } else {
      alert('無法開啟攝影機：' + (data.error || '未知錯誤'));
      return;
    }
    syncCameraUI();
    console.debug('[toggle] after syncCameraUI: _previewActive=' + _previewActive
                  + ' stream.src=' + (document.getElementById('camera-stream')?.getAttribute('src') || '(none)'));
  } finally {
    cameraStartPending = false;
    btn.disabled       = false;
  }
}

function syncCameraUI() {
  const btn         = document.getElementById('btn-camera');
  const btnRec      = document.getElementById('btn-recognize');
  const stream      = document.getElementById('camera-stream');
  const preview     = document.getElementById('preview-img');
  const placeholder = document.getElementById('placeholder');

  if (cameraActive) {
    btn.textContent = '⏹ 關閉攝影機';
    btn.classList.add('active');
    stream.onerror = _onStreamError;   // reset _previewActive if stream dies
    stream.style.display = 'block';
    preview.style.display = 'none';
    placeholder.classList.remove('visible');
    // Only set a new stream URL when one isn't already active.
    // Recognition-toggle calls to syncCameraUI() must not reassign img.src
    // mid-stream, which would restart the connection unnecessarily.
    if (!_previewActive) {
      startCameraPreview();
    }
    // Recognition button: toggles recognition, always enabled when camera is on
    btnRec.disabled = false;
    console.debug('[syncUI] cameraActive=true recActive=' + recognitionActive
      + ' previewActive=' + _previewActive + ' btnRec.disabled=' + false);
    if (recognitionActive) {
      btnRec.textContent = '⏹ 停止辨識';
      btnRec.classList.add('active');
    } else {
      btnRec.textContent = '⚐ 開始辨識';
      btnRec.classList.remove('active');
    }
  } else {
    btn.textContent = '◉ 開啟攝影機';
    btn.classList.remove('active');
    stopCameraPreview();
    stream.style.display = 'none';
    placeholder.classList.add('visible');
    preview.style.display = 'none';
    // Recognition button: only enabled when a file is pending for upload detection
    btnRec.textContent = '⚐ 開始辨識';
    btnRec.classList.remove('active');
    btnRec.disabled = !pendingFile;
    console.debug('[syncUI] cameraActive=false pendingFile=' + !!pendingFile
      + ' btnRec.disabled=' + btnRec.disabled);
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
