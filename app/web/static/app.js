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
let stdPendingPackage = null;   // package a pending standards edit belongs to
let uwPendingPackage  = null;   // package a pending unit-weight edit belongs to
let stdInFlight       = null;   // promise for a standards POST already on the wire
let uwInFlight        = null;   // promise for a unit-weight POST already on the wire
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
let activePackageId         = null;   // id of the active model package
let availablePackages       = [];     // /api/model-packages payload
let activePreset            = null;   // selected surgery preset, if the package has any
let availablePresets        = [];     // /api/inventory-presets payload
let presetModified          = false;  // standards hand-edited away from the preset
// Whether the numbers on screen came from an inference for the ACTIVE package.
// "No recognition yet" and "recognised, found nothing" must never both render
// as an empty table with a 盤點符合 verdict attached.
let hasCurrentResult        = false;

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  if (appInitialized) {
    console.warn('[init] DOMContentLoaded fired again — ignoring duplicate CLIENT_ID=' + CLIENT_ID);
    return;
  }
  appInitialized = true;
  console.debug('[init] app start CLIENT_ID=' + CLIENT_ID);
  await loadPackages();
  await loadPresets();
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

// ── Inventory model ───────────────────────────────────────────────────────
// One pure function turns counts + standards into everything the UI shows, so
// the hero verdict, the summary cards and the table can never disagree.
//
// UNITS are the headline number, not classes: "缺少 2 支" is what the operator
// has to go and find. "缺少 1 類" hides that two Towel-Clamps are missing.
function calculateInventorySummary(counts, standardsMap) {
  counts = counts || {};
  standardsMap = standardsMap || {};

  const names = new Set(Object.keys(standardsMap).concat(Object.keys(counts)));
  const rows = [];
  let missingUnits = 0, extraUnits = 0, expectedUnits = 0, detectedUnits = 0;

  names.forEach(cls => {
    const detected = Number(counts[cls]) || 0;
    const standard = Number(standardsMap[cls]) || 0;
    // Neither expected nor seen: auto-registered noise, not part of this tray.
    if (detected === 0 && standard === 0) return;

    expectedUnits += standard;
    detectedUnits += detected;

    let status = 'normal';
    let delta = 0;
    if (detected < standard) {
      status = 'missing';
      delta = standard - detected;
      missingUnits += delta;
    } else if (detected > standard) {
      status = 'extra';
      delta = detected - standard;
      extraUnits += delta;
    }
    rows.push({ cls, detected, standard, status, delta });
  });

  const order = { missing: 0, extra: 1, normal: 2 };
  rows.sort((a, b) => (order[a.status] - order[b.status])
    || (a.cls < b.cls ? -1 : a.cls > b.cls ? 1 : 0));

  const byStatus = st => rows.filter(r => r.status === st);
  return {
    rows,
    missingRows: byStatus('missing'),
    extraRows: byStatus('extra'),
    normalRows: byStatus('normal'),
    missingClasses: byStatus('missing').length,
    extraClasses: byStatus('extra').length,
    normalClasses: byStatus('normal').length,
    missingUnits, extraUnits, expectedUnits, detectedUnits,
  };
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ── Standards ─────────────────────────────────────────────────────────────
async function fetchStandards() {
  try {
    const res = await fetch('/standards');
    standards = await res.json();
  } catch (e) { console.error('fetchStandards:', e); }
}

// Edits are debounced, so a keystroke can still be in flight when the operator
// switches packages.  Each pending edit remembers which package it was typed
// against, and that identity travels with the request — the backend refuses it
// with 409 rather than stamping one department's quantities onto another's.
function scheduleStandardsSync() {
  clearTimeout(stdDebounce);
  stdPendingPackage = activePackageId;
  stdDebounce = setTimeout(() => { stdDebounce = null; pushStandards(stdPendingPackage); }, 300);
}

// A hand-edited standard must change the verdict NOW, not at the next
// inference: the operator is typing "2" precisely to find out whether the tray
// is short.  The row itself is patched in place rather than re-rendered, so the
// caret does not jump out of the input mid-keystroke.
function onStandardEdit(input) {
  const cls = input.dataset.cls;
  if (cls) standards[cls] = parseInt(input.value) || 0;

  const summary = calculateInventorySummary(lastCounts, standards);
  _lastSummary = summary;
  renderSummaryCards(summary);
  renderOverallStatus();

  const row = summary.rows.find(r => r.cls === cls);
  const tr  = input.closest('tr');
  if (row && tr) {
    const badge = tr.querySelector('td:last-child');
    if (badge) {
      if (row.status === 'missing') {
        badge.innerHTML = `<span class="badge badge-missing">缺少 ${row.delta} 支</span>`;
      } else if (row.status === 'extra') {
        badge.innerHTML = `<span class="badge badge-extra">多出 ${row.delta} 支</span>`;
      } else {
        badge.innerHTML = '<span class="badge badge-normal">正常</span>';
      }
    }
    tr.classList.remove('row-missing', 'row-extra', 'row-normal');
    tr.classList.add('row-' + row.status);
  }

  // Expected weight is standards × class_weights, so it moves with the edit too.
  refreshWeightVerification();
  scheduleStandardsSync();
}

function collectStandardsInputs() {
  const payload = {};
  document.querySelectorAll('.std-input').forEach(inp => {
    payload[inp.dataset.cls] = parseInt(inp.value) || 0;
  });
  return payload;
}

// Read the backend's error message, falling back to the status code.
async function readError(res) {
  try {
    const data = await res.json();
    if (data && data.error) return data.error;
  } catch (e) { /* not JSON */ }
  return 'HTTP ' + res.status;
}

async function pushStandards(packageId) {
  const payload = collectStandardsInputs();
  const target = packageId || activePackageId;
  const request = (async () => {
    try {
      const res = await fetch('/standards', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Model-Package': target || '',
        },
        body: JSON.stringify({ package_id: target, values: payload }),
      });
      if (res.status === 409) {
        // The package changed under this edit; the write was refused on purpose.
        console.warn('[standards] rejected — package changed since the edit');
        await fetchStandards();
        rerenderTable(lastCounts);
        refreshWeightVerification();
        return { ok: false, conflict: true, error: await readError(res) };
      }
      if (!res.ok) {
        // Rejected (bad value, incompatible class, ...).  Do NOT pretend the
        // local payload was stored — resync from the server instead, or the UI
        // would show numbers that exist only in the browser.
        const error = await readError(res);
        console.error('[standards] rejected: ' + error);
        await fetchStandards();
        rerenderTable(lastCounts);
        refreshWeightVerification();
        return { ok: false, error };
      }
      let data = {};
      try { data = await res.json(); } catch (e) { /* empty body */ }
      standards = (data && data.standards) ? data.standards : payload;
      rerenderTable(lastCounts);
      refreshWeightVerification();
      return { ok: true };
    } catch (e) {
      console.error('pushStandards:', e);
      return { ok: false, error: e.message };
    }
  })();

  stdInFlight = request;
  try {
    return await request;
  } finally {
    if (stdInFlight === request) stdInFlight = null;
  }
}

// Send anything still pending before switching packages: both edits sitting in
// a debounce timer AND requests already on the wire.  Waiting only on the timer
// would let a request that fired 50 ms ago be forgotten, so the operator's last
// keystroke could be lost (or rejected) without anyone noticing.
async function flushPendingEdits() {
  const jobs = [];
  if (stdDebounce !== null) {
    clearTimeout(stdDebounce);
    stdDebounce = null;
    jobs.push(pushStandards(stdPendingPackage));
  } else if (stdInFlight) {
    jobs.push(stdInFlight);
  }
  if (uwDebounce !== null) {
    clearTimeout(uwDebounce);
    uwDebounce = null;
    jobs.push(pushUnitWeights(uwPendingPackage));
  } else if (uwInFlight) {
    jobs.push(uwInFlight);
  }
  if (!jobs.length) return { ok: true };

  console.debug('[edits] flushing ' + jobs.length + ' pending edit(s) before switch');
  const settled = await Promise.all(
    jobs.map(p => Promise.resolve(p).catch(e => ({ ok: false, error: e && e.message })))
  );
  const failure = settled.find(r => r && r.ok === false);
  return failure ? { ok: false, error: failure.error || '設定儲存失敗' } : { ok: true };
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

// ── Model packages ────────────────────────────────────────────────────────
// The active package decides which instruments exist, what they weigh, and how
// many are expected.  Switching it invalidates every cached result on screen.
async function loadPackages() {
  try {
    const res  = await fetch('/api/model-packages');
    const data = await res.json();
    availablePackages = data.packages || [];
    activePackageId   = data.active || null;
    renderPackageSelect();
  } catch (e) {
    console.error('loadPackages:', e);
  }
}

function renderPackageSelect() {
  const sel = document.getElementById('package-select');
  if (!sel) return;
  if (!availablePackages.length) {
    sel.innerHTML = '<option value="">（無可用套件）</option>';
    return;
  }
  sel.innerHTML = availablePackages.map(p => {
    // A template or a broken manifest is listed but cannot be selected — the
    // operator should see it exists and needs configuring, not silently miss it.
    const disabled = p.activatable ? '' : ' disabled';
    let suffix = '';
    if (p.template)   suffix = '（範本，未設定）';
    else if (!p.valid) suffix = '（設定錯誤）';
    const selected = p.id === activePackageId ? ' selected' : '';
    return `<option value="${p.id}"${disabled}${selected}>${p.display_name}${suffix}</option>`;
  }).join('');
}

// ── Surgery presets ───────────────────────────────────────────────────────
// A preset changes only WHAT IS EXPECTED. The model does not reload, the camera
// keeps running, and the model generation does not advance — this is not a
// package switch.
async function loadPresets() {
  try {
    const res  = await fetch('/api/inventory-presets');
    const data = await res.json();
    availablePresets = data.presets || [];
    activePreset     = data.active_preset || null;
    presetModified   = data.preset_modified === true;
    renderPresetSelect();
  } catch (e) {
    console.error('loadPresets:', e);
  }
}

function renderPresetSelect() {
  const bar    = document.getElementById('surgery-bar');
  const select = document.getElementById('preset-select');
  const meta   = document.getElementById('preset-meta');
  if (!bar || !select) return;

  // Packages with one fixed tray get no control at all, rather than a dead one.
  if (!availablePresets.length) {
    bar.style.display = 'none';
    select.innerHTML = '';
    if (meta) meta.textContent = '';
    renderOverallStatus();
    return;
  }

  bar.style.display = 'flex';
  select.innerHTML = '<option value="" disabled>請選擇手術類型</option>'
    + availablePresets.map(p => {
      const selected = p.id === activePreset ? ' selected' : '';
      return `<option value="${p.id}"${selected}>${escapeHtml(p.display_name)}</option>`;
    }).join('');
  if (activePreset === null) select.value = '';
  // Nothing chosen yet is a state the operator must resolve before any number
  // on screen means anything — make the control itself say so.
  select.classList.toggle('unset', activePreset === null);

  if (meta) {
    const current = availablePresets.find(p => p.id === activePreset);
    if (!current) {
      meta.textContent = '';
      meta.className = '';
    } else if (presetModified) {
      // The operator hand-edited the table; say so instead of showing a factory
      // total that no longer matches what is expected.
      meta.textContent = '已修改';
      meta.className = 'modified';
    } else {
      meta.textContent = `${current.instrument_count} 支 · `
        + `${current.expected_weight.toFixed(0)} g`;
      meta.className = '';
    }
  }
  renderOverallStatus();
}

async function onPresetChange(evt) {
  const select   = evt.target;
  const targetId = select.value;
  if (!targetId || (targetId === activePreset && !presetModified)) return;

  const previous = activePreset;
  select.disabled = true;
  setPackageStatus('套用手術類型…', 'busy');
  try {
    // Same rule as a package switch: a debounced standards edit must land (or
    // fail loudly) before the tray is replaced under it.
    const flushed = await flushPendingEdits();
    if (!flushed.ok) {
      alert('設定尚未儲存，未切換手術類型：\n' + (flushed.error || '未知錯誤'));
      select.value = previous || '';
      setPackageStatus('設定未儲存', 'error');
      return;
    }

    const res = await fetch('/api/inventory-preset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Model-Package': activePackageId || '',
      },
      body: JSON.stringify({ id: targetId, package_id: activePackageId }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      alert('切換手術類型失敗：' + (data.error || 'HTTP ' + res.status));
      select.value = previous || '';
      setPackageStatus('切換失敗', 'error');
      await loadPresets();
      return;
    }

    activePreset   = data.active_preset || targetId;
    presetModified = data.preset_modified === true;
    availablePresets = data.presets || availablePresets;
    standards = data.standards || standards;

    // The counts were produced for the previous tray; the backend has already
    // invalidated them, so drop what is on screen too.
    lastCounts = {};
    hasCurrentResult = false;
    rerenderTable({});
    setUpdateTime('');
    resetWeightDisplay();
    await fetchStandards();
    await fetchUnitWeights();
    rerenderTable({});
    renderPresetSelect();
    setPackageStatus('已套用', 'ok');
    setTimeout(() => setPackageStatus('', ''), 3000);
  } catch (e) {
    alert('連線錯誤：' + e.message);
    select.value = previous || '';
    setPackageStatus('連線錯誤', 'error');
  } finally {
    select.disabled = false;
  }
}

function setPackageStatus(text, cls) {
  const el = document.getElementById('package-status');
  if (!el) return;
  el.textContent = text || '';
  el.className   = cls || '';
}

async function onPackageChange(evt) {
  const sel = evt.target;
  const targetId = sel.value;
  if (!targetId || targetId === activePackageId) return;

  const previous = activePackageId;
  sel.disabled = true;
  setPackageStatus('切換中…', 'busy');
  try {
    // Send any pending edit first, tagged with the package it was typed
    // against.  If it cannot be saved, abort the switch rather than silently
    // discarding the operator's last change.
    const flushed = await flushPendingEdits();
    if (!flushed.ok) {
      alert('設定尚未儲存，未切換器械套件：\n' + (flushed.error || '未知錯誤'));
      sel.value = previous || '';
      setPackageStatus('設定未儲存', 'error');
      return;
    }

    const res  = await fetch('/api/model-package', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: targetId }),
    });
    const data = await res.json();
    if (data.ok) {
      // The switch already completed on the server, so everything derived from
      // the old package must be dropped before the next poll paints it again.
      await applyPackageSwitch(targetId, data);
      setPackageStatus(data.recognition_resumed ? '已切換 · 辨識中' : '已切換', 'ok');
      setTimeout(() => setPackageStatus('', ''), 3000);
    } else {
      alert('切換器械套件失敗：' + (data.error || '未知錯誤'));
      sel.value = previous || '';
      setPackageStatus('切換失敗', 'error');
      // The backend rolled the old model back and may have resumed
      // recognition; resync rather than assuming anything.
      await syncRuntimeState(data);
    }
  } catch (e) {
    alert('連線錯誤：' + e.message);
    sel.value = previous || '';
    setPackageStatus('連線錯誤', 'error');
    await syncRuntimeState(null);
  } finally {
    sel.disabled = false;
  }
}

async function applyPackageSwitch(newId, response) {
  activePackageId = newId;
  // Any pending edit now belongs to a package that is no longer active.
  clearTimeout(stdDebounce); stdDebounce = null;
  clearTimeout(uwDebounce);  uwDebounce  = null;
  lastCounts = {};
  hasCurrentResult = false;
  rerenderTable({});
  setUpdateTime('');
  resetWeightDisplay();
  await fetchStandards();
  await fetchUnitWeights();
  await fetchClassWeights();
  await loadPackages();
  // Presets belong to the package, so the selector is rebuilt (or hidden) here.
  await loadPresets();
  await syncRuntimeState(response);
}

// Camera and recognition state come from the backend, never from an assumption.
// The backend decides whether recognition was resumed after a switch, so the UI
// must read that rather than hard-coding "recognition is now off".
async function syncRuntimeState(response) {
  let camRunning = null;
  let recRunning = null;

  // Always re-read the hardware.  A package switch can take ten seconds or
  // more, and the operator may have closed the camera during it — a snapshot
  // taken before the switch started is not evidence of anything now.
  try {
    const d = await (await fetch('/camera/status')).json();
    camRunning = d.running === true && d.stopping !== true;
    recRunning = d.recognition_running === true && d.stopping !== true;
  } catch (e) {
    console.warn('[sync] /camera/status failed:', e.message);
    if (response && typeof response.camera_running === 'boolean') {
      camRunning = response.camera_running;
      recRunning = response.recognition_resumed === true;
    } else {
      return;
    }
  }

  cameraActive           = camRunning;
  recognitionActive      = recRunning;
  recognitionStopPending = false;
  cameraStopPending      = false;
  console.debug('[sync] camera=' + camRunning + ' recognition=' + recRunning);

  if (recRunning) startWeightPolling('runtime-state-sync');
  else stopWeightPolling('runtime-state-sync');
  syncCameraUI();
}

function scheduleUnitWeightsSync() {
  clearTimeout(uwDebounce);
  uwPendingPackage = activePackageId;
  uwDebounce = setTimeout(() => { uwDebounce = null; pushUnitWeights(uwPendingPackage); }, 300);
}

async function pushUnitWeights(packageId) {
  const payload = {};
  document.querySelectorAll('.uw-input').forEach(inp => {
    payload[inp.dataset.cls] = parseFloat(inp.value) || 0;
  });
  const target = packageId || activePackageId;
  const request = (async () => {
    try {
      const res = await fetch('/unit_weights', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Model-Package': target || '',
        },
        body: JSON.stringify({ package_id: target, values: payload }),
      });
      if (res.status === 409) {
        console.warn('[unit_weights] rejected — package changed since the edit');
        await fetchUnitWeights();
        refreshWeightVerification();
        return { ok: false, conflict: true, error: await readError(res) };
      }
      if (!res.ok) {
        const error = await readError(res);
        console.error('[unit_weights] rejected: ' + error);
        await fetchUnitWeights();
        refreshWeightVerification();
        return { ok: false, error };
      }
      let data = {};
      try { data = await res.json(); } catch (e) { /* empty body */ }
      unitWeights = (data && data.unit_weights) ? data.unit_weights : payload;
      refreshWeightVerification();
      return { ok: true };
    } catch (e) {
      console.error('pushUnitWeights:', e);
      return { ok: false, error: e.message };
    }
  })();

  uwInFlight = request;
  try {
    return await request;
  } finally {
    if (uwInFlight === request) uwInFlight = null;
  }
}

// ── Weight verification display ───────────────────────────────────────────
// Five distinct states.  A reading that has not settled is NOT rendered as a
// red 不符合: an unstable scale is an unfinished measurement, not a failed
// inventory, and painting it red teaches operators to ignore red.
const WEIGHT_STATE_LABEL = {
  waiting_weight:     '等待重量',
  stabilizing:        '重量穩定中',
  no_standard_weight: '標準重量未設定',
  passed:             '重量符合',
  failed:             '重量不符合',
};

const WEIGHT_STATE_ICON = {
  waiting_weight:     '…',
  stabilizing:        '◔',
  no_standard_weight: '?',
  passed:             '✓',
  failed:             '✕',
};

const WEIGHT_STATE_MESSAGE = {
  waiting_weight:     '等待重量讀值',
  stabilizing:        '重量穩定中',
  no_standard_weight: '尚未設定標準重量',
  passed:             '重量在容許範圍內',
  failed:             '重量超出容許範圍',
};

function weightStateClass(state) {
  if (state === 'passed')      return 'wt-passed';
  if (state === 'failed')      return 'wt-failed';
  if (state === 'stabilizing') return 'wt-stabilizing';
  return 'wt-waiting';
}

function grams(value) {
  return (value == null || !Number.isFinite(Number(value)))
    ? '—' : Number(value).toFixed(1) + ' g';
}

function renderWeightVerification(wv) {
  if (!wv) return;
  const state = wv.state || 'waiting_weight';
  const card    = document.getElementById('weight-card');
  const icon    = document.getElementById('weight-icon');
  const title   = document.getElementById('weight-title');
  const numbers = document.getElementById('weight-numbers');
  const diff    = document.getElementById('weight-diff');
  if (!card) return;

  card.className = weightStateClass(state);
  if (icon)  icon.textContent  = WEIGHT_STATE_ICON[state] || '…';
  if (title) title.textContent = WEIGHT_STATE_LABEL[state] || '重量';

  if (numbers) {
    if (state === 'no_standard_weight') {
      const missing = wv.missing_class_weights || [];
      numbers.textContent = missing.length
        ? '缺少單重：' + missing.slice(0, 3).join('、') + (missing.length > 3 ? '…' : '')
        : '尚未設定標準重量';
    } else if (wv.actual == null) {
      numbers.textContent = '標準 ' + grams(wv.expected);
    } else {
      // A cached value from before the cable was pulled must not read as live.
      const stale = wv.fresh === false ? '（逾時）' : '';
      numbers.textContent = '實際 ' + grams(wv.actual) + stale
        + '　標準 ' + grams(wv.expected);
    }
  }

  if (diff) {
    if (wv.actual != null && wv.expected != null && state !== 'no_standard_weight') {
      const delta = Number(wv.actual) - Number(wv.expected);
      const sign  = delta > 0 ? '+' : '';
      diff.textContent = '差異 ' + sign + delta.toFixed(1) + ' g'
        + (wv.tolerance != null ? '（容許 ±' + Number(wv.tolerance).toFixed(1) + ' g）' : '');
    } else {
      diff.textContent = '';
    }
  }

  console.debug('[wv] state=' + state + ' actual=' + wv.actual
    + ' expected=' + wv.expected + ' passed=' + wv.passed
    + ' stable=' + wv.stable + ' fresh=' + wv.fresh);

  renderOverallStatus();
}

// Copy only the fields the backend owns (scale reading + gate) into the cached
// verification.  ``expected`` is deliberately NOT copied — it is recomputed
// locally so the 2-second poll never overwrites a manual standards edit.
function mergeBackendWV(src) {
  if (!src) return;
  if (_lastKnownWV == null) {
    _lastKnownWV = { ...src };
    return;
  }
  _lastKnownWV.actual        = src.actual;
  _lastKnownWV.tolerance     = src.tolerance;
  _lastKnownWV.stable        = src.stable;
  _lastKnownWV.fresh         = src.fresh;
  _lastKnownWV.sample_age    = src.sample_age;
  _lastKnownWV.sample_reason = src.sample_reason;
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
  const missingWeights = [];
  Object.entries(standards).forEach(([cls, std]) => {
    const qty = Number(std);   // Number("0")=0  Number("")=0  Number(undefined)=NaN
    if (!Number.isFinite(qty) || qty <= 0) return;   // 0 contributes nothing; NaN is invalid
    const uw = Number(classWeights[cls]);
    if (!Number.isFinite(uw) || uw <= 0) {
      // NOT treated as 0 g: skipping it would lower the expected total, which
      // is exactly the direction that lets an incomplete tray pass.
      missingWeights.push(cls);
      return;
    }
    expected += qty * uw;
    debugParts.push(cls + ':' + qty + '×' + uw + '=' + (qty * uw).toFixed(1));
  });

  const actual    = _lastKnownWV.actual;
  const tolerance = _lastKnownWV.tolerance;
  // Mirror the backend's stability gate exactly, but against the locally
  // recomputed expected weight.  Undefined stable/fresh (older payloads) are
  // treated as usable so nothing regresses to a permanent "measuring" state.
  const fresh  = _lastKnownWV.fresh  !== false;
  const stable = _lastKnownWV.stable !== false;

  let state, diff = null, passed = null, ready = false;
  if (missingWeights.length) {
    // A configuration fault, not a measurement state: the expected total is
    // wrong until every expected instrument has a unit weight.
    state = 'no_standard_weight';
    _lastKnownWV.message = '標準重量未完整設定：缺少 '
      + missingWeights.slice(0, 5).join('、')
      + (missingWeights.length > 5 ? '…' : '') + ' 的單重';
  } else if (actual == null || !fresh) {
    state = 'waiting_weight';
  } else if (!stable) {
    state = 'stabilizing';
  } else if (!(expected > 0)) {
    state = 'no_standard_weight';
  } else if (tolerance == null) {
    state = 'waiting_weight';
  } else {
    diff   = Math.abs(actual - expected);
    passed = diff <= tolerance;
    ready  = true;
    state  = passed ? 'passed' : 'failed';
  }

  // Patch _lastKnownWV so any subsequent call sees the fresh expected value.
  _lastKnownWV.expected              = expected;
  _lastKnownWV.difference            = diff;
  _lastKnownWV.passed                = passed;
  _lastKnownWV.ready                 = ready;
  _lastKnownWV.state                 = state;
  _lastKnownWV.missing_class_weights = missingWeights;
  if (!missingWeights.length) {
    _lastKnownWV.message = WEIGHT_STATE_MESSAGE[state] || '';
  }

  console.debug('[wv] recalc source=' + source
    + ' expected=' + expected.toFixed(1)
    + ' actual=' + actual
    + ' state=' + state
    + (debugParts.length
        ? '  classes=[' + debugParts.slice(0, 6).join(', ')
          + (debugParts.length > 6 ? '…' : '') + ']'
        : '  [no contributing classes]'));

  renderWeightVerification(_lastKnownWV);
}

let _lastKnownWV = null;

// ── Results table ─────────────────────────────────────────────────────────
// Rows are the UNION of expected instruments and detected ones.  Iterating only
// what the model saw is how an instrument that was missed entirely disappears
// from the table instead of being reported 缺少 — the single most important row
// in an inventory system.
//
// Order is missing → extra → normal, because at 600 px high the operator should
// never have to scroll to find out what is wrong.  Normal rows are folded away
// by default for the same reason.
let normalRowsCollapsed = true;   // frontend-only; not persisted
let _lastSummary = null;

function rerenderTable(counts) {
  lastCounts = counts || {};
  const summary = calculateInventorySummary(lastCounts, standards);
  _lastSummary = summary;
  renderInventoryRows(summary);
  renderSummaryCards(summary);
  renderOverallStatus();
}

function toggleNormalRows() {
  normalRowsCollapsed = !normalRowsCollapsed;
  if (_lastSummary) renderInventoryRows(_lastSummary);
}

function countCell(row) {
  // detected / [editable standard] — the operator never has to do the
  // subtraction themselves, and the standard stays editable.
  return `<td class="cls-count">`
    + `<span class="detected-num">${row.detected}</span>`
    + `<span class="count-sep">/</span>`
    + `<input type="number" min="0" step="1" class="std-input" data-cls="${escapeHtml(row.cls)}"`
    + ` value="${row.standard}" oninput="onStandardEdit(this)"></td>`;
}

function inventoryRow(row) {
  const name = escapeHtml(row.cls);
  let badge, cssClass;
  if (row.status === 'missing') {
    badge = `<span class="badge badge-missing">缺少 ${row.delta} 支</span>`;
    cssClass = 'row-missing';
  } else if (row.status === 'extra') {
    badge = `<span class="badge badge-extra">多出 ${row.delta} 支</span>`;
    cssClass = 'row-extra';
  } else {
    badge = '<span class="badge badge-normal">正常</span>';
    cssClass = 'row-normal' + (normalRowsCollapsed ? ' collapsed' : '');
  }
  return `<tr class="${cssClass}">`
    + `<td class="cls-name" title="${name}">${name}</td>`
    + countCell(row)
    + `<td>${badge}</td></tr>`;
}

function groupRow(cssClass, label, onclick) {
  const handler = onclick ? ` onclick="${onclick}"` : '';
  return `<tr class="group-row ${cssClass}"${handler}><td colspan="3">${label}</td></tr>`;
}

function renderInventoryRows(summary) {
  const tbody = document.getElementById('results-tbody');
  if (!tbody) return;

  if (!summary.rows.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="empty-row">尚無辨識結果</td></tr>';
    return;
  }

  let html = '';
  if (summary.missingRows.length) {
    html += groupRow('group-missing',
      `✕ 缺少 ${summary.missingUnits} 支 · ${summary.missingClasses} 類`);
    summary.missingRows.forEach(r => { html += inventoryRow(r); });
  }
  if (summary.extraRows.length) {
    html += groupRow('group-extra',
      `! 多出 ${summary.extraUnits} 支 · ${summary.extraClasses} 類`);
    summary.extraRows.forEach(r => { html += inventoryRow(r); });
  }
  if (summary.normalRows.length) {
    const normalUnits = summary.normalRows.reduce((n, r) => n + r.detected, 0);
    html += groupRow('group-normal',
      `✓ 正常 ${summary.normalClasses} 類 / ${normalUnits} 支`
      + `　<span class="fold-hint">[${normalRowsCollapsed ? '展開' : '收合'}]</span>`,
      'toggleNormalRows()');
    summary.normalRows.forEach(r => { html += inventoryRow(r); });
  }
  tbody.innerHTML = html;
}

function renderSummaryCards(summary) {
  const set = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  set('sum-missing-units', summary.missingUnits);
  set('sum-missing-classes', summary.missingClasses + ' 類');
  set('sum-extra-units', summary.extraUnits);
  set('sum-extra-classes', summary.extraClasses + ' 類');
  set('sum-normal-classes', summary.normalClasses);

  // Only light the card up when there is something to act on; a permanently
  // red "缺少 0" trains the operator to ignore red.
  const missingCard = document.getElementById('sum-missing');
  if (missingCard) missingCard.classList.toggle('active', summary.missingUnits > 0);
  const extraCard = document.getElementById('sum-extra');
  if (extraCard) extraCard.classList.toggle('active', summary.extraUnits > 0);
}

// ── Overall verdict ───────────────────────────────────────────────────────
// Strict precedence, worst first.  The hero must never say 盤點符合 while the
// table shows a missing instrument, and it must never claim a verdict at all
// before there is something to judge.
function overallStatus(summary, wv, opts) {
  opts = opts || {};
  if (opts.needsPreset) {
    return { cls: 'hero-warn', icon: '▾', title: '請選擇手術類型',
             detail: '未選擇手術類型前，標準數量與重量皆無意義' };
  }
  if (!opts.hasResult) {
    return { cls: 'hero-idle', icon: '…', title: '尚未辨識',
             detail: '請上傳圖片或開始辨識' };
  }

  const miss  = summary ? summary.missingUnits : 0;
  const extra = summary ? summary.extraUnits : 0;
  if (miss > 0 && extra > 0) {
    return { cls: 'hero-fail', icon: '✕', title: '盤點異常',
             detail: `缺少 ${miss} 支、多出 ${extra} 支` };
  }
  if (miss > 0) {
    return { cls: 'hero-fail', icon: '✕', title: '器械缺少',
             detail: `缺少 ${miss} 支 · ${summary.missingClasses} 類` };
  }
  if (extra > 0) {
    return { cls: 'hero-warn', icon: '!', title: '器械多出',
             detail: `多出 ${extra} 支 · ${summary.extraClasses} 類` };
  }

  // Counts are clean — the weight check is the only thing left that can fail.
  const state = wv && wv.state ? wv.state : 'waiting_weight';
  if (state === 'no_standard_weight') {
    return { cls: 'hero-warn', icon: '?', title: '標準重量未設定',
             detail: '數量相符，但無法核對重量' };
  }
  if (state === 'waiting_weight') {
    return { cls: 'hero-warn', icon: '…', title: '等待重量',
             detail: '數量相符，請將器械置於磅秤上' };
  }
  if (state === 'stabilizing') {
    return { cls: 'hero-warn', icon: '◔', title: '重量穩定中',
             detail: '數量相符，重量量測中' };
  }
  if (state === 'failed') {
    const d = (wv && wv.difference != null)
      ? `差異 ${Number(wv.difference) > 0 ? '+' : ''}${Number(wv.difference).toFixed(1)} g`
      : '重量超出容許範圍';
    return { cls: 'hero-fail', icon: '✕', title: '重量不符合',
             detail: `數量相符，但${d}` };
  }
  // PASS is the one verdict that must be earned, not defaulted into: it needs a
  // weight check that actually completed AND passed.  Any other state above has
  // already returned, so an unrecognised state falls through to 等待重量 here
  // rather than being rendered as a green tick.
  if (state === 'passed' && wv && wv.ready === true && wv.passed === true) {
    const n = summary ? summary.detectedUnits : 0;
    return { cls: 'hero-pass', icon: '✓', title: '盤點符合',
             detail: `${n} / ${n} 支 · 重量符合` };
  }
  return { cls: 'hero-warn', icon: '…', title: '等待重量',
           detail: '數量相符，等待重量核對' };
}

function renderOverallStatus() {
  const hero = document.getElementById('hero');
  if (!hero) return;
  const st = overallStatus(_lastSummary, _lastKnownWV, {
    needsPreset: availablePresets.length > 0 && activePreset === null,
    hasResult: hasCurrentResult,
  });
  hero.className = st.cls;
  const icon   = document.getElementById('hero-icon');
  const title  = document.getElementById('hero-title');
  const detail = document.getElementById('hero-detail');
  if (icon)   icon.textContent   = st.icon;
  if (title)  title.textContent  = st.title;
  if (detail) detail.textContent = st.detail;
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

    // The package can change from another tab or a server restart.  Everything
    // on screen belongs to the old package at that point, so resync before
    // rendering anything else.
    if (data.active_package && data.active_package !== activePackageId) {
      if (activePackageId === null) {
        activePackageId = data.active_package;
        renderPackageSelect();
      } else {
        console.debug('[poll] package switched', activePackageId, '→', data.active_package);
        await applyPackageSwitch(data.active_package);
        return;
      }
    }

    // The backend clears counts that belong to a superseded package; drop
    // whatever is still painted so the table cannot outlive its package.
    if (data.result_stale) {
      console.debug('[poll] result is stale (package '
        + data.result_package_id + ' gen ' + data.result_model_generation + ')');
      if (hasCurrentResult || Object.keys(lastCounts).length) {
        lastCounts = {};
        hasCurrentResult = false;
        rerenderTable({});
        setUpdateTime('');
        resetWeightDisplay();
      }
    }

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
    // Keyed on timestamp, not on counts being non-empty: a recognition that
    // detected nothing at all is a real result — and the one where every
    // expected instrument must show as 缺少.
    if (shouldRenderCounts && data.timestamp) {
      hasCurrentResult = data.has_result === true;
      rerenderTable(data.counts || {});
      setUpdateTime(data.timestamp);
    }

    // 標準重量: update whenever new recognition results arrive via /status.
    // Only take actual weight and tolerance from the backend response.
    // Expected is always recomputed from current frontend standards × classWeights so
    // that manual standard edits are never overwritten by the 2-second poll cycle.
    if (shouldRenderCounts && data.weight_verification) {
      mergeBackendWV(data.weight_verification);
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
  _lastKnownWV = null;
  renderWeightVerification({ state: 'waiting_weight', actual: null, expected: null });
  _lastKnownWV = null;
  renderOverallStatus();
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
      // Update actual weight + stability from the scale; recompute expected locally.
      mergeBackendWV(data.weight_verification);
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
        hasCurrentResult = true;
        rerenderTable(data.counts);
        setUpdateTime(data.timestamp);
        showAnnotatedImage(data.annotated_image);
        if (data.weight_verification) {
          // Seed from the backend's scale reading; recompute expected from the
          // current standards × classWeights.
          mergeBackendWV(data.weight_verification);
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
// History export: always available — it reads immutable records.
function downloadReport() { window.location.href = '/report'; }

// BOM export: /bom_report returns 409 when there is no valid result for the
// active package, so a plain navigation would dump a JSON error page onto a
// kiosk with no back button.  Fetch first, then hand over a real file.
async function downloadBOM() {
  try {
    const res = await fetch('/bom_report');
    if (res.status === 409) {
      const data = await res.json().catch(() => ({}));
      alert('無法產生 BOM 報表：\n' + (data.error || '目前沒有有效的辨識結果'));
      return;
    }
    if (!res.ok) {
      alert('產生 BOM 報表失敗：' + await readError(res));
      return;
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'bom_' + new Date().toISOString().replace(/[:.]/g, '-') + '.csv';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {
    alert('產生 BOM 報表失敗：' + e.message);
  }
}

async function requestShutdown() {
  if (!confirm('確定要關機？\n系統將停止所有程式並安全關閉。')) return;
  const btn = document.getElementById('btn-shutdown');
  if (btn) { btn.disabled = true; btn.textContent = '關機中…'; }
  try {
    const res = await fetch('/system/shutdown', { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      alert('關機指令已送出，系統即將關閉。');
    } else {
      alert('關機失敗：' + (data.error || '未知錯誤'));
      if (btn) { btn.disabled = false; btn.textContent = '⏻ 安全關機'; }
    }
  } catch (err) {
    alert('關機請求失敗：' + err);
    if (btn) { btn.disabled = false; btn.textContent = '⏻ 安全關機'; }
  }
}
