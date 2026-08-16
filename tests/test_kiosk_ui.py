"""Frontend tests for the 1024x600 exception-first kiosk UI.

There is no Node runtime on the Jetson and the task forbids installing one, so
this module tests the frontend two ways:

* **Static source assertions** — structure that must exist (or must *not* exist,
  like the removed donut chart) in index.html / style.css / app.js.
* **Real execution in the kiosk's own Chromium**, headless, at 1024x600.  The
  browser that ships the UI is the browser that runs the logic tests, so
  ``calculateInventorySummary`` and the hero-verdict precedence are exercised as
  the real page runs them rather than as a Python re-implementation.

The Chromium tests skip cleanly if the binary is missing, so the suite still
runs on a dev box without a kiosk browser.
"""

import json
import os
import re
import shutil
import subprocess
import textwrap

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEB = os.path.join(_HERE, os.pardir, "app", "web")
_TEMPLATES = os.path.join(_WEB, "templates")
_STATIC = os.path.join(_WEB, "static")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def index_html():
    return _read(os.path.join(_TEMPLATES, "index.html"))


@pytest.fixture(scope="module")
def app_js():
    return _read(os.path.join(_STATIC, "app.js"))


@pytest.fixture(scope="module")
def style_css():
    return _read(os.path.join(_STATIC, "style.css"))


# ── 1-10, 20-23: static structure ────────────────────────────────────────────

def test_index_contains_overall_hero_status(index_html):
    """The whole-tray verdict must be a real element, not a repurposed stat box."""
    assert 'id="hero"' in index_html
    assert 'id="hero-icon"' in index_html
    assert 'id="hero-title"' in index_html
    assert 'id="hero-detail"' in index_html


def test_index_contains_surgery_selector(index_html):
    assert 'id="surgery-bar"' in index_html
    assert 'id="preset-select"' in index_html
    assert 'onPresetChange' in index_html


def test_donut_canvas_does_not_exist(index_html):
    assert "donut-chart" not in index_html
    assert "<canvas" not in index_html
    assert "chart-legend" not in index_html


def test_chartjs_cdn_is_gone(index_html, app_js):
    """No CDN at all: the kiosk must render its core UI with the network down."""
    assert "chart.js" not in index_html.lower()
    # No off-box asset of any kind — scripts, stylesheets or fonts.
    assert not re.search(r'(src|href)\s*=\s*["\'](?:https?:)?//', index_html)
    for symbol in ("donutChart", "initChart", "updateChart", "new Chart("):
        assert symbol not in app_js, symbol


def test_weight_is_a_single_card(index_html):
    assert 'id="weight-card"' in index_html
    assert 'id="weight-title"' in index_html
    assert 'id="weight-numbers"' in index_html
    assert 'id="weight-diff"' in index_html
    # The three same-weight boxes are gone.
    for old in ("stat-exp-wt", "stat-act-wt", "stat-wt-ok", "stat-boxes"):
        assert old not in index_html, old


def test_missing_and_extra_summary_elements_exist(index_html):
    for element_id in ("sum-missing", "sum-missing-units", "sum-missing-classes",
                       "sum-extra", "sum-extra-units", "sum-extra-classes",
                       "sum-normal-classes"):
        assert 'id="%s"' % element_id in index_html, element_id


def test_normal_fold_control_exists(app_js, style_css):
    assert "toggleNormalRows" in app_js
    assert "normalRowsCollapsed" in app_js
    assert "tr.row-normal.collapsed" in style_css


def test_normal_rows_default_to_collapsed(app_js):
    assert re.search(r"let normalRowsCollapsed\s*=\s*true", app_js)


def test_status_order_is_missing_then_extra_then_normal(app_js):
    """The sort key itself is asserted; the rendered order is covered in Chromium."""
    match = re.search(r"const order\s*=\s*\{([^}]*)\}", app_js)
    assert match, "inventory sort order not found"
    order = match.group(1)
    assert order.index("missing") < order.index("extra") < order.index("normal")


def test_no_result_and_zero_detection_are_distinct(app_js):
    """SI-PLATFORM-004 semantics: counts={} is not the same as 'not recognised'."""
    assert "hasCurrentResult" in app_js
    assert "data.has_result === true" in app_js
    # An empty counts map with standards>0 must still produce missing rows, so
    # the union must be built from standards as well as counts.
    assert "Object.keys(standardsMap).concat(Object.keys(counts))" in app_js


def test_long_class_names_stay_readable(index_html, app_js, style_css):
    assert "cls-name" in app_js
    assert 'title="${name}"' in app_js          # tooltip carries the full name
    assert "-webkit-line-clamp" in style_css    # up to two lines on exception rows


def test_manual_standards_editing_survives(index_html, app_js):
    assert "std-input" in app_js
    assert 'type="number"' in app_js
    assert "onStandardEdit(this)" in app_js
    assert "scheduleStandardsSync" in app_js
    assert "flushPendingEdits" in app_js
    # The edit must move the verdict immediately, not at the next inference.
    edit_fn = app_js[app_js.index("function onStandardEdit"):]
    edit_fn = edit_fn[:edit_fn.index("\nfunction ")]
    assert "renderSummaryCards" in edit_fn
    assert "renderOverallStatus" in edit_fn
    assert "refreshWeightVerification" in edit_fn


def test_package_switch_hooks_survive(index_html, app_js):
    assert "onPackageChange" in index_html
    assert 'id="package-select"' in index_html
    assert 'id="package-status"' in index_html
    for fn in ("applyPackageSwitch", "setPackageStatus", "loadPackages"):
        assert fn in app_js, fn
    # Switching must drop the previous package's result, not repaint it.
    switch_fn = app_js[app_js.index("async function applyPackageSwitch"):]
    switch_fn = switch_fn[:switch_fn.index("\nasync function ")]
    assert "hasCurrentResult = false" in switch_fn
    assert "resetWeightDisplay" in switch_fn


def test_preset_switch_hooks_survive(app_js):
    assert "async function onPresetChange" in app_js
    switch_fn = app_js[app_js.index("async function onPresetChange"):]
    switch_fn = switch_fn[:switch_fn.index("\nfunction setPackageStatus")]
    # SurgeryA counts must never be judged against SurgeryB standards.
    assert "flushPendingEdits" in switch_fn
    assert "hasCurrentResult = false" in switch_fn
    assert "rerenderTable({})" in switch_fn
    assert "resetWeightDisplay" in switch_fn


def test_recognize_is_the_dominant_action(index_html, style_css):
    assert 'id="btn-recognize"' in index_html
    assert 'class="btn-primary"' in index_html
    # 25 / 25 / 50-ish: the two setup buttons share a quarter each.
    assert "#btn-upload, #btn-camera { flex: 1 1 25%" in style_css
    recognize = style_css[style_css.index("#btn-recognize {"):]
    assert "flex: 2" in recognize[:200]


def test_tools_are_secondary(index_html):
    toolbar = index_html[index_html.index('id="toolbar"'):]
    toolbar = toolbar[:toolbar.index("</div>")]
    for fn in ("openHistory", "downloadReport", "requestShutdown"):
        assert fn in toolbar, fn
    assert "danger" in toolbar   # shutdown keeps an explicit danger style


def test_no_body_scroll_at_kiosk_size(style_css):
    body = re.search(r"\bbody\s*\{([^}]*)\}", style_css)
    assert body, "no body rule"
    assert "overflow: hidden" in body.group(1)
    # Only the instrument detail table may scroll.
    assert "#table-wrap" in style_css
    wrap = style_css[style_css.index("#table-wrap {"):]
    assert "overflow-y: auto" in wrap[:300]


# ── 11-20: behaviour, executed in the kiosk's own Chromium ───────────────────

CHROMIUM = shutil.which("chromium-browser") or shutil.which("chromium")

requires_chromium = pytest.mark.skipif(
    CHROMIUM is None, reason="no Chromium available to execute the kiosk JS")

# Scenarios are declared as data so the same harness answers every question in
# one browser launch — starting Chromium on a Jetson costs seconds.
SCENARIOS = json.dumps([
    {
        "name": "missing_units",
        "counts": {"A-Forceps": 0, "B-Clamp": 1},
        "standards": {"A-Forceps": 2, "B-Clamp": 2},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "passed", "ready": True, "passed": True},
    },
    {
        "name": "extra_units",
        "counts": {"A-Forceps": 3},
        "standards": {"A-Forceps": 1},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "passed", "ready": True, "passed": True},
    },
    {
        "name": "no_result",
        "counts": {}, "standards": {"A-Forceps": 2},
        "hasResult": False, "presets": True, "activePreset": "SurgeryA",
        "wv": None,
    },
    {
        "name": "zero_detection",
        "counts": {}, "standards": {"A-Forceps": 2, "B-Clamp": 6},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "waiting_weight"},
    },
    {
        "name": "preset_null",
        "counts": {}, "standards": {},
        "hasResult": True, "presets": True, "activePreset": None,
        "wv": {"state": "passed", "ready": True, "passed": True},
    },
    {
        "name": "weight_stabilizing",
        "counts": {"A-Forceps": 2}, "standards": {"A-Forceps": 2},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "stabilizing", "ready": False, "passed": None,
               "actual": 307.4, "expected": 309.0},
    },
    {
        "name": "weight_failed",
        "counts": {"A-Forceps": 2}, "standards": {"A-Forceps": 2},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "failed", "ready": True, "passed": False,
               "actual": 345.0, "expected": 309.0, "difference": 36.0},
    },
    {
        "name": "missing_but_weight_passed",
        "counts": {"A-Forceps": 0}, "standards": {"A-Forceps": 2},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "passed", "ready": True, "passed": True,
               "actual": 309.0, "expected": 309.0},
    },
    {
        "name": "all_pass",
        "counts": {"A-Forceps": 2, "B-Clamp": 6},
        "standards": {"A-Forceps": 2, "B-Clamp": 6},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "passed", "ready": True, "passed": True,
               "actual": 309.0, "expected": 309.0, "difference": 0.0},
    },
    {
        "name": "mixed",
        "counts": {"Mayo-Hegar-Needle-Holder-TC": 0, "Knife-Handle-No.3": 2,
                   "Adson-Smooth-Tissue-Forceps": 1},
        "standards": {"Mayo-Hegar-Needle-Holder-TC": 1, "Knife-Handle-No.3": 0,
                      "Adson-Smooth-Tissue-Forceps": 1},
        "hasResult": True, "presets": True, "activePreset": "SurgeryA",
        "wv": {"state": "passed", "ready": True, "passed": True},
    },
])

HARNESS_TAIL = textwrap.dedent("""
    <script>
    // fetch() is stubbed to a promise that never settles, so app.js's normal
    // start-up parks at its first await and cannot race the scenarios below.
    window.__scenarios = %s;
    window.addEventListener('load', function () {
      const out = [];
      window.__scenarios.forEach(function (sc) {
        standards        = sc.standards;
        lastCounts       = sc.counts;
        hasCurrentResult = sc.hasResult;
        availablePresets = sc.presets
          ? [{ id: 'SurgeryA', display_name: 'SurgeryA',
               expected_weight: 309, instrument_count: 8 }]
          : [];
        activePreset = sc.activePreset;
        _lastKnownWV = sc.wv;
        normalRowsCollapsed = true;

        const summary = calculateInventorySummary(sc.counts, sc.standards);
        _lastSummary = summary;
        renderInventoryRows(summary);
        renderSummaryCards(summary);
        renderOverallStatus();

        const tbody = document.getElementById('results-tbody');
        out.push({
          name: sc.name,
          missingUnits: summary.missingUnits,
          extraUnits: summary.extraUnits,
          missingClasses: summary.missingClasses,
          extraClasses: summary.extraClasses,
          normalClasses: summary.normalClasses,
          expectedUnits: summary.expectedUnits,
          detectedUnits: summary.detectedUnits,
          rowOrder: summary.rows.map(function (r) { return r.status; }),
          heroClass: document.getElementById('hero').className,
          heroTitle: document.getElementById('hero-title').textContent,
          heroDetail: document.getElementById('hero-detail').textContent,
          sumMissingUnits: document.getElementById('sum-missing-units').textContent,
          sumExtraUnits: document.getElementById('sum-extra-units').textContent,
          tbodyHtml: tbody.innerHTML,
        });
      });
      const sink = document.createElement('div');
      sink.id = 'TESTOUT';
      sink.textContent = JSON.stringify(out);
      document.body.appendChild(sink);
    });
    </script>
""")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, index_html):
    if CHROMIUM is None:
        pytest.skip("no Chromium")
    work = tmp_path_factory.mktemp("kiosk")
    shutil.copy(os.path.join(_STATIC, "app.js"), str(work / "app.js"))
    shutil.copy(os.path.join(_STATIC, "style.css"), str(work / "style.css"))

    page = (index_html
            .replace("/static/style.css", "style.css")
            .replace("/static/app.js", "app.js")
            .replace("<script src=\"app.js\"></script>",
                     "<script>window.fetch = function () "
                     "{ return new Promise(function () {}); };</script>\n"
                     "<script src=\"app.js\"></script>\n"
                     + HARNESS_TAIL % SCENARIOS))
    harness = work / "harness.html"
    harness.write_text(page, encoding="utf-8")

    proc = subprocess.run(
        [CHROMIUM, "--headless", "--no-sandbox", "--disable-gpu",
         "--disable-dev-shm-usage", "--window-size=1024,600",
         "--virtual-time-budget=8000", "--dump-dom",
         "file://" + str(harness)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=180)
    dom = proc.stdout.decode("utf-8", "replace")
    match = re.search(r'<div id="TESTOUT">(.*?)</div>', dom, re.S)
    assert match, "kiosk JS did not run to completion in Chromium"
    payload = json.loads(_unescape(match.group(1)))
    return {row["name"]: row for row in payload}


def _unescape(text):
    for entity, char in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'")):
        text = text.replace(entity, char)
    return text


@requires_chromium
def test_missing_units_counts_pieces_not_classes(rendered):
    """Two short of one class and one of another is '缺少 3 支', not '缺少 2 類'."""
    row = rendered["missing_units"]
    assert row["missingUnits"] == 3
    assert row["missingClasses"] == 2
    assert row["sumMissingUnits"] == "3"
    assert "3 支" in row["heroDetail"]


@requires_chromium
def test_extra_units_counts_pieces(rendered):
    row = rendered["extra_units"]
    assert row["extraUnits"] == 2
    assert row["extraClasses"] == 1
    assert row["sumExtraUnits"] == "2"


@requires_chromium
def test_no_result_shows_not_yet_recognised(rendered):
    row = rendered["no_result"]
    assert row["heroTitle"] == "尚未辨識"
    assert row["heroClass"] == "hero-idle"


@requires_chromium
def test_zero_detection_with_standards_is_missing_not_no_result(rendered):
    """counts={} with a real result means every expected instrument is 缺少."""
    row = rendered["zero_detection"]
    assert row["heroTitle"] == "器械缺少"
    assert row["heroClass"] == "hero-fail"
    assert row["missingUnits"] == 8
    assert row["rowOrder"] == ["missing", "missing"]


@requires_chromium
def test_null_preset_blocks_any_verdict(rendered):
    """All-zero standards plus zero detections must not read as PASS."""
    row = rendered["preset_null"]
    assert row["heroTitle"] == "請選擇手術類型"
    assert row["heroClass"] != "hero-pass"


@requires_chromium
def test_stabilizing_weight_is_not_a_failure(rendered):
    row = rendered["weight_stabilizing"]
    assert row["heroTitle"] == "重量穩定中"
    assert row["heroClass"] == "hero-warn"


@requires_chromium
def test_failed_weight_cannot_show_overall_pass(rendered):
    row = rendered["weight_failed"]
    assert row["heroClass"] == "hero-fail"
    assert row["heroTitle"] == "重量不符合"


@requires_chromium
def test_missing_instrument_cannot_show_overall_pass(rendered):
    """The weight can be perfect while an instrument is still on the floor."""
    row = rendered["missing_but_weight_passed"]
    assert row["heroClass"] == "hero-fail"
    assert row["heroTitle"] == "器械缺少"


@requires_chromium
def test_exact_inventory_and_passing_weight_is_pass(rendered):
    row = rendered["all_pass"]
    assert row["heroClass"] == "hero-pass"
    assert row["heroTitle"] == "盤點符合"
    assert row["missingUnits"] == 0 and row["extraUnits"] == 0


@requires_chromium
def test_rows_render_missing_then_extra_then_normal(rendered):
    row = rendered["mixed"]
    assert row["rowOrder"] == ["missing", "extra", "normal"]
    html = row["tbodyHtml"]
    assert html.index("row-missing") < html.index("row-extra") < html.index("row-normal")
    assert html.index("group-missing") < html.index("group-extra") < html.index("group-normal")


@requires_chromium
def test_normal_rows_render_collapsed_and_exceptions_do_not(rendered):
    html = rendered["mixed"]["tbodyHtml"]
    assert "row-normal collapsed" in html
    assert "row-missing collapsed" not in html
    assert "row-extra collapsed" not in html


@requires_chromium
def test_exception_rows_show_detected_over_standard(rendered):
    """The operator must never have to do the subtraction themselves."""
    html = rendered["mixed"]["tbodyHtml"]
    assert "detected-num" in html
    assert "缺少 1 支" in html
    assert "多出 2 支" in html


@requires_chromium
def test_long_class_names_keep_a_tooltip(rendered):
    html = rendered["mixed"]["tbodyHtml"]
    assert 'title="Mayo-Hegar-Needle-Holder-TC"' in html
    assert 'class="cls-name"' in html


@requires_chromium
def test_mixed_exceptions_report_both_sides(rendered):
    row = rendered["mixed"]
    assert row["heroTitle"] == "盤點異常"
    assert "缺少 1 支" in row["heroDetail"]
    assert "多出 2 支" in row["heroDetail"]
