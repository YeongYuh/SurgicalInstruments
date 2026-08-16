"""Profile edits must not cross packages, and must not lose updates."""

from __future__ import annotations

import json
import threading

import pytest

import app.web as web_pkg
from app.inference.manager import PackageMismatchError
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import make_manager, write_package


@pytest.fixture
def api(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "ortho",
                  class_weights={"widget": 10.0}, standards={"widget": 2},
                  adapter_options={"counts": {"widget": 2}})
    write_package(manager.packages_dir, "obgyn",
                  class_weights={"forceps": 7.5}, standards={"forceps": 4},
                  adapter_options={"counts": {"forceps": 4}})
    manager.activate("ortho")

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.reset_latest_state()
    web_pkg.app.config["TESTING"] = True
    with web_pkg.app.test_client() as client:
        yield client, manager


def _json(response):
    return json.loads(response.data.decode("utf-8"))


# ── 16 & 17. an edit for the old package must never land on the new one ─────

def test_standards_edit_for_a_superseded_package_is_refused(api):
    """The 300 ms debounce means this really happens on a touchscreen.

    Operator edits orthopaedic standards, immediately picks obstetrics, and the
    queued POST arrives after the switch.  Writing it would put orthopaedic
    quantities on an obstetric tray.
    """
    client, manager = api

    client.post("/api/model-package", json={"id": "obgyn"})

    response = client.post("/standards",
                           json={"package_id": "ortho", "values": {"widget": 99}})

    assert response.status_code == 409
    payload = _json(response)
    assert payload["ok"] is False
    assert payload["requested_package"] == "ortho"
    assert payload["active_package"] == "obgyn"
    # Nothing was written anywhere.
    assert _json(client.get("/standards")) == {"forceps": 4}
    manager.activate("ortho")
    assert _json(client.get("/standards")) == {"widget": 2}


def test_unit_weights_edit_for_a_superseded_package_is_refused(api):
    client, manager = api
    client.post("/api/model-package", json={"id": "obgyn"})

    response = client.post("/unit_weights",
                           json={"package_id": "ortho", "values": {"widget": 55.0}})

    assert response.status_code == 409
    assert "widget" not in _json(client.get("/unit_weights"))
    manager.activate("ortho")
    assert _json(client.get("/unit_weights")).get("widget") == 10.0


def test_edit_targeting_the_active_package_succeeds(api):
    client, _ = api
    response = client.post("/standards",
                           json={"package_id": "ortho", "values": {"widget": 7}})
    assert response.status_code == 200
    assert _json(response)["package_id"] == "ortho"
    assert _json(client.get("/standards")) == {"widget": 7}


# ── 18. the package identity can travel in a header too ─────────────────────

def test_package_identity_can_be_sent_as_a_header(api):
    client, _ = api
    client.post("/api/model-package", json={"id": "obgyn"})

    stale = client.post("/standards", json={"widget": 99},
                        headers={"X-Model-Package": "ortho"})
    assert stale.status_code == 409

    fresh = client.post("/standards", json={"forceps": 6},
                        headers={"X-Model-Package": "obgyn"})
    assert fresh.status_code == 200
    assert _json(client.get("/standards")) == {"forceps": 6}


def test_legacy_untagged_edit_still_applies_to_the_active_package(api):
    """Older clients send a bare mapping; they keep working."""
    client, _ = api
    assert client.post("/standards", json={"widget": 5}).status_code == 200
    assert _json(client.get("/standards")) == {"widget": 5}


def test_manager_rejects_cross_package_writes_directly(api):
    _client, manager = api
    with pytest.raises(PackageMismatchError):
        manager.update_active_profile("standards", {"widget": 1}, package_id="obgyn")
    assert manager.state().standards == {"widget": 2}


# ── 19 & 20. concurrent writes must not lose each other ─────────────────────

def test_concurrent_edit_and_class_registration_preserve_both(api):
    """An operator edit and an inference's class registration collide often.

    Both mutate the same JSON file; if the write happened outside the lock the
    older snapshot could rename last and silently drop the newer change.
    """
    _client, manager = api
    profile = manager.active_profile
    start = threading.Barrier(2, timeout=10)
    errors = []

    def edit():
        try:
            start.wait()
            for i in range(40):
                profile.update_standards({"widget": 100 + i})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def register():
        try:
            start.wait()
            for i in range(40):
                profile.register_classes(["discovered-%02d" % i])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=edit), threading.Thread(target=register)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not errors, errors
    memory = profile.standards
    assert memory["widget"] == 139                       # the operator's last edit
    assert len([k for k in memory if k.startswith("discovered-")]) == 40

    on_disk = json.loads(profile.standards_path.read_text(encoding="utf-8"))
    assert on_disk == memory, "disk and memory diverged under concurrent writes"


def test_concurrent_unit_weight_writes_are_linearised(api):
    _client, manager = api
    profile = manager.active_profile
    start = threading.Barrier(3, timeout=10)

    def writer(offset):
        start.wait()
        for i in range(30):
            profile.update_unit_weights({"w%d" % offset: float(i)})

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    start.wait()
    for t in threads:
        t.join(timeout=20)

    memory = profile.unit_weights
    on_disk = json.loads(profile.unit_weights_path.read_text(encoding="utf-8"))
    assert memory["w0"] == 29.0
    assert memory["w1"] == 29.0
    assert on_disk == memory


def test_no_temp_files_are_left_behind(api):
    _client, manager = api
    profile = manager.active_profile
    profile.update_standards({"widget": 3})
    profile.register_classes(["another"])

    leftovers = list(profile.standards_path.parent.glob("*.tmp"))
    assert leftovers == []


# ── SI-PLATFORM-003: 19-24. runtime input is as strict as the manifest ──────

@pytest.mark.parametrize("body,fragment", [
    ('{"widget": 1.7}', "whole number"),
    ('{"widget": true}', "boolean"),
    ('{"widget": -1}', "negative"),
    ('{"widget": NaN}', "finite"),
    ('{"widget": Infinity}', "finite"),
    ('{"widget": "two"}', "number"),
    ('{"widget": null}', "number"),
    ('{"widget": [1]}', "number"),
])
def test_standards_rejects_values_the_manifest_would_reject(api, body, fragment):
    """1.7 instruments is a mistake, not a 1.

    Silently rounding it would change what the tray is expected to hold, and the
    operator would never see the correction.
    """
    client, manager = api
    before = _json(client.get("/standards"))

    response = client.post("/standards", data=body,
                           content_type="application/json")

    assert response.status_code == 400
    assert fragment in _json(response)["error"]
    assert _json(client.get("/standards")) == before      # nothing was written


def test_standards_accepts_a_whole_float(api):
    client, _ = api
    response = client.post("/standards", data='{"widget": 3.0}',
                           content_type="application/json")
    assert response.status_code == 200
    assert _json(client.get("/standards")) == {"widget": 3}


@pytest.mark.parametrize("body,fragment", [
    ('{"widget": -0.5}', "negative"),
    ('{"widget": NaN}', "finite"),
    ('{"widget": Infinity}', "finite"),
    ('{"widget": -Infinity}', "finite"),
    ('{"widget": true}', "boolean"),
    ('{"widget": "heavy"}', "number"),
])
def test_unit_weights_reject_invalid_numbers(api, body, fragment):
    client, _ = api
    before = _json(client.get("/unit_weights"))

    response = client.post("/unit_weights", data=body,
                           content_type="application/json")

    assert response.status_code == 400
    assert fragment in _json(response)["error"]
    assert _json(client.get("/unit_weights")) == before


def test_unit_weights_accept_zero_and_fractions(api):
    client, _ = api
    assert client.post("/unit_weights", json={"widget": 0}).status_code == 200
    assert client.post("/unit_weights", json={"widget": 12.75}).status_code == 200
    assert _json(client.get("/unit_weights"))["widget"] == 12.75


def test_profile_loading_drops_a_corrupt_value_rather_than_rounding_it(tmp_path):
    """A stored 1.7 is corrupt data; turning it into 1 would invent a standard."""
    from app.inference.profile import _coerce_ints

    cleaned = _coerce_ints({"good": 2, "fractional": 1.7, "negative": -3,
                            "text": "x", "nan": float("nan")})

    assert cleaned == {"good": 2}


# ── 25 & 26. an expectation the model could never satisfy ───────────────────

def test_standard_for_an_unpriced_class_is_rejected(api):
    """Without a unit weight the expected total would silently be too low."""
    client, _ = api

    response = client.post("/standards", json={"widget": 1, "mystery": 2})

    assert response.status_code == 400
    payload = _json(response)
    assert "mystery" in payload["error"]
    assert "mystery" in payload["invalid"]
    assert _json(client.get("/standards")) == {"widget": 2}   # untouched


def test_zero_standard_for_an_unpriced_class_is_allowed(api):
    """Turning an unused class off must always be possible."""
    client, _ = api
    response = client.post("/standards", json={"widget": 2, "mystery": 0})
    assert response.status_code == 200
    assert _json(client.get("/standards"))["mystery"] == 0


def test_standard_for_a_class_the_model_cannot_detect_is_rejected(tmp_path, monkeypatch):
    """Configured now rather than discovered at the next restart."""
    manager = make_manager(tmp_path / "cls")
    write_package(manager.packages_dir, "listed",
                  adapter="class_listing",
                  class_weights={"scissors": 5.0, "ghost": 1.0},
                  standards={"scissors": 1},
                  adapter_options={"model_classes": ["scissors"]})
    manager.activate("listed")

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.app.config["TESTING"] = True

    with web_pkg.app.test_client() as client:
        response = client.post("/standards", json={"scissors": 1, "ghost": 2})
        payload = _json(response)

        assert response.status_code == 400
        assert "ghost" in payload["error"]
        assert "無法辨識" in payload["error"]
        assert _json(client.get("/standards")) == {"scissors": 1}

        # The same class at zero is fine.
        assert client.post("/standards", json={"ghost": 0}).status_code == 200
