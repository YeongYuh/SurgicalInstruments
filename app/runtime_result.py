"""Is a stored inference result still valid for the package that is active now?

A result is only meaningful together with the package that produced it.  After a
switch, last inventory's counts describe a different instrument family entirely:
re-reading them against the new package's standards would report an obstetric
tray as an incomplete orthopaedic set, confidently and wrongly.

Every runtime reader (``/status``, ``/camera/result``, ``/bom_report``) passes
its stored result through here.  The package switch also clears application
state inside its own transaction; this is the second line of defence that makes
the combination impossible rather than merely unlikely.

History records are exempt: they are immutable snapshots that carry their own
standards, so they are read with the package they were taken under.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def result_matches_active(package_id: Optional[str],
                          model_generation: Optional[int],
                          active_state) -> bool:
    """True only when the result came from exactly the active activation.

    Both halves matter.  The id alone is not enough: reloading the same package
    produces a new generation, and the profile may have been reseeded, so a
    result from before the reload is not comparable either.
    """
    if active_state is None or not getattr(active_state, "configured", False):
        return False
    if not package_id or package_id != active_state.package_id:
        return False
    if model_generation is None:
        return False
    return int(model_generation) == int(active_state.generation)


#: No inference has been published for the active package yet.
STATUS_NO_RESULT = "no_result"
#: A published result that belongs to the active activation.
STATUS_CURRENT = "current"
#: A published result from a package/activation that has been replaced.
STATUS_STALE = "stale"


def is_empty_result(result: Mapping[str, Any]) -> bool:
    """Has anything been published at all?

    Keyed on the timestamp, which is written only when an inference actually
    commits — NOT on the counts.  "The model looked and found nothing" is a
    real, valid inventory result whose counts are legitimately empty, and
    confusing it with "nothing has run yet" is what makes a blank BOM read as a
    tray with every instrument missing.
    """
    return not result.get("timestamp")


def has_current_result(result: Mapping[str, Any], active_state) -> bool:
    """True only for a published result belonging to the active activation."""
    if is_empty_result(result):
        return False
    return result_matches_active(result.get("package_id"),
                                 result.get("model_generation"), active_state)


def sanitize_runtime_result(result: Mapping[str, Any], active_state) -> Dict[str, Any]:
    """Return the result with model-dependent fields cleared if it is stale.

    Adds three flags every consumer needs:

    ``has_result``      an inference for the active package has been published
    ``result_status``   no_result / current / stale
    ``result_stale``    the result exists but belongs to a superseded package

    Diagnostic fields (the result's own package identity) are kept so the UI and
    logs can explain *why* a panel went blank rather than silently showing
    nothing.
    """
    payload: Dict[str, Any] = dict(result)
    package_id = payload.get("package_id")
    generation = payload.get("model_generation")
    payload["result_package_id"] = package_id
    payload["result_model_generation"] = generation

    if is_empty_result(payload):
        payload["result_stale"] = False
        payload["has_result"] = False
        payload["result_status"] = STATUS_NO_RESULT
        payload["counts"] = {}
        payload["weight_verification"] = None
        return payload

    current = result_matches_active(package_id, generation, active_state)
    payload["result_stale"] = not current
    payload["has_result"] = current
    payload["result_status"] = STATUS_CURRENT if current else STATUS_STALE

    if not current:
        payload["counts"] = {}
        payload["weight"] = None
        payload["weight_verification"] = None
        payload["annotated_b64"] = None
        payload["timestamp"] = ""
        payload["package_id"] = None
        payload["package_display_name"] = None
    return payload
