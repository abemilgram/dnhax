"""Causal summaries for branching, graph-constrained trajectory rollouts.

The caller owns trajectory construction and supplies branches that already obey
the authored navigation graph.  This module only summarizes those branches: it
never averages paths, creates a new horizon, or turns model support into a
calibrated probability claim.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_EPS = 1e-7
_BIN_SIZE_M = 0.5
_MAX_BINS = 64
_SNAPSHOT_OFFSETS_S = (0.0, 3.0, 6.0, 10.0)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _intent_records(geometry: Any) -> list[tuple[str, str, str]]:
    prior = getattr(geometry, "prior", None)
    intents = getattr(prior, "intents", None)
    if not isinstance(intents, (list, tuple)) or not intents:
        raise ValueError("geometry.prior.intents is required")
    records: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for intent in intents:
        intent_id = _value(intent, "id")
        label = _value(intent, "label")
        kind = _value(intent, "kind")
        if not isinstance(intent_id, str) or not intent_id:
            raise ValueError("intent IDs must be non-empty strings")
        if intent_id in seen:
            raise ValueError(f"duplicate intent ID {intent_id!r}")
        if not isinstance(label, str) or not label:
            raise ValueError(f"intent {intent_id!r} needs a label")
        if kind not in {"hold", "route"}:
            raise ValueError(f"intent {intent_id!r} has an unsupported kind")
        seen.add(intent_id)
        records.append((intent_id, label, kind))
    return records


def _check_keys(mapping: dict[str, Any], known: set[str], name: str) -> None:
    unknown = set(mapping) - known
    if unknown:
        raise ValueError(f"{name} contains unknown intent IDs: {sorted(unknown)!r}")


def _validated_support(
    support: dict[str, float], known: set[str]
) -> dict[str, float]:
    if not isinstance(support, dict):
        raise ValueError("support must be a mapping")
    _check_keys(support, known, "support")
    if set(support) != known:
        raise ValueError("support must provide one normalized value per intent")
    result = {intent_id: _finite(support[intent_id], f"support[{intent_id!r}]") for intent_id in known}
    if any(value < 0.0 for value in result.values()):
        raise ValueError("support values must be non-negative")
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("support must sum to one; it is not renormalized here")
    return result


def _validated_invalid(invalid: dict[str, int], known: set[str]) -> dict[str, int]:
    if not isinstance(invalid, dict):
        raise ValueError("invalid must be a mapping")
    _check_keys(invalid, known, "invalid")
    result: dict[str, int] = {}
    for intent_id in known:
        value = invalid.get(intent_id, 0)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise ValueError(f"invalid[{intent_id!r}] must be a non-negative integer")
        result[intent_id] = int(value)
    return result


def _validated_branch(branch: Any, intent_id: str, horizon: float) -> dict[str, Any]:
    times_raw = _value(branch, "times")
    xyz_raw = _value(branch, "xyz")
    modes_raw = _value(branch, "modes")
    times = np.asarray(times_raw, dtype=float)
    xyz = np.asarray(xyz_raw, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError(f"trajectory {intent_id!r} has invalid times")
    if abs(float(times[0])) > _EPS or abs(float(times[-1]) - horizon) > _EPS:
        raise ValueError(f"trajectory {intent_id!r} times must include 0 and the configured horizon")
    if np.any(np.diff(times) <= _EPS):
        raise ValueError(f"trajectory {intent_id!r} times must increase strictly")
    if xyz.shape != (len(times), 3) or not np.isfinite(xyz).all():
        raise ValueError(f"trajectory {intent_id!r} has invalid xyz samples")
    if not isinstance(modes_raw, (list, tuple)) or len(modes_raw) != len(times) or not all(
        isinstance(mode, str) and mode for mode in modes_raw
    ):
        raise ValueError(f"trajectory {intent_id!r} has invalid mode samples")
    speed = _finite(_value(branch, "speed"), f"trajectory {intent_id!r}.speed")
    if speed < 0.0:
        raise ValueError(f"trajectory {intent_id!r}.speed must be non-negative")
    first_arrival_raw = _value(branch, "first_arrival_s")
    first_arrival = None if first_arrival_raw is None else _finite(
        first_arrival_raw, f"trajectory {intent_id!r}.first_arrival_s"
    )
    if first_arrival is not None and first_arrival < 0.0:
        raise ValueError(f"trajectory {intent_id!r}.first_arrival_s must be non-negative")
    reached_goal_node = _value(branch, "reached_goal_node")
    if reached_goal_node is not None and (not isinstance(reached_goal_node, str) or not reached_goal_node):
        raise ValueError(f"trajectory {intent_id!r}.reached_goal_node must be a non-empty string or None")
    reached_goal_s_raw = _value(branch, "reached_goal_s")
    reached_goal_s = None if reached_goal_s_raw is None else _finite(
        reached_goal_s_raw, f"trajectory {intent_id!r}.reached_goal_s"
    )
    if (reached_goal_node is None) != (reached_goal_s is None):
        raise ValueError(f"trajectory {intent_id!r} must provide both reached goal fields or neither")
    if reached_goal_s is not None and not 0.0 <= reached_goal_s <= horizon + _EPS:
        raise ValueError(f"trajectory {intent_id!r}.reached_goal_s must lie within the forecast horizon")
    reverse_count = _value(branch, "reverse_count")
    if (
        isinstance(reverse_count, (bool, np.bool_))
        or not isinstance(reverse_count, (int, np.integer))
        or int(reverse_count) < 0
    ):
        raise ValueError(f"trajectory {intent_id!r}.reverse_count must be a non-negative integer")
    positions = _value(branch, "positions")
    if not callable(positions):
        raise ValueError(f"trajectory {intent_id!r}.positions must be callable")
    return {
        "object": branch,
        "times": times,
        "xyz": xyz,
        "modes": list(modes_raw),
        "speed": speed,
        "first_arrival_s": first_arrival,
        "reached_goal_node": reached_goal_node,
        "reached_goal_s": reached_goal_s,
        "reverse_count": int(reverse_count),
    }


def _validated_trajectories(
    trajectories: dict[str, list[Any]],
    weights: dict[str, np.ndarray],
    known: set[str],
    horizon: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    if not isinstance(trajectories, dict) or not isinstance(weights, dict):
        raise ValueError("trajectories and weights must be mappings")
    _check_keys(trajectories, known, "trajectories")
    _check_keys(weights, known, "weights")
    normalized_trajectories: dict[str, list[dict[str, Any]]] = {}
    normalized_weights: dict[str, np.ndarray] = {}
    for intent_id in known:
        raw_branches = trajectories.get(intent_id, [])
        if not isinstance(raw_branches, (list, tuple)):
            raise ValueError(f"trajectories[{intent_id!r}] must be a list")
        branches = [_validated_branch(branch, intent_id, horizon) for branch in raw_branches]
        raw_weights = weights.get(intent_id, np.empty(0, dtype=float))
        branch_weights = np.asarray(raw_weights, dtype=float)
        if branch_weights.ndim != 1 or len(branch_weights) != len(branches) or not np.isfinite(branch_weights).all():
            raise ValueError(f"weights[{intent_id!r}] must match the finite branch count")
        if np.any(branch_weights < 0.0):
            raise ValueError(f"weights[{intent_id!r}] must be non-negative")
        if branches:
            if not math.isclose(float(branch_weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"weights[{intent_id!r}] must sum to one within the intent")
        elif len(branch_weights):
            raise ValueError(f"weights[{intent_id!r}] cannot contain weights without trajectories")
        normalized_trajectories[intent_id] = branches
        normalized_weights[intent_id] = branch_weights
    return normalized_trajectories, normalized_weights


def _position(branch: dict[str, Any], elapsed: float, intent_id: str) -> np.ndarray:
    try:
        value = np.asarray(_value(branch["object"], "positions")(float(elapsed)), dtype=float)
    except Exception as exc:
        raise ValueError(f"trajectory {intent_id!r}.positions failed") from exc
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"trajectory {intent_id!r}.positions must return a finite 3-vector")
    return value


def _positions(branch: dict[str, Any], elapsed: np.ndarray, intent_id: str) -> np.ndarray:
    """Evaluate one branch at an array of elapsed times in one call."""

    try:
        value = np.asarray(_value(branch["object"], "positions")(elapsed), dtype=float)
    except Exception as exc:
        raise ValueError(f"trajectory {intent_id!r}.positions failed") from exc
    if value.shape != (len(elapsed), 3) or not np.isfinite(value).all():
        raise ValueError(f"trajectory {intent_id!r}.positions must return finite [N,3] samples")
    return value


def _offsets(remaining: float, step: float) -> list[float]:
    if remaining <= _EPS:
        return []
    values = [float(value) for value in np.arange(0.0, remaining + _EPS, step)]
    if not values:
        values = [0.0]
    if values[-1] < remaining - _EPS:
        values.append(float(remaining))
    stride = max(1, int(math.ceil(len(values) / 20)))
    sampled = values[::stride]
    if sampled[-1] < values[-1] - _EPS:
        sampled.append(values[-1])
    return sampled


def _event_offsets(branch: dict[str, Any], elapsed: float, horizon: float) -> list[float] | None:
    """Return branch event breakpoints relative to now, including both endpoints."""

    event_times = np.asarray(branch["times"], dtype=float)
    current_events = event_times[(event_times >= elapsed - _EPS) & (event_times <= horizon + _EPS)]
    values = [float(elapsed), *[float(value) for value in current_events], float(horizon)]
    values = sorted({round(value, 9) for value in values})
    if len(values) > 128:
        return None
    return [float(value - elapsed) for value in values]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float | None:
    if len(values) == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(sorted_weights.sum())
    if total <= _EPS:
        return None
    threshold = min(total, max(0.0, quantile) * total)
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def _uncalibrated_meaning() -> str:
    return (
        "Conditional model mass from valid graph-constrained branches and the supplied speed/branch weights; "
        "arrival timing and support are uncalibrated and are not event probabilities."
    )


def _arrival_summary(
    branches: list[dict[str, Any]],
    branch_weights: np.ndarray,
    elapsed: float,
    horizon: float,
) -> dict[str, Any]:
    if not branches:
        return {
            "status": "unavailable",
            "p10": None,
            "p50": None,
            "p90": None,
            "future_conditional": {"p10": None, "p50": None, "p90": None},
            "already_arrived_mass": None,
            "future_arrival_within_horizon_mass": None,
            "no_arrival_within_horizon_mass": None,
            "uncalibrated": True,
            "meaning": "No valid trajectory branches were available for this route intent.",
        }
    arrivals = np.asarray(
        [np.inf if branch["first_arrival_s"] is None else branch["first_arrival_s"] for branch in branches],
        dtype=float,
    )
    arrived = np.isfinite(arrivals) & (arrivals <= elapsed + _EPS) & (arrivals <= horizon + _EPS)
    future = np.isfinite(arrivals) & (arrivals > elapsed + _EPS) & (arrivals <= horizon + _EPS)
    no_arrival = ~(arrived | future)
    already_mass = float(np.dot(branch_weights, arrived.astype(float)))
    future_mass = float(np.dot(branch_weights, future.astype(float)))
    no_mass = float(np.dot(branch_weights, no_arrival.astype(float)))
    future_values = arrivals[future] - elapsed
    future_weights = branch_weights[future]
    quantiles = {
        f"p{int(q * 100)}": _weighted_quantile(future_values, future_weights, q)
        for q in (0.1, 0.5, 0.9)
    }
    return {
        "status": "modeled",
        "p10": quantiles["p10"],
        "p50": quantiles["p50"],
        "p90": quantiles["p90"],
        "future_conditional": quantiles,
        "already_arrived_mass": already_mass,
        "future_arrival_within_horizon_mass": future_mass,
        "no_arrival_within_horizon_mass": no_mass,
        "uncalibrated": True,
        "meaning": _uncalibrated_meaning(),
    }


def _goal_labels(geometry: Any, intent_records: list[tuple[str, str, str]]) -> dict[str, str]:
    """Collect declared route goals and retain a stable fallback label."""

    labels: dict[str, str] = {}
    intents = getattr(getattr(geometry, "prior", None), "intents", ())
    for intent in intents:
        goal_node = _value(intent, "goal_node")
        if goal_node is None:
            continue
        if not isinstance(goal_node, str) or not goal_node:
            raise ValueError("declared route goal nodes must be non-empty strings")
        label = _value(intent, "label")
        if goal_node not in labels:
            labels[goal_node] = str(label) if isinstance(label, str) and label else goal_node
    return labels


def _exit_forecasts(
    geometry: Any,
    support: dict[str, float],
    trajectories: dict[str, list[dict[str, Any]]],
    weights: dict[str, np.ndarray],
    intent_records: list[tuple[str, str, str]],
    elapsed: float,
    horizon: float,
) -> list[dict[str, Any]]:
    """Aggregate absorbing reached-goal events across intent behavior mixtures."""

    labels = _goal_labels(geometry, intent_records)
    for branches in trajectories.values():
        for branch in branches:
            goal_node = branch["reached_goal_node"]
            if goal_node is not None:
                labels.setdefault(goal_node, goal_node)
    unsupported = float(sum(support[intent_id] for intent_id, _label, _kind in intent_records if not trajectories[intent_id]))
    valid_mass = float(1.0 - unsupported)
    forecasts: list[dict[str, Any]] = []
    for goal_node in sorted(labels):
        future_values: list[float] = []
        future_weights: list[float] = []
        already_mass = 0.0
        future_mass = 0.0
        for intent_id, _label, _kind in intent_records:
            for branch, branch_weight in zip(trajectories[intent_id], weights[intent_id]):
                arrival = branch["reached_goal_s"] if branch["reached_goal_node"] == goal_node else None
                mass = float(support[intent_id] * branch_weight)
                if arrival is None:
                    continue
                if arrival <= elapsed + _EPS and arrival <= horizon + _EPS:
                    already_mass += mass
                elif arrival <= horizon + _EPS:
                    future_mass += mass
                    future_values.append(float(arrival - elapsed))
                    future_weights.append(mass)
        no_mass = max(0.0, valid_mass - already_mass - future_mass)
        if valid_mass <= _EPS:
            status = "unavailable"
            no_mass = 0.0
        else:
            status = "modeled"
        values = np.asarray(future_values, dtype=float)
        value_weights = np.asarray(future_weights, dtype=float)
        quantiles = {
            f"p{int(q * 100)}": _weighted_quantile(values, value_weights, q)
            for q in (0.1, 0.5, 0.9)
        }
        total = already_mass + future_mass + no_mass + unsupported
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"exit forecast mass does not account for one model mass unit: {total}")
        forecasts.append({
            "goal_node": goal_node,
            "label": labels[goal_node],
            "status": status,
            "already_arrived_mass": float(already_mass),
            "future_arrival_mass": float(future_mass),
            "conditional_future_quantiles": quantiles,
            "no_arrival_here_by_horizon_mass": float(no_mass),
            "unsupported_mass": float(unsupported),
            "uncalibrated": True,
            "meaning": (
                "Mass is aggregated across behavior branches and intent support; reaching this exit is a model "
                "forecast, not a calibrated event probability."
            ),
        })
    return forecasts


def _exposure(
    geometry: Any,
    player: Any,
    branches: list[dict[str, Any]],
    branch_weights: np.ndarray,
    intent_id: str,
    elapsed: float,
    offsets: list[float],
) -> float:
    if player is None or not offsets:
        raise RuntimeError("exposure is not requested")
    pose = np.asarray(_value(player, "camera_to_world"), dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("player camera pose must be a finite 4x4 matrix")
    origin = pose[:3, 3]
    branch_targets = []
    for branch in branches:
        points = _positions(branch, elapsed + np.asarray(offsets, dtype=float), intent_id)
        branch_targets.append(points + np.asarray([0.0, 0.0, 0.9]))
    targets = np.concatenate(branch_targets, axis=0)
    visible = np.asarray(geometry.line_of_sight_many(origin, targets), dtype=bool)
    if visible.shape != (len(branches) * len(offsets),):
        raise ValueError("geometry.line_of_sight_many returned an invalid shape")
    values = visible.reshape(len(branches), len(offsets)).mean(axis=1)
    return float(np.dot(branch_weights, values))


def _occupancy_snapshot(
    geometry: Any,
    support: dict[str, float],
    trajectories: dict[str, list[dict[str, Any]]],
    weights: dict[str, np.ndarray],
    intent_records: list[tuple[str, str, str]],
    elapsed: float,
    now: float,
    dt: float,
) -> dict[str, Any]:
    bins: dict[tuple[int, int, int], dict[str, Any]] = {}
    unsupported_mass = 0.0
    for intent_id, _label, _kind in intent_records:
        branches = trajectories[intent_id]
        intent_mass = support[intent_id]
        if not branches:
            unsupported_mass += intent_mass
            continue
        for branch_index, (branch, branch_weight) in enumerate(zip(branches, weights[intent_id])):
            mass = float(intent_mass * branch_weight)
            if mass <= 0.0:
                continue
            xyz = _position(branch, elapsed + dt, intent_id)
            voxel = tuple(np.floor(xyz / _BIN_SIZE_M).astype(int).tolist())
            current = bins.get(voxel)
            if current is None:
                bins[voxel] = {
                    "voxel_index": [int(value) for value in voxel],
                    "xyz": [float(value) for value in xyz],
                    "probability_model_mass": mass,
                    "_representative_mass": mass,
                    "_representative_order": branch_index,
                }
            else:
                # Preserve the mass already accumulated in this voxel when a
                # larger branch takes over as its representative sample.
                current["probability_model_mass"] += mass
                if mass > current["_representative_mass"]:
                    current["xyz"] = [float(value) for value in xyz]
                    current["_representative_mass"] = mass
    ordered = sorted(
        bins.values(),
        key=lambda item: (-item["probability_model_mass"], item["voxel_index"]),
    )
    kept = ordered[:_MAX_BINS]
    omitted_mass = float(sum(item["probability_model_mass"] for item in ordered[_MAX_BINS:]))
    clean_bins = []
    for item in kept:
        clean_bins.append({
            "voxel_index": item["voxel_index"],
            "xyz": item["xyz"],
            "probability_model_mass": float(item["probability_model_mass"]),
        })
    bin_mass = float(sum(item["probability_model_mass"] for item in clean_bins))
    total = bin_mass + unsupported_mass + omitted_mass
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"occupancy mass does not account for one model mass unit: {total}")
    return {
        "dt": float(dt),
        "t": float(now + dt),
        "bins": clean_bins,
        "unsupported_intent_mass": float(unsupported_mass),
        "omitted_mass": omitted_mass,
        "mass_accounting": {
            "bin_mass": bin_mass,
            "unsupported_intent_mass": float(unsupported_mass),
            "omitted_mass": omitted_mass,
            "total": float(total),
        },
    }


def _occupancy(
    geometry: Any,
    config: Any,
    support: dict[str, float],
    trajectories: dict[str, list[dict[str, Any]]],
    weights: dict[str, np.ndarray],
    intent_records: list[tuple[str, str, str]],
    anchor_t: float,
    now: float,
    elapsed: float,
    remaining: float,
) -> dict[str, Any]:
    expiry_t = anchor_t + _finite(_value(config, "horizon_s"), "config.horizon_s")
    if remaining <= _EPS:
        return {
            "status": "expired",
            "unavailable_reason": "forecast_horizon_expired",
            "anchor_t": anchor_t,
            "now": now,
            "expires_at": expiry_t,
            "remaining_horizon_s": 0.0,
            "bin_size_m": _BIN_SIZE_M,
            "max_bins": _MAX_BINS,
            "snapshots": [],
            "unsupported_intent_mass": float(sum(support[i] for i in trajectories if not trajectories[i])),
            "uncalibrated": True,
            "provenance": "Actual samples from authored graph-constrained trajectory branches; no path averaging.",
        }
    offsets = [dt for dt in _SNAPSHOT_OFFSETS_S if dt <= remaining + _EPS]
    snapshots = [
        _occupancy_snapshot(geometry, support, trajectories, weights, intent_records, elapsed, now, dt)
        for dt in offsets
    ]
    unsupported = float(sum(support[i] for i in trajectories if not trajectories[i]))
    return {
        "status": "available" if any(trajectories[i] for i, _label, _kind in intent_records) else "unsupported",
        "anchor_t": anchor_t,
        "now": now,
        "expires_at": expiry_t,
        "remaining_horizon_s": remaining,
        "bin_size_m": _BIN_SIZE_M,
        "max_bins": _MAX_BINS,
        "snapshot_offsets_s": [float(dt) for dt in offsets],
        "snapshots": snapshots,
        "unsupported_intent_mass": unsupported,
        "uncalibrated": True,
        "provenance": "Actual samples from authored graph-constrained trajectory branches; no path averaging.",
    }


def summarize_branching(
    geometry: Any,
    config: Any,
    trajectories: dict[str, list[Any]],
    weights: dict[str, np.ndarray],
    support: dict[str, float],
    invalid: dict[str, int],
    anchor_t: float,
    now: float,
    player: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return compatible intent ranking plus bounded model-mass occupancy."""

    intent_records = _intent_records(geometry)
    known = {intent_id for intent_id, _label, _kind in intent_records}
    support_values = _validated_support(support, known)
    invalid_values = _validated_invalid(invalid, known)
    horizon = _finite(_value(config, "horizon_s"), "config.horizon_s")
    step = _finite(_value(config, "step_s"), "config.step_s")
    if horizon <= 0.0 or step <= 0.0:
        raise ValueError("config horizon and step must be positive")
    anchor = _finite(anchor_t, "anchor_t")
    current = _finite(now, "now")
    elapsed = max(0.0, current - anchor)
    remaining = max(0.0, horizon - elapsed)
    normalized_trajectories, normalized_weights = _validated_trajectories(
        trajectories, weights, known, horizon
    )

    ranking: list[dict[str, Any]] = []
    for intent_id, label, kind in intent_records:
        branches = normalized_trajectories[intent_id]
        branch_weights = normalized_weights[intent_id]
        available = bool(branches) and remaining > _EPS
        item: dict[str, Any] = {
            "intent_id": intent_id,
            "label": label,
            "kind": kind,
            "relative_support": float(support_values[intent_id]),
            "samples": len(branches),
            "rejected_samples": invalid_values[intent_id],
            "available": available,
            "remaining_horizon_s": float(remaining),
            "arrival_remaining_s": None,
            "exposure_fraction": None,
            "rank_score": None,
            "path_preview": [],
        }
        if kind == "route" and remaining > _EPS:
            item["arrival_remaining_s"] = _arrival_summary(branches, branch_weights, elapsed, horizon)
        if available:
            representative = int(np.argmax(branch_weights))
            event_offsets = _event_offsets(branches[representative], elapsed, horizon)
            if event_offsets is None:
                item["path_preview_status"] = "omitted_event_cap"
            else:
                preview = []
                for dt in event_offsets:
                    point = _position(branches[representative], elapsed + dt, intent_id)
                    preview.append({"dt": float(dt), "xyz": [float(value) for value in point]})
                item["path_preview"] = preview
                item["path_preview_status"] = "actual_event_breakpoints"
            if player is not None:
                offsets = _offsets(remaining, step)
                item["exposure_fraction"] = _exposure(
                    geometry,
                    player,
                    branches,
                    branch_weights,
                    intent_id,
                    elapsed,
                    offsets,
                )
                item["rank_score"] = float(
                    support_values[intent_id]
                    * (0.5 + 0.5 * item["exposure_fraction"])
                )
            else:
                item["rank_score"] = float(support_values[intent_id])
        ranking.append(item)
    ranking.sort(
        key=lambda item: (
            -(item["rank_score"] if item["rank_score"] is not None else -1.0),
            item["intent_id"],
        )
    )
    occupancy = _occupancy(
        geometry,
        config,
        support_values,
        normalized_trajectories,
        normalized_weights,
        intent_records,
        anchor,
        current,
        elapsed,
        remaining,
    )
    occupancy["model_provenance"] = {
        "multimodal": True,
        "branching": "Intent-conditioned graph trajectories with supplied within-intent weights",
        "support": "Relative model support normalized over intents; not calibrated probability",
        "arrival": _uncalibrated_meaning(),
        "occupancy": "Global model mass equals intent support times within-intent branch weight",
        "representatives": "Each retained bin reports an actual trajectory sample, never a voxel center or path mean",
        "expiry": "No new horizon is created on presentation ticks; forecast expires at anchor_t + horizon_s",
    }
    occupancy["exit_forecasts"] = _exit_forecasts(
        geometry,
        support_values,
        normalized_trajectories,
        normalized_weights,
        intent_records,
        elapsed,
        horizon,
    ) if remaining > _EPS else []
    return ranking, occupancy


__all__ = ["summarize_branching"]
