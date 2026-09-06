"""Heuristic continuity between repeated image windows; no cumulative fusion."""

import numpy as np
from .geometry import depth_edge_mask, register, transform


def world_pixels(prediction, index):
    if "world_points" in prediction:
        points = prediction["world_points"][index]
        c = prediction["confidence"][index].squeeze()
        valid = np.isfinite(points).all(-1) & np.isfinite(c) & (c > 1e-5)
        if "extrinsics" in prediction:
            ex = prediction["extrinsics"][index]
            camera_points = points @ ex[:3, :3].T + ex[:3, 3]
            camera_depth = camera_points[..., 2]
            valid &= (camera_depth > 0) & ~depth_edge_mask(camera_depth)
        if valid.any():
            valid &= c >= np.percentile(c[valid], 40)
        return points, valid

    d = prediction["depth"][index]
    h, w = d.shape
    y, x = np.mgrid[:h, :w]
    ex = prediction["extrinsics"][index]
    rays = (
        np.stack([x, y, np.ones_like(x)], -1)
        @ np.linalg.inv(prediction["intrinsics"][index]).T
    )
    points = (rays * d[..., None] - ex[:3, 3]) @ ex[:3, :3]
    c = prediction["confidence"][index].squeeze()
    valid = (
        np.isfinite(points).all(-1)
        & np.isfinite(c)
        & (c > 1e-5)
        & (d > 0)
        & ~depth_edge_mask(d)
    )
    if valid.any():
        valid &= c >= np.percentile(c[valid], 40)
    return points, valid


def continuity(current, ids, previous, old_ids):
    shared = sorted(set(ids) & set(old_ids))
    rejected = {
        "status": "new_segment",
        "reason": "Need two shared frames for continuity.",
        "shared_frames": len(shared),
    }
    if len(shared) < 2:
        return np.eye(4), rejected
    groups = []
    for identity in shared:
        a, va = world_pixels(current, ids.index(identity))
        b, vb = world_pixels(previous, old_ids.index(identity))
        if a.shape != b.shape:
            return np.eye(4), {**rejected, "reason": "Preprocessed dimensions changed."}
        grid = np.zeros(va.shape, dtype=bool)
        grid[:: max(1, va.shape[0] // 20), :: max(1, va.shape[1] // 20)] = True
        mask = va & vb & grid
        if mask.sum() >= 50:
            groups.append((a[mask], b[mask]))
    if len(groups) < 2:
        return np.eye(4), {
            **rejected,
            "reason": "Insufficient distributed overlap geometry.",
        }
    a, b = (
        np.concatenate([g[0] for g in groups]),
        np.concatenate([g[1] for g in groups]),
    )
    extent = float(
        np.linalg.norm(np.percentile(b, 95, axis=0) - np.percentile(b, 5, axis=0))
    )
    if extent < 1e-6:
        return np.eye(4), {**rejected, "reason": "Degenerate overlap geometry."}
    try:
        matrix, diagnostics = register(a, b, extent * 0.02)
        errors = [
            float(np.median(np.linalg.norm(transform(x, matrix) - y, axis=1)) / extent)
            for x, y in groups
        ]
        accepted = (
            diagnostics["status"] == "accepted"
            and 0.8 <= diagnostics["relative_scale"] <= 1.25
            and all(e < 0.02 for e in errors)
        )
        return (matrix if accepted else np.eye(4)), {
            **diagnostics,
            "status": "accepted" if accepted else "new_segment",
            "shared_frames": len(groups),
            "normalized_frame_errors": errors,
            "reason": "Overlap continuity passed; physical accuracy unverified."
            if accepted
            else "Continuity limits exceeded; camera reset for a new segment.",
        }
    except ValueError as exc:
        return np.eye(4), {**rejected, "reason": str(exc)}


def apply_similarity(prediction, matrix):
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    rotation = matrix[:3, :3] / scale
    if "world_points" in prediction:
        shape = prediction["world_points"].shape
        prediction["world_points"] = transform(
            prediction["world_points"].reshape(-1, 3), matrix
        ).reshape(shape)
    ex = prediction["extrinsics"].copy()
    ex[:, :3, :3] = ex[:, :3, :3] @ rotation.T
    ex[:, :3, 3] = (
        scale * prediction["extrinsics"][:, :3, 3] - ex[:, :3, :3] @ matrix[:3, 3]
    )
    prediction["extrinsics"] = ex
    if "depth" in prediction:
        prediction["depth"] = prediction["depth"] * scale
