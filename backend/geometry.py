import numpy as np


def fit_similarity(source, target):
    """Return positive-scale source -> target transform, without reflection."""
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("At least three corresponding 3D points are required.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Landmarks must be finite numbers.")
    ac, bc = a - a.mean(0), b - b.mean(0)
    if np.linalg.matrix_rank(ac) < 2 or np.linalg.matrix_rank(bc) < 2:
        raise ValueError("Landmarks must not lie on a single line.")
    u, singular, vt = np.linalg.svd(bc.T @ ac / len(a))
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = (u * signs) @ vt
    scale = float((singular * signs).sum() / (ac * ac).sum(axis=1).mean())
    if scale <= 0:
        raise ValueError("Invalid relative scale.")
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = b.mean(0) - scale * rotation @ a.mean(0)
    return matrix


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def register(source, target, threshold=0.08):
    """Robust fit with deterministic held-out validation. Errors use target units."""
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 8:
        raise ValueError(
            "Provide at least eight landmark pairs; two or more are held out for validation."
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite landmarks.")
    rng = np.random.default_rng(41)
    order = rng.permutation(len(a))
    ntest = max(2, len(a) // 5)
    test, train = order[:ntest], order[ntest:]
    best = np.zeros(len(train), dtype=bool)
    for _ in range(256):
        ids = rng.choice(train, 3, replace=False)
        try:
            matrix = fit_similarity(a[ids], b[ids])
        except ValueError:
            continue
        mask = (
            np.linalg.norm(transform(a[train], matrix) - b[train], axis=1) < threshold
        )
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 3:
        raise ValueError("No stable alignment. Use more widely distributed landmarks.")
    matrix = fit_similarity(a[train[best]], b[train[best]])
    residuals = np.linalg.norm(transform(a, matrix) - b, axis=1)
    inliers = residuals[train] < threshold
    heldout = float(np.median(residuals[test]))
    ratio = float(inliers.mean())
    accepted = ratio >= 0.6 and heldout < threshold
    return matrix, {
        "status": "accepted" if accepted else "provisional",
        "method": "RANSAC similarity + held-out validation",
        "matches": len(a),
        "inliers": int(inliers.sum()),
        "fit_count": len(train),
        "inlier_ratio": ratio,
        "median_error": float(np.median(residuals[train][inliers]))
        if inliers.any()
        else None,
        "p90_error": float(np.percentile(residuals[train][inliers], 90))
        if inliers.any()
        else None,
        "heldout_error": heldout,
        "heldout_count": len(test),
        "threshold": threshold,
        "relative_scale": float(np.cbrt(np.linalg.det(matrix[:3, :3]))),
        "units": "reconstruction units",
        "spatial_extent": np.ptp(b[train[best]], axis=0).tolist(),
    }


def save_cloud(folder, points, colors):
    folder.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype="<f4")
    colors = np.asarray(colors, dtype="uint8")
    if len(points) > 1000000:
        ids = np.linspace(0, len(points) - 1, 1000000).astype(int)
        points = points[ids]
        colors = colors[ids]
    points.tofile(folder / "points.bin")
    colors.tofile(folder / "colors.bin")
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    packed = np.empty(len(points), dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    packed["xyz"] = points
    packed["rgb"] = colors
    with (folder / "cloud.ply").open("wb") as out:
        out.write(header.encode())
        out.write(packed.tobytes())
    return len(points)


def depth_edge_mask(depth, relative_threshold=0.03):
    """Reject neighborhoods crossing depth discontinuities before rendering."""
    depth = np.asarray(depth, dtype=float)
    padded = np.pad(depth, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    low = windows.min(axis=(-2, -1))
    high = windows.max(axis=(-2, -1))
    return (
        ((high - low) / np.maximum(np.abs(depth), 1e-6) > relative_threshold)
        | ~np.isfinite(low)
        | ~np.isfinite(high)
    )
