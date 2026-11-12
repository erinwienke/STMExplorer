import numpy as np


def select_fraction(scores, fraction, *, largest=False):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError("Pair scores must be a square matrix for at least two spots")
    if not 0 <= fraction <= 1 or not np.isfinite(scores).all():
        raise ValueError("Fraction must be in [0, 1] and pair scores must be finite")
    mask = np.zeros_like(scores, dtype=np.float32)
    if fraction == 0:
        return mask
    n = len(scores)
    count = min(n - 1, int(np.ceil(fraction * (n - 1))))
    ranked = -scores.copy() if largest else scores.copy()
    np.fill_diagonal(ranked, np.inf)
    indices = np.argsort(ranked, axis=1, kind="stable")[:, :count]
    mask[np.arange(n)[:, None], indices] = 1.0
    return mask


def positive_blocks(mask, views):
    if views not in {"cross", "same", "both"}:
        raise ValueError("Unknown positive view selection")
    mask = np.asarray(mask, dtype=np.float32).copy()
    np.fill_diagonal(mask, 0)
    n = len(mask)
    result = np.zeros((2*n, 2*n), dtype=np.float32)
    if views in {"cross", "both"}:
        result[:n, n:] = mask
        result[n:, :n] = mask
    if views in {"same", "both"}:
        result[:n, :n] = mask
        result[n:, n:] = mask
    return result


def embedding_positive_blocks(z_a, z_b, fraction, views):
    z_a, z_b = np.asarray(z_a, dtype=float), np.asarray(z_b, dtype=float)
    if z_a.shape != z_b.shape or z_a.ndim != 2:
        raise ValueError("The two embedding arrays must have equal N-by-D shapes")
    z_a = z_a / np.maximum(np.linalg.norm(z_a, axis=1, keepdims=True), 1e-12)
    z_b = z_b / np.maximum(np.linalg.norm(z_b, axis=1, keepdims=True), 1e-12)
    n = len(z_a)
    result = np.zeros((2*n, 2*n), dtype=np.float32)
    if views not in {"cross", "same", "both"}:
        raise ValueError("Unknown positive view selection")
    if views in {"same", "both"}:
        result[:n, :n] = select_fraction(z_a @ z_a.T, fraction, largest=True)
        result[n:, n:] = select_fraction(z_b @ z_b.T, fraction, largest=True)
    if views in {"cross", "both"}:
        result[:n, n:] = select_fraction(z_a @ z_b.T, fraction, largest=True)
        result[n:, :n] = select_fraction(z_b @ z_a.T, fraction, largest=True)
    return result
