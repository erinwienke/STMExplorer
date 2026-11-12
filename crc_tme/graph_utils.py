import numpy as np

eps = 1e-12

def normalize_minmax(mat, clip_to_finite=True):
    m = np.array(mat, dtype=float, copy=True)
    if clip_to_finite:
        finite = np.isfinite(m)
        if not finite.all():
            finite_vals = m[finite]
            if finite_vals.size == 0:
                return np.zeros_like(m)
            minv = float(np.nanmin(finite_vals))
            m[~finite] = minv
    mn = m.min()
    mx = m.max()
    if mx - mn < 1e-12:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn + eps)

def keep_top_k_per_row(mat, k):
    if k <= 0:
        return np.zeros_like(mat)
    n = mat.shape[0]
    if k >= n - 1:
        out = mat.copy()
        np.fill_diagonal(out, 0.0)
        return out
    out = np.zeros_like(mat)
    for i in range(n):
        row = mat[i]
        inds = np.argsort(row)[::-1]
        kept = []
        for idx in inds:
            if idx == i:
                continue
            kept.append(idx)
            if len(kept) >= k:
                break
        if kept:
            out[i, kept] = row[kept]
    np.fill_diagonal(out, 0.0)
    return out

def combine_affinities(matrices, k, *, weights=None, neighborhoods=None):
    if not matrices or k < 1:
        raise ValueError("At least one active modality and a positive k are required")
    weights = weights if weights is not None else {name: 1.0 for name in matrices}
    neighborhoods = neighborhoods if neighborhoods is not None else {}
    shape = np.asarray(next(iter(matrices.values()))).shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 2:
        raise ValueError("Affinity matrices must be square with at least two spots")
    score, union, total = np.zeros(shape), np.zeros(shape, dtype=bool), 0.0
    for name, matrix in matrices.items():
        matrix = np.asarray(matrix, dtype=float)
        weight = float(weights.get(name, 1.0))
        if matrix.shape != shape or not np.isfinite(weight) or weight < 0:
            raise ValueError("Affinity shapes must match and weights must be finite and nonnegative")
        if weight == 0:
            continue
        selected = neighborhoods.get(name)
        if selected is None:
            selected = keep_top_k_per_row(matrix, min(k, shape[0] - 1))
        union |= (selected > 0) | (selected.T > 0)
        score += weight * normalize_minmax(matrix)
        total += weight
    if total <= 0:
        raise ValueError("At least one modality must have positive weight")
    score = score / total * union
    score = np.maximum(score, score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)
    return score
