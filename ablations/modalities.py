import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def optional_affinities(adata, deconv_path=None, ligrec_path=None, *, include_deconv=True, include_lr=None):
    if include_lr is None:
        include_lr = bool(ligrec_path)
    matrices = []
    if include_lr and (deconv_path is None or ligrec_path is None):
        raise ValueError("LR edges require both a deconvolution CSV and an LR CSV")
    if not include_deconv and not include_lr:
        return matrices
    if deconv_path is None:
        return matrices
    table = pd.read_csv(deconv_path, index_col=0)
    table.index = table.index.astype(str)
    if table.index.duplicated().any():
        raise ValueError("Deconvolution IDs must be unique")
    missing = adata.obs_names.difference(table.index)
    if len(missing):
        raise ValueError(f"Deconvolution lacks {len(missing)} retained spot IDs; use exact obs_names")
    W = table.loc[adata.obs_names].apply(pd.to_numeric).to_numpy(dtype=float)
    if not np.isfinite(W).all() or (W < 0).any():
        raise ValueError("Deconvolution values must be finite and nonnegative")
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    if include_deconv:
        D = cosine_similarity(W)
        np.fill_diagonal(D, 0)
        matrices.append(D)
    if include_lr:
        rules = pd.read_csv(ligrec_path)
        if not {"sender", "receiver", "score"}.issubset(rules.columns):
            raise ValueError("LR CSV requires sender, receiver, score")
        types = {str(name): i for i, name in enumerate(table.columns)}
        R = np.zeros((W.shape[1], W.shape[1]))
        for row in rules.itertuples(index=False):
            if str(row.sender) not in types or str(row.receiver) not in types:
                raise ValueError("LR sender/receiver must match deconvolution column names")
            score = float(row.score)
            if not np.isfinite(score) or score < 0:
                raise ValueError("LR scores must be finite and nonnegative")
            R[types[str(row.sender)], types[str(row.receiver)]] += score
        L = W @ R @ W.T
        L = np.maximum(L, L.T)
        np.fill_diagonal(L, 0)
        matrices.append(L)
    return matrices
