import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from ..graph_utils import normalize_minmax, keep_top_k_per_row

EPS = 1e-12


def moran_affinity(X, coords, k, graph="spatial-expression"):
    if graph not in {"spatial", "spatial-expression"}:
        raise ValueError("Unknown Moran graph")
    if k < 1:
        raise ValueError("k must be positive")
    distances = cdist(coords, coords)
    upper = distances[np.triu_indices_from(distances, k=1)]
    sigma = np.median(upper) if len(upper) else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.mean(upper) if len(upper) else 1.0
    sigma = max(float(sigma), 1e-6)
    spatial = np.exp(-distances**2 / (2 * sigma**2))
    np.fill_diagonal(spatial, 0)
    spatial = keep_top_k_per_row(spatial, k)
    spatial = normalize_minmax(np.maximum(spatial, spatial.T))
    if graph == "spatial":
        return sparse.csr_matrix(spatial)
    expression = cosine_similarity(X)
    np.fill_diagonal(expression, 0)
    expression = keep_top_k_per_row(expression, k)
    expression = normalize_minmax(np.maximum(expression, expression.T))
    A = 0.5 * spatial + 0.5 * expression
    np.fill_diagonal(A, 0)
    return sparse.csr_matrix(A)


def feature_moran_values(X, affinity):
    X = np.asarray(X, dtype=float)
    A = sparse.csr_matrix(affinity, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1 or A.shape != (len(X), len(X)):
        raise ValueError("Feature and affinity dimensions must agree")
    if not np.isfinite(X).all() or not np.isfinite(A.data).all() or (A.data < 0).any():
        raise ValueError("Features and affinities must be finite; Moran weights must be nonnegative")
    W = sparse.diags(1 / (np.asarray(A.sum(axis=1)).ravel() + EPS)) @ A
    weight_sum = float(W.sum())
    values = np.full(X.shape[1], np.nan)
    if weight_sum <= EPS:
        return values
    centered = X - X.mean(axis=0)
    denom = np.square(centered).sum(axis=0)
    numer = np.einsum("ij,ij->j", centered, W @ centered)
    valid = denom > EPS
    values[valid] = len(X) / (weight_sum + EPS) * numer[valid] / (denom[valid] + EPS)
    return values


def spatial_binary_knn(coords, k):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if coords.ndim != 2 or n < 2 or k < 1 or not np.isfinite(coords).all():
        raise ValueError("Finite spatial coordinates for at least two spots and positive k are required")
    count = min(k, n - 1)
    neighbors = NearestNeighbors(n_neighbors=count + 1).fit(coords).kneighbors(coords, return_distance=False)
    rows, cols = [], []
    for i, candidates in enumerate(neighbors):
        selected = candidates[candidates != i][:count]
        rows.extend([i] * len(selected))
        cols.extend(selected.tolist())
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return graph.maximum(graph.T)


def spatial_eigenspectrum(coords, k, n_values=20):
    if n_values < 2:
        raise ValueError("At least two eigenvalues are needed to compute an eigengap")
    graph = spatial_binary_knn(coords, k)
    L = laplacian(graph, normed=True).tocsr()
    count = min(int(n_values), L.shape[0])
    if count == L.shape[0]:
        values = np.linalg.eigvalsh(L.toarray())
    else:
        v0 = np.random.default_rng(42).normal(size=L.shape[0])
        values = eigsh(L, k=count, which="SM", return_eigenvectors=False, v0=v0)
    values = np.sort(values)
    return values, np.diff(values)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--cluster-file")
    p.add_argument("--features-mode", choices=["clusters", "hvg", "pca", "all", "hvg2000", "pca50"], default="clusters")
    p.add_argument("--hvg", type=int, default=3000)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--moran-graph", choices=["spatial-expression", "spatial"], default="spatial-expression")
    p.add_argument("--n-eigenvalues", type=int, default=20)
    p.add_argument("--out-dir", type=Path, default=Path("results/neighborhood"))
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    if args.k < 1 or args.hvg < 2 or args.pca_components < 1 or args.n_eigenvalues < 2:
        p.error("k and PCA components must be positive; hvg and eigenvalue count must be at least 2")
    import scanpy as sc
    from ..features import prepare_model_features
    adata = sc.read_h5ad(args.h5ad)
    if not adata.uns.get("crc_tme_preprocessed", False):
        raise ValueError("Use the output from crc_tme.preprocess")
    if "spatial" not in adata.obsm:
        raise ValueError("Input must contain obsm['spatial']")
    if args.features_mode == "clusters" and not args.cluster_file:
        args.cluster_file = str(args.h5ad.with_suffix("")) + "_clusters_single_with_sizes.csv"
    X = prepare_model_features(adata, args)
    coords = adata.obsm["spatial"]
    values = feature_moran_values(X, moran_affinity(X, coords, args.k, args.moran_graph))
    eigenvalues, gaps = spatial_eigenspectrum(coords, args.k, args.n_eigenvalues)
    valid = np.isfinite(values)
    summary = dict(h5ad=str(args.h5ad), k=args.k, effective_k=min(args.k, len(X)-1),
                   moran_graph=args.moran_graph, avg_moran=float(values[valid].mean()) if valid.any() else np.nan,
                   valid_moran_features=int(valid.sum()), max_eigengap=float(gaps.max()),
                   eigengap_after_index=int(gaps.argmax()+1), n_eigenvalues=len(eigenvalues))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.out_dir / "neighborhood_metrics.csv", index=False)
    pd.DataFrame({"feature_index": np.arange(X.shape[1]), "moran_i": values}).to_csv(args.out_dir / "moran_per_feature.csv", index=False)
    pd.DataFrame({"eigenvalue_index": np.arange(1, len(eigenvalues)+1), "eigenvalue": eigenvalues,
                  "gap_to_next": np.r_[gaps, np.nan]}).to_csv(args.out_dir / "laplacian_eigenvalues.csv", index=False)
    with (args.out_dir / "configuration.json").open("w") as f:
        json.dump(vars(args), f, default=str, indent=2)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved diagnostic values to {args.out_dir.resolve()}; no k was selected")


if __name__ == "__main__":
    main()
