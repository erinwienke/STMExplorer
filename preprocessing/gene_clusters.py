from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def cluster_genes(adata, output_prefix, k=3000):
    if not adata.uns.get("crc_tme_preprocessed", False):
        raise ValueError("Input must be produced by crc_tme.preprocess")

    if k < 1 or k > adata.n_vars:
        raise ValueError(f"Gene groups ({k}) must be between 1 and {adata.n_vars}")

    X = adata.X.T

    if sparse.issparse(X):
        X = X.tocsr()

    #Identify gene clusters
    features = X.astype(np.float32)
    print(f"Clustering {adata.n_vars} genes into {k} groups with full KMeans", flush=True)
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=300).fit_predict(features)
    rows = []

    #find median expression value of each gene clusters member genes across spots
    for cid in np.unique(labels):
        ix = np.flatnonzero(labels == cid)
        block = X[ix, :]
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        rows.append(dict(cluster_id=int(cid), genes=",".join(adata.var_names[ix]),
                         median_across_spots=float(np.median(np.median(block.astype(np.float32), axis=0))),
                         n_genes=len(ix)))

    #Return gene cluster files    
    result = pd.DataFrame(rows)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result.drop(columns="n_genes").to_csv(str(prefix) + "_clusters_single.csv", index=False)
    result.to_csv(str(prefix) + "_clusters_single_with_sizes.csv", index=False)
    print(f"Saved gene-cluster CSVs under {prefix.parent.resolve()}", flush=True)
    return result

