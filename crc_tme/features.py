import numpy as np
import pandas as pd
import scanpy as sc


def dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def prepare_model_features(adata, args):
    mode = args.features_mode
    # Select HVGs and return their expression without PCA.
    if mode == "hvg":
        sample = adata.copy()
        sc.pp.highly_variable_genes(
            sample,
            n_top_genes=args.hvg,
            subset=True,
        )
        return dense(sample.X)

    # Select HVGs, then compute PCA on those genes.
    if mode == "pca":
        requested_components = getattr(args, "pca_components", args.pca)

        if requested_components < 1:
            raise ValueError("The requested PCA component count must be positive")

        sample = adata.copy()
        sc.pp.highly_variable_genes(
            sample,
            n_top_genes=args.hvg,
            subset=True,
        )
        
        n_components = min(
            requested_components,
            sample.n_obs - 1,
            sample.n_vars - 1,
        )

        if n_components < 1:
            raise ValueError("Too few spots or genes for PCA")

        sc.pp.pca(
            sample,
            n_comps=n_components,
            svd_solver="arpack",
        )
        return sample.obsm["X_pca"]

    # Only the clusters mode should reach gene-group aggregation.
    if mode != "clusters":
        raise ValueError(f"Unknown feature mode: {mode}")

    if not args.cluster_file:
        raise ValueError("--cluster-file is required for clusters mode")

    groups = pd.read_csv(
        args.cluster_file,
        sep=None,
        engine="python",
    )

    if "cluster_id" not in groups.columns and "index" in groups.columns:
        groups = groups.rename(columns={"index": "cluster_id"})

    if not {"cluster_id", "genes"}.issubset(groups.columns):
        raise ValueError("Gene-cluster CSV requires cluster_id and genes")

    X = dense(adata.X)
    index = {str(g): i for i, g in enumerate(adata.var_names)}
    upper = {str(g).upper(): i for i, g in enumerate(adata.var_names)}

    features = []

    for gene_list in groups["genes"].fillna(""):
        genes = [
            gene.strip()
            for gene in str(gene_list).split(",")
            if gene.strip()
        ]

        indices = [
            index.get(gene, upper.get(gene.upper()))
            for gene in genes
        ]
        indices = [i for i in indices if i is not None]

        if indices:
            values = np.median(X[:, indices], axis=1)
        else:
            values = np.zeros(adata.n_obs)

        features.append(values)

    if not features:
        raise ValueError("Gene-cluster CSV is empty")

    return np.column_stack(features)
