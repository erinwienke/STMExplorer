import argparse
from pathlib import Path
import scanpy as sc


#Remove spots with fewer than 200 detected genes and genes observed in fewer than 10 spots, then normalize and log-transform the data.
def preprocess(adata, min_genes=200, min_cells=10, target_sum=10_000):
    adata = adata.copy()
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError("Too few spots or genes remain after filtering")
    if "spatial" not in adata.obsm:
        raise ValueError("Input .h5ad must contain obsm['spatial']")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.uns["crc_tme_preprocessed"] = True
    return adata


def main():
    parser = argparse.ArgumentParser(description="Preprocess one spatial transcriptomics sample")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input to preserve the raw sample")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata = preprocess(sc.read_h5ad(args.input))
    adata.write_h5ad(args.output)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"Failed to write preprocessed sample: {args.output.resolve()}")
    print(f"Saved preprocessed .h5ad: {args.output.resolve()} ({adata.n_obs} spots, {adata.n_vars} genes)", flush=True)
    from .gene_clusters import cluster_genes
    print("Generating 3000 gene clusters; this may take a while...", flush=True)
    try:
        cluster_genes(adata, args.output.with_suffix(""), k=3000)
    except Exception as exc:
        raise RuntimeError(f"Gene clustering failed, but the preprocessed .h5ad was saved at {args.output.resolve()}") from exc


if __name__ == "__main__":
    main()
