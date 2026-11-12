import argparse
from pathlib import Path

DEFAULTS = dict(mask_rate=0.8, hvg=3000, epochs=200, lr=0.001,
                temperature=1.0, beta=1.0, lambda_dec=0.0, dec_start_epoch=50,
                dec_update_interval=5, dec_alpha=1.0, alpha_spatial=1.0,
                alpha_expr=1.0, out_dim=32, hidden_dim=128,
                use_graph_positives=False, positive_views="cross",
                positive_fraction=0.0, positive_mining="node",
                contrastive_neg_mode="both", contrastive_top_pct=0.1,
                contrastive_per_view=False, contrastive_refresh_k=1,
                contrastive_symmetric=True, features_mode="clusters",
                loss="2n", use_contrastive=True,
                lambda_smooth=0.0, smooth_graph="spatial_knn",
                lambda_recon_adj=0.0, recon_adj_source="combined",
                lambda_recon_x=0.0, recon_x_target="masked",
                pca_components=50, edges="spatial-expression",
                include_deconv=False, include_lr=False,
                deconv=None, ligrec_file=None)


def input_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--h5ad", required=True, type=Path)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--labels", help="Pathologist annotation CSV")
    labels.add_argument("--no_annotations", action="store_true")
    parser.add_argument("--n-clusters", required=True, type=int)
    parser.add_argument("--k", type=int, help="Number of clusters for k-means", default=10)
    parser.add_argument("--cluster-file", help="Optional explicit path to gene-cluster CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser


def regular_config(args, parser):
    if args.n_clusters < 2:
        parser.error("--n-clusters must be at least 2")
    for key, value in DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args
