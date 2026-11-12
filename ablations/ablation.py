import math
from ..config import input_parser, regular_config


def build_parser():
    p = input_parser("Run one CRC TME ablation configuration")
    p.add_argument("--k", type=int, default=10, help="Neighbors in both kNN graphs")
    p.add_argument("--features-mode", choices=["clusters", "hvg", "pca", "all", "hvg2000", "pca50"], default="clusters")
    p.add_argument("--hvg", type=int, default=3000)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--edges", choices=["spatial-expression", "spatial", "expression", "deconv", "lr", "all"], default="spatial-expression")
    p.add_argument("--deconv", help="Spot-by-cell-type CSV; first column is spot ID")
    p.add_argument("--ligrec-file", help="sender, receiver, score CSV; requires --deconv")
    p.add_argument("--negative-views", dest="contrastive_neg_mode", choices=["cross", "same", "both"], default=None)
    p.add_argument("--negative-fraction", dest="contrastive_top_pct", type=float, default=0.1,
                   help="Fraction of farthest spatial/expression pairs; -1 uses all negatives")
    p.add_argument("--per-view-mining", dest="contrastive_per_view", action="store_true",
                   help="Mine same-view negative blocks using each view's embedding similarities")
    p.add_argument("--asymmetric", action="store_true", help="Use first-view anchors only")
    p.add_argument("--graph-positives", dest="use_graph_positives", action="store_true")
    p.add_argument("--positive-fraction", "--pos-top-pct", type=float, default=0.0,
                   help="Closest fraction of other spots as additional positives; 0 disables")
    p.add_argument("--positive-mining", choices=["node", "view"], default="node",
                   help="Rank positives by spatial/expression distance or by each embedding view block")
    p.add_argument("--positive-views", choices=["cross", "same", "both"], default="cross",
                   help="View blocks for extra positives; matched spots remain cross-view positives")
    p.add_argument("--loss", choices=["2n", "cross-view"], default="2n")
    contrastive = p.add_mutually_exclusive_group()
    contrastive.add_argument("--use-contrastive", dest="use_contrastive", action="store_true")
    contrastive.add_argument("--no-contrastive", dest="use_contrastive", action="store_false")
    p.set_defaults(use_contrastive=True)
    p.add_argument("--beta", "--contrastive-weight", type=float, default=1.0)
    p.add_argument("--dec-weight", "--lambda-dec", dest="lambda_dec", type=float, default=0.0)
    p.add_argument("--dec-start-epoch", type=int, default=50)
    p.add_argument("--dec-update-interval", type=int, default=5)
    p.add_argument("--dec-alpha", type=float, default=1.0)
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--smooth-graph", choices=["spatial_knn", "spatial_full", "combined"], default="spatial_knn")
    p.add_argument("--lambda-recon-adj", type=float, default=0.0)
    p.add_argument("--recon-adj-source", choices=["combined", "spatial"], default="combined")
    p.add_argument("--lambda-recon-x", type=float, default=0.0)
    p.add_argument("--recon-x-target", choices=["masked", "clean"], default="masked")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--mask-rate", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=1.0)
    return p


def parse_args(argv=None):
    p = build_parser()
    args = regular_config(p.parse_args(argv), p)
    args.alpha_spatial = float(args.edges in {"spatial-expression", "spatial", "all"})
    args.alpha_expr = float(args.edges in {"spatial-expression", "expression", "all"})
    args.include_deconv = bool(args.deconv) and args.edges != "lr"
    args.include_lr = bool(args.ligrec_file) and args.edges != "deconv"
    args.contrastive_symmetric = not args.asymmetric if args.loss == "2n" else False
    if args.contrastive_neg_mode is None:
        args.contrastive_neg_mode = "cross" if args.loss == "cross-view" else "both"
    numeric = [args.lr, args.mask_rate, args.temperature, args.beta, args.lambda_dec,
               args.dec_alpha, args.lambda_smooth, args.lambda_recon_adj, args.lambda_recon_x,
               args.positive_fraction, args.contrastive_top_pct]
    if not all(math.isfinite(value) for value in numeric):
        p.error("Numeric controls must be finite")
    if args.k < 1 or args.epochs < 1 or args.hvg < 2 or args.pca_components < 1:
        p.error("k, epochs, and pca-components must be positive; hvg must be at least 2")
    if not 0 <= args.mask_rate < 1 or args.temperature <= 0 or args.lr <= 0:
        p.error("mask-rate must be in [0, 1); temperature and lr must be positive")
    if min(args.beta, args.lambda_dec, args.lambda_smooth, args.lambda_recon_adj, args.lambda_recon_x) < 0:
        p.error("Loss weights must be nonnegative")
    if args.dec_alpha <= 0 or args.dec_update_interval < 1 or args.dec_start_epoch < 1:
        p.error("DEC alpha, interval, and start epoch must be positive")
    if args.lambda_dec > 0 and args.dec_start_epoch > args.epochs:
        p.error("DEC starts after the last training epoch")
    if args.ligrec_file and not args.deconv:
        p.error("--ligrec-file requires --deconv for cell-type proportions")
    if args.edges in {"deconv", "lr", "all"} and not args.deconv:
        p.error(f"--edges {args.edges} requires --deconv")
    if args.edges in {"lr", "all"} and not args.ligrec_file:
        p.error(f"--edges {args.edges} requires --ligrec-file")
    if args.edges == "deconv" and args.ligrec_file:
        p.error("--edges deconv excludes LR edges; omit --ligrec-file")
    if args.contrastive_top_pct != -1 and not 0 < args.contrastive_top_pct <= 1:
        p.error("negative-fraction must be -1 or in (0, 1]")
    if not 0 <= args.positive_fraction <= 1:
        p.error("positive-fraction must be in [0, 1]")
    if not args.use_graph_positives and args.positive_fraction == 0 and args.positive_views != "cross":
        p.error("positive-views requires --graph-positives or a positive --positive-fraction")
    if args.positive_mining == "view" and args.positive_fraction == 0:
        p.error("View-specific positive mining requires a positive --positive-fraction")
    if args.contrastive_per_view and (args.contrastive_neg_mode == "cross" or args.contrastive_top_pct == -1):
        p.error("per-view mining requires same/both negative views and a positive negative fraction")
    if args.loss == "cross-view" and (args.contrastive_per_view or args.contrastive_neg_mode != "cross" or args.positive_views != "cross"):
        p.error("cross-view loss requires cross negative and positive views without same-view mining")
    immediate = (args.use_contrastive and args.beta > 0) or max(args.lambda_smooth, args.lambda_recon_adj, args.lambda_recon_x) > 0
    if not immediate and not (args.lambda_dec > 0 and args.dec_start_epoch == 1):
        p.error("At least one objective must be active at epoch 1; DEC alone must start at epoch 1")
    return args


def main():
    args = parse_args()
    from ..engine import run_analysis
    run_analysis(args)


if __name__ == "__main__":
    main()
