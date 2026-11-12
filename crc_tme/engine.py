import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
)
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, add_self_loops
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial.distance import cdist

eps = 1e-12

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def apply_feature_mask(X_np, mask_rate=0.05, seed=None):
    rng = np.random.RandomState(seed)
    N, Fdim = X_np.shape
    mask = rng.rand(N, Fdim) >= mask_rate
    X_masked = X_np * mask.astype(X_np.dtype)
    return X_masked

#Create GCN
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=16, dropout=0.1, reconstruct_features=False):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=False)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=False)
        self.dropout = dropout
        if reconstruct_features:
            self.decoder = nn.Linear(out_dim, in_dim, bias=True)
    def forward(self, data: Data):
        x, ei = data.x, data.edge_index
        x = self.conv1(x, ei)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, ei)
        return x

from .contrastive import info_nce_2n, build_node_mask_top_p, per_view_top_p_masks_from_embeddings
from .features import prepare_model_features
from .graph_utils import normalize_minmax, keep_top_k_per_row, combine_affinities

def run_analysis(args):
    if not args.cluster_file and args.features_mode == "clusters":
        args.cluster_file = str(args.h5ad.with_suffix("")) + "_clusters_single_with_sizes.csv"
    if args.cluster_file and args.features_mode == "clusters" and not Path(args.cluster_file).is_file():
        raise ValueError(f"Gene-cluster file not found: {args.cluster_file}")
    biological_only = args.alpha_spatial == 0 and args.alpha_expr == 0
    args.pair_reference = "spatial-expression" if biological_only else "active-spatial-expression"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with (args.out_dir / "configuration.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #Load and Preprocess
    adata = sc.read_h5ad(args.h5ad)
    if not adata.uns.get("crc_tme_preprocessed", False):
        raise ValueError("Input must be produced by crc_tme.preprocess")
    if args.n_clusters >= adata.n_obs:
        raise ValueError("--n-clusters must be smaller than the number of spots")
    spot_ids = adata.obs_names.to_numpy()
    n_spots = len(spot_ids)
    print(f"Loaded {args.h5ad}: {n_spots} spots")

    feature_matrix_for_model = prepare_model_features(adata, args)

    coords = adata.obsm.get("spatial", None)
    if coords is None:
        raise RuntimeError("adata.obsm['spatial'] required for spatial edges.")

    #Spatial set-up
    n_spots = feature_matrix_for_model.shape[0]
    spatial_full = np.zeros((n_spots, n_spots), dtype=float)
    spatial_knn = np.zeros_like(spatial_full)
    need_spatial_targets = (args.lambda_smooth > 0 and args.smooth_graph != "combined") or (
        args.lambda_recon_adj > 0 and args.recon_adj_source == "spatial")
    if args.alpha_spatial > 0 or need_spatial_targets or biological_only:
        try:
            dists = cdist(coords, coords, metric="euclidean")
            med = np.median(dists[np.triu_indices_from(dists, k=1)]) if n_spots > 1 else 0.0
            if med <= 0 or not np.isfinite(med):
                med = np.mean(dists[np.triu_indices_from(dists, k=1)]) if n_spots > 1 else 1.0
            sigma = max(med, 1e-6)
            spatial_full = np.exp(-(dists**2) / (2.0 * (sigma**2) + eps))
            np.fill_diagonal(spatial_full, 0.0)
            k_spatial = min(args.k, n_spots - 1)
            S_sp_k = keep_top_k_per_row(spatial_full, k_spatial)
            spatial_knn = np.maximum(S_sp_k, S_sp_k.T)
            print(f"Built spatial (FULL RBF) and KNN (k={k_spatial})")
        except Exception as e:
            print("Warning building spatial:", e)
            spatial_full = np.zeros_like(spatial_full)
            spatial_knn = np.zeros_like(spatial_knn)
    else:
        spatial_full = np.zeros_like(spatial_full)
        spatial_knn = np.zeros_like(spatial_knn)

    #Expression set-up
    expr_full = np.zeros((n_spots, n_spots), dtype=float)
    expr_knn = np.zeros_like(expr_full)
    if args.alpha_expr > 0 or biological_only:
        try:
            #Uses gene clusters unless otherwise indicated
            expr_full = cosine_similarity(feature_matrix_for_model)
            np.fill_diagonal(expr_full, 0.0)
            k_expr = min(args.k, n_spots - 1)
            S_k = keep_top_k_per_row(expr_full, k_expr)
            expr_knn = np.maximum(S_k, S_k.T)
            print(f"Built expression FULL + KNN cosine (k={k_expr})")
        except Exception as e:
            print("Warning building expression:", e)
            expr_full = np.zeros_like(expr_full)
            expr_knn = np.zeros_like(expr_knn)
    else:
        expr_full = np.zeros_like(expr_full)
        expr_knn = np.zeros_like(expr_knn)

    spatial_full_n = normalize_minmax(spatial_full)
    expr_full_n = normalize_minmax(expr_full)
    matrices, weights, neighborhoods = {}, {}, {}
    if args.alpha_spatial > 0:
        matrices["spatial"] = spatial_full
        weights["spatial"] = args.alpha_spatial
        neighborhoods["spatial"] = spatial_knn
    if args.alpha_expr > 0:
        matrices["expression"] = expr_full
        weights["expression"] = args.alpha_expr
        neighborhoods["expression"] = expr_knn
    if args.include_deconv or args.include_lr:
        from .ablations.modalities import optional_affinities
        extra = optional_affinities(adata, args.deconv, args.ligrec_file,
                                    include_deconv=args.include_deconv, include_lr=args.include_lr)
        names = (["deconv"] if args.include_deconv else []) + (["lr"] if args.include_lr else [])
        for name, matrix in zip(names, extra):
            matrices[name] = matrix
            weights[name] = 1.0
    combined_sym = combine_affinities(matrices, args.k, weights=weights, neighborhoods=neighborhoods)
    A = combined_sym.copy()
    if np.all(A == 0):
        raise ValueError("Graph has no edges; check feature matrix and k")
    dense_density = (A > 0).sum() / float(A.size)
    combined_min = float(A.min()) if A.size>0 else 0.0
    combined_max = float(A.max()) if A.size>0 else 0.0
    print(f"Combined adjacency stats: min={combined_min:.6e}, max={combined_max:.6f}, mean={A.mean():.6f}, density={dense_density:.6f}")

    rows_e, cols_e = np.where(combined_sym > 0)
    if rows_e.size == 0:
        idx_i, idx_j = np.triu_indices(n_spots, k=1)
        rows_e = np.concatenate([idx_i, idx_j])
        cols_e = np.concatenate([idx_j, idx_i])
        data_vals = np.ones(rows_e.shape[0]) * 1e-9
    else:
        data_vals = combined_sym[rows_e, cols_e]
    coo_e = coo_matrix((data_vals.astype(float), (rows_e.astype(int), cols_e.astype(int))), shape=(n_spots, n_spots))
    coo_sym_e = coo_e.maximum(coo_e.T).tocoo()
    src = coo_sym_e.row.astype(np.int64)
    dst = coo_sym_e.col.astype(np.int64)
    e_comb_np = np.vstack([src, dst]) if src.size > 0 else np.empty((2, 0), dtype=np.int64)
    e_comb = torch.tensor(e_comb_np, dtype=torch.long)
    e_comb, _ = add_self_loops(e_comb, num_nodes=n_spots)
    e_comb = to_undirected(e_comb)
    print("Combined PyG edge_index length:", e_comb.shape[1])

    adata.obsp["connectivities"] = csr_matrix(A)

    edge_index_device = e_comb.to(device)
    model = GCN(
        in_dim=feature_matrix_for_model.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=0.1,
        reconstruct_features=args.lambda_recon_x > 0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    auxiliary = None
    if max(args.lambda_smooth, args.lambda_recon_adj, args.lambda_recon_x) > 0:
        from .ablations.auxiliary_losses import AuxiliaryObjectives
        auxiliary = AuxiliaryObjectives(args, spatial_knn, spatial_full_n, A,
                                         feature_matrix_for_model, device)

    if args.no_annotations:
        if "gt" in adata.obs:
            del adata.obs["gt"]
    else:
        labels_df = pd.read_csv(args.labels)
        if "Barcode" not in labels_df.columns:
            cand = [c for c in labels_df.columns if c.lower() in ("barcode","barcodes","spot","spot_id")]
            if cand:
                labels_df = labels_df.rename(columns={cand[0]: "Barcode"})
        if "Pathologist Annotations" not in labels_df.columns:
            cand = [c for c in labels_df.columns if "annotation" in c.lower()]
            if cand:
                labels_df = labels_df.rename(columns={cand[0]: "Pathologist Annotations"})
        if "Barcode" not in labels_df.columns or "Pathologist Annotations" not in labels_df.columns:
            raise ValueError("Labels CSV needs Barcode and Pathologist Annotations columns")
        label_map = dict(zip(labels_df["Barcode"].astype(str), labels_df["Pathologist Annotations"]))
        adata.obs["gt"] = pd.Index(spot_ids).map(label_map)

    spatial_for_pairs = spatial_full_n if args.alpha_spatial > 0 or biological_only else np.zeros_like(spatial_full_n)
    expr_for_pairs = expr_full_n if args.alpha_expr > 0 or biological_only else np.zeros_like(expr_full_n)
    pair_distance = normalize_minmax((1.0 - spatial_for_pairs) + (1.0 - expr_for_pairs))
    static_positives = None
    if args.use_contrastive and (args.use_graph_positives or args.positive_fraction > 0):
        from .ablations.pair_selection import select_fraction, positive_blocks, embedding_positive_blocks
    if args.use_contrastive and (args.use_graph_positives or (args.positive_fraction > 0 and args.positive_mining == "node")):
        node_positives = np.zeros_like(A)
        if args.use_graph_positives:
            node_positives = (A > 0).astype(float)
        if args.positive_fraction > 0 and args.positive_mining == "node":
            node_positives = np.maximum(node_positives, select_fraction(pair_distance, args.positive_fraction))
        static_positives = torch.tensor(positive_blocks(node_positives, args.positive_views), device=device)

    log_rows = []
    centers = None
    Kc = args.n_clusters

    node_mask_t = None
    per_view_masks = None
    #Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        seed_a = epoch * 2 + 1
        seed_b = epoch * 2 + 2
        X_a = apply_feature_mask(feature_matrix_for_model, mask_rate=args.mask_rate, seed=seed_a)
        X_b = apply_feature_mask(feature_matrix_for_model, mask_rate=args.mask_rate, seed=seed_b)
        x_a = torch.tensor(X_a, dtype=torch.float32).to(device)
        x_b = torch.tensor(X_b, dtype=torch.float32).to(device)
        data_a = Data(x=x_a, edge_index=edge_index_device).to(device)
        data_b = Data(x=x_b, edge_index=edge_index_device).to(device)
        z_a = model(data_a)
        z_b = model(data_b)
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)

        #Pair selection
        extra_pos_mask_2n = static_positives
        if args.use_contrastive and args.positive_fraction > 0 and args.positive_mining == "view":
            selected = embedding_positive_blocks(z_a.detach().cpu().numpy(), z_b.detach().cpu().numpy(),
                                                  args.positive_fraction, args.positive_views)
            view_positives = torch.tensor(selected, device=device)
            extra_pos_mask_2n = view_positives if static_positives is None else torch.maximum(static_positives, view_positives)
        if args.use_contrastive and args.contrastive_top_pct >= 0:
            if args.contrastive_per_view:
                per_view_masks = per_view_top_p_masks_from_embeddings(z_a.detach(), z_b.detach(), p=args.contrastive_top_pct)
            else:
                node_mask_t = torch.tensor(build_node_mask_top_p(pair_distance, args.contrastive_top_pct), device=device)
        loss_contrast = z_a.new_zeros(())
        if args.use_contrastive:
            if args.loss == "cross-view":
                from .contrastive import cross_view_loss
                extra_cross = extra_pos_mask_2n[:n_spots, n_spots:] if extra_pos_mask_2n is not None else None
                loss_contrast = cross_view_loss(z_a, z_b, args.temperature,
                                                 neg_mask=node_mask_t, pos_mask_override=extra_cross)
            else:
                loss_contrast = info_nce_2n(
                    z_a, z_b, temperature=args.temperature, neg_mode=args.contrastive_neg_mode,
                    node_mask=node_mask_t, per_view_mask=per_view_masks,
                    extra_pos_mask_2n=extra_pos_mask_2n, symmetric=args.contrastive_symmetric)
        #DEC iterative loss
        loss_dec = torch.tensor(0.0, device=device)
        if args.lambda_dec > 0 and epoch >= args.dec_start_epoch:
            with torch.no_grad():
                z_mean = ((z_a + z_b) / 2.0).detach()
                if centers is None or (epoch - args.dec_start_epoch) % args.dec_update_interval == 0:
                    z_np = z_mean.cpu().numpy()
                    km_init = KMeans(n_clusters=Kc, random_state=42, n_init=10).fit(z_np)
                    centers = torch.tensor(km_init.cluster_centers_, dtype=torch.float32, device=device)
            q, p = compute_q_p_from_centers((z_a + z_b) / 2.0, centers, alpha=args.dec_alpha)
            loss_dec = dec_kl_loss(q, p)

        terms = auxiliary.compute(model, z_a, z_b, x_a, x_b) if auxiliary is not None else dict(smooth=z_a.new_zeros(()), recon_adj=z_a.new_zeros(()), recon_x=z_a.new_zeros(()))
        total_loss = (args.beta * loss_contrast + args.lambda_dec * loss_dec
                      + args.lambda_smooth * terms["smooth"]
                      + args.lambda_recon_adj * terms["recon_adj"]
                      + args.lambda_recon_x * terms["recon_x"])
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        total_loss.backward()
        opt.step()

        #Save each raw component and its weighted contribution for every epoch.
        components = dict(contrastive=loss_contrast, dec=loss_dec, **terms)
        weights = dict(contrastive=args.beta if args.use_contrastive else 0.0,
                       dec=args.lambda_dec, smooth=args.lambda_smooth,
                       recon_adj=args.lambda_recon_adj, recon_x=args.lambda_recon_x)
        row = {"epoch": epoch, "total_loss": float(total_loss.detach())}
        for name, value in components.items():
            row[name] = float(value.detach())
            row[f"weighted_{name}"] = weights[name] * row[name]
        log_rows.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}: total={row['total_loss']:.6f} "
                  + " ".join(f"{name}={row[name]:.6f}" for name in components), flush=True)

    #Save training log
    log_df = pd.DataFrame(log_rows)
    log_path = args.out_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)
    print("Saved training log to", log_path)

    #Final embeddings on full X
    model.eval()
    with torch.no_grad():
        data_full = Data(x=torch.tensor(feature_matrix_for_model.copy(), dtype=torch.float32).to(device), edge_index=edge_index_device).to(device)
        z_final = model(data_full).cpu().numpy()

    from .clustering import cluster_embeddings
    pred, wcss = cluster_embeddings(z_final, args.n_clusters)
    adata.obs["cluster_pred"] = pred.astype(str)
    assignments = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "cluster_assignment": pred})
    assignments.to_csv(args.out_dir / "cluster_assignments.csv", index=False)

    #Metrics helper
    def compute_all_metrics(true_series, pred_labels, embeddings, method_name, wcss=np.nan):
        out = {"Method": method_name}
        try:
            ami_val, nmi_val, ari_val = evaluate_ami_nmi(true_series, pred_labels)
        except Exception as e:
            print(f"Warning computing AMI/NMI for {method_name}: {e}")
            ami_val = None; nmi_val = None; ari_val = None
        sil = None
        dbi = None
        try:
            unique_labels = np.unique(pred_labels)
            if (len(unique_labels) >= 2) and (len(unique_labels) < embeddings.shape[0]):
                sil = silhouette_score(embeddings, pred_labels)
                dbi = davies_bouldin_score(embeddings, pred_labels)
        except Exception as e:
            print(f"Warning computing silhouette for {method_name}: {e}")
            sil = None
        out.update({"AMI": ami_val, "NMI": nmi_val, "ARI": ari_val, "Silhouette": sil, "DBI": dbi,
                    "WCSS": wcss})
        return out

    metrics_rows = [compute_all_metrics(adata.obs.get("gt", None), pred, z_final,
                                       method_name="GCN_kmeans", wcss=wcss)]
    metrics_table = pd.DataFrame(metrics_rows, columns=["Method","AMI","NMI","ARI","Silhouette","DBI","WCSS"])
    print("\n==================== Metrics Table ====================")
    print(metrics_table.to_string(index=False))
    metrics_table.to_csv(args.out_dir / "metrics_table.csv", index=False)
    print("Saved metrics table to", args.out_dir / "metrics_table.csv")

    #Save final embeddings to a known file in the current working directory
    try:
        sample_name = args.h5ad.stem
        emb_path = args.out_dir / "embeddings.npy"
        np.save(emb_path, z_final)
        print(f"Saved embeddings to {emb_path}")
        #Write h5ad copy
        try:
            adata.obsm["X_gcn"] = z_final
            adata_out = args.out_dir / f"{sample_name}_with_gcn.h5ad"
            adata.write_h5ad(adata_out)
            print(f"Saved AnnData with embeddings to {adata_out}")
        except Exception as _exc:
            print(f"[warn] Could not write AnnData with embeddings: {_exc}")
    except Exception as _exc:
        print(f"[warn] Could not save embeddings: {_exc}")

    #Save summary
    def mt_val(name, col):
        try:
            return metrics_table.loc[metrics_table["Method"]==name, col].values[0]
        except Exception:
            return None
    results = {
        "h5ad": args.h5ad,
        "k": args.k,
        "n_spots": n_spots,
        "n_clusters": args.n_clusters,
        "mask_rate": args.mask_rate,
        "hvg": args.hvg,
        "AMI_GCN": mt_val("GCN_kmeans","AMI"),
        "NMI_GCN": mt_val("GCN_kmeans","NMI"),
        "Sil_GCN": mt_val("GCN_kmeans","Silhouette"),
        "ARI_GCN": mt_val("GCN_kmeans","ARI"),
        "DBI_GCN": mt_val("GCN_kmeans","DBI"),
        "WCSS_GCN": wcss,
        "contrastive_neg_mode": args.contrastive_neg_mode,
        "contrastive_top_pct": args.contrastive_top_pct,
        "contrastive_per_view": bool(args.contrastive_per_view),
        "contrastive_refresh_k": int(args.contrastive_refresh_k),
        "contrastive_symmetric": bool(args.contrastive_symmetric),
        "loss": args.loss,
        "clustering": "kmeans",
        "dec_enabled": args.lambda_dec > 0,
        "lambda_dec": args.lambda_dec,
        "lambda_smooth": args.lambda_smooth,
        "lambda_recon_adj": args.lambda_recon_adj,
        "lambda_recon_x": args.lambda_recon_x,
        "positive_fraction": args.positive_fraction,
        "positive_mining": args.positive_mining,
        "features_mode": args.features_mode,
    }
    summary_path = args.out_dir / "results.csv"
    pd.DataFrame([results]).to_csv(summary_path, index=False)
    print("Saved results to", summary_path)

#DEC helpers
def compute_q_p_from_centers(z, centers, alpha=1.0):
    d = torch.cdist(z, centers, p=2)
    d2 = d.pow(2)
    q_num = (1.0 + d2 / (alpha + eps)).pow(-(alpha + 1.0) / 2.0)
    q = q_num / (q_num.sum(dim=1, keepdim=True) + eps)
    f = q.sum(dim=0)
    p_num = (q * q) / (f.unsqueeze(0) + eps)
    p = p_num / (p_num.sum(dim=1, keepdim=True) + eps)
    return q, p

def dec_kl_loss(q, p):
    q = torch.clamp(q, min=eps)
    p = torch.clamp(p, min=eps)
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()
    return kl

def evaluate_ami_nmi(true_series: pd.Series, pred_labels: np.ndarray):
    if true_series is None:
        return np.nan, np.nan, np.nan
    labels = pd.Series(true_series).reset_index(drop=True)
    mask = labels.notna() & labels.astype(str).str.strip().ne("") & labels.astype(str).str.lower().str.strip().ne("exclude")
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    y_true = labels[mask].astype(str).to_numpy()
    y_pred = np.asarray(pred_labels)[mask.to_numpy()]
    return adjusted_mutual_info_score(y_true, y_pred), normalized_mutual_info_score(y_true, y_pred), adjusted_rand_score(y_true, y_pred)


