import numpy as np
import torch
import torch.nn.functional as F

def info_nce_2n(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.5,
    neg_mode: str = "cross",
    node_mask: torch.Tensor = None, 
    per_view_mask: dict = None,
    extra_pos_mask_2n: torch.Tensor = None,
    symmetric: bool = True
) -> torch.Tensor:

    assert neg_mode in {"cross", "same", "both"}
    device = z_a.device
    N = z_a.shape[0]

    #Normalize and compute 2N×2N 
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    Z = torch.cat([z_a, z_b], dim=0)                 
    S = torch.matmul(Z, Z.T) / (temperature + 1e-12)   

    #Default cross-view positives
    pos_mask = torch.zeros_like(S)
    rows = torch.arange(N, device=device)
    pos_mask[rows, rows + N] = 1.0
    pos_mask[rows + N, rows] = 1.0

    #Add extra positives if selected
    if extra_pos_mask_2n is not None:
        m = extra_pos_mask_2n.to(device).float()
        m.fill_diagonal_(0.0)
        pos_mask = torch.clamp(pos_mask + m, max=1.0)

    #No self loops
    eye = torch.eye(2*N, device=device)
    base_mask = torch.ones_like(S) - eye

    #Allowed negative blocks by mode
    AA = (slice(0, N), slice(0, N))
    AB = (slice(0, N), slice(N, 2*N))
    BA = (slice(N, 2*N), slice(0, N))
    BB = (slice(N, 2*N), slice(N, 2*N))

    neg_allowed = torch.zeros_like(S)
    if neg_mode in {"both", "cross"}:
        neg_allowed[AB] = 1.0
        neg_allowed[BA] = 1.0
        #Remove the single positives from neg
        neg_allowed[rows, rows + N] = 0.0
        neg_allowed[rows + N, rows] = 0.0
    if neg_mode in {"both", "same"}:
        neg_allowed[AA] = 1.0
        neg_allowed[BB] = 1.0

    #Shared node-level mask (N,N) → broadcast to AA/BB and AB/BA
    if node_mask is not None:
        nm = node_mask.to(device).float().clone()
        nm.fill_diagonal_(0.0)
        nm_AA = nm_BB = nm_AB = nm_BA = torch.zeros((N, N), device=device)
        if neg_mode in {"both", "same"}:
            nm_AA = nm
            nm_BB = nm
        if neg_mode in {"both", "cross"}:
            nm_AB = nm
            nm_BA = nm
        nm_full = torch.zeros_like(S)
        nm_full[AA] = nm_AA
        nm_full[AB] = nm_AB
        nm_full[BA] = nm_BA
        nm_full[BB] = nm_BB
        neg_allowed = neg_allowed * nm_full

    #Per-view masks (A and B) for same-view experiments or view-specific mining
    if per_view_mask is not None:
        mA = per_view_mask.get("A", None)
        mB = per_view_mask.get("B", None)
        pv = torch.ones_like(S)
        if mA is not None:
            mA = mA.to(device).float().clone()
            mA.fill_diagonal_(0.0)
            pv[AA] = pv[AA] * mA
        if mB is not None:
            mB = mB.to(device).float().clone()
            mB.fill_diagonal_(0.0)
            pv[BB] = pv[BB] * mB
        neg_allowed = neg_allowed * pv

    #Final allowed negatives & anchors
    neg_allowed = neg_allowed * base_mask * (1.0 - pos_mask)
    anchor_rows = torch.arange(2*N, device=device) if symmetric else torch.arange(N, device=device)

    log_numer = torch.logsumexp(S.masked_fill(pos_mask <= 0, -torch.inf), dim=1)[anchor_rows]
    log_denom = torch.logsumexp(S.masked_fill((pos_mask + neg_allowed) <= 0, -torch.inf), dim=1)[anchor_rows]
    loss = (log_denom - log_numer).mean()
    return loss

def build_node_mask_top_p(D: np.ndarray, p: float) -> np.ndarray:
    N = D.shape[0]
    M = np.zeros_like(D, dtype=np.float32)
    D_ = D.copy()
    np.fill_diagonal(D_, -np.inf)
    k = max(1, int(np.ceil(p * (N - 1))))
    idx = np.argpartition(-D_, kth=k-1, axis=1)[:, :k]
    rows = np.arange(N)[:, None]
    M[rows, idx] = 1.0
    np.fill_diagonal(M, 0.0)
    return M

@torch.no_grad()
def per_view_top_p_masks_from_embeddings(z_a: torch.Tensor,
                                         z_b: torch.Tensor,
                                         p: float) -> dict:
    za = F.normalize(z_a, dim=1)
    zb = F.normalize(z_b, dim=1)
    score_A = za @ za.T
    score_B = zb @ zb.T
    big_neg = -1e9

    N = za.shape[0]
    k = max(1, int(np.ceil(p * (N - 1))))
    score_A = score_A.clone()
    score_B = score_B.clone()
    score_A.fill_diagonal_(big_neg)
    score_B.fill_diagonal_(big_neg)
    idx_A = torch.topk(score_A, k=k, dim=1, largest=True).indices
    idx_B = torch.topk(score_B, k=k, dim=1, largest=True).indices
    M_A = torch.zeros_like(score_A).scatter_(1, idx_A, 1.0)
    M_B = torch.zeros_like(score_B).scatter_(1, idx_B, 1.0)
    M_A.fill_diagonal_(0.0)
    M_B.fill_diagonal_(0.0)
    return {"A": M_A, "B": M_B}



def cross_view_loss(z_a, z_b, temperature=1.0, neg_mask=None, pos_mask_override=None):
    logits = F.normalize(z_a, dim=1) @ F.normalize(z_b, dim=1).T / (temperature + 1e-12)
    n = len(z_a)
    positives = torch.eye(n, device=z_a.device, dtype=torch.bool)
    if pos_mask_override is not None:
        positives = positives | pos_mask_override.to(z_a.device).bool()
    if neg_mask is None:
        allowed = torch.ones_like(positives)
    else:
        allowed = positives | neg_mask.to(z_a.device).bool()
    log_numer = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    log_denom = torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)
    return (log_denom - log_numer).mean()
