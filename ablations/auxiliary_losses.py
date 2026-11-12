import numpy as np
import torch
import torch.nn.functional as F


def spatial_smoothness_loss(z, edges, weights):
    if weights.numel() == 0:
        return z.sum() * 0.0
    differences = z[edges[0]] - z[edges[1]]
    return (differences.square().sum(dim=1) * weights).sum() / weights.sum().clamp_min(1e-12)


def adjacency_reconstruction_loss(z, adjacency):
    z = F.normalize(z, dim=1)
    return F.mse_loss(z @ z.T, adjacency)


def feature_reconstruction_loss(decoder, z_a, z_b, target_a, target_b):
    return 0.5 * (F.mse_loss(decoder(z_a), target_a) + F.mse_loss(decoder(z_b), target_b))


class AuxiliaryObjectives:
    def __init__(self, args, spatial_knn, spatial_full_n, combined, X, device):
        self.args = args
        self.edges = self.weights = self.adjacency = self.clean = None
        if args.lambda_smooth > 0:
            graph = {"spatial_knn": spatial_knn, "spatial_full": spatial_full_n,
                     "combined": combined}[args.smooth_graph]
            rows, cols = np.where(graph > 0)
            self.edges = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
            self.weights = torch.tensor(graph[rows, cols], dtype=torch.float32, device=device)
            if not len(rows):
                raise ValueError("The requested smoothness graph has no edges")
        if args.lambda_recon_adj > 0:
            graph = combined if args.recon_adj_source == "combined" else spatial_full_n
            self.adjacency = torch.tensor(graph, dtype=torch.float32, device=device)
        if args.lambda_recon_x > 0 and args.recon_x_target == "clean":
            self.clean = torch.tensor(X, dtype=torch.float32, device=device)

    def compute(self, model, z_a, z_b, x_a, x_b):
        zero = z_a.new_zeros(())
        terms = dict(smooth=zero, recon_adj=zero, recon_x=zero)
        z_mean = (z_a + z_b) / 2.0
        if self.edges is not None:
            terms["smooth"] = spatial_smoothness_loss(z_mean, self.edges, self.weights)
        if self.adjacency is not None:
            terms["recon_adj"] = adjacency_reconstruction_loss(z_mean, self.adjacency)
        if self.args.lambda_recon_x > 0:
            target_a, target_b = (self.clean, self.clean) if self.clean is not None else (x_a, x_b)
            terms["recon_x"] = feature_reconstruction_loss(model.decoder, z_a, z_b, target_a, target_b)
        return terms
