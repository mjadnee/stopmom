import torch
import numpy as np
import math
from scipy.linalg import eigh


class SOM:
    def __init__(self, map_h, map_w, Ne,
                 geometry="rectangular", device="cpu", seed=None, feature_weights=None):
        if seed is not None:
            torch.manual_seed(seed)
        if geometry == "toroidal" and map_h % 2:
            raise ValueError("toroidal hexagonal SOMs require an even map_h")
        self.map_h, self.map_w, self.Ne = map_h, map_w, Ne
        self.geometry, self.device = geometry, device
        if feature_weights is None:
            self.feature_weights = torch.ones(Ne, device=device)
        else:
            self.feature_weights = torch.as_tensor(feature_weights, dtype=torch.float32, device=device).flatten()
            if self.feature_weights.numel() != Ne:
                raise ValueError("feature_weights must contain one value for each spectral feature")
            if (self.feature_weights < 0).any():
                raise ValueError("feature_weights must be non-negative")

        # Hex lattice: odd rows shifted right by 1/2, neighbours at distance 1
        r, c = torch.meshgrid(torch.arange(map_h), torch.arange(map_w), indexing="ij")
        x = (c + 0.5 * (r % 2)).flatten().float()
        y = (r * math.sqrt(3) / 2).flatten().float()
        self.xy = torch.stack([x, y], 1)  # (M, 2) node positions, for plotting

        dx = x[:, None] - x[None, :]
        if geometry in {"cylindrical", "toroidal"}:
            dx = torch.remainder(dx + map_w / 2, map_w) - map_w / 2  # shortest way round the seam
        dy = y[:, None] - y[None, :]
        if geometry == "toroidal":
            map_height = map_h * math.sqrt(3) / 2
            dy = torch.remainder(dy + map_height / 2, map_height) - map_height / 2
        self.D2 = (dx**2 + dy**2).to(device)  # (M, M) squared lattice distances
        self.diameter = self.D2.max().sqrt().item()
 
        self.weights = torch.zeros(map_h, map_w, Ne, device=device)

    def init_pca(self, X):
        """Lay the map out along the top principal components of X."""
        _, S, V = torch.pca_lowrank(X, q=3, center=False)  # X is already zero-mean
        lam = S**2 / (len(X) - 1)  # variance along PC1..PC3
        x, y = self.xy[:, 0].to(X.device), self.xy[:, 1].to(X.device)
        a = 2 * (x - x.min()) / (x.max() - x.min()) - 1  # along map_w, in [-1, 1]
        b = 2 * y / y.max() - 1  # along map_h, in [-1, 1]
 
        if self.geometry == "cylindrical":
            # Loop in the PC1-PC2 plane, open axis along PC3
            t = 2 * math.pi * x / self.map_w
            coef = torch.stack([(2 * lam[0]).sqrt() * t.cos(),
                                (2 * lam[1]).sqrt() * t.sin(),
                                (3 * lam[2]).sqrt() * b], 1)
        elif self.geometry == "toroidal":
            # Two periodic axes embedded as a torus in the first three PCs
            u = 2 * math.pi * x / self.map_w
            v = 2 * math.pi * y / (self.map_h * math.sqrt(3) / 2)
            major_radius = (2 * lam[0]).sqrt()
            minor_radius = (2 * lam[1]).sqrt()
            coef = torch.stack([(major_radius + minor_radius * v.cos()) * u.cos(),
                                (major_radius + minor_radius * v.cos()) * u.sin(),
                                minor_radius * v.sin()], 1)
        else:
            # Plane spanned by PC1 (along map_w) and PC2 (along map_h)
            coef = torch.stack([(3 * lam[0]).sqrt() * a,
                                (3 * lam[1]).sqrt() * b,
                                torch.zeros_like(a)], 1)
        self.weights = (coef @ V.T).view(self.map_h, self.map_w, self.Ne)

    def _bmu(self, X, chunk=16384):
        """Best-matching unit and squared distance to it, for each row of X."""
        W = self.weights.reshape(-1, self.Ne)
        w2 = (W * W * self.feature_weights).sum(1)
        idx, d2 = [], []
        for Xc in X.split(chunk):
            weighted_Xc = Xc * self.feature_weights
            s = torch.addmm(w2, weighted_Xc, W.T, alpha=-2)  # ||w||^2 - 2 x.w
            m, i = s.min(1)
            idx.append(i)
            d2.append(m + (Xc * Xc * self.feature_weights).sum(1))
        return torch.cat(idx), torch.cat(d2).clamp_min(0)
 
    def _phase(self, X, epochs, sigma_start, sigma_end, subset_fraction, name):
        M = self.map_h * self.map_w
        k = int(subset_fraction * len(X))
        for ep, sigma in enumerate(np.linspace(sigma_start, sigma_end, epochs)):
            Xe = X if k >= len(X) else X[torch.randperm(len(X), device=X.device)[:k]]
            bmu, d2 = self._bmu(Xe)
 
            # Batch update: each node becomes the neighbourhood-weighted mean of the data
            counts = torch.bincount(bmu, minlength=M).to(X.dtype)
            sums = torch.zeros(M, self.Ne, device=X.device).index_add_(0, bmu, Xe)
            H = torch.exp(-self.D2 / (2 * sigma**2))
            num, den = H @ sums, H @ counts
            W = self.weights.reshape(-1, self.Ne)
            W = torch.where(den[:, None] > 0, num / den[:, None], W)
            self.weights = W.view(self.map_h, self.map_w, self.Ne)
 
            print(f"{name} {ep + 1}/{epochs} | sigma={sigma:.2f} | QE={d2.sqrt().mean():.4f}")

    def errors(self, X, chunk=16384):
        """Quantization error (mean distance to BMU) and topographic error
        (fraction of samples whose 1st and 2nd BMUs aren't hex neighbours)."""
        X = X.to(self.device, torch.float32)
        W = self.weights.reshape(-1, self.Ne)
        w2 = (W * W * self.feature_weights).sum(1)
        qe, te = 0.0, 0
        for Xc in X.split(chunk):
            weighted_Xc = Xc * self.feature_weights
            d2, idx = torch.addmm(w2, weighted_Xc, W.T, alpha=-2).topk(2, dim=1, largest=False)
            qe += (d2[:, 0] + (Xc * Xc * self.feature_weights).sum(1)).clamp_min(0).sqrt().sum().item()
            te += (self.D2[idx[:, 0], idx[:, 1]] > 1.5).sum().item()
        return qe / len(X), te / len(X)
 
    def fit(self, X, ordering_epochs=20, tuning_epochs=60, sigma_end=0.5, subset_fraction=1.0):
        """PCA init, then ordering (sigma: diameter/4 -> diameter/16) and tuning (-> sigma_end)."""
        X = X.to(self.device, torch.float32)
        self.init_pca(X)
        sigma0 = self.diameter / 4
        sigma1 = max(sigma_end, sigma0 / 4)
        self._phase(X, ordering_epochs, sigma0, sigma1, subset_fraction, "Ordering")
        self._phase(X, tuning_epochs, sigma1, sigma_end, subset_fraction, "Tuning")
 
    def predict(self, X):
        """(row, col) of the BMU for each row of X."""
        bmu, _ = self._bmu(X.to(self.device, torch.float32))
        return torch.stack([bmu // self.map_w, bmu % self.map_w], 1).cpu()
 
    def get_u_matrix(self):
        """Mean weight distance from each node to its hex neighbours (seam-aware)."""
        W = self.weights.reshape(-1, self.Ne)
        adj = (self.D2 > 0) & (self.D2 < 1.5)
        weighted_W = W * self.feature_weights.sqrt()
        U = (torch.cdist(weighted_W, weighted_W) * adj).sum(1) / adj.sum(1)
        return U.view(self.map_h, self.map_w).cpu().numpy()
 
    def _get_plot_coordinates(self, i, j):
        return j + 0.5 * (i % 2), i * math.sqrt(3) / 2



def build_som_grid(FLAT_DATA, geometry="rectangular"):
    N, D = FLAT_DATA.shape
    n = int(3 * np.sqrt(N))

    # Find eigenvalues of PCA covariance on a subsample:
    l1, l2, l3 = eigh(FLAT_DATA.T @ FLAT_DATA,
                        eigvals_only = True,
                        subset_by_index=[D-3, D-1])[::-1]

    if geometry in {'rectangular', 'toroidal'}:
        ratio = np.sqrt(l1 / l2)
    elif geometry == 'cylindrical':
        ratio = 2 * np.pi * np.sqrt((l1 + l2) / (12 * l3))
        ratio = min(ratio, 4)
    else:
        raise ValueError("geometry must be cylindrical, rectangular, or toroidal")

    map_h = round(np.sqrt(n / ratio))
    if geometry == "toroidal" and map_h % 2:
        map_h += 1
    map_w = round(n / map_h)

    return map_h, map_w


def save_som(som, filename, hyperparameters=None):
    torch.save({"weights": som.weights.cpu(), "map_h": som.map_h, "map_w": som.map_w,
                "Ne": som.Ne, "geometry": som.geometry,
                "feature_weights": som.feature_weights.cpu(),
                "hyperparameters": hyperparameters}, filename)
    print(f"SOM saved to {filename}")

def load_som(filename, device="cpu"):
    s = torch.load(filename, map_location=device)
    som = SOM(s["map_h"], s["map_w"], s["Ne"], geometry=s["geometry"], device=device,
              feature_weights=s.get("feature_weights"))
    som.weights = s["weights"].to(device)
    print(f"SOM loaded from {filename}")
    return som

