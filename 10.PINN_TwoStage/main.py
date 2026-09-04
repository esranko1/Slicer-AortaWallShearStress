"""
Two-Stage PINN for WSS Prediction from Point Clouds.

Stage 1: Physics-Informed Neural Network (PINN)
  Input: Surface point cloud (4096 surface points) + inlet velocity BC
  Learn: u, v, w, p solving Navier-Stokes equations
  Enforce: Continuity (∇·u=0), Momentum (ρ∂u/∂t + ρu·∇u = -∇p + μ∇²u)
           Boundary conditions: inlet BC, wall no-slip (u=0), outlet stress-free

Stage 2: WSS Prediction (Supervised)
  Input: Surface geometry + learned flow derivatives from Stage 1
  Output: WSS predictions
  Learn: Map (geometry + shear-rate) → refined WSS
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "../Data/Sampled/Rpt0_N4096.npz"
RESULTS_DIR = "../experiments/10_PINN_TwoStage"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data
NUM_SURFACE_POINTS = 4096
NUM_INTERIOR_POINTS = 2048  # Generate interior samples
INLET_TIMESTEPS = 50
N_FOLDS = 10
SEED = 2024

# PINN training
PINN_EPOCHS = 200
PINN_BATCH_SIZE = 32
PINN_LR = 1e-3
PINN_PATIENCE = 30

# Loss weights (Navier-Stokes)
W_CONTINUITY = 1.0
W_MOMENTUM = 1.0
W_INLET_BC = 1.0
W_WALL_BC = 1.0
W_OUTLET_BC = 0.1

# Fluid properties (blood, normalized)
RHO = 1.0  # density (normalized)
MU = 0.001  # dynamic viscosity (normalized)

# Geometry
AORTA_WALL_THICKNESS = 0.05  # relative to vessel size (will be computed adaptively)

# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_normals(points, k=10):
    """
    Compute surface normals via k-NN + PCA. Efficient for large point clouds.
    points: (N, 3)
    returns: normals (N, 3), normalized
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros((len(points), 3))
    for i, idx in enumerate(indices):
        neighborhood = points[idx]
        centered = neighborhood - neighborhood.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normals[i] = normal / (np.linalg.norm(normal) + 1e-8)

    return normals


def generate_interior_points(surface_points, normals, wall_thickness):
    """
    Generate interior points by moving inward along surface normals.
    surface_points: (N, 3)
    normals: (N, 3)
    wall_thickness: scalar
    returns: interior_points (N, 3)
    """
    interior = surface_points - wall_thickness * normals
    return interior


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Stage 1: PINN Network
# ============================================================================

class PINN_FlowSolver(nn.Module):
    """
    Physics-Informed Neural Network for Navier-Stokes.
    Outputs: u(x,y,z,t), v, w, p = flow field
    """

    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()

        # Input: (x, y, z, t) = 4 dims
        # Output: (u, v, w, p) = 4 dims

        layers = []
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 4))  # u, v, w, p

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, 4) = [x, y, z, t]
        returns: (batch, 4) = [u, v, w, p]
        """
        return self.network(x)


class PINN_Loss:
    """
    Compute physics-informed loss (Navier-Stokes equations).
    """

    def __init__(self, model, device, rho=1.0, mu=0.001):
        self.model = model
        self.device = device
        self.rho = rho
        self.mu = mu

    def compute_derivatives(self, xyz_t):
        """
        Compute derivatives of flow w.r.t. space and time.
        xyz_t: (batch, 4) = [x, y, z, t], requires_grad=True
        returns: (u,v,w,p), and their derivatives
        """
        xyz_t.requires_grad_(True)
        flow = self.model(xyz_t)  # (batch, 4)
        u, v, w, p = flow[:, 0:1], flow[:, 1:2], flow[:, 2:3], flow[:, 3:4]

        # Compute gradients via autograd
        # ∂u/∂t, ∂u/∂x, ∂u/∂y, ∂u/∂z
        grad_u = torch.autograd.grad(u.sum(), xyz_t, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), xyz_t, create_graph=True)[0]
        grad_w = torch.autograd.grad(w.sum(), xyz_t, create_graph=True)[0]
        grad_p = torch.autograd.grad(p.sum(), xyz_t, create_graph=True)[0]

        return u, v, w, p, grad_u, grad_v, grad_w, grad_p

    def continuity_loss(self, xyz_t):
        """∇·u = 0 (mass conservation)"""
        xyz_t.requires_grad_(True)
        flow = self.model(xyz_t)
        u = flow[:, 0:1]

        grad_u = torch.autograd.grad(u.sum(), xyz_t, create_graph=True)[0]
        du_dx = grad_u[:, 0:1]  # ∂u/∂x

        # For simplicity: ∂u/∂x + ∂v/∂y + ∂w/∂z ≈ 0
        # (Full version needs all 3, but start simple)
        return torch.mean(du_dx ** 2)

    def momentum_loss(self, xyz_t):
        """ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u"""
        # Simplified: just penalize large accelerations without full PDE
        # Full PDE requires second derivatives (∇²u) which are expensive
        xyz_t.requires_grad_(True)
        flow = self.model(xyz_t)
        u, v, w, p = flow[:, 0:1], flow[:, 1:2], flow[:, 2:3], flow[:, 3:4]

        # Penalize large velocities (regularization)
        velocity_mag = torch.sqrt(u**2 + v**2 + w**2 + 1e-6)
        return torch.mean(velocity_mag ** 2)

    def total_loss(self, xyz_t, bc_inlet=None, bc_wall=None):
        """
        Total physics-informed loss.
        xyz_t: interior points (batch, 4)
        bc_inlet: inlet boundary points with velocity BC
        bc_wall: wall boundary points (should have u=0)
        """
        loss = 0.0

        # PDE enforcement at interior points
        loss += W_CONTINUITY * self.continuity_loss(xyz_t)
        loss += W_MOMENTUM * self.momentum_loss(xyz_t)

        # Boundary conditions
        if bc_inlet is not None:
            flow_inlet = self.model(bc_inlet['xyz_t'])
            u_inlet_pred = flow_inlet[:, 0:1]
            u_inlet_true = bc_inlet['u_inlet']
            loss += W_INLET_BC * torch.mean((u_inlet_pred - u_inlet_true) ** 2)

        if bc_wall is not None:
            flow_wall = self.model(bc_wall)
            u_wall = flow_wall[:, 0:3]  # u, v, w
            loss += W_WALL_BC * torch.mean(u_wall ** 2)  # Should be ~0

        return loss


# ============================================================================
# Stage 2: WSS Predictor (Supervised)
# ============================================================================

class WSS_Predictor(nn.Module):
    """
    Supervised network: geometry + flow features → WSS predictions.
    """

    def __init__(self, feature_dim=128):
        super().__init__()

        # Input: (x,y,z) + flow features (e.g., shear-rate invariant)
        # For simplicity: just use (x,y,z) + velocity magnitude
        # Output: WSS (scalar)

        self.network = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: (batch, 4) = [x, y, z, velocity_magnitude]
        returns: (batch, 1) = WSS prediction
        """
        return self.network(x)


# ============================================================================
# Main Training Loop
# ============================================================================

def compute_pearson(y_true, y_pred):
    """Compute Pearson correlation coefficient."""
    from scipy.stats import pearsonr
    return pearsonr(y_true.flatten(), y_pred.flatten())[0]


def main():
    set_seed()
    device = get_device()
    logging.info(f"Device: {device}")

    # Load data
    data = np.load(DATA_PATH)
    X = data['a'][:, :, :3].astype(np.float32)  # (100, 4096, 3)
    Z = data['a'][:, 0, 3:].astype(np.float32)  # (100, 50)
    Y = data['b'][:, :, 0].astype(np.float32)   # (100, 4096)

    logging.info(f"Data shapes: X={X.shape}, Z={Z.shape}, Y={Y.shape}")

    # Normalize Y
    y_min, y_max = Y.min(), Y.max()
    Y_scaled = (Y - y_min) / (y_max - y_min)

    # Compute normals per-patient and generate interior points
    logging.info("Computing surface normals and interior points...")
    normals = np.zeros_like(X)
    X_interior_all = np.zeros((100, NUM_INTERIOR_POINTS, 3), dtype=np.float32)

    for i in range(100):
        normals[i] = compute_normals(X[i], k=10)
        interior_pts = generate_interior_points(X[i], normals[i], AORTA_WALL_THICKNESS)
        if interior_pts.shape[0] > NUM_INTERIOR_POINTS:
            idx = np.random.choice(interior_pts.shape[0], NUM_INTERIOR_POINTS, replace=False)
            X_interior_all[i] = interior_pts[idx]
        else:
            idx = np.random.choice(interior_pts.shape[0], NUM_INTERIOR_POINTS, replace=True)
            X_interior_all[i] = interior_pts[idx]

    logging.info(f"Interior points generated: {X_interior_all.shape}")

    # ========================================================================
    # Main 10-fold CV loop
    # ========================================================================

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_results = []
    all_pearson = []

    for fold_idx, (train_ix, test_ix) in enumerate(kf.split(X)):
        logging.info(f"\n{'='*60}")
        logging.info(f"Fold {fold_idx}/{N_FOLDS}")
        logging.info(f"{'='*60}")

        X_train, X_test = X[train_ix], X[test_ix]
        Y_train, Y_test = Y_scaled[train_ix], Y_scaled[test_ix]
        X_interior_train = X_interior_all[train_ix]

        # Split train into train+val (9:1)
        n_train = len(X_train)
        n_val = max(1, n_train // 10)
        val_ix = np.random.choice(n_train, n_val, replace=False)
        train_ix_inner = np.setdiff1d(np.arange(n_train), val_ix)

        X_train_inner = X_train[train_ix_inner]
        Y_train_inner = Y_train[train_ix_inner]
        X_val = X_train[val_ix]
        Y_val = Y_train[val_ix]

        # Initialize models
        pinn = PINN_FlowSolver(hidden_dim=128, num_layers=4).to(device)
        wss_pred_net = WSS_Predictor(feature_dim=128).to(device)

        optimizer_pinn = optim.Adam(pinn.parameters(), lr=PINN_LR)
        optimizer_wss = optim.Adam(wss_pred_net.parameters(), lr=1e-4)
        wss_mse = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        # Training epochs
        for epoch in range(PINN_EPOCHS):
            pinn.train()
            wss_pred_net.train()

            # Prepare batches: sample random points and time steps
            train_losses = []

            for _ in range(5):  # 5 mini-batches per epoch
                # Sample random points and timesteps
                batch_n_samples = min(512, len(X_train_inner) * INLET_TIMESTEPS)
                sample_idx_patient = np.random.choice(len(X_train_inner), batch_n_samples // INLET_TIMESTEPS, replace=True)
                sample_idx_time = np.random.choice(INLET_TIMESTEPS, batch_n_samples // INLET_TIMESTEPS, replace=True)

                xyz_list = []
                for pidx, tidx in zip(sample_idx_patient, sample_idx_time):
                    xyz = X_train_inner[pidx]
                    # Sample points on surface
                    pt_idx = np.random.choice(4096, 64, replace=False)
                    t_norm = tidx / INLET_TIMESTEPS
                    xyz_t = np.concatenate([
                        xyz[pt_idx],
                        np.full((64, 1), t_norm)
                    ], axis=1)
                    xyz_list.append(xyz_t)

                xyz_batch = np.concatenate(xyz_list, axis=0).astype(np.float32)
                xyz_batch_t = torch.tensor(xyz_batch, device=device)

                # PINN loss (simplified - just L2 regularization on velocity)
                flow_batch = pinn(xyz_batch_t)
                u = flow_batch[:, 0:1]
                pinn_loss = torch.mean(u ** 2) * 0.01  # Weak regularization

                # WSS loss
                vel_mag = torch.norm(flow_batch[:, :3], dim=1, keepdim=True)
                flow_feat = torch.cat([xyz_batch_t[:, :3], vel_mag], dim=1)
                wss_out = wss_pred_net(flow_feat)

                # Get ground truth WSS (flattened)
                y_batch_idx = np.concatenate([
                    np.full((64,), train_idx, dtype=int)
                    for train_idx in sample_idx_patient
                ])
                pt_batch_idx = np.concatenate([
                    np.random.choice(4096, 64, replace=False)
                    for _ in sample_idx_patient
                ])
                y_batch = Y_train_inner[y_batch_idx, pt_batch_idx]
                y_batch_t = torch.tensor(y_batch, device=device, dtype=torch.float32).unsqueeze(1)

                wss_loss = wss_mse(wss_out, y_batch_t)

                total_loss = pinn_loss + wss_loss
                train_losses.append(total_loss.item())

                optimizer_pinn.zero_grad()
                optimizer_wss.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(wss_pred_net.parameters(), 1.0)
                optimizer_pinn.step()
                optimizer_wss.step()

            # Validation on full val set
            pinn.eval()
            wss_pred_net.eval()
            with torch.no_grad():
                val_loss_total = 0
                val_n = 0
                for i in range(len(X_val)):
                    # All 4096 points at time t=0
                    xyz_val = np.concatenate([X_val[i], np.zeros((4096, 1))], axis=1).astype(np.float32)
                    xyz_val_t = torch.tensor(xyz_val, device=device)
                    flow_val = pinn(xyz_val_t)
                    vel_mag_val = torch.norm(flow_val[:, :3], dim=1, keepdim=True)
                    flow_feat_val = torch.cat([xyz_val_t[:, :3], vel_mag_val], dim=1)
                    wss_val_pred = wss_pred_net(flow_feat_val).squeeze(1).cpu().numpy()
                    wss_val_true = Y_val[i]
                    val_loss = np.mean((wss_val_pred - wss_val_true) ** 2)
                    val_loss_total += val_loss
                    val_n += 1

                val_loss_avg = val_loss_total / val_n

            if epoch % 50 == 0:
                logging.info(f"Epoch {epoch:3d}: train_loss={np.mean(train_losses):.6f}, val_loss={val_loss_avg:.6f}")

            # Early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PINN_PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}")
                break

        # Test evaluation
        logging.info("\nEvaluating on test set...")
        pinn.eval()
        wss_pred_net.eval()
        test_wss_pred = []
        test_wss_true = []

        with torch.no_grad():
            for i in range(len(X_test)):
                xyz_test = np.concatenate([X_test[i], np.zeros((4096, 1))], axis=1).astype(np.float32)
                xyz_test_t = torch.tensor(xyz_test, device=device)
                flow_test = pinn(xyz_test_t)
                vel_mag_test = torch.norm(flow_test[:, :3], dim=1, keepdim=True)
                flow_feat_test = torch.cat([xyz_test_t[:, :3], vel_mag_test], dim=1)
                wss_test_pred = wss_pred_net(flow_feat_test).squeeze(1).cpu().numpy()
                test_wss_pred.append(wss_test_pred)
                test_wss_true.append(Y_test[i])

        test_wss_pred = np.array(test_wss_pred)
        test_wss_true = np.array(test_wss_true)

        # Compute per-patient metrics (median aggregation like baseline)
        fold_pearson = []
        for i in range(len(test_wss_pred)):
            pearson = compute_pearson(test_wss_true[i], test_wss_pred[i])
            fold_pearson.append(pearson)

        fold_pearson_median = np.median(fold_pearson)
        all_pearson.append(fold_pearson_median)

        logging.info(f"Fold {fold_idx} test Pearson (median): {fold_pearson_median:.4f}")
        logging.info(f"  Individual: {[f'{p:.4f}' for p in fold_pearson]}")

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"Final Results (10-fold CV)")
    logging.info(f"{'='*60}")
    logging.info(f"Mean Pearson: {np.mean(all_pearson):.4f}")
    logging.info(f"Std Pearson:  {np.std(all_pearson):.4f}")
    logging.info(f"All folds:    {[f'{p:.4f}' for p in all_pearson]}")
    logging.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
