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
    Physics-Informed Neural Network for Navier-Stokes with Fourier features.
    Outputs: u(x,y,z,t), v, w, p = flow field
    """

    def __init__(self, hidden_dim=128, num_layers=4, fourier_features=32):
        super().__init__()

        self.fourier_features = fourier_features
        # Fourier feature matrix for positional encoding
        self.register_buffer(
            'fourier_matrix',
            torch.randn(4, fourier_features) * np.pi
        )

        # Input: 4 (xyz,t) + 2*fourier_features (sin/cos of Fourier features)
        input_dim = 4 + 2 * fourier_features

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
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
        # Fourier feature encoding
        fourier_x = x @ self.fourier_matrix  # (batch, fourier_features)
        fourier_feat = torch.cat([
            torch.sin(fourier_x),
            torch.cos(fourier_x)
        ], dim=1)  # (batch, 2*fourier_features)

        x_encoded = torch.cat([x, fourier_feat], dim=1)
        return self.network(x_encoded)


class PINN_Loss:
    """
    Full Navier-Stokes loss with second derivatives for viscous term ∇²u.
    """

    def __init__(self, model, device, rho=1.0, mu=0.001):
        self.model = model
        self.device = device
        self.rho = rho
        self.mu = mu

    def navier_stokes_loss(self, xyz_t):
        """
        Enforce full Navier-Stokes: ρ(∂u/∂t + u·∇u) + ∇p - μ∇²u = 0
        Computes residual at interior points.
        xyz_t: (batch, 4) = [x, y, z, t], requires_grad=True
        """
        xyz_t.requires_grad_(True)
        flow = self.model(xyz_t)
        u, v, w, p = flow[:, 0:1], flow[:, 1:2], flow[:, 2:3], flow[:, 3:4]

        # First derivatives w.r.t. space and time
        grad_u = torch.autograd.grad(u.sum(), xyz_t, create_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), xyz_t, create_graph=True)[0]
        grad_w = torch.autograd.grad(w.sum(), xyz_t, create_graph=True)[0]
        grad_p = torch.autograd.grad(p.sum(), xyz_t, create_graph=True)[0]

        du_dt, du_dx, du_dy, du_dz = grad_u[:, 3:4], grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
        dv_dt, dv_dx, dv_dy, dv_dz = grad_v[:, 3:4], grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
        dw_dt, dw_dx, dw_dy, dw_dz = grad_w[:, 3:4], grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]
        dp_dx, dp_dy, dp_dz = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

        # Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), xyz_t, create_graph=True)[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(du_dy.sum(), xyz_t, create_graph=True)[0][:, 1:2]
        d2u_dz2 = torch.autograd.grad(du_dz.sum(), xyz_t, create_graph=True)[0][:, 2:3]
        laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2

        d2v_dx2 = torch.autograd.grad(dv_dx.sum(), xyz_t, create_graph=True)[0][:, 0:1]
        d2v_dy2 = torch.autograd.grad(dv_dy.sum(), xyz_t, create_graph=True)[0][:, 1:2]
        d2v_dz2 = torch.autograd.grad(dv_dz.sum(), xyz_t, create_graph=True)[0][:, 2:3]
        laplacian_v = d2v_dx2 + d2v_dy2 + d2v_dz2

        d2w_dx2 = torch.autograd.grad(dw_dx.sum(), xyz_t, create_graph=True)[0][:, 0:1]
        d2w_dy2 = torch.autograd.grad(dw_dy.sum(), xyz_t, create_graph=True)[0][:, 1:2]
        d2w_dz2 = torch.autograd.grad(dw_dz.sum(), xyz_t, create_graph=True)[0][:, 2:3]
        laplacian_w = d2w_dx2 + d2w_dy2 + d2w_dz2

        # Convection terms: u·∇u
        convection_u = u * du_dx + v * du_dy + w * du_dz
        convection_v = u * dv_dx + v * dv_dy + w * dv_dz
        convection_w = u * dw_dx + v * dw_dy + w * dw_dz

        # Momentum residuals
        momentum_u = self.rho * (du_dt + convection_u) + dp_dx - self.mu * laplacian_u
        momentum_v = self.rho * (dv_dt + convection_v) + dp_dy - self.mu * laplacian_v
        momentum_w = self.rho * (dw_dt + convection_w) + dp_dz - self.mu * laplacian_w

        # Continuity residual: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
        continuity = du_dx + dv_dy + dw_dz

        # L2 norm of residuals
        ns_loss = torch.mean(momentum_u**2 + momentum_v**2 + momentum_w**2)
        cont_loss = torch.mean(continuity**2)

        return ns_loss, cont_loss

    def total_loss(self, xyz_interior, xyz_inlet, u_inlet_target, xyz_wall):
        """
        Total physics-informed loss with boundary conditions.
        xyz_interior: (batch, 4) interior points
        xyz_inlet: (batch, 4) inlet boundary points
        u_inlet_target: (batch, 1) inlet velocity BC value
        xyz_wall: (batch, 4) wall boundary points
        """
        # PDE residuals at interior
        ns_loss, cont_loss = self.navier_stokes_loss(xyz_interior)

        # Inlet boundary condition: u_normal = Z(t)
        flow_inlet = self.model(xyz_inlet)
        u_inlet_pred = flow_inlet[:, 0:1]  # Normal component
        inlet_loss = torch.mean((u_inlet_pred - u_inlet_target)**2)

        # Wall boundary condition: u = 0 (no-slip)
        flow_wall = self.model(xyz_wall)
        u_wall = flow_wall[:, 0:3]  # u, v, w
        wall_loss = torch.mean(u_wall**2)

        # Total loss
        total = (W_CONTINUITY * cont_loss +
                 W_MOMENTUM * ns_loss +
                 W_INLET_BC * inlet_loss +
                 W_WALL_BC * wall_loss)

        return total, ns_loss, cont_loss, inlet_loss, wall_loss


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
        Z_train_inner = Z_train[train_ix_inner]
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

            train_losses = []
            pinn_loss_log = []

            for _ in range(3):  # 3 mini-batches per epoch (reduced due to autodiff cost)
                # === Sample PINN training points ===
                n_interior = 128
                n_boundary = 64

                # Interior points (random points from interior + random times)
                interior_pt_idx = np.random.choice(NUM_INTERIOR_POINTS, n_interior, replace=True)
                interior_patient_idx = np.random.choice(len(X_train_inner), n_interior, replace=True)
                interior_time_idx = np.random.choice(INLET_TIMESTEPS, n_interior, replace=True)

                xyz_interior_list = []
                for pidx, ptidx, tidx in zip(interior_patient_idx, interior_pt_idx, interior_time_idx):
                    pt = X_interior_train[pidx, ptidx]
                    t_norm = tidx / INLET_TIMESTEPS
                    xyz_t = np.concatenate([pt, [t_norm]])
                    xyz_interior_list.append(xyz_t)
                xyz_interior = np.array(xyz_interior_list, dtype=np.float32)
                xyz_interior_t = torch.tensor(xyz_interior, device=device, requires_grad=True)

                # Inlet boundary points (surface + inlet times)
                inlet_pt_idx = np.random.choice(4096, n_boundary, replace=True)
                inlet_patient_idx = np.random.choice(len(X_train_inner), n_boundary, replace=True)
                inlet_time_idx = np.random.choice(INLET_TIMESTEPS, n_boundary, replace=True)

                xyz_inlet_list = []
                for pidx, ptidx, tidx in zip(inlet_patient_idx, inlet_pt_idx, inlet_time_idx):
                    pt = X_train_inner[pidx, ptidx]
                    t_norm = tidx / INLET_TIMESTEPS
                    xyz_t = np.concatenate([pt, [t_norm]])
                    xyz_inlet_list.append(xyz_t)
                xyz_inlet = np.array(xyz_inlet_list, dtype=np.float32)
                xyz_inlet_t = torch.tensor(xyz_inlet, device=device, requires_grad=True)

                # Inlet velocity BC target (from Z waveform)
                u_inlet_target = torch.tensor(
                    Z_train_inner[inlet_patient_idx, inlet_time_idx],
                    device=device, dtype=torch.float32
                ).unsqueeze(1)

                # Wall boundary points
                wall_pt_idx = np.random.choice(4096, n_boundary, replace=True)
                wall_patient_idx = np.random.choice(len(X_train_inner), n_boundary, replace=True)
                wall_time_idx = np.random.choice(INLET_TIMESTEPS, n_boundary, replace=True)

                xyz_wall_list = []
                for pidx, ptidx, tidx in zip(wall_patient_idx, wall_pt_idx, wall_time_idx):
                    pt = X_train_inner[pidx, ptidx]
                    t_norm = tidx / INLET_TIMESTEPS
                    xyz_t = np.concatenate([pt, [t_norm]])
                    xyz_wall_list.append(xyz_t)
                xyz_wall = np.array(xyz_wall_list, dtype=np.float32)
                xyz_wall_t = torch.tensor(xyz_wall, device=device, requires_grad=True)

                # === PINN loss ===
                pinn_loss_fn = PINN_Loss(pinn, device, rho=RHO, mu=MU)
                pinn_total, ns_l, cont_l, inlet_l, wall_l = pinn_loss_fn.total_loss(
                    xyz_interior_t, xyz_inlet_t, u_inlet_target, xyz_wall_t
                )

                # === WSS prediction loss ===
                # Sample surface points at t=0 for WSS prediction
                wss_pt_idx = np.random.choice(4096, 128, replace=True)
                wss_patient_idx = np.random.choice(len(X_train_inner), 128, replace=True)

                xyz_wss_list = []
                for pidx, ptidx in zip(wss_patient_idx, wss_pt_idx):
                    pt = X_train_inner[pidx, ptidx]
                    xyz_t = np.concatenate([pt, [0.0]])  # t=0
                    xyz_wss_list.append(xyz_t)
                xyz_wss = np.array(xyz_wss_list, dtype=np.float32)
                xyz_wss_t = torch.tensor(xyz_wss, device=device)

                with torch.no_grad():
                    flow_wss = pinn(xyz_wss_t)
                    vel_mag_wss = torch.norm(flow_wss[:, :3], dim=1, keepdim=True)
                    flow_feat_wss = torch.cat([xyz_wss_t[:, :3], vel_mag_wss], dim=1)

                wss_pred_out = wss_pred_net(flow_feat_wss)
                y_wss = Y_train_inner[wss_patient_idx, wss_pt_idx]
                y_wss_t = torch.tensor(y_wss, device=device, dtype=torch.float32).unsqueeze(1)
                wss_loss = wss_mse(wss_pred_out, y_wss_t)

                # Combined loss
                total_loss = pinn_total + wss_loss

                # Backprop
                optimizer_pinn.zero_grad()
                optimizer_wss.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(wss_pred_net.parameters(), 1.0)
                optimizer_pinn.step()
                optimizer_wss.step()

                train_losses.append(total_loss.item())
                pinn_loss_log.append((ns_l.item(), cont_l.item(), inlet_l.item(), wall_l.item()))

            # Validation on val set (sample to save time)
            pinn.eval()
            wss_pred_net.eval()
            with torch.no_grad():
                val_wss_loss = 0
                val_n = min(len(X_val), 5)  # Sample max 5 patients for speed
                val_idx = np.random.choice(len(X_val), val_n, replace=False)
                for i in val_idx:
                    # Sample points at t=0
                    pt_idx = np.random.choice(4096, 256, replace=False)
                    xyz_val = np.concatenate([X_val[i, pt_idx], np.zeros((256, 1))], axis=1).astype(np.float32)
                    xyz_val_t = torch.tensor(xyz_val, device=device)
                    flow_val = pinn(xyz_val_t)
                    vel_mag_val = torch.norm(flow_val[:, :3], dim=1, keepdim=True)
                    flow_feat_val = torch.cat([xyz_val_t[:, :3], vel_mag_val], dim=1)
                    wss_val_pred = wss_pred_net(flow_feat_val).squeeze(1).cpu().numpy()
                    wss_val_true = Y_val[i, pt_idx]
                    val_wss_loss += np.mean((wss_val_pred - wss_val_true) ** 2)

                val_loss_avg = val_wss_loss / len(val_idx)

            if epoch % 25 == 0:
                ns_avg, cont_avg, inlet_avg, wall_avg = np.mean(pinn_loss_log, axis=0)
                logging.info(f"Epoch {epoch:3d}: train={np.mean(train_losses):.6f}, "
                           f"NS={ns_avg:.6f}, Cont={cont_avg:.6f}, Inlet={inlet_avg:.6f}, Wall={wall_avg:.6f}, "
                           f"val_WSS={val_loss_avg:.6f}")

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
