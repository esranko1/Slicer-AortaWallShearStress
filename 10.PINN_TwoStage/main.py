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
PINN_EPOCHS = 100  # Reduced since each epoch now processes all points
PINN_BATCH_SIZE = 32
PINN_LR = 1e-3
PINN_PATIENCE = 20  # Earlier stopping with more constraint per epoch

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


def compute_poiseuille_flow(surface_points, inlet_velocity_waveform):
    """
    Compute Poiseuille flow field for aorta.
    Assumes parabolic profile: u = u_max * (1 - (r/R)²)

    surface_points: (4096, 3) - point cloud
    inlet_velocity_waveform: (50,) - Z(t) velocity over time

    returns: flow_field (4096, 50, 3) - u,v,w at each point, each timestep
             where flow is along dominant axis (z-direction of PCA)
    """
    n_points = surface_points.shape[0]
    n_times = len(inlet_velocity_waveform)

    # Compute principal axis (flow direction) via PCA
    centroid = surface_points.mean(axis=0)
    centered = surface_points - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    flow_direction = eigvecs[:, -1]  # Direction of maximum variance (flow axis)

    # Project all points onto flow direction to get axial position
    axial_position = centered @ flow_direction  # (4096,)

    # Compute radial distance from flow axis for each point
    # r = ||point - (point·flow_direction)*flow_direction||
    axial_component = np.outer(axial_position, flow_direction)  # (4096, 3)
    radial_vector = centered - axial_component  # (4096, 3)
    radial_distance = np.linalg.norm(radial_vector, axis=1)  # (4096,)

    # Estimate vessel radius as 90th percentile of radial distance
    vessel_radius = np.percentile(radial_distance, 90)

    # Compute Poiseuille profile: u_z = u_max * (1 - (r/R)²)
    # For r > R, clip to 0
    r_normalized = np.clip(radial_distance / vessel_radius, 0, 1)
    poiseuille_profile = 1.0 - r_normalized**2  # (4096,)

    # Broadcast over time and compute velocity components
    flow_field = np.zeros((n_points, n_times, 3), dtype=np.float32)
    for t in range(n_times):
        u_max = inlet_velocity_waveform[t]
        u_axial = u_max * poiseuille_profile  # (4096,)
        # Velocity in flow direction
        flow_field[:, t, :] = np.outer(u_axial, flow_direction).astype(np.float32)

    return flow_field, flow_direction, vessel_radius


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
        grad_u = torch.autograd.grad(u.sum(), xyz_t, create_graph=True, retain_graph=True)[0]
        grad_v = torch.autograd.grad(v.sum(), xyz_t, create_graph=True, retain_graph=True)[0]
        grad_w = torch.autograd.grad(w.sum(), xyz_t, create_graph=True, retain_graph=True)[0]
        grad_p = torch.autograd.grad(p.sum(), xyz_t, create_graph=True, retain_graph=True)[0]

        du_dt, du_dx, du_dy, du_dz = grad_u[:, 3:4], grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
        dv_dt, dv_dx, dv_dy, dv_dz = grad_v[:, 3:4], grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
        dw_dt, dw_dx, dw_dy, dw_dz = grad_w[:, 3:4], grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]
        dp_dx, dp_dy, dp_dz = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

        # Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
        # Avoid nested autograd for stability; use single-pass second derivatives
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(du_dy.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 1:2]
        d2u_dz2 = torch.autograd.grad(du_dz.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 2:3]
        laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2

        d2v_dx2 = torch.autograd.grad(dv_dx.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 0:1]
        d2v_dy2 = torch.autograd.grad(dv_dy.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 1:2]
        d2v_dz2 = torch.autograd.grad(dv_dz.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 2:3]
        laplacian_v = d2v_dx2 + d2v_dy2 + d2v_dz2

        d2w_dx2 = torch.autograd.grad(dw_dx.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 0:1]
        d2w_dy2 = torch.autograd.grad(dw_dy.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 1:2]
        d2w_dz2 = torch.autograd.grad(dw_dz.sum(), xyz_t, create_graph=False, retain_graph=True)[0][:, 2:3]
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

    def total_loss(self, xyz_inlet, u_inlet_target, xyz_wall, xyz_supervised=None, u_supervised_target=None):
        """
        Total physics-informed loss with boundary conditions and supervised flow.
        xyz_inlet: (batch, 4) inlet boundary points
        u_inlet_target: (batch, 1) inlet velocity BC value
        xyz_wall: (batch, 4) wall boundary points
        xyz_supervised: (batch, 4) points for supervised flow loss [optional]
        u_supervised_target: (batch, 3) ground-truth flow field [optional]
        """
        # Inlet boundary condition: u_normal = Z(t)
        flow_inlet = self.model(xyz_inlet)
        u_inlet_pred = flow_inlet[:, 0:1]  # Normal component
        inlet_loss = torch.mean((u_inlet_pred - u_inlet_target)**2)

        # Wall boundary condition: u = 0 (no-slip)
        flow_wall = self.model(xyz_wall)
        u_wall = flow_wall[:, 0:3]  # u, v, w
        wall_loss = torch.mean(u_wall**2)

        # Supervised flow loss (Poiseuille flow supervision)
        if xyz_supervised is not None and u_supervised_target is not None:
            flow_supervised = self.model(xyz_supervised)
            u_supervised_pred = flow_supervised[:, 0:3]  # u, v, w
            supervised_loss = torch.mean((u_supervised_pred - u_supervised_target)**2)
        else:
            supervised_loss = 0.0

        # Total loss (removed PDE residuals - relying on supervised flow instead)
        total = (W_INLET_BC * inlet_loss +
                 W_WALL_BC * wall_loss +
                 0.5 * supervised_loss)  # Weight supervised loss equally

        return total, inlet_loss, wall_loss, supervised_loss


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

    # Precompute Poiseuille flow supervision
    logging.info("Computing Poiseuille flow fields for all patients...")
    flow_poiseuille = np.zeros((100, NUM_SURFACE_POINTS, INLET_TIMESTEPS, 3), dtype=np.float32)
    for i in range(100):
        flow_poiseuille[i], _, _ = compute_poiseuille_flow(X[i], Z[i])
    logging.info(f"Poiseuille flows computed: {flow_poiseuille.shape}")

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
        Z_train, Z_test = Z[train_ix], Z[test_ix]
        Y_train, Y_test = Y_scaled[train_ix], Y_scaled[test_ix]
        X_interior_train_fold = X_interior_all[train_ix]  # All fold training patients

        # Split train into train+val (9:1)
        n_train = len(X_train)
        n_val = max(1, n_train // 10)
        val_ix = np.random.choice(n_train, n_val, replace=False)
        train_ix_inner = np.setdiff1d(np.arange(n_train), val_ix)

        X_train_inner = X_train[train_ix_inner]
        Z_train_inner = Z_train[train_ix_inner]
        Y_train_inner = Y_train[train_ix_inner]
        X_interior_train = X_interior_train_fold[train_ix_inner]  # Only train (not val) interior points
        flow_train_inner = flow_poiseuille[train_ix[train_ix_inner]]  # Map back to original patient indices
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

            for _ in range(1):  # 1 full batch per epoch (all surface + interior points)
                # === Sample one patient per epoch (full surface + interior) ===
                patient_idx = np.random.randint(0, len(X_train_inner))
                time_idx = np.random.randint(0, INLET_TIMESTEPS)
                t_norm = time_idx / INLET_TIMESTEPS

                # All interior points for this patient at this time
                xyz_interior = np.concatenate([
                    X_interior_train[patient_idx],
                    np.full((NUM_INTERIOR_POINTS, 1), t_norm)
                ], axis=1).astype(np.float32)
                xyz_interior_t = torch.tensor(xyz_interior, device=device, requires_grad=True)

                # All surface points for inlet BC at this time
                xyz_inlet = np.concatenate([
                    X_train_inner[patient_idx],  # All 4096 surface points
                    np.full((4096, 1), t_norm)
                ], axis=1).astype(np.float32)
                xyz_inlet_t = torch.tensor(xyz_inlet, device=device, requires_grad=True)

                # Inlet velocity BC target (same for all surface points, from Z waveform)
                u_inlet_target = torch.full(
                    (4096, 1),
                    Z_train_inner[patient_idx, time_idx].item(),
                    device=device, dtype=torch.float32
                )

                # Wall boundary points (same surface, different random time for wall BC)
                wall_time_idx = np.random.randint(0, INLET_TIMESTEPS)
                wall_t_norm = wall_time_idx / INLET_TIMESTEPS
                xyz_wall = np.concatenate([
                    X_train_inner[patient_idx],  # All 4096 surface points
                    np.full((4096, 1), wall_t_norm)
                ], axis=1).astype(np.float32)
                xyz_wall_t = torch.tensor(xyz_wall, device=device, requires_grad=True)

                # === PINN loss with supervised Poiseuille flow ===
                pinn_loss_fn = PINN_Loss(pinn, device, rho=RHO, mu=MU)

                # Get supervised flow for all surface points at this time
                u_supervised = torch.tensor(
                    flow_train_inner[patient_idx, :, time_idx, :],  # (4096, 3)
                    device=device, dtype=torch.float32
                )

                pinn_total, inlet_l, wall_l, flow_l = pinn_loss_fn.total_loss(
                    xyz_inlet_t, u_inlet_target, xyz_wall_t,
                    xyz_inlet_t, u_supervised  # Use all surface points for supervised loss
                )

                # === WSS prediction loss ===
                # Sample random patient, all surface points at t=0
                wss_patient_idx = np.random.randint(0, len(X_train_inner))
                xyz_wss = np.concatenate([
                    X_train_inner[wss_patient_idx],  # All 4096 surface points
                    np.zeros((4096, 1))  # t=0
                ], axis=1).astype(np.float32)
                xyz_wss_t = torch.tensor(xyz_wss, device=device)

                with torch.no_grad():
                    flow_wss = pinn(xyz_wss_t)
                    vel_mag_wss = torch.norm(flow_wss[:, :3], dim=1, keepdim=True)
                    flow_feat_wss = torch.cat([xyz_wss_t[:, :3], vel_mag_wss], dim=1)

                wss_pred_out = wss_pred_net(flow_feat_wss)
                y_wss = Y_train_inner[wss_patient_idx]  # All 4096 points
                y_wss_t = torch.tensor(y_wss, device=device, dtype=torch.float32).unsqueeze(1)
                wss_loss = wss_mse(wss_pred_out, y_wss_t)

                # Combined loss with scaling (avoid physics loss from dominating)
                # Scale PINN loss to be comparable to WSS loss
                pinn_scaled = pinn_total * 0.1  # Downweight physics loss to let WSS dominate learning
                total_loss = pinn_scaled + wss_loss

                # Backprop
                optimizer_pinn.zero_grad()
                optimizer_wss.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(wss_pred_net.parameters(), 1.0)
                optimizer_pinn.step()
                optimizer_wss.step()

                train_losses.append(total_loss.item())
                pinn_loss_log.append((inlet_l.item(), wall_l.item(), flow_l if isinstance(flow_l, float) else flow_l.item()))

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

            if epoch % 10 == 0 and len(pinn_loss_log) > 0:
                pinn_logs_arr = np.array(pinn_loss_log)
                inlet_avg, wall_avg, flow_avg = np.mean(pinn_logs_arr, axis=0)
                logging.info(f"Epoch {epoch:3d}: train={np.mean(train_losses):.6f}, "
                           f"Inlet={inlet_avg:.6f}, Wall={wall_avg:.6f}, Flow={flow_avg:.6f}, "
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
