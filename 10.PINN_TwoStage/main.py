"""
Two-Stage PINN for WSS Prediction from Point Clouds.

Stage 1: Physics-Informed Neural Network (PINN)
  Input: Surface point cloud (4096 surface points) + inlet velocity BC
  Learn: u, v, w, p solving Navier-Stokes equations
  Enforce: Continuity (∇·u=0), Momentum (ρ∂u/∂t + ρu·∇u = -∇p + μ∇²u)
           Boundary conditions: wall no-slip (u=0) on the surface,
           Poiseuille-flow supervision + NS residual in the interior

Stage 2: WSS Prediction (Supervised)
  Input: Surface geometry + wall shear rate (∂u/∂n) from Stage 1
  Output: WSS predictions
  Learn: Map (geometry + shear-rate) → refined WSS
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
NUM_INTERIOR_POINTS = 2048  # Interior samples, spanning wall -> lumen centerline
NS_SUBSAMPLE = 512  # Interior points used per step for the 2nd-derivative NS residual (cost control)
INLET_TIMESTEPS = 50
N_FOLDS = 10
SEED = 2024

# PINN training
PINN_EPOCHS = 100
PINN_STEPS_PER_EPOCH = 8  # Multiple patient samples per epoch so a fold's ~90 patients get real coverage
PINN_LR = 1e-3
PINN_PATIENCE = 20

# Loss weights (Navier-Stokes)
W_WALL_BC = 1.0        # No-slip at the true wall surface
W_CONTINUITY = 1.0     # ∇·u = 0, enforced in the interior
W_MOMENTUM = 1.0       # Navier-Stokes momentum residual, enforced in the interior
W_SUPERVISED = 1.0     # Poiseuille-flow supervision, enforced in the interior

# Fluid properties (blood, normalized)
RHO = 1.0  # density (normalized)
MU = 0.001  # dynamic viscosity (normalized)

# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_normals(points, k=10):
    """
    Compute surface normals via k-NN + PCA. Efficient for large point clouds.

    The raw PCA eigenvector has an arbitrary sign per point (not consistent
    across the surface), so normals are oriented outward from the point
    cloud's centroid. This matters for generate_interior_points, which needs
    a reliable "inward" direction to land points inside the lumen rather
    than outside it; compute_wall_shear_feature is unaffected since it only
    uses the squared normal-derivative magnitude.

    points: (N, 3)
    returns: normals (N, 3), normalized, outward-oriented
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    _, indices = nbrs.kneighbors(points)

    centroid = points.mean(axis=0)
    normals = np.zeros((len(points), 3))
    for i, idx in enumerate(indices):
        neighborhood = points[idx]
        centered = neighborhood - neighborhood.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        if np.dot(normal, points[i] - centroid) < 0:
            normal = -normal
        normals[i] = normal

    return normals


def compute_flow_geometry(surface_points):
    """
    PCA-based estimate of the vessel's flow axis, shared by the interior
    point sampler and the Poiseuille supervision target below.

    surface_points: (N, 3)
    returns: flow_direction (3,), axial_position (N,), radial_distance (N,),
             vessel_radius (scalar)
    """
    centroid = surface_points.mean(axis=0)
    centered = surface_points - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    flow_direction = eigvecs[:, -1]  # Direction of maximum variance (flow axis)

    axial_position = centered @ flow_direction
    axial_component = np.outer(axial_position, flow_direction)
    radial_vector = centered - axial_component
    radial_distance = np.linalg.norm(radial_vector, axis=1)
    vessel_radius = np.percentile(radial_distance, 90)

    return flow_direction, axial_position, radial_distance, vessel_radius


def generate_interior_points(surface_points, normals, radial_distance):
    """
    Sample points inside the vessel lumen by moving inward from each surface
    (wall) point along its normal, by a random depth up to that point's own
    distance to the flow axis. This spans the full radial range from just
    under the wall to near the centerline, instead of a thin near-wall shell
    (where the Poiseuille profile is ~0 and physics is barely constrained).

    surface_points: (N, 3)
    normals: (N, 3)
    radial_distance: (N,) distance of each surface point to the flow axis
    returns: interior_points (N, 3), interior_radial_distance (N,)
    """
    depth_fraction = np.random.uniform(0.05, 0.95, size=len(surface_points))
    depth = depth_fraction * radial_distance
    interior = surface_points - depth[:, None] * normals
    interior_radial_distance = radial_distance * (1.0 - depth_fraction)
    return interior.astype(np.float32), interior_radial_distance.astype(np.float32)


def compute_poiseuille_profile(radial_distance, vessel_radius, flow_direction, inlet_velocity_waveform):
    """
    Poiseuille flow target for a set of points at known radial distance from
    the flow axis: u = u_max * (1 - (r/R)^2), directed along the flow axis.

    radial_distance: (N,)
    vessel_radius: scalar
    flow_direction: (3,)
    inlet_velocity_waveform: (T,) - Z(t) velocity over time
    returns: flow_field (N, T, 3)
    """
    n_points = radial_distance.shape[0]
    n_times = len(inlet_velocity_waveform)

    r_normalized = np.clip(radial_distance / vessel_radius, 0, 1)
    poiseuille_profile = 1.0 - r_normalized**2

    flow_field = np.zeros((n_points, n_times, 3), dtype=np.float32)
    for t in range(n_times):
        u_axial = inlet_velocity_waveform[t] * poiseuille_profile
        flow_field[:, t, :] = np.outer(u_axial, flow_direction).astype(np.float32)

    return flow_field


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


def compute_wall_shear_feature(model, xyz_t, wall_normals):
    """
    Approximate wall shear rate |dU/dn| via autograd: the wall-normal
    derivative of the velocity field. WSS is physically proportional to this
    (tau_w = mu * du_tangential/dn); raw velocity magnitude AT the wall is
    not usable as a feature since no-slip drives it to ~0 by construction.

    Must be called outside any torch.no_grad() context (needs a live graph
    to differentiate w.r.t. the input coordinates). The returned feature is
    detached, so it does not backprop into the PINN.

    xyz_t: (batch, 4) = [x, y, z, t]
    wall_normals: (batch, 3)
    returns: (batch, 1) shear-rate magnitude
    """
    xyz_t = xyz_t.clone().requires_grad_(True)
    flow = model(xyz_t)
    u, v, w = flow[:, 0:1], flow[:, 1:2], flow[:, 2:3]

    grad_u = torch.autograd.grad(u.sum(), xyz_t, retain_graph=True)[0][:, :3]
    grad_v = torch.autograd.grad(v.sum(), xyz_t, retain_graph=True)[0][:, :3]
    grad_w = torch.autograd.grad(w.sum(), xyz_t, retain_graph=True)[0][:, :3]

    dudn = (grad_u * wall_normals).sum(dim=1, keepdim=True)
    dvdn = (grad_v * wall_normals).sum(dim=1, keepdim=True)
    dwdn = (grad_w * wall_normals).sum(dim=1, keepdim=True)

    shear_rate = torch.sqrt(dudn**2 + dvdn**2 + dwdn**2 + 1e-12)
    return shear_rate.detach()


class PINN_Loss:
    """
    Wall no-slip on the true surface, Navier-Stokes residual + Poiseuille
    supervision in the interior.
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
        xyz_t: (batch, 4) = [x, y, z, t]
        returns: ns_loss, cont_loss, uvw (batch, 3) - both differentiable
                 w.r.t. model parameters, for reuse by the caller.
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
        # create_graph=True is required here: otherwise these 2nd-derivative
        # tensors are detached from the graph and the viscous term carries
        # zero gradient back to the network (silently ignored by backward()).
        d2u_dx2 = torch.autograd.grad(du_dx.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(du_dy.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 1:2]
        d2u_dz2 = torch.autograd.grad(du_dz.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 2:3]
        laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2

        d2v_dx2 = torch.autograd.grad(dv_dx.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 0:1]
        d2v_dy2 = torch.autograd.grad(dv_dy.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 1:2]
        d2v_dz2 = torch.autograd.grad(dv_dz.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 2:3]
        laplacian_v = d2v_dx2 + d2v_dy2 + d2v_dz2

        d2w_dx2 = torch.autograd.grad(dw_dx.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 0:1]
        d2w_dy2 = torch.autograd.grad(dw_dy.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 1:2]
        d2w_dz2 = torch.autograd.grad(dw_dz.sum(), xyz_t, create_graph=True, retain_graph=True)[0][:, 2:3]
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

        ns_loss = torch.mean(momentum_u**2 + momentum_v**2 + momentum_w**2)
        cont_loss = torch.mean(continuity**2)

        return ns_loss, cont_loss, torch.cat([u, v, w], dim=1)

    def total_loss(self, xyz_wall, xyz_interior, u_supervised_target):
        """
        xyz_wall: (batch, 4) - all surface points; the vessel wall, so
                  no-slip (u=0) applies everywhere here.
        xyz_interior: (batch, 4) - points inside the lumen; NS residual and
                  Poiseuille supervision both apply here.
        u_supervised_target: (batch, 3) - Poiseuille flow target at xyz_interior
        """
        flow_wall = self.model(xyz_wall)
        wall_loss = torch.mean(flow_wall[:, 0:3]**2)

        ns_loss, cont_loss, uvw_interior = self.navier_stokes_loss(xyz_interior)
        supervised_loss = torch.mean((uvw_interior - u_supervised_target)**2)

        total = (W_WALL_BC * wall_loss +
                 W_CONTINUITY * cont_loss +
                 W_MOMENTUM * ns_loss +
                 W_SUPERVISED * supervised_loss)

        return total, wall_loss, cont_loss, ns_loss, supervised_loss


# ============================================================================
# Stage 2: WSS Predictor (Supervised)
# ============================================================================

class WSS_Predictor(nn.Module):
    """
    Supervised network: geometry + flow features → WSS predictions.
    """

    def __init__(self, feature_dim=128):
        super().__init__()

        # Input: (x, y, z, wall shear rate)
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
        x: (batch, 4) = [x, y, z, wall_shear_rate]
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

    n_patients = X.shape[0]

    # Center each patient at their own centroid (raw scanner coordinates
    # carry an arbitrary absolute offset that doesn't generalize across
    # patients) and scale everyone by ONE global constant (median patient
    # extent) so coordinates land in a roughly O(1) range - required for
    # the Fourier-feature positional encoding (sin/cos of raw, unscaled
    # coordinates aliases into near-noise) while still preserving genuine
    # cross-patient size differences, which correlate with WSS.
    X_centroids = X.mean(axis=1, keepdims=True)  # (n_patients, 1, 3)
    X_centered = X - X_centroids
    patient_extent = np.linalg.norm(X_centered, axis=2).max(axis=1)  # (n_patients,)
    coord_scale = np.median(patient_extent) + 1e-8
    X = (X_centered / coord_scale).astype(np.float32)
    logging.info(f"Coordinates centered per-patient, scaled by global median extent={coord_scale:.4f}")

    # Normalize the inlet velocity waveform to O(1) too: raw Z drove the
    # Poiseuille supervision target (Flow loss) into the hundreds while
    # Wall/Cont/NS losses sat around 0.01-1, so it dominated and
    # destabilized the shared PINN's training.
    z_scale = np.max(np.abs(Z)) + 1e-8
    Z = Z / z_scale
    logging.info(f"Z waveform normalized by scale={z_scale:.4f}")

    # Compute normals, flow geometry, and interior lumen points per patient
    logging.info("Computing surface normals, flow geometry, and interior points...")
    normals = np.zeros_like(X)
    flow_directions = np.zeros((n_patients, 3), dtype=np.float32)
    vessel_radii = np.zeros(n_patients, dtype=np.float32)
    X_interior_all = np.zeros((n_patients, NUM_INTERIOR_POINTS, 3), dtype=np.float32)
    interior_radial_all = np.zeros((n_patients, NUM_INTERIOR_POINTS), dtype=np.float32)

    for i in range(n_patients):
        normals[i] = compute_normals(X[i], k=10)
        flow_dir, _, radial_dist, vessel_r = compute_flow_geometry(X[i])
        flow_directions[i] = flow_dir
        vessel_radii[i] = vessel_r

        if X[i].shape[0] >= NUM_INTERIOR_POINTS:
            idx = np.random.choice(X[i].shape[0], NUM_INTERIOR_POINTS, replace=False)
        else:
            idx = np.random.choice(X[i].shape[0], NUM_INTERIOR_POINTS, replace=True)

        interior_pts, interior_radial = generate_interior_points(
            X[i][idx], normals[i][idx], radial_dist[idx]
        )
        X_interior_all[i] = interior_pts
        interior_radial_all[i] = interior_radial

    logging.info(f"Interior points generated: {X_interior_all.shape}")

    # Precompute Poiseuille flow supervision at the interior points (not at
    # the wall, where the profile is ~0 by definition and just duplicates
    # the no-slip constraint)
    logging.info("Computing Poiseuille flow supervision for all patients...")
    flow_poiseuille_interior = np.zeros((n_patients, NUM_INTERIOR_POINTS, INLET_TIMESTEPS, 3), dtype=np.float32)
    for i in range(n_patients):
        flow_poiseuille_interior[i] = compute_poiseuille_profile(
            interior_radial_all[i], vessel_radii[i], flow_directions[i], Z[i]
        )
    logging.info(f"Poiseuille flows computed: {flow_poiseuille_interior.shape}")

    # ========================================================================
    # Main 10-fold CV loop
    # ========================================================================

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_pearson = []

    for fold_idx, (train_ix, test_ix) in enumerate(kf.split(X)):
        logging.info(f"\n{'='*60}")
        logging.info(f"Fold {fold_idx}/{N_FOLDS}")
        logging.info(f"{'='*60}")

        X_train, X_test = X[train_ix], X[test_ix]

        # Normalize Y using train-fold statistics only (no test-fold leakage)
        y_min, y_max = Y[train_ix].min(), Y[train_ix].max()
        Y_train = (Y[train_ix] - y_min) / (y_max - y_min)
        Y_test = (Y[test_ix] - y_min) / (y_max - y_min)

        X_interior_train_fold = X_interior_all[train_ix]
        flow_interior_train_fold = flow_poiseuille_interior[train_ix]
        normals_train_fold = normals[train_ix]
        normals_test = normals[test_ix]

        # Split train into train+val (9:1)
        n_train = len(X_train)
        n_val = max(1, n_train // 10)
        val_ix = np.random.choice(n_train, n_val, replace=False)
        train_ix_inner = np.setdiff1d(np.arange(n_train), val_ix)

        X_train_inner = X_train[train_ix_inner]
        Y_train_inner = Y_train[train_ix_inner]
        X_interior_train = X_interior_train_fold[train_ix_inner]
        flow_train_inner = flow_interior_train_fold[train_ix_inner]
        normals_train_inner = normals_train_fold[train_ix_inner]
        X_val = X_train[val_ix]
        Y_val = Y_train[val_ix]
        normals_val = normals_train_fold[val_ix]

        # Initialize models
        pinn = PINN_FlowSolver(hidden_dim=128, num_layers=4).to(device)
        wss_pred_net = WSS_Predictor(feature_dim=128).to(device)

        optimizer_pinn = optim.Adam(pinn.parameters(), lr=PINN_LR)
        optimizer_wss = optim.Adam(wss_pred_net.parameters(), lr=1e-4)
        wss_mse = nn.MSELoss()

        best_val_loss = float('inf')
        best_pinn_state = None
        best_wss_state = None
        patience_counter = 0
        val_loss_avg = float('inf')

        pinn_loss_fn = PINN_Loss(pinn, device, rho=RHO, mu=MU)

        # Training epochs
        for epoch in range(PINN_EPOCHS):
            pinn.train()
            wss_pred_net.train()

            train_losses = []
            pinn_loss_log = []

            for _ in range(PINN_STEPS_PER_EPOCH):
                # === Sample one patient per step (full surface + interior) ===
                patient_idx = np.random.randint(0, len(X_train_inner))
                time_idx = np.random.randint(0, INLET_TIMESTEPS)
                t_norm = time_idx / INLET_TIMESTEPS

                # Wall no-slip: every surface point is a true wall point
                xyz_wall = np.concatenate([
                    X_train_inner[patient_idx],
                    np.full((NUM_SURFACE_POINTS, 1), t_norm)
                ], axis=1).astype(np.float32)
                xyz_wall_t = torch.tensor(xyz_wall, device=device, requires_grad=True)

                # Interior: NS residual + Poiseuille supervision
                interior_idx = np.random.choice(NUM_INTERIOR_POINTS, NS_SUBSAMPLE, replace=False)
                xyz_interior = np.concatenate([
                    X_interior_train[patient_idx][interior_idx],
                    np.full((NS_SUBSAMPLE, 1), t_norm)
                ], axis=1).astype(np.float32)
                xyz_interior_t = torch.tensor(xyz_interior, device=device, requires_grad=True)

                u_supervised = torch.tensor(
                    flow_train_inner[patient_idx][interior_idx, time_idx, :],
                    device=device, dtype=torch.float32
                )

                pinn_total, wall_l, cont_l, ns_l, flow_l = pinn_loss_fn.total_loss(
                    xyz_wall_t, xyz_interior_t, u_supervised
                )

                # === WSS prediction loss ===
                # Feature = wall shear RATE (dU/dn), not raw velocity, since
                # velocity at the wall is driven to ~0 by no-slip.
                wss_patient_idx = np.random.randint(0, len(X_train_inner))
                xyz_wss = np.concatenate([
                    X_train_inner[wss_patient_idx],
                    np.zeros((NUM_SURFACE_POINTS, 1))  # t=0
                ], axis=1).astype(np.float32)
                xyz_wss_t = torch.tensor(xyz_wss, device=device)
                wss_normals_t = torch.tensor(
                    normals_train_inner[wss_patient_idx], device=device, dtype=torch.float32
                )

                shear_feat = compute_wall_shear_feature(pinn, xyz_wss_t, wss_normals_t)
                flow_feat_wss = torch.cat([xyz_wss_t[:, :3], shear_feat], dim=1)

                wss_pred_out = wss_pred_net(flow_feat_wss)
                y_wss = Y_train_inner[wss_patient_idx]
                y_wss_t = torch.tensor(y_wss, device=device, dtype=torch.float32).unsqueeze(1)
                wss_loss = wss_mse(wss_pred_out, y_wss_t)

                # Combined loss with scaling (avoid physics loss from dominating)
                pinn_scaled = pinn_total * 0.1
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
                pinn_loss_log.append((wall_l.item(), cont_l.item(), ns_l.item(), flow_l.item()))

            # Validation on val set (sample to save time)
            pinn.eval()
            wss_pred_net.eval()
            val_wss_loss = 0
            val_n = min(len(X_val), 5)
            val_idx = np.random.choice(len(X_val), val_n, replace=False)
            for i in val_idx:
                pt_idx = np.random.choice(NUM_SURFACE_POINTS, 256, replace=False)
                xyz_val = np.concatenate([X_val[i, pt_idx], np.zeros((256, 1))], axis=1).astype(np.float32)
                xyz_val_t = torch.tensor(xyz_val, device=device)
                val_normals_t = torch.tensor(normals_val[i, pt_idx], device=device, dtype=torch.float32)

                shear_feat_val = compute_wall_shear_feature(pinn, xyz_val_t, val_normals_t)
                flow_feat_val = torch.cat([xyz_val_t[:, :3], shear_feat_val], dim=1)
                with torch.no_grad():
                    wss_val_pred = wss_pred_net(flow_feat_val).squeeze(1).cpu().numpy()
                wss_val_true = Y_val[i, pt_idx]
                val_wss_loss += np.mean((wss_val_pred - wss_val_true) ** 2)

            val_loss_avg = val_wss_loss / len(val_idx)

            if epoch % 10 == 0 and len(pinn_loss_log) > 0:
                pinn_logs_arr = np.array(pinn_loss_log)
                wall_avg, cont_avg, ns_avg, flow_avg = np.mean(pinn_logs_arr, axis=0)
                logging.info(f"Epoch {epoch:3d}: train={np.mean(train_losses):.6f}, "
                           f"Wall={wall_avg:.6f}, Cont={cont_avg:.6f}, NS={ns_avg:.6f}, Flow={flow_avg:.6f}, "
                           f"val_WSS={val_loss_avg:.6f}")

            # Early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_pinn_state = {k: v.detach().clone() for k, v in pinn.state_dict().items()}
                best_wss_state = {k: v.detach().clone() for k, v in wss_pred_net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PINN_PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}")
                break

        # Test evaluation: restore the best-validated weights, not whatever
        # state training happened to be in when patience ran out.
        if best_pinn_state is not None:
            pinn.load_state_dict(best_pinn_state)
            wss_pred_net.load_state_dict(best_wss_state)

        logging.info("\nEvaluating on test set...")
        pinn.eval()
        wss_pred_net.eval()
        test_wss_pred = []
        test_wss_true = []

        for i in range(len(X_test)):
            xyz_test = np.concatenate([X_test[i], np.zeros((NUM_SURFACE_POINTS, 1))], axis=1).astype(np.float32)
            xyz_test_t = torch.tensor(xyz_test, device=device)
            test_normals_t = torch.tensor(normals_test[i], device=device, dtype=torch.float32)

            shear_feat_test = compute_wall_shear_feature(pinn, xyz_test_t, test_normals_t)
            flow_feat_test = torch.cat([xyz_test_t[:, :3], shear_feat_test], dim=1)
            with torch.no_grad():
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
