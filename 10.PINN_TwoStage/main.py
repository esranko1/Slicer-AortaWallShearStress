"""
Two-Stage PINN for WSS Prediction from Point Clouds.

Stage 1: Physics-Informed Neural Network (PINN)
  Input: Surface point cloud (4096 surface points) + inlet velocity BC
  Learn: u, v, w, p solving Navier-Stokes equations
  Enforce: Continuity (∇·u=0), Momentum (ρ∂u/∂t + ρu·∇u = -∇p + μ∇²u) at all
           interior points; wall no-slip (u=0) on the surface; Poiseuille-
           flow supervision only near the (heuristically identified) inlet
           end of the vessel, not forced across the whole interior

Stage 2: WSS Prediction (Supervised)
  Input: Surface geometry + wall shear rate (tangential viscous traction) from Stage 1
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

# PINN training - two SEQUENTIAL stages, not interleaved: Stage 1 trains the
# flow solver alone to convergence; Stage 2 freezes it and trains the WSS
# predictor on its now-stable output. Previously both were updated together
# in one loop, so Stage 2 was chasing a moving target that kept changing
# every step for the entire training budget.
STAGE1_EPOCHS = 60      # PINN physics-only training
STAGE1_PATIENCE = 15
STAGE2_EPOCHS = 60      # WSS_Predictor training on the frozen PINN
STAGE2_PATIENCE = 20
PINN_STEPS_PER_EPOCH = 8  # Multiple patient samples per epoch so a fold's ~90 patients get real coverage
PATIENTS_PER_BATCH = 4  # Mix multiple patients into every gradient step (real mini-batch diversity,
                         # not a single patient's gradient dominating each update)
PINN_LR = 1e-3

# Points sampled per patient within a batch, split so the total points/step
# (and hence compute cost) stays about the same as a single full-patient step
WALL_POINTS_PER_PATIENT = NUM_SURFACE_POINTS // PATIENTS_PER_BATCH
NS_POINTS_PER_PATIENT = NS_SUBSAMPLE // PATIENTS_PER_BATCH
WSS_POINTS_PER_PATIENT = NUM_SURFACE_POINTS // PATIENTS_PER_BATCH

INLET_AXIAL_PERCENTILE = 15  # % of each patient's interior points, nearest the inlet end, that get
                             # the Poiseuille supervision target - the rest is governed by NS + no-slip
                             # alone, instead of being forced to look like a straight pipe everywhere.

# Patient conditioning: [inlet_velocity_at_t, vessel_radius, flow_dir_x,
# flow_dir_y, flow_dir_z], appended to every point fed to the PINN. Without
# this, one shared f(x,y,z,t) is fit across ~90 different patients with
# nothing telling it which patient a point belongs to - two patients can
# land at nearly the same normalized coordinates with different Poiseuille
# targets, and the network can only average across the conflict.
COND_DIM = 5

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
    than outside it; compute_wall_shear_feature is unaffected since its
    tangential-traction magnitude is invariant to the normal's sign.

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


def build_cond(waveform_value, vessel_radius, flow_direction, n_points):
    """
    Broadcast one patient's conditioning vector - [inlet_velocity_at_t,
    vessel_radius, flow_dir_x, flow_dir_y, flow_dir_z] - across n_points,
    so every point fed to the PINN carries which patient's geometry/flow
    condition it belongs to.

    waveform_value: scalar, Z[patient, time_idx]
    vessel_radius: scalar
    flow_direction: (3,)
    returns: (n_points, COND_DIM)
    """
    cond_vec = np.concatenate([
        np.array([waveform_value], dtype=np.float32),
        np.array([vessel_radius], dtype=np.float32),
        flow_direction.astype(np.float32)
    ])
    return np.tile(cond_vec, (n_points, 1))


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Stage 1: PINN Network
# ============================================================================

class PINN_FlowSolver(nn.Module):
    """
    Physics-Informed Neural Network for Navier-Stokes with Fourier features.
    Outputs: u(x,y,z,t), v, w, p = flow field

    Input carries COND_DIM extra columns after [x,y,z,t]: the patient's
    inlet velocity at this t, vessel radius, and flow direction. Only the
    first 4 columns are Fourier-encoded / differentiated for the PDE
    residual (navier_stokes_loss and compute_wall_shear_feature already
    only ever slice columns 0-3 out of any gradient they take, so appending
    extra trailing columns needed no changes there) - the conditioning
    columns are passed straight through to the MLP.
    """

    def __init__(self, hidden_dim=128, num_layers=4, fourier_features=32, cond_dim=COND_DIM):
        super().__init__()

        self.fourier_features = fourier_features
        self.cond_dim = cond_dim
        # Fourier feature matrix for positional encoding (spatial+time only)
        self.register_buffer(
            'fourier_matrix',
            torch.randn(4, fourier_features) * np.pi
        )

        # Input: 4 (xyz,t) + 2*fourier_features (sin/cos) + cond_dim
        input_dim = 4 + 2 * fourier_features + cond_dim

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
        x: (batch, 4 + cond_dim) = [x, y, z, t, <patient conditioning>]
        returns: (batch, 4) = [u, v, w, p]
        """
        spatial = x[:, :4]
        cond = x[:, 4:]

        # Fourier feature encoding (spatial+time only)
        fourier_x = spatial @ self.fourier_matrix  # (batch, fourier_features)
        fourier_feat = torch.cat([
            torch.sin(fourier_x),
            torch.cos(fourier_x)
        ], dim=1)  # (batch, 2*fourier_features)

        x_encoded = torch.cat([spatial, fourier_feat, cond], dim=1)
        return self.network(x_encoded)


def compute_wall_shear_feature(model, xyz_t, wall_normals):
    """
    Wall shear rate from the full velocity Jacobian, not just |dU/dn|: build
    the symmetric strain-rate tensor D = grad(u) + grad(u)^T, project it onto
    the wall normal to get the surface traction, then strip out the traction's
    normal component to leave the tangential part - the quantity WSS is
    actually proportional to (tau_w = mu * tangential traction), rather than
    the raw per-component normal derivative. At an exact no-slip wall with a
    perfect normal these coincide (tangential velocity derivatives vanish
    there), but this version is more robust to the normal estimate being
    imperfect (it comes from PCA over a point cloud, not exact geometry).

    Must be called outside any torch.no_grad() context (needs a live graph
    to differentiate w.r.t. the input coordinates) - this does NOT require
    the model's parameters to have requires_grad=True, only the input, so
    it works fine on a frozen model. The returned feature is detached, so
    it does not backprop into the PINN.

    xyz_t: (batch, 4 + COND_DIM) = [x, y, z, t, <patient conditioning>]
    wall_normals: (batch, 3)
    returns: (batch, 1) shear-rate magnitude
    """
    xyz_t = xyz_t.clone().requires_grad_(True)
    flow = model(xyz_t)
    u, v, w = flow[:, 0:1], flow[:, 1:2], flow[:, 2:3]

    grad_u = torch.autograd.grad(u.sum(), xyz_t, retain_graph=True)[0][:, :3]
    grad_v = torch.autograd.grad(v.sum(), xyz_t, retain_graph=True)[0][:, :3]
    grad_w = torch.autograd.grad(w.sum(), xyz_t, retain_graph=True)[0][:, :3]

    jacobian = torch.stack([grad_u, grad_v, grad_w], dim=1)  # (batch, 3, 3), row i = grad of component i
    strain_rate = jacobian + jacobian.transpose(1, 2)

    traction = torch.einsum('bij,bj->bi', strain_rate, wall_normals)  # (batch, 3)
    normal_traction = (traction * wall_normals).sum(dim=1, keepdim=True) * wall_normals
    tangential_traction = traction - normal_traction

    shear_rate = torch.norm(tangential_traction, dim=1, keepdim=True)
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
        xyz_t: (batch, 4 + COND_DIM) = [x, y, z, t, <patient conditioning>] -
               only columns 0-3 are ever sliced out of a gradient below, so
               the trailing conditioning columns don't affect this method.
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

    def total_loss(self, xyz_wall, xyz_interior, u_supervised_target, inlet_mask):
        """
        xyz_wall: (batch, 4 + COND_DIM) - all surface points; the vessel
                  wall, so no-slip (u=0) applies everywhere here.
        xyz_interior: (batch, 4 + COND_DIM) - points inside the lumen; the
                  NS residual + continuity apply to ALL of them.
        u_supervised_target: (batch, 3) - Poiseuille flow target at
                  xyz_interior, only meaningful where inlet_mask is True.
        inlet_mask: (batch,) bool - which xyz_interior points are near the
                  inlet end of the vessel and get pulled toward the
                  Poiseuille profile. Real vessels are curved/branching, not
                  straight pipes, so everywhere else is left to the NS
                  residual + no-slip alone rather than forced into a
                  straight-pipe shape throughout the whole interior.
        """
        flow_wall = self.model(xyz_wall)
        wall_loss = torch.mean(flow_wall[:, 0:3]**2)

        ns_loss, cont_loss, uvw_interior = self.navier_stokes_loss(xyz_interior)

        if inlet_mask.any():
            supervised_loss = torch.mean((uvw_interior[inlet_mask] - u_supervised_target[inlet_mask])**2)
        else:
            supervised_loss = torch.zeros((), device=xyz_interior.device)

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
    axial_positions = np.zeros((n_patients, NUM_SURFACE_POINTS), dtype=np.float32)
    radial_distances = np.zeros((n_patients, NUM_SURFACE_POINTS), dtype=np.float32)
    X_interior_all = np.zeros((n_patients, NUM_INTERIOR_POINTS, 3), dtype=np.float32)
    interior_radial_all = np.zeros((n_patients, NUM_INTERIOR_POINTS), dtype=np.float32)
    interior_idx_all = np.zeros((n_patients, NUM_INTERIOR_POINTS), dtype=np.int64)

    for i in range(n_patients):
        normals[i] = compute_normals(X[i], k=10)
        flow_dir, axial_pos, radial_dist, vessel_r = compute_flow_geometry(X[i])
        flow_directions[i] = flow_dir
        vessel_radii[i] = vessel_r
        axial_positions[i] = axial_pos
        radial_distances[i] = radial_dist

        if X[i].shape[0] >= NUM_INTERIOR_POINTS:
            idx = np.random.choice(X[i].shape[0], NUM_INTERIOR_POINTS, replace=False)
        else:
            idx = np.random.choice(X[i].shape[0], NUM_INTERIOR_POINTS, replace=True)
        interior_idx_all[i] = idx

        interior_pts, interior_radial = generate_interior_points(
            X[i][idx], normals[i][idx], radial_dist[idx]
        )
        X_interior_all[i] = interior_pts
        interior_radial_all[i] = interior_radial

    logging.info(f"Interior points generated: {X_interior_all.shape}")

    # flow_direction is a PCA axis, not a direction - its sign is arbitrary
    # per patient. Without fixing this, two patients with equivalent
    # anatomy can get oppositely-signed Poiseuille targets in the same
    # normalized coordinate frame, which is its own source of contradictory
    # supervision. Resolve to a consistent sign using the standard trick for
    # averaging "headless" vectors: average the sign-invariant outer product
    # d.d^T across patients, then flip each patient to align with the
    # dominant eigenvector of that average. This only guarantees consistency
    # across patients, not that the result is the true anatomical forward
    # direction - but consistency is what the shared network needs.
    outer_sum = np.einsum('pi,pj->ij', flow_directions, flow_directions)
    _, ref_eigvecs = np.linalg.eigh(outer_sum)
    consensus_direction = ref_eigvecs[:, -1]
    flip_mask = (flow_directions @ consensus_direction) < 0
    flow_directions[flip_mask] *= -1
    axial_positions[flip_mask] *= -1  # keep axial position consistent with the now-flipped direction
    logging.info(f"Flow direction sign-consistency: flipped {flip_mask.sum()}/{n_patients} patients")

    # Which end of the (now sign-consistent) flow axis is the inlet? This
    # wall-only point cloud has no literal inlet-cap points to check
    # directly. Real aortas taper from a wider proximal (inlet) segment to
    # a narrower distal one, so use a radius-vs-axial-position correlation
    # aggregated across patients as the convention: negative means radius
    # shrinks as axial position increases, i.e. the LOW-axial-position end
    # is the wider/inlet end.
    taper_correlations = [
        compute_pearson(axial_positions[i], radial_distances[i])
        for i in range(n_patients)
        if axial_positions[i].std() > 1e-8 and radial_distances[i].std() > 1e-8
    ]
    mean_taper_corr = np.mean(taper_correlations) if taper_correlations else 0.0
    inlet_is_low_axial = mean_taper_corr <= 0
    logging.info(f"Vessel taper check: mean radius-vs-axial-position correlation={mean_taper_corr:.4f}, "
                 f"assuming inlet at {'low' if inlet_is_low_axial else 'high'}-axial-position end")

    # Only interior points near that end get the Poiseuille supervision
    # target; the rest of the interior is governed by the NS residual +
    # continuity alone, instead of the whole interior being forced to look
    # like a straight pipe (real vessels are curved/branching, not that).
    interior_axial_all = np.zeros((n_patients, NUM_INTERIOR_POINTS), dtype=np.float32)
    inlet_mask_all = np.zeros((n_patients, NUM_INTERIOR_POINTS), dtype=bool)
    for i in range(n_patients):
        interior_axial_all[i] = axial_positions[i][interior_idx_all[i]]
        if inlet_is_low_axial:
            threshold = np.percentile(interior_axial_all[i], INLET_AXIAL_PERCENTILE)
            inlet_mask_all[i] = interior_axial_all[i] <= threshold
        else:
            threshold = np.percentile(interior_axial_all[i], 100 - INLET_AXIAL_PERCENTILE)
            inlet_mask_all[i] = interior_axial_all[i] >= threshold
    logging.info(f"Inlet region: targeting {INLET_AXIAL_PERCENTILE}% of interior points per patient "
                 f"(actual mean {inlet_mask_all.mean() * 100:.1f}%)")

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
        inlet_mask_train_fold = inlet_mask_all[train_ix]
        normals_train_fold = normals[train_ix]
        normals_test = normals[test_ix]
        Z_train_fold = Z[train_ix]
        vessel_radii_train_fold = vessel_radii[train_ix]
        flow_directions_train_fold = flow_directions[train_ix]
        Z_test = Z[test_ix]
        vessel_radii_test = vessel_radii[test_ix]
        flow_directions_test = flow_directions[test_ix]

        # Split train into train+val (9:1)
        n_train = len(X_train)
        n_val = max(1, n_train // 10)
        val_ix = np.random.choice(n_train, n_val, replace=False)
        train_ix_inner = np.setdiff1d(np.arange(n_train), val_ix)

        X_train_inner = X_train[train_ix_inner]
        Y_train_inner = Y_train[train_ix_inner]
        X_interior_train = X_interior_train_fold[train_ix_inner]
        flow_train_inner = flow_interior_train_fold[train_ix_inner]
        inlet_mask_train_inner = inlet_mask_train_fold[train_ix_inner]
        normals_train_inner = normals_train_fold[train_ix_inner]
        Z_train_inner = Z_train_fold[train_ix_inner]
        vessel_radii_train_inner = vessel_radii_train_fold[train_ix_inner]
        flow_directions_train_inner = flow_directions_train_fold[train_ix_inner]
        X_val = X_train[val_ix]
        Y_val = Y_train[val_ix]
        normals_val = normals_train_fold[val_ix]
        X_interior_val = X_interior_train_fold[val_ix]
        flow_interior_val = flow_interior_train_fold[val_ix]
        inlet_mask_val = inlet_mask_train_fold[val_ix]
        Z_val = Z_train_fold[val_ix]
        vessel_radii_val = vessel_radii_train_fold[val_ix]
        flow_directions_val = flow_directions_train_fold[val_ix]

        # Initialize models
        pinn = PINN_FlowSolver(hidden_dim=128, num_layers=4).to(device)
        wss_pred_net = WSS_Predictor(feature_dim=128).to(device)

        optimizer_pinn = optim.Adam(pinn.parameters(), lr=PINN_LR)
        optimizer_wss = optim.Adam(wss_pred_net.parameters(), lr=1e-4)
        wss_mse = nn.MSELoss()

        pinn_loss_fn = PINN_Loss(pinn, device, rho=RHO, mu=MU)

        # ====================================================================
        # Stage 1: train the PINN flow solver alone, to convergence, on
        # physics + Poiseuille supervision only. No WSS network involved yet.
        # ====================================================================
        best_phys_val = float('inf')
        best_pinn_state = None
        patience_counter = 0

        for epoch in range(STAGE1_EPOCHS):
            pinn.train()
            train_losses = []
            pinn_loss_log = []

            for _ in range(PINN_STEPS_PER_EPOCH):
                # Mix PATIENTS_PER_BATCH patients into this one gradient step
                # (each at its own random time) so the update direction
                # reflects several geometries at once, instead of one
                # patient's field dominating the step.
                batch_patients = np.random.randint(0, len(X_train_inner), size=PATIENTS_PER_BATCH)

                wall_parts, interior_parts, u_sup_parts, inlet_mask_parts = [], [], [], []
                for p in batch_patients:
                    time_idx = np.random.randint(0, INLET_TIMESTEPS)
                    t_norm = time_idx / INLET_TIMESTEPS

                    wall_cond = build_cond(
                        Z_train_inner[p, time_idx], vessel_radii_train_inner[p],
                        flow_directions_train_inner[p], WALL_POINTS_PER_PATIENT
                    )
                    wall_pt_idx = np.random.choice(NUM_SURFACE_POINTS, WALL_POINTS_PER_PATIENT, replace=False)
                    wall_parts.append(np.concatenate([
                        X_train_inner[p][wall_pt_idx],
                        np.full((WALL_POINTS_PER_PATIENT, 1), t_norm),
                        wall_cond
                    ], axis=1))

                    interior_cond = build_cond(
                        Z_train_inner[p, time_idx], vessel_radii_train_inner[p],
                        flow_directions_train_inner[p], NS_POINTS_PER_PATIENT
                    )
                    interior_idx = np.random.choice(NUM_INTERIOR_POINTS, NS_POINTS_PER_PATIENT, replace=False)
                    interior_parts.append(np.concatenate([
                        X_interior_train[p][interior_idx],
                        np.full((NS_POINTS_PER_PATIENT, 1), t_norm),
                        interior_cond
                    ], axis=1))
                    u_sup_parts.append(flow_train_inner[p][interior_idx, time_idx, :])
                    inlet_mask_parts.append(inlet_mask_train_inner[p][interior_idx])

                xyz_wall = np.concatenate(wall_parts, axis=0).astype(np.float32)
                xyz_wall_t = torch.tensor(xyz_wall, device=device, requires_grad=True)

                xyz_interior = np.concatenate(interior_parts, axis=0).astype(np.float32)
                xyz_interior_t = torch.tensor(xyz_interior, device=device, requires_grad=True)

                u_supervised = torch.tensor(
                    np.concatenate(u_sup_parts, axis=0), device=device, dtype=torch.float32
                )
                inlet_mask_t = torch.tensor(
                    np.concatenate(inlet_mask_parts, axis=0), device=device, dtype=torch.bool
                )

                pinn_total, wall_l, cont_l, ns_l, flow_l = pinn_loss_fn.total_loss(
                    xyz_wall_t, xyz_interior_t, u_supervised, inlet_mask_t
                )

                optimizer_pinn.zero_grad()
                pinn_total.backward()
                torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
                optimizer_pinn.step()

                train_losses.append(pinn_total.item())
                pinn_loss_log.append((wall_l.item(), cont_l.item(), ns_l.item(), flow_l.item()))

            # Physics-only validation on held-out patients - no WSS labels
            # involved, this only checks the flow solver generalizes.
            # Subsampled to the same per-patient point counts a training
            # step uses (not the full surface/interior sets): the 2nd-
            # derivative NS residual is expensive, and validating on ~9
            # patients at full size would cost several times more compute
            # than a whole training epoch.
            pinn.eval()
            phys_val_losses = []
            for i in range(len(X_val)):
                time_idx = np.random.randint(0, INLET_TIMESTEPS)
                t_norm = time_idx / INLET_TIMESTEPS

                wall_cond_val = build_cond(
                    Z_val[i, time_idx], vessel_radii_val[i], flow_directions_val[i], WALL_POINTS_PER_PATIENT
                )
                wall_pt_idx = np.random.choice(NUM_SURFACE_POINTS, WALL_POINTS_PER_PATIENT, replace=False)
                xyz_wall_val = np.concatenate([
                    X_val[i][wall_pt_idx], np.full((WALL_POINTS_PER_PATIENT, 1), t_norm), wall_cond_val
                ], axis=1).astype(np.float32)
                xyz_wall_val_t = torch.tensor(xyz_wall_val, device=device, requires_grad=True)

                interior_cond_val = build_cond(
                    Z_val[i, time_idx], vessel_radii_val[i], flow_directions_val[i], NS_POINTS_PER_PATIENT
                )
                interior_idx = np.random.choice(NUM_INTERIOR_POINTS, NS_POINTS_PER_PATIENT, replace=False)
                xyz_interior_val = np.concatenate([
                    X_interior_val[i][interior_idx], np.full((NS_POINTS_PER_PATIENT, 1), t_norm), interior_cond_val
                ], axis=1).astype(np.float32)
                xyz_interior_val_t = torch.tensor(xyz_interior_val, device=device, requires_grad=True)

                u_supervised_val = torch.tensor(
                    flow_interior_val[i][interior_idx, time_idx, :], device=device, dtype=torch.float32
                )
                inlet_mask_val_t = torch.tensor(
                    inlet_mask_val[i][interior_idx], device=device, dtype=torch.bool
                )

                val_total, _, _, _, _ = pinn_loss_fn.total_loss(
                    xyz_wall_val_t, xyz_interior_val_t, u_supervised_val, inlet_mask_val_t
                )
                phys_val_losses.append(val_total.item())

            phys_val_avg = np.mean(phys_val_losses)

            if epoch % 10 == 0 and len(pinn_loss_log) > 0:
                pinn_logs_arr = np.array(pinn_loss_log)
                wall_avg, cont_avg, ns_avg, flow_avg = np.mean(pinn_logs_arr, axis=0)
                logging.info(f"[Stage 1] Epoch {epoch:3d}: train={np.mean(train_losses):.6f}, "
                           f"Wall={wall_avg:.6f}, Cont={cont_avg:.6f}, NS={ns_avg:.6f}, Flow={flow_avg:.6f}, "
                           f"phys_val={phys_val_avg:.6f}")

            if phys_val_avg < best_phys_val:
                best_phys_val = phys_val_avg
                best_pinn_state = {k: v.detach().clone() for k, v in pinn.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= STAGE1_PATIENCE:
                logging.info(f"[Stage 1] Early stopping at epoch {epoch}")
                break

        if best_pinn_state is not None:
            pinn.load_state_dict(best_pinn_state)

        # Freeze the PINN: Stage 2 learns from its now-stable flow field
        # instead of continuing to tune it, so the WSS predictor's regression
        # target stops moving underneath it every step. compute_wall_shear_
        # feature only needs requires_grad on the input coordinates (not the
        # model's parameters) to compute the spatial gradient, so freezing
        # here doesn't break it.
        for p in pinn.parameters():
            p.requires_grad = False
        pinn.eval()

        # ====================================================================
        # Calibrate WHICH point in the cardiac cycle Y's "instantaneous WSS"
        # was taken at - Y never says, and it need not be t=0. Scan candidate
        # times and keep whichever one makes the frozen flow field's raw
        # shear feature correlate best with the true WSS on training
        # patients, before Stage 2's MLP has been trained on anything - if
        # the flow field is any good, its shear feature should already track
        # WSS reasonably well at the right phase and much less well off it.
        # ====================================================================
        calib_patients = np.random.choice(len(X_train_inner), min(10, len(X_train_inner)), replace=False)
        best_calib_time = 0
        best_calib_score = -np.inf
        for cand_time in range(0, INLET_TIMESTEPS, max(1, INLET_TIMESTEPS // 10)):
            t_norm = cand_time / INLET_TIMESTEPS
            scores = []
            for p in calib_patients:
                cond = build_cond(
                    Z_train_inner[p, cand_time], vessel_radii_train_inner[p],
                    flow_directions_train_inner[p], NUM_SURFACE_POINTS
                )
                xyz = np.concatenate([
                    X_train_inner[p], np.full((NUM_SURFACE_POINTS, 1), t_norm), cond
                ], axis=1).astype(np.float32)
                xyz_t = torch.tensor(xyz, device=device)
                cand_normals_t = torch.tensor(normals_train_inner[p], device=device, dtype=torch.float32)

                shear = compute_wall_shear_feature(pinn, xyz_t, cand_normals_t).squeeze(1).cpu().numpy()
                y_true = Y_train_inner[p]
                if shear.std() > 1e-8 and y_true.std() > 1e-8:
                    score_p = compute_pearson(y_true, shear)
                    if np.isfinite(score_p):
                        scores.append(score_p)

            if scores and np.median(scores) > best_calib_score:
                best_calib_score = np.median(scores)
                best_calib_time = cand_time

        logging.info(f"Calibrated WSS reference time: index {best_calib_time}/{INLET_TIMESTEPS} "
                     f"(shear-vs-Y correlation={best_calib_score:.4f})")

        # ====================================================================
        # Stage 2: train the WSS predictor on the frozen PINN's wall shear
        # feature, using the real WSS labels.
        # ====================================================================
        best_val_loss = float('inf')
        best_wss_state = None
        patience_counter = 0
        val_loss_avg = float('inf')

        for epoch in range(STAGE2_EPOCHS):
            wss_pred_net.train()
            train_losses = []

            for _ in range(PINN_STEPS_PER_EPOCH):
                wss_batch_patients = np.random.randint(0, len(X_train_inner), size=PATIENTS_PER_BATCH)

                wss_xyz_parts, wss_normal_parts, wss_y_parts = [], [], []
                for p in wss_batch_patients:
                    wss_cond = build_cond(
                        Z_train_inner[p, best_calib_time], vessel_radii_train_inner[p],
                        flow_directions_train_inner[p], WSS_POINTS_PER_PATIENT
                    )
                    wss_pt_idx = np.random.choice(NUM_SURFACE_POINTS, WSS_POINTS_PER_PATIENT, replace=False)
                    wss_xyz_parts.append(np.concatenate([
                        X_train_inner[p][wss_pt_idx],
                        np.full((WSS_POINTS_PER_PATIENT, 1), best_calib_time / INLET_TIMESTEPS),
                        wss_cond
                    ], axis=1))
                    wss_normal_parts.append(normals_train_inner[p][wss_pt_idx])
                    wss_y_parts.append(Y_train_inner[p][wss_pt_idx])

                xyz_wss = np.concatenate(wss_xyz_parts, axis=0).astype(np.float32)
                xyz_wss_t = torch.tensor(xyz_wss, device=device)
                wss_normals_t = torch.tensor(
                    np.concatenate(wss_normal_parts, axis=0), device=device, dtype=torch.float32
                )

                shear_feat = compute_wall_shear_feature(pinn, xyz_wss_t, wss_normals_t)
                flow_feat_wss = torch.cat([xyz_wss_t[:, :3], shear_feat], dim=1)

                wss_pred_out = wss_pred_net(flow_feat_wss)
                y_wss_t = torch.tensor(
                    np.concatenate(wss_y_parts, axis=0), device=device, dtype=torch.float32
                ).unsqueeze(1)
                wss_loss = wss_mse(wss_pred_out, y_wss_t)

                optimizer_wss.zero_grad()
                wss_loss.backward()
                torch.nn.utils.clip_grad_norm_(wss_pred_net.parameters(), 1.0)
                optimizer_wss.step()

                train_losses.append(wss_loss.item())

            # Validation over the FULL val set (previously a noisy 5-patient
            # sample, which made the early-stopping decision itself noisy).
            wss_pred_net.eval()
            val_wss_loss = 0
            for i in range(len(X_val)):
                val_cond = build_cond(Z_val[i, best_calib_time], vessel_radii_val[i], flow_directions_val[i], 256)
                pt_idx = np.random.choice(NUM_SURFACE_POINTS, 256, replace=False)
                xyz_val = np.concatenate([
                    X_val[i, pt_idx], np.full((256, 1), best_calib_time / INLET_TIMESTEPS), val_cond
                ], axis=1).astype(np.float32)
                xyz_val_t = torch.tensor(xyz_val, device=device)
                val_normals_t = torch.tensor(normals_val[i, pt_idx], device=device, dtype=torch.float32)

                shear_feat_val = compute_wall_shear_feature(pinn, xyz_val_t, val_normals_t)
                flow_feat_val = torch.cat([xyz_val_t[:, :3], shear_feat_val], dim=1)
                with torch.no_grad():
                    wss_val_pred = wss_pred_net(flow_feat_val).squeeze(1).cpu().numpy()
                wss_val_true = Y_val[i, pt_idx]
                val_wss_loss += np.mean((wss_val_pred - wss_val_true) ** 2)

            val_loss_avg = val_wss_loss / len(X_val)

            if epoch % 10 == 0:
                logging.info(f"[Stage 2] Epoch {epoch:3d}: train_WSS={np.mean(train_losses):.6f}, "
                           f"val_WSS={val_loss_avg:.6f}")

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_wss_state = {k: v.detach().clone() for k, v in wss_pred_net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= STAGE2_PATIENCE:
                logging.info(f"[Stage 2] Early stopping at epoch {epoch}")
                break

        if best_wss_state is not None:
            wss_pred_net.load_state_dict(best_wss_state)

        logging.info("\nEvaluating on test set...")
        pinn.eval()
        wss_pred_net.eval()
        test_wss_pred = []
        test_wss_true = []

        for i in range(len(X_test)):
            test_cond = build_cond(
                Z_test[i, best_calib_time], vessel_radii_test[i], flow_directions_test[i], NUM_SURFACE_POINTS
            )
            xyz_test = np.concatenate([
                X_test[i], np.full((NUM_SURFACE_POINTS, 1), best_calib_time / INLET_TIMESTEPS), test_cond
            ], axis=1).astype(np.float32)
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
