"""
PI's original PointNet architecture (shape -> WSS), ported from TensorFlow/Keras to
PyTorch, with one addition: the inlet velocity waveform is encoded and fused into the
global shape feature, so predictions depend on both geometry and flow condition, not
shape alone.

Kept from the original: the learned input/feature transformation (alignment) blocks,
the SSIM-based loss, early stopping + best-checkpoint restoration, 10-fold CV.
New: the velocity encoder and its fusion into the global feature.

Expects Data/Sampled/Rpt0_N4096.npz (built by 1.Data_preprocessing/1.3.convert_mat_to_npz.py),
with arrays:
    a -> (100, 4096, 53) : xyz (cols 0-2) + inlet velocity waveform (cols 3-52)
    b -> (100, 4096, 1)  : SWSS
    c -> (100,)          : patient identifiers
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_PATH = "../Data/Sampled/Rpt0_N4096.npz"
RESULTS_DIR = "../experiments/6_PointNet_Velocity"
NUM_POINTS = 4096
VELOCITY_DIM = 50
N_FOLDS = 10
MAX_EPOCHS = 3000
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 50  # epochs of no train-loss improvement before stopping
SEED = 2024


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Model — ported from the PI's conv_block / mlp_block / transformation_block /
# pointNet_model, with a new velocity encoder fused into the global feature.
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d(kernel_size=3, same padding) + BatchNorm + SiLU — matches PI's conv_block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.5)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MLPBlock(nn.Module):
    """Dense + BatchNorm + SiLU — matches PI's mlp_block."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, momentum=0.5)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.fc(x)))


class TransformationNet(nn.Module):
    """Learns a num_features x num_features alignment matrix. Initialized to the
    identity (zero weights, identity bias), matching the PI's kernel_initializer=
    "zeros" / bias_initializer=identity setup, so training starts as a no-op alignment
    and learns to correct for real misalignment from there."""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv1 = ConvBlock(num_features, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 1024)
        self.mlp1 = MLPBlock(1024, 512)
        self.mlp2 = MLPBlock(512, 256)
        self.fc_final = nn.Linear(256, num_features * num_features)
        nn.init.zeros_(self.fc_final.weight)
        with torch.no_grad():
            self.fc_final.bias.copy_(torch.eye(num_features).flatten())

    def forward(self, x):
        # x: (batch, num_features, num_points)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = h.max(dim=2)[0]
        h = self.mlp1(h)
        h = self.mlp2(h)
        matrix = self.fc_final(h)
        return matrix.view(-1, self.num_features, self.num_features)


class TransformationBlock(nn.Module):
    """Applies the learned alignment matrix to every point. Returns both the
    transformed features and the raw matrix (needed for the orthogonality
    regularization term in the loss)."""
    def __init__(self, num_features):
        super().__init__()
        self.transform_net = TransformationNet(num_features)

    def forward(self, x):
        # x: (batch, num_features, num_points)
        matrix = self.transform_net(x)
        x_t = x.transpose(1, 2)                    # (batch, num_points, num_features)
        transformed = torch.bmm(x_t, matrix)        # (batch, num_points, num_features)
        return transformed.transpose(1, 2), matrix  # back to (batch, num_features, num_points)


class AortaPointNetVelocity(nn.Module):
    def __init__(self, velocity_dim=VELOCITY_DIM):
        super().__init__()
        self.input_transform = TransformationBlock(3)
        self.conv64 = ConvBlock(3, 64)
        self.conv128_1 = ConvBlock(64, 128)
        self.conv128_2 = ConvBlock(128, 128)
        self.feature_transform = TransformationBlock(128)
        self.conv512 = ConvBlock(128, 512)
        self.conv2048 = ConvBlock(512, 2048)

        # New: encode the inlet velocity waveform into an embedding that gets
        # fused with the global shape feature.
        self.velocity_encoder = nn.Sequential(
            nn.Linear(velocity_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )

        seg_input_channels = 64 + 128 + 128 + 128 + 512 + (2048 + 128)
        self.seg_conv128 = ConvBlock(seg_input_channels, 128)
        self.seg_conv64 = ConvBlock(128, 64)
        self.seg_conv32 = ConvBlock(64, 32)
        self.pred_head = nn.Conv1d(32, 1, kernel_size=1)
        self.pred_act = nn.SiLU()

    def forward(self, points, velocity):
        # points: (batch, num_points, 3), velocity: (batch, velocity_dim)
        x = points.transpose(1, 2)  # (batch, 3, num_points)
        num_points = x.shape[2]

        x_t, matrix1 = self.input_transform(x)
        f64 = self.conv64(x_t)
        f128_1 = self.conv128_1(f64)
        f128_2 = self.conv128_2(f128_1)
        f_transformed, matrix2 = self.feature_transform(f128_2)
        f512 = self.conv512(f_transformed)
        f2048 = self.conv2048(f512)

        global_feat = f2048.max(dim=2)[0]              # (batch, 2048)
        vel_embed = self.velocity_encoder(velocity)     # (batch, 128)
        global_combined = torch.cat([global_feat, vel_embed], dim=1)  # (batch, 2176)
        global_broadcast = global_combined.unsqueeze(2).expand(-1, -1, num_points)

        seg_input = torch.cat([f64, f128_1, f128_2, f_transformed, f512, global_broadcast], dim=1)
        s = self.seg_conv128(seg_input)
        s = self.seg_conv64(s)
        s = self.seg_conv32(s)
        out = self.pred_act(self.pred_head(s))  # (batch, 1, num_points)
        return out.squeeze(1), matrix1, matrix2  # (batch, num_points)


# ---------------------------------------------------------------------------
# Loss — SSIM (spatial structure) + MAE + orthogonality regularization on the
# learned alignment matrices, matching the PI's SSIMLoss + OrthogonalRegularizer.
# ---------------------------------------------------------------------------

def compute_ssim(pred, target, data_range=1.0, window_size=3):
    """Lightweight single-scale SSIM (average-pooling window). pred/target: (batch, 1, H, W)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return ssim_map.mean()


def orthogonal_regularization(matrix, l2reg=0.01):
    """Penalizes the learned alignment matrix for deviating from orthogonal
    (MM^T != I), matching the PI's OrthogonalRegularizer."""
    num_features = matrix.shape[-1]
    identity = torch.eye(num_features, device=matrix.device).unsqueeze(0)
    mmt = torch.bmm(matrix, matrix.transpose(1, 2))
    return l2reg * torch.mean(torch.sum((mmt - identity) ** 2, dim=(1, 2)))


def compute_loss(pred, target, matrix1, matrix2, data_range=1.0):
    batch_size = pred.shape[0]
    pred_grid = pred.reshape(batch_size, 1, 32, 128)
    target_grid = target.reshape(batch_size, 1, 32, 128)

    mae = torch.mean(torch.abs(pred - target))
    ssim_val = compute_ssim(pred_grid, target_grid, data_range=data_range)
    ssim_loss = 2 * mae + 1 - ssim_val

    reg_loss = orthogonal_regularization(matrix1) + orthogonal_regularization(matrix2)
    return ssim_loss + reg_loss


# ---------------------------------------------------------------------------
# Training — one fold, with manual early stopping (train-loss plateau) and
# explicit best-validation-loss checkpoint restoration, matching the PI's
# EarlyStopping(monitor='loss') + ModelCheckpoint(monitor='val_loss') pairing.
# ---------------------------------------------------------------------------

def calculate_r2(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    return 1 - ss_res / ss_tot


def train_one_fold(X_train, Y_train, Z_train, X_test, Y_test, Z_test, device, fold_idx):
    model = AortaPointNetVelocity().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    Z_train_t = torch.tensor(Z_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)
    Z_test_t = torch.tensor(Z_test, dtype=torch.float32, device=device)

    n_train = X_train_t.shape[0]
    best_val_loss = float('inf')
    best_state = None
    epochs_without_train_improvement = 0
    best_train_loss = float('inf')

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_train_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            pred, m1, m2 = model(X_train_t[idx], Z_train_t[idx])
            loss = compute_loss(pred, Y_train_t[idx], m1, m2)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        epoch_train_loss /= n_batches

        model.eval()
        with torch.no_grad():
            val_pred, vm1, vm2 = model(X_test_t, Z_test_t)
            val_loss = compute_loss(val_pred, Y_test_t, vm1, vm2).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch_train_loss < best_train_loss - 1e-5:
            best_train_loss = epoch_train_loss
            epochs_without_train_improvement = 0
        else:
            epochs_without_train_improvement += 1

        if epoch % 50 == 0 or epochs_without_train_improvement >= EARLY_STOP_PATIENCE:
            logging.info(f"  Fold {fold_idx} epoch {epoch}: train_loss={epoch_train_loss:.4f} val_loss={val_loss:.4f}")

        if epochs_without_train_improvement >= EARLY_STOP_PATIENCE:
            logging.info(f"  Fold {fold_idx}: early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred, _, _ = model(X_test_t, Z_test_t)
    return final_pred.cpu().numpy(), best_val_loss


# ---------------------------------------------------------------------------
# Main — 10-fold CV, matching the PI's original evaluation methodology.
# ---------------------------------------------------------------------------

def main():
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    data = np.load(DATA_PATH)
    a, b = data['a'], data['b']
    X = a[:, :, :3].astype(np.float32)          # (100, 4096, 3)
    Z = a[:, 0, 3:].astype(np.float32)           # (100, 50)
    Y = b[:, :, 0].astype(np.float32)            # (100, 4096)

    logging.info(f"X: {X.shape}, Z: {Z.shape}, Y: {Y.shape}")
    logging.info(f"Y range: [{Y.min():.4f}, {Y.max():.4f}]")

    # Scale Y to [0, 1] for training stability; inverse-transform for reporting.
    y_min, y_max = Y.min(), Y.max()
    Y_scaled = (Y - y_min) / (y_max - y_min)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_true, all_pred = [], []
    fold_r2s = []

    for fold_idx, (train_ix, test_ix) in enumerate(kf.split(X)):
        logging.info(f"\n=== Fold {fold_idx} ===")
        pred_scaled, best_val_loss = train_one_fold(
            X[train_ix], Y_scaled[train_ix], Z[train_ix],
            X[test_ix], Y_scaled[test_ix], Z[test_ix],
            device, fold_idx
        )
        pred = pred_scaled * (y_max - y_min) + y_min
        true = Y[test_ix]

        fold_r2 = calculate_r2(true.flatten(), pred.flatten())
        fold_r2s.append(fold_r2)
        logging.info(f"Fold {fold_idx}: best_val_loss={best_val_loss:.4f}, R2={fold_r2:.4f}")

        all_true.append(true)
        all_pred.append(pred)

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    overall_r2 = calculate_r2(all_true.flatten(), all_pred.flatten())
    overall_mae = np.mean(np.abs(all_true.flatten() - all_pred.flatten()))
    overall_rmse = np.sqrt(np.mean((all_true.flatten() - all_pred.flatten()) ** 2))

    patient_true = np.median(all_true, axis=1)
    patient_pred = np.median(all_pred, axis=1)
    pearson_r = np.corrcoef(patient_true, patient_pred)[0, 1]
    spearman_r, spearman_p = spearmanr(patient_true, patient_pred)

    logging.info("\n=== Overall results (pooled across all 10 folds) ===")
    logging.info(f"Per-fold R2: {[f'{r:.3f}' for r in fold_r2s]}")
    logging.info(f"Point-wise MAE: {overall_mae:.4f}")
    logging.info(f"Point-wise RMSE: {overall_rmse:.4f}")
    logging.info(f"Point-wise R2: {overall_r2:.4f}")
    logging.info(f"Patient-level Pearson r: {pearson_r:.4f}")
    logging.info(f"Patient-level Spearman rho: {spearman_r:.4f} (p={spearman_p:.4f})")

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "results.npz"),
        all_true=all_true, all_pred=all_pred, fold_r2s=fold_r2s,
    )
    logging.info(f"\nSaved results to {RESULTS_DIR}/results.npz")


if __name__ == "__main__":
    main()
