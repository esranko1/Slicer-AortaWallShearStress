"""
Final, configurable version of the region-specialist PointNet for the paper:
13.PointNet_Velocity_ValSelect (leak-free) + three options. 6 and 13 are left
untouched.

    python main.py <region> [--cv 10|loo] [--no-flow] [--seed N] [--pwv-weight W]

    region     pasc | arch | desc | abda (specialist, loss on that region only)
               whole (single whole-vessel model: the baseline)
    --cv loo   leave-one-out cross-validation (100 folds) instead of 10-fold
    --no-flow  ablation: drop the inlet velocity waveform input (geometry only)
    --seed     repeat runs with different seeds
    --pwv-weight  weight of the pulse wave velocity loss (default 1; 0 = WSS only)
    --no-retrain  skip the retrain step below (report the epoch-selection model)

Retrain step (default): per fold, the best epoch is first chosen on 10% of the
training patients (validation). Then a fresh model is trained on ALL the
fold's training patients (train + validation) for that many epochs, and only
that model predicts the test patient(s). The test patients are never used for
training, epoch choice or scaling.

PWV: besides the WSS map, the model predicts one pulse wave velocity per
patient (PWVreal from dataForPy9.mat, via extract_pwv.py), from the same global
shape feature (+ the inlet-flow embedding when flow is used), trained jointly
with WSS. Scaled with the training patients' range only, like WSS.

Results go to ../experiments/14_Final_<region>_<cv>_<flow|noflow>[_seed<N>],
so every combination has its own folder and cached folds never mix.
stitch.py builds the composite of the four specialists; compare_runs.py runs
the paired bootstrap between any two settings, with both Pearson metrics.

Below: the docstring of 13, then of 6.

COPY of 6.PointNet_Velocity/main.py (left untouched) with the two test-fold
leaks removed, so the reported numbers are held-out in the strict sense:
  1. Best-epoch selection: 6 keeps the epoch with the lowest loss on the TEST
     fold, then reports that same fold. Here 10% of each fold's training
     patients are set aside as a validation set; the epoch is chosen on them,
     and the test fold is only used once, for the final predictions.
  2. Target scaling: 6 takes Y's min/max over all 100 patients, test fold
     included. Here they come from the fold's training patients only.
Everything else (architecture, loss, folds, seed, early stopping on train loss)
is identical. Results go to ../experiments/13_ValSelect_roi_<region>; the region
is chosen on the command line: python main.py pasc

Original docstring follows.

PI's original PointNet architecture (shape -> WSS), ported from TensorFlow/Keras to
PyTorch, with the inlet velocity waveform encoded through a small dense (Linear) MLP
and fused into the global shape feature, so predictions depend on both geometry and
flow condition, not shape alone.

Kept from the original: the learned input/feature transformation (alignment) blocks,
the SSIM-based loss, early stopping + best-checkpoint restoration, 10-fold CV.
New: mid-fold checkpointing so an interruption mid-training doesn't lose partial progress.

Expects Data/Sampled/Rpt0_N4096.npz (built by 1.Data_preprocessing/1.3.convert_mat_to_npz.py),
with arrays:
    a -> (100, 4096, 53) : xyz (cols 0-2) + inlet velocity waveform (cols 3-52)
    b -> (100, 4096, 1)  : SWSS
    c -> (100,)          : patient identifiers
"""

import argparse
import gc
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
NUM_POINTS = 4096
VELOCITY_DIM = 50

# Flip to False for the full, PI-comparable 10-fold evaluation run. True gives a fast
# 3-fold smoke test — enough to tell whether the new architecture (multi-task TAWSS +
# non-negativity) trains stably and looks directionally promising before committing to
# the full run's wall-clock cost. Fold filenames encode N_FOLDS, so quick-test and
# full-run results never collide on disk.
QUICK_TEST = False

_ap = argparse.ArgumentParser(description="Region-specialist PointNet (final, leak-free)")
_ap.add_argument("region", choices=["pasc", "arch", "desc", "abda", "whole"])
_ap.add_argument("--cv", choices=["10", "loo"], default="10", help="10-fold or leave-one-out")
_ap.add_argument("--no-flow", action="store_true", help="ablation: no inlet velocity input")
_ap.add_argument("--seed", type=int, default=2024)
_ap.add_argument("--pwv-weight", type=float, default=1.0, help="PWV loss weight; 0 = no PWV output")
_ap.add_argument("--no-retrain", action="store_true", help="skip retraining on all training patients")
ARGS = _ap.parse_args()

if QUICK_TEST:
    N_FOLDS = 3
    MAX_EPOCHS = 250
    EARLY_STOP_PATIENCE = 30
else:
    N_FOLDS = None  # set from the data size for leave-one-out, below
    MAX_EPOCHS = 400
    EARLY_STOP_PATIENCE = 50
CV_LOO = ARGS.cv == "loo"
USE_FLOW = not ARGS.no_flow
PWV_WEIGHT = ARGS.pwv_weight
USE_PWV = PWV_WEIGHT > 0
RETRAIN = not ARGS.no_retrain
PWV_PATH = "../Data/Sampled/pwv_N100.npz"  # from extract_pwv.py

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 1.0      # caps gradient norm, guards against the val-loss spikes seen last run
SEED = ARGS.seed
CHECKPOINT_EVERY = 25     # epochs between mid-fold resume checkpoints

# Option B: append precomputed surface normals + inlet-distance to the raw xyz input.
# Kept off by default so RESULTS_DIR stays the proven shape-only baseline; flipping this
# on writes to a separate _geo results dir so it can never clobber the 0.782 baseline's
# cached fold results.
USE_GEOMETRIC_FEATURES = False
GEOMETRIC_FEATURES_PATH = "../X_with_features.npy"

# Project focus (per PI direction): train one specialist model per aortic region
# rather than the whole vessel. Points outside the current ROI still pass through the
# network as geometric context (full 4096-point input is unchanged) — only the loss
# and best-checkpoint selection are restricted to this region. Longitudinal column
# ranges of the 32 (circumferential) x 128 (longitudinal) grid match the existing
# regional diagnostic breakdown further down. Switch ROI_NAME to train each region's
# specialist in turn; each gets its own isolated results dir, so none can collide.
USE_ROI_LOSS = ARGS.region != "whole"
ROI_REGIONS = {
    "pasc": slice(0, 36),    # Proximal Ascending
    "arch": slice(36, 60),   # Thoracic Arch
    "desc": slice(60, 96),   # Descending Aorta
    "abda": slice(96, 128),  # Abdominal Aorta
}
ROI_NAME = ARGS.region
VAL_FRACTION = 0.1  # share of each fold's training patients used to pick the best epoch
ROI_LONGITUDINAL_SLICE = ROI_REGIONS.get(ROI_NAME, slice(0, 128))
RESULTS_DIR = (f"../experiments/14_Final_{ROI_NAME}_{'loo' if CV_LOO else 'cv10'}_"
               f"{'flow' if USE_FLOW else 'noflow'}" + (f"_seed{SEED}" if SEED != 2024 else "")
               + ("" if PWV_WEIGHT == 1.0 else f"_pwvw{PWV_WEIGHT:g}")
               + ("" if RETRAIN else "_noretrain"))

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Model — ported from the PI's conv_block / mlp_block / transformation_block /
# pointNet_model, with a new velocity encoder fused into the global feature.
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d(kernel_size=3, same padding) + GroupNorm + SiLU. Matches PI's conv_block,
    but uses GroupNorm instead of BatchNorm: GroupNorm normalizes using the current
    input directly (no accumulated running statistics), so there's no train/eval-mode
    discrepancy — this is what was causing the validation-loss spikes (BatchNorm's
    running stats, built from small/noisy batches on this small dataset, occasionally
    miscalibrated for the full eval-mode forward pass)."""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MLPBlock(nn.Module):
    """Dense + GroupNorm + SiLU — same GroupNorm-over-BatchNorm reasoning as ConvBlock."""
    def __init__(self, in_features, out_features, num_groups=8):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.GroupNorm(num_groups, out_features)
        self.act = nn.SiLU()

    def forward(self, x):
        # GroupNorm expects (batch, channels, *); x here is (batch, features), so add
        # and then drop a dummy trailing dim around the normalization.
        h = self.fc(x)
        h = self.norm(h.unsqueeze(-1)).squeeze(-1)
        return self.act(h)


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
    def __init__(self, velocity_dim=VELOCITY_DIM, input_channels=3, use_flow=True, use_pwv=True):
        super().__init__()
        self.use_flow = use_flow
        self.use_pwv = use_pwv
        self.input_transform = TransformationBlock(input_channels)
        self.conv64 = ConvBlock(input_channels, 64)
        self.conv128_1 = ConvBlock(64, 128)
        self.conv128_2 = ConvBlock(128, 128)
        self.feature_transform = TransformationBlock(128)
        self.conv512 = ConvBlock(128, 512)
        self.conv2048 = ConvBlock(512, 2048)

        # Encode the inlet velocity waveform into an embedding that gets fused with
        # the global shape feature — per PI direction: just a couple of dense layers,
        # no conv/attention machinery over the waveform itself.
        self.velocity_encoder = nn.Sequential(
            nn.Linear(velocity_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )

        # --no-flow ablation: the velocity embedding is left out of the global feature
        seg_input_channels = 64 + 128 + 128 + 128 + 512 + (2048 + (128 if use_flow else 0))
        self.seg_conv128 = ConvBlock(seg_input_channels, 128)
        self.seg_conv64 = ConvBlock(128, 64)
        self.seg_conv32 = ConvBlock(64, 32)
        self.pred_head = nn.Conv1d(32, 1, kernel_size=1)
        self.pred_act = nn.SiLU()

        # One pulse wave velocity per patient, from the global feature
        if use_pwv:
            self.pwv_head = nn.Sequential(
                nn.Linear(2048 + (128 if use_flow else 0), 128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )

    def forward(self, points, velocity):
        # points: (batch, num_points, input_channels), velocity: (batch, velocity_dim)
        x = points.transpose(1, 2)  # (batch, input_channels, num_points)
        num_points = x.shape[2]

        x_t, matrix1 = self.input_transform(x)
        f64 = self.conv64(x_t)
        f128_1 = self.conv128_1(f64)
        f128_2 = self.conv128_2(f128_1)
        f_transformed, matrix2 = self.feature_transform(f128_2)
        f512 = self.conv512(f_transformed)
        f2048 = self.conv2048(f512)

        global_feat = f2048.max(dim=2)[0]               # (batch, 2048)
        if self.use_flow:
            vel_embed = self.velocity_encoder(velocity)      # (batch, 128)
            global_combined = torch.cat([global_feat, vel_embed], dim=1)  # (batch, 2176)
        else:
            global_combined = global_feat                    # (batch, 2048), geometry only
        global_broadcast = global_combined.unsqueeze(2).expand(-1, -1, num_points)

        seg_input = torch.cat([f64, f128_1, f128_2, f_transformed, f512, global_broadcast], dim=1)
        s = self.seg_conv128(seg_input)
        s = self.seg_conv64(s)
        s = self.seg_conv32(s)
        out = self.pred_act(self.pred_head(s))  # (batch, 1, num_points)
        pwv = self.pwv_head(global_combined).squeeze(-1) if self.use_pwv else None  # (batch,)
        return out.squeeze(1), matrix1, matrix2, pwv  # (batch, num_points), ..., (batch,)


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


def compute_loss(pred, target, matrix1, matrix2, data_range=1.0, roi_slice=None):
    """
    SSIM + MAE + orthogonality regularization, matching the PI's SSIMLoss +
    OrthogonalRegularizer. When roi_slice is given (a slice over the 128 longitudinal
    grid columns), MAE and SSIM are computed only over that region — points outside it
    still pass through the network as geometric context, they just don't contribute
    to the loss, so training optimizes specifically for the region of interest without
    discarding the rest of the vessel as model input.
    """
    batch_size = pred.shape[0]
    pred_grid = pred.reshape(batch_size, 1, 32, 128)
    target_grid = target.reshape(batch_size, 1, 32, 128)

    if roi_slice is not None:
        pred_grid = pred_grid[:, :, :, roi_slice]
        target_grid = target_grid[:, :, :, roi_slice]

    mae = torch.mean(torch.abs(pred_grid - target_grid))
    ssim_val = compute_ssim(pred_grid, target_grid, data_range=data_range)
    ssim_loss = 2 * mae + 1 - ssim_val

    reg_loss = orthogonal_regularization(matrix1) + orthogonal_regularization(matrix2)
    return ssim_loss + reg_loss


# ---------------------------------------------------------------------------
# Training — one fold, with manual early stopping (train-loss plateau) and
# explicit best-validation-loss checkpoint restoration, matching the PI's
# EarlyStopping(monitor='loss') + ModelCheckpoint(monitor='val_loss') pairing.
# ---------------------------------------------------------------------------

def fold_results_path(fold_idx):
    # N_FOLDS is baked into the filename deliberately: switching N_FOLDS changes what
    # the train/test partitions actually are (3-fold and 10-fold split the same 100
    # patients completely differently), so results from one N_FOLDS setting are never
    # valid to reuse for another — this naming makes that impossible to mix up by accident.
    return os.path.join(RESULTS_DIR, f"fold{fold_idx}_of{N_FOLDS}_results.npz")


def fold_model_path(fold_idx):
    return os.path.join(RESULTS_DIR, f"fold{fold_idx}_of{N_FOLDS}_model.pth")


def resume_checkpoint_path(fold_idx):
    # Mid-fold progress, separate from fold_model_path: this holds optimizer state and
    # epoch/early-stopping bookkeeping, gets updated every CHECKPOINT_EVERY epochs
    # while a fold is still training, and is deleted once the fold finishes cleanly.
    return os.path.join(RESULTS_DIR, f"fold{fold_idx}_of{N_FOLDS}_resume.pth")


def calculate_r2(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    return 1 - ss_res / ss_tot


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient — matches the PI's original function."""
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def pwv_loss(pwv_pred, pwv_true):
    """MAE on the [0, 1]-scaled PWV, weighted against the WSS loss."""
    if pwv_pred is None:
        return 0.0
    return PWV_WEIGHT * torch.mean(torch.abs(pwv_pred - pwv_true))


def train_one_fold(X_train, Y_train, Z_train, X_val, Y_val, Z_val, X_test, Z_test, device, fold_idx,
                   input_channels=3, P_train=None, P_val=None):
    model = AortaPointNetVelocity(input_channels=input_channels, use_flow=USE_FLOW, use_pwv=USE_PWV).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    roi_slice = ROI_LONGITUDINAL_SLICE if USE_ROI_LOSS else None

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    Z_train_t = torch.tensor(Z_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)
    Z_val_t = torch.tensor(Z_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Z_test_t = torch.tensor(Z_test, dtype=torch.float32, device=device)
    P_train_t = torch.tensor(P_train, dtype=torch.float32, device=device) if USE_PWV else None
    P_val_t = torch.tensor(P_val, dtype=torch.float32, device=device) if USE_PWV else None

    # Verify everything is on GPU
    if device.type == 'cuda':
        logging.info(f"  Fold {fold_idx}: Data on GPU ✓ ({X_train_t.device}, {Y_train_t.device})")
        logging.info(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB used")

    n_train = X_train_t.shape[0]
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    epochs_without_train_improvement = 0
    best_train_loss = float('inf')

    # Resume mid-fold training if this fold was interrupted (crash/reboot) after at
    # least one checkpoint interval — avoids losing partial progress within a fold,
    # on top of the existing per-fold (whole-fold) checkpointing in main().
    resume_path = resume_checkpoint_path(fold_idx)
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        torch.set_rng_state(checkpoint['rng_state'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_state = checkpoint['best_state']
        best_epoch = checkpoint.get('best_epoch', 0)
        best_train_loss = checkpoint['best_train_loss']
        epochs_without_train_improvement = checkpoint['epochs_without_train_improvement']
        logging.info(f"  Fold {fold_idx}: resuming mid-fold checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_train_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            pred, m1, m2, pwv = model(X_train_t[idx], Z_train_t[idx])
            loss = compute_loss(pred, Y_train_t[idx], m1, m2, roi_slice=roi_slice)
            if USE_PWV:
                loss = loss + pwv_loss(pwv, P_train_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        epoch_train_loss /= n_batches

        model.eval()
        with torch.no_grad():
            # Epoch selection on the validation patients, never on the test fold
            val_pred, vm1, vm2, val_pwv = model(X_val_t, Z_val_t)
            val_loss = compute_loss(val_pred, Y_val_t, vm1, vm2, roi_slice=roi_slice)
            if USE_PWV:
                val_loss = val_loss + pwv_loss(val_pwv, P_val_t)
            val_loss = float(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch_train_loss < best_train_loss - 1e-5:
            best_train_loss = epoch_train_loss
            epochs_without_train_improvement = 0
        else:
            epochs_without_train_improvement += 1

        if epoch % 50 == 0 or epochs_without_train_improvement >= EARLY_STOP_PATIENCE:
            logging.info(f"  Fold {fold_idx} epoch {epoch}: train_loss={epoch_train_loss:.4f} val_loss={val_loss:.4f}")

        stop_early = epochs_without_train_improvement >= EARLY_STOP_PATIENCE
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or stop_early:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_state': best_state,
                'best_epoch': best_epoch,
                'best_train_loss': best_train_loss,
                'epochs_without_train_improvement': epochs_without_train_improvement,
            }, resume_path)

        if stop_early:
            logging.info(f"  Fold {fold_idx}: early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred, _, _, final_pwv = model(X_test_t, Z_test_t)

    # Fold finished cleanly — the mid-fold resume checkpoint is now superseded by
    # fold_results_path()/fold_model_path() (saved by the caller), so drop it.
    if os.path.exists(resume_path):
        os.remove(resume_path)

    final_pwv = final_pwv.cpu().numpy() if final_pwv is not None else None
    return final_pred.cpu().numpy(), final_pwv, best_val_loss, best_state, best_epoch


def retrain_fixed_epochs(X_all, Y_all, Z_all, P_all, n_epochs, X_test, Z_test, device, fold_idx,
                         input_channels=3):
    """
    Train a fresh model on all of the fold's training patients for a fixed number
    of epochs (chosen beforehand on the validation patients), then predict the
    test patients. No validation and no early stopping here: the epoch count is
    already decided, so the test patients play no part in it.
    """
    torch.manual_seed(SEED + 1000 + fold_idx)
    model = AortaPointNetVelocity(input_channels=input_channels, use_flow=USE_FLOW, use_pwv=USE_PWV).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    roi_slice = ROI_LONGITUDINAL_SLICE if USE_ROI_LOSS else None
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    X_t, Y_t, Z_t = to(X_all), to(Y_all), to(Z_all)
    P_t = to(P_all) if USE_PWV else None
    n = X_t.shape[0]
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            pred, m1, m2, pwv = model(X_t[idx], Z_t[idx])
            loss = compute_loss(pred, Y_t[idx], m1, m2, roi_slice=roi_slice)
            if USE_PWV:
                loss = loss + pwv_loss(pwv, P_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        if epoch % 50 == 0:
            logging.info(f"  Fold {fold_idx} retrain epoch {epoch}/{n_epochs}: train_loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        pred, _, _, pwv = model(to(X_test), to(Z_test))
    return pred.cpu().numpy(), (pwv.cpu().numpy() if pwv is not None else None), model.state_dict()


# ---------------------------------------------------------------------------
# Main — 10-fold CV, matching the PI's original evaluation methodology.
# ---------------------------------------------------------------------------

def main():
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU (Metal), e.g. M-series Macs
    else:
        device = torch.device('cpu')
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Available: {torch.cuda.is_available()}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        logging.info("Using Apple MPS (Metal) backend")
    else:
        logging.warning("⚠️  RUNNING ON CPU — This will be slow!")

    data = np.load(DATA_PATH)
    a, b = data['a'], data['b']
    X = a[:, :, :3].astype(np.float32)          # (100, 4096, 3)
    Z = a[:, 0, 3:].astype(np.float32)           # (100, 50)
    Y = b[:, :, 0].astype(np.float32)            # (100, 4096)

    if USE_GEOMETRIC_FEATURES:
        geo = np.load(GEOMETRIC_FEATURES_PATH)  # (100, 4096, 7): x,y,z,nx,ny,nz,dist
        geo_feats = geo[:, :, 3:7].astype(np.float32)  # [nx, ny, nz, dist]
        X = np.concatenate([X, geo_feats], axis=2)  # (100, 4096, 7)
        logging.info(f"Using geometric features from {GEOMETRIC_FEATURES_PATH}")
    input_channels = X.shape[-1]

    PWV = None
    if USE_PWV:
        if not os.path.exists(PWV_PATH):
            raise SystemExit(f"{PWV_PATH} not found: run extract_pwv.py first (or --pwv-weight 0 for WSS only).")
        pwv_file = np.load(PWV_PATH)
        if not np.array_equal(pwv_file['c'].astype(str), data['c'].astype(str)):
            raise SystemExit(f"Patient order in {PWV_PATH} does not match {DATA_PATH}.")
        PWV = pwv_file['pwv'].astype(np.float32)  # (100,) m/s
        logging.info(f"PWV: {PWV.shape}, range [{PWV.min():.2f}, {PWV.max():.2f}] m/s, loss weight {PWV_WEIGHT}")

    logging.info(f"X: {X.shape}, Z: {Z.shape}, Y: {Y.shape}")
    logging.info(f"Y range: [{Y.min():.4f}, {Y.max():.4f}]")

    global N_FOLDS
    if N_FOLDS is None:
        N_FOLDS = X.shape[0] if CV_LOO else 10
    logging.info(f"Region {ROI_NAME}, {'leave-one-out' if CV_LOO else '10-fold'} CV ({N_FOLDS} folds), "
                 f"{'with' if USE_FLOW else 'WITHOUT'} inlet flow input, seed {SEED} -> {RESULTS_DIR}")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_true, all_pred = [], []
    all_pwv_true, all_pwv_pred = [], []
    fold_r2s = []
    fold_mses = []

    for fold_idx, (train_ix, test_ix) in enumerate(kf.split(X)):
        results_path = fold_results_path(fold_idx)

        if os.path.exists(results_path):
            logging.info(f"\n=== Fold {fold_idx} === (already completed — loading cached results)")
            cached = np.load(results_path)
            cached_retrained = bool(cached['retrained']) if 'retrained' in cached.files else False
            if cached_retrained != RETRAIN:
                raise SystemExit(f"{results_path} was made {'with' if not RETRAIN else 'without'} the retrain step; "
                                 f"delete {RESULTS_DIR} (or use the matching --no-retrain setting) and rerun.")
            if USE_FLOW and 'flow_scaled' not in cached.files:
                raise SystemExit(f"{results_path} was made before the inlet-flow input was scaled; "
                                 f"delete {RESULTS_DIR} and rerun.")
            pred, true, fold_r2 = cached['pred'], cached['true'], float(cached['r2'])
            pwv_pred = cached['pwv_pred'] if 'pwv_pred' in cached.files else None
            pwv_true = cached['pwv_true'] if 'pwv_true' in cached.files else None
        else:
            logging.info(f"\n=== Fold {fold_idx} ===")
            # Validation patients come out of this fold's training patients only
            rng = np.random.default_rng(SEED + fold_idx)
            n_val = max(1, int(round(VAL_FRACTION * len(train_ix))))
            val_ix = rng.choice(train_ix, n_val, replace=False)
            fit_ix = np.setdiff1d(train_ix, val_ix)
            logging.info(f"  {len(fit_ix)} train / {len(val_ix)} validation (epoch selection) / {len(test_ix)} test patients")

            # Scale Y to [0, 1] with the training patients' range only
            y_min, y_max = Y[fit_ix].min(), Y[fit_ix].max()
            scale = lambda v: (v - y_min) / (y_max - y_min)  # noqa: E731

            # Inlet flow waveform scaled by its largest magnitude among the training
            # patients (raw values reach the hundreds, which saturates the flow
            # encoder). Shape and timing are unchanged.
            z_max = float(np.abs(Z[fit_ix]).max()) or 1.0
            z_scale = lambda v: v / z_max  # noqa: E731

            # PWV scaled with the training patients' range only, like WSS
            if USE_PWV:
                p_min, p_max = PWV[fit_ix].min(), PWV[fit_ix].max()
                p_scale = lambda v: (v - p_min) / (p_max - p_min)  # noqa: E731

            pred_scaled, pwv_scaled, best_val_loss, best_state, best_epoch = train_one_fold(
                X[fit_ix], scale(Y[fit_ix]), z_scale(Z[fit_ix]),
                X[val_ix], scale(Y[val_ix]), z_scale(Z[val_ix]),
                X[test_ix], z_scale(Z[test_ix]),
                device, fold_idx, input_channels=input_channels,
                P_train=p_scale(PWV[fit_ix]) if USE_PWV else None,
                P_val=p_scale(PWV[val_ix]) if USE_PWV else None,
            )
            if RETRAIN:
                # Fresh model on all training patients (train + validation) for the
                # chosen number of epochs; scaling now from all training patients.
                n_epochs = best_epoch + 1
                logging.info(f"  Best epoch {best_epoch} on validation -> retraining on all "
                             f"{len(train_ix)} training patients for {n_epochs} epochs")
                # scale / p_scale read y_min.. at call time, so they now use this range
                y_min, y_max = Y[train_ix].min(), Y[train_ix].max()
                z_max = float(np.abs(Z[train_ix]).max()) or 1.0
                if USE_PWV:
                    p_min, p_max = PWV[train_ix].min(), PWV[train_ix].max()
                pred_scaled, pwv_scaled, best_state = retrain_fixed_epochs(
                    X[train_ix], scale(Y[train_ix]), z_scale(Z[train_ix]),
                    p_scale(PWV[train_ix]) if USE_PWV else None,
                    n_epochs, X[test_ix], z_scale(Z[test_ix]), device, fold_idx, input_channels=input_channels)
            pred = pred_scaled * (y_max - y_min) + y_min
            true = Y[test_ix]
            pwv_pred = pwv_scaled * (p_max - p_min) + p_min if USE_PWV else None
            pwv_true = PWV[test_ix] if USE_PWV else None
            fold_r2 = calculate_r2(true.flatten(), pred.flatten())

            # Save immediately — if the run gets interrupted later, this fold's work
            # is not lost, and a re-run will skip straight past it.
            extra = dict(pwv_pred=pwv_pred, pwv_true=pwv_true) if USE_PWV else {}
            np.savez_compressed(results_path, pred=pred, true=true, r2=fold_r2, best_val_loss=best_val_loss,
                                best_epoch=best_epoch, retrained=RETRAIN, flow_scaled=True, **extra)
            torch.save(best_state, fold_model_path(fold_idx))
            logging.info(f"Fold {fold_idx}: best_val_loss={best_val_loss:.4f}, R2={fold_r2:.4f} (saved to {results_path})")

            # Free this fold's models before the next one: with 100 leave-one-out
            # folds, GPU memory that is not handed back promptly adds up.
            del best_state, pred_scaled, pwv_scaled
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                logging.info(f"  GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Whole-vessel R2 above no longer reflects what the model is optimized for once
        # USE_ROI_LOSS is on — it's expected to look bad/negative since the model isn't
        # being trained to fit those other regions anymore. This is the number that
        # actually matters for the current project focus.
        n_test = true.shape[0]
        true_roi = true.reshape(n_test, 32, 128)[:, :, ROI_LONGITUDINAL_SLICE]
        pred_roi = pred.reshape(n_test, 32, 128)[:, :, ROI_LONGITUDINAL_SLICE]
        fold_r2_roi = calculate_r2(true_roi.flatten(), pred_roi.flatten())
        logging.info(f"Fold {fold_idx}: {ROI_NAME} ROI R2={fold_r2_roi:.4f} "
                     f"(whole-vessel R2={fold_r2:.4f}, not the current optimization target)")

        fold_r2s.append(fold_r2)
        fold_mses.append(np.mean((true.flatten() - pred.flatten()) ** 2))
        all_true.append(true)
        all_pred.append(pred)
        if pwv_pred is not None:
            all_pwv_true.append(pwv_true)
            all_pwv_pred.append(pwv_pred)

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    overall_r2 = calculate_r2(all_true.flatten(), all_pred.flatten())
    overall_mae = np.mean(np.abs(all_true.flatten() - all_pred.flatten()))
    overall_mse = np.mean((all_true.flatten() - all_pred.flatten()) ** 2)
    overall_rmse = np.sqrt(overall_mse)

    patient_true = np.median(all_true, axis=1)
    patient_pred = np.median(all_pred, axis=1)
    pearson_r = np.corrcoef(patient_true, patient_pred)[0, 1]
    spearman_r, spearman_p = spearmanr(patient_true, patient_pred)
    ccc = concordance_correlation_coefficient(patient_true, patient_pred)
    slope, intercept = np.polyfit(patient_true, patient_pred, 1)

    # Region-wise point-level Spearman — same 32 (circumferential) x 128 (longitudinal)
    # grid slicing the PI's original notebook uses, on the longitudinal axis.
    n_total = all_true.shape[0]
    true_grid = all_true.reshape(n_total, 32, 128)
    pred_grid = all_pred.reshape(n_total, 32, 128)
    acc_pasc, pval_pasc = spearmanr(true_grid[:, :, 0:36].flatten(), pred_grid[:, :, 0:36].flatten())
    acc_arch, pval_arch = spearmanr(true_grid[:, :, 36:60].flatten(), pred_grid[:, :, 36:60].flatten())
    acc_desc, pval_desc = spearmanr(true_grid[:, :, 60:96].flatten(), pred_grid[:, :, 60:96].flatten())
    acc_abda, pval_abda = spearmanr(true_grid[:, :, 96:128].flatten(), pred_grid[:, :, 96:128].flatten())

    logging.info(f"\n=== Overall results (pooled across all {N_FOLDS} folds) ===")
    logging.info(f"Per-fold R2: {[f'{r:.3f}' for r in fold_r2s]}")
    logging.info(f"Per-fold MSE: {np.mean(fold_mses):.4f} (+/- {np.std(fold_mses):.4f})")
    logging.info(f"Point-wise MAE: {overall_mae:.4f}")
    logging.info(f"Point-wise MSE: {overall_mse:.4f}")
    logging.info(f"Point-wise RMSE: {overall_rmse:.4f}")
    logging.info(f"Point-wise R2: {overall_r2:.4f}")

    logging.info("\n--- Patient-level diagnostics (matching PI's reporting) ---")
    logging.info(f"Pearsons correlation: {pearson_r:.3f}")
    logging.info(f"Spearmans correlation: {spearman_r:.3f} {spearman_p:.3f}")
    logging.info(f"CCC: {ccc:.3f}")
    logging.info(f"Slope, Intercept: {slope:.3f} {intercept:.3f}")

    logging.info(
        "\nSpearmans correlations:\n"
        f" Proximal Ascending: {acc_pasc:.3f} {pval_pasc:.3f}\n"
        f" Thoracic Arch: {acc_arch:.3f} {pval_arch:.3f}\n"
        f" Descending Aorta: {acc_desc:.3f} {pval_desc:.3f}\n"
        f" Abdominal Aorta: {acc_abda:.3f} {pval_abda:.3f}"
    )

    # Region-wise patient-level Pearson (per-patient median within the region, then
    # correlated across patients) — same regions as the Spearman breakdown above, but
    # Pearson to match how the whole-vessel number is reported, since the project is
    # now specifically focused on maximizing the Proximal Ascending region.
    def region_pearson(col_slice):
        t = np.median(true_grid[:, :, col_slice].reshape(n_total, -1), axis=1)
        p = np.median(pred_grid[:, :, col_slice].reshape(n_total, -1), axis=1)
        return np.corrcoef(t, p)[0, 1]

    pearson_pasc = region_pearson(slice(0, 36))
    pearson_arch = region_pearson(slice(36, 60))
    pearson_desc = region_pearson(slice(60, 96))
    pearson_abda = region_pearson(slice(96, 128))

    region_labels = {"pasc": "Proximal Ascending", "arch": "Thoracic Arch",
                     "desc": "Descending Aorta", "abda": "Abdominal Aorta"}
    focus_marker = {name: ("  <-- current project focus" if USE_ROI_LOSS and name == ROI_NAME else "")
                    for name in region_labels}

    logging.info(
        "\nPatient-level Pearson by region:\n"
        f" Proximal Ascending: {pearson_pasc:.3f}{focus_marker['pasc']}\n"
        f" Thoracic Arch: {pearson_arch:.3f}{focus_marker['arch']}\n"
        f" Descending Aorta: {pearson_desc:.3f}{focus_marker['desc']}\n"
        f" Abdominal Aorta: {pearson_abda:.3f}{focus_marker['abda']}"
    )

    pwv_out = {}
    if all_pwv_true:
        pt, pp = np.concatenate(all_pwv_true), np.concatenate(all_pwv_pred)
        diff = pp - pt
        logging.info(
            "\nPulse wave velocity (one value per patient, held-out):\n"
            f" Pearson r: {np.corrcoef(pt, pp)[0, 1]:.3f}   Spearman: {spearmanr(pt, pp)[0]:.3f}\n"
            f" MAE: {np.mean(np.abs(diff)):.2f} m/s   RMSE: {np.sqrt(np.mean(diff ** 2)):.2f} m/s\n"
            f" Bland-Altman: bias {diff.mean():+.2f} m/s, 95% limits of agreement "
            f"[{diff.mean() - 1.96 * diff.std():+.2f}, {diff.mean() + 1.96 * diff.std():+.2f}] m/s"
        )
        pwv_out = dict(all_pwv_true=pt, all_pwv_pred=pp)

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "results.npz"),
        all_true=all_true, all_pred=all_pred, fold_r2s=fold_r2s, fold_mses=fold_mses,
        pearson_pasc=pearson_pasc, pearson_arch=pearson_arch,
        pearson_desc=pearson_desc, pearson_abda=pearson_abda, **pwv_out,
    )
    logging.info(f"\nSaved results to {RESULTS_DIR}/results.npz")


if __name__ == "__main__":
    main()
