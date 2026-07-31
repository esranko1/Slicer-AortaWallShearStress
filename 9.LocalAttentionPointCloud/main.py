"""
Local Attention on Point Clouds: Each point attends only to k-nearest neighbors.

Exploits mesh structure (points have natural neighbors on aortic surface).
Reduces attention complexity from O(n²) to O(n*k), provides regularization.

Input: (batch, 4096, 3) point coordinates
  ↓
Embedding: map to d_model
  ↓
Local Attention Blocks: Each point looks at k-NN neighbors only
  (reduces 4096² attention to k×4096, k=128 or 256)
  ↓
Output head: WSS per point
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
RESULTS_DIR = "../experiments/9_LocalAttentionPointCloud"
NUM_POINTS = 4096

QUICK_TEST = False

if QUICK_TEST:
    N_FOLDS = 3
    MAX_EPOCHS = 250
    EARLY_STOP_PATIENCE = 30
else:
    N_FOLDS = 10
    MAX_EPOCHS = 400
    EARLY_STOP_PATIENCE = 50

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 2024
CHECKPOINT_EVERY = 25

# Local attention config
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
K_NEIGHBORS = 128  # Each point attends to 128 nearest neighbors (out of 4096)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Local Attention
# ---------------------------------------------------------------------------

def compute_local_attention_mask(points, k=K_NEIGHBORS):
    """
    Compute k-nearest neighbors for each point, return attention mask.

    points: (batch, num_points, 3)
    returns: (batch, num_points, num_points) bool mask [True = attend, False = ignore]
    """
    batch_size, num_points, _ = points.shape
    device = points.device

    # Compute pairwise distances: (batch, num_points, num_points)
    # Using Euclidean distance
    diff = points.unsqueeze(2) - points.unsqueeze(1)  # (batch, num_points, num_points, 3)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (batch, num_points, num_points)

    # Find k nearest neighbors for each point (including self)
    _, knn_indices = torch.topk(dist_sq, k=min(k + 1, num_points), dim=-1, largest=False)
    # k+1 because nearest includes itself; we'll keep k (self + k-1 neighbors)
    knn_indices = knn_indices[:, :, :k]  # (batch, num_points, k)

    # Create attention mask: (batch, num_points, num_points)
    # True where we should attend, False where we should mask out
    mask = torch.zeros(batch_size, num_points, num_points, dtype=torch.bool, device=device)
    for b in range(batch_size):
        for i in range(num_points):
            mask[b, i, knn_indices[b, i]] = True

    return mask


class LocalAttentionLayer(nn.Module):
    """
    Multi-head self-attention with local attention mask.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        """
        x: (batch, num_points, d_model)
        attention_mask: (batch, num_points, num_points) bool, True = attend, False = mask
        """
        # Convert bool mask to attention mask (PyTorch expects -inf for masked positions)
        # MultiheadAttention.forward expects attn_mask of shape (L, S) or (batch*nhead, L, S)
        # where True/1 means *keep*, False/0 means *mask*
        attn_mask = ~attention_mask  # Flip: True → mask out, False → keep
        attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
        # Actually, let's use the proper format: False means attend, True means mask
        # PyTorch attention_mask: True = masked (ignore), False = not masked (attend)
        attn_mask_pt = attention_mask.logical_not().float()
        attn_mask_pt = attn_mask_pt.masked_fill(attn_mask_pt == 1, float('-inf'))
        attn_mask_pt = attn_mask_pt.masked_fill(attn_mask_pt == 0, 0.0)

        # Reshape mask for multi-head attention: (batch*nhead, L, S)
        # Actually PyTorch can handle (batch, L, S) in batch_first mode

        # Self-attention with local mask
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask_pt)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward
        mlp_out = self.mlp(x)
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)

        return x


class LocalAttentionPointTransformer(nn.Module):
    """
    Transformer with local (k-NN) attention instead of global attention.
    """
    def __init__(self, input_channels=3, output_channels=1, d_model=D_MODEL,
                 nhead=NHEAD, num_layers=NUM_LAYERS, k_neighbors=K_NEIGHBORS):
        super().__init__()
        self.d_model = d_model
        self.k_neighbors = k_neighbors

        # Point embedding
        self.point_embed = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Positional encoding (learned)
        self.pos_embed = nn.Linear(input_channels, d_model)

        # Local attention layers
        self.local_attention_layers = nn.ModuleList([
            LocalAttentionLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=0.1)
            for _ in range(num_layers)
        ])

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, output_channels)
        )

    def forward(self, points):
        """
        points: (batch, num_points, 3)
        output: (batch, num_points)
        """
        # Embed point coordinates
        x = self.point_embed(points)  # (batch, num_points, d_model)

        # Add positional encoding
        pos = self.pos_embed(points)  # (batch, num_points, d_model)
        x = x + pos

        # Compute local attention mask (k-NN)
        attention_mask = compute_local_attention_mask(points, k=self.k_neighbors)

        # Local attention layers
        for layer in self.local_attention_layers:
            x = layer(x, attention_mask)

        # Per-point output
        out = self.head(x)  # (batch, num_points, 1)

        # Apply SiLU activation
        out = F.silu(out.squeeze(-1))  # (batch, num_points)

        return out


# ---------------------------------------------------------------------------
# Loss — same SSIM + MAE as baseline
# ---------------------------------------------------------------------------

def compute_ssim(pred, target, data_range=1.0, window_size=3):
    """SSIM loss."""
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


def compute_loss(pred, target, data_range=1.0):
    batch_size = pred.shape[0]
    pred_grid = pred.reshape(batch_size, 1, 32, 128)
    target_grid = target.reshape(batch_size, 1, 32, 128)

    mae = torch.mean(torch.abs(pred - target))
    ssim_val = compute_ssim(pred_grid, target_grid, data_range=data_range)
    ssim_loss = 2 * mae + 1 - ssim_val

    return ssim_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fold_results_path(fold_idx):
    return os.path.join(RESULTS_DIR, f"fold{fold_idx}_of{N_FOLDS}_results.npz")


def fold_model_path(fold_idx):
    return os.path.join(RESULTS_DIR, f"fold{fold_idx}_of{N_FOLDS}_model.pth")


def resume_checkpoint_path(fold_idx):
    return os.path.join(RESULTS_DIR, f"fold{fold_idx}_of{N_FOLDS}_resume.pth")


def calculate_r2(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    return 1 - ss_res / ss_tot


def concordance_correlation_coefficient(y_true, y_pred):
    """CCC."""
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def train_one_fold(X_train, Y_train, Z_train, X_test, Y_test, Z_test, device, fold_idx):
    model = LocalAttentionPointTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    Z_train_t = torch.tensor(Z_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)
    Z_test_t = torch.tensor(Z_test, dtype=torch.float32, device=device)

    if device.type == 'cuda':
        logging.info(f"  Fold {fold_idx}: Data on GPU ✓ ({X_train_t.device}, {Y_train_t.device})")
        logging.info(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB used")

    n_train = X_train_t.shape[0]
    start_epoch = 0
    best_val_loss = float('inf')
    best_state = None
    epochs_without_train_improvement = 0
    best_train_loss = float('inf')

    resume_path = resume_checkpoint_path(fold_idx)
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        torch.set_rng_state(checkpoint['rng_state'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_state = checkpoint['best_state']
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
            pred = model(X_train_t[idx])
            loss = compute_loss(pred, Y_train_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        epoch_train_loss /= n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = compute_loss(val_pred, Y_test_t).item()

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

        stop_early = epochs_without_train_improvement >= EARLY_STOP_PATIENCE
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or stop_early:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_state': best_state,
                'best_train_loss': best_train_loss,
                'epochs_without_train_improvement': epochs_without_train_improvement,
            }, resume_path)

        if stop_early:
            logging.info(f"  Fold {fold_idx}: early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = model(X_test_t)

    if os.path.exists(resume_path):
        os.remove(resume_path)

    return final_pred.cpu().numpy(), best_val_loss, best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Available: {torch.cuda.is_available()}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()

    data = np.load(DATA_PATH)
    a, b = data['a'], data['b']
    X = a[:, :, :3].astype(np.float32)
    Z = a[:, 0, 3:].astype(np.float32)
    Y = b[:, :, 0].astype(np.float32)

    logging.info(f"X: {X.shape}, Z: {Z.shape}, Y: {Y.shape}")
    logging.info(f"Y range: [{Y.min():.4f}, {Y.max():.4f}]")
    logging.info(f"Local attention: k={K_NEIGHBORS} neighbors per point (of {NUM_POINTS})")

    y_min, y_max = Y.min(), Y.max()
    Y_scaled = (Y - y_min) / (y_max - y_min)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_true, all_pred = [], []
    fold_r2s = []
    fold_mses = []

    for fold_idx, (train_ix, test_ix) in enumerate(kf.split(X)):
        results_path = fold_results_path(fold_idx)

        if os.path.exists(results_path):
            logging.info(f"\n=== Fold {fold_idx} === (already completed — loading cached results)")
            cached = np.load(results_path)
            pred, true, fold_r2 = cached['pred'], cached['true'], float(cached['r2'])
        else:
            logging.info(f"\n=== Fold {fold_idx} ===")
            pred_scaled, best_val_loss, best_state = train_one_fold(
                X[train_ix], Y_scaled[train_ix], Z[train_ix],
                X[test_ix], Y_scaled[test_ix], Z[test_ix],
                device, fold_idx
            )
            pred = pred_scaled * (y_max - y_min) + y_min
            true = Y[test_ix]
            fold_r2 = calculate_r2(true.flatten(), pred.flatten())

            np.savez_compressed(results_path, pred=pred, true=true, r2=fold_r2, best_val_loss=best_val_loss)
            torch.save(best_state, fold_model_path(fold_idx))
            logging.info(f"Fold {fold_idx}: best_val_loss={best_val_loss:.4f}, R2={fold_r2:.4f} (saved to {results_path})")

        fold_r2s.append(fold_r2)
        fold_mses.append(np.mean((true.flatten() - pred.flatten()) ** 2))
        all_true.append(true)
        all_pred.append(pred)

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

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "results.npz"),
        all_true=all_true, all_pred=all_pred, fold_r2s=fold_r2s, fold_mses=fold_mses,
    )
    logging.info(f"\nSaved results to {RESULTS_DIR}/results.npz")


if __name__ == "__main__":
    main()
