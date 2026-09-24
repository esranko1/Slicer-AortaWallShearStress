"""
Simple WSS inference - directly loads trained model without complex imports
(no dependency on main.py or sklearn/scipy).
Run from project root:
    python infer_wss_new_aorta.py --pointcloud my_aorta_pointcloud.npz --visualize
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
from pathlib import Path

N_FOLDS = 10
DATA_PATH = "Data/Sampled/Rpt0_N4096.npz"
REGIONS = {
    "pasc": slice(0, 36),
    "arch": slice(36, 60),
    "desc": slice(60, 96),
    "abda": slice(96, 128),
}


# ---------------------------------------------------------------------------
# Model — must match main.py's AortaPointNetVelocity exactly (module names and
# shapes) so the trained fold checkpoints' state_dicts load without remapping.
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, num_groups=8):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.GroupNorm(num_groups, out_features)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.fc(x)
        h = self.norm(h.unsqueeze(-1)).squeeze(-1)
        return self.act(h)


class TransformationNet(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv1 = ConvBlock(num_features, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 1024)
        self.mlp1 = MLPBlock(1024, 512)
        self.mlp2 = MLPBlock(512, 256)
        self.fc_final = nn.Linear(256, num_features * num_features)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = h.max(dim=2)[0]
        h = self.mlp1(h)
        h = self.mlp2(h)
        matrix = self.fc_final(h)
        return matrix.view(-1, self.num_features, self.num_features)


class TransformationBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.transform_net = TransformationNet(num_features)

    def forward(self, x):
        matrix = self.transform_net(x)
        x_t = x.transpose(1, 2)
        transformed = torch.bmm(x_t, matrix)
        return transformed.transpose(1, 2), matrix


class AortaPointNetVelocity(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.input_transform = TransformationBlock(input_channels)
        self.conv64 = ConvBlock(input_channels, 64)
        self.conv128_1 = ConvBlock(64, 128)
        self.conv128_2 = ConvBlock(128, 128)
        self.feature_transform = TransformationBlock(128)
        self.conv512 = ConvBlock(128, 512)
        self.conv2048 = ConvBlock(512, 2048)

        seg_input_channels = 64 + 128 + 128 + 128 + 512 + 2048
        self.seg_conv128 = ConvBlock(seg_input_channels, 128)
        self.seg_conv64 = ConvBlock(128, 64)
        self.seg_conv32 = ConvBlock(64, 32)
        self.pred_head = nn.Conv1d(32, 1, kernel_size=1)
        self.pred_act = nn.SiLU()

    def forward(self, points):
        x = points.transpose(1, 2)
        num_points = x.shape[2]

        x_t, _ = self.input_transform(x)
        f64 = self.conv64(x_t)
        f128_1 = self.conv128_1(f64)
        f128_2 = self.conv128_2(f128_1)
        f_transformed, _ = self.feature_transform(f128_2)
        f512 = self.conv512(f_transformed)
        f2048 = self.conv2048(f512)

        global_feat = f2048.max(dim=2)[0]
        global_broadcast = global_feat.unsqueeze(2).expand(-1, -1, num_points)

        seg_input = torch.cat([f64, f128_1, f128_2, f_transformed, f512, global_broadcast], dim=1)
        s = self.seg_conv128(seg_input)
        s = self.seg_conv64(s)
        s = self.seg_conv32(s)
        out = self.pred_act(self.pred_head(s))
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# Data / inference
# ---------------------------------------------------------------------------

def load_pointcloud(npz_path):
    """Load a (4096, 3) point cloud from an NPZ file."""
    data = np.load(npz_path)
    pts = data['pts']

    if pts.shape[0] == 3:
        pts = pts.transpose(1, 2, 0).reshape(-1, 3)
    else:
        pts = pts.reshape(-1, 3)
    pts = pts.astype(np.float32)

    if pts.shape != (4096, 3):
        raise ValueError(
            f"Expected 4096 points in (x,y,z), got shape {pts.shape}. The model needs "
            "exactly 4096 points in the same 32x128 circumferential-by-longitudinal grid "
            "ordering used at training time — a mismatched count or ordering gives "
            "silently wrong predictions, not just less accurate ones."
        )
    print(f"Loaded point cloud: {pts.shape}")
    return pts


def _get_y_scaling(data_path=DATA_PATH):
    """Recomputes the [0,1] target scaling used at training time (main.py never saves
    it), so predictions can be mapped back to real WSS units. Returns None if the
    training data isn't available on this machine."""
    if not Path(data_path).exists():
        return None
    data = np.load(data_path)
    Y = data['b'][:, :, 0].astype(np.float32)
    return float(Y.min()), float(Y.max())


def _predict_region_ensemble(region_key, points_tensor, device):
    """Averages predictions across all available per-fold checkpoints for one region
    (a 10-fold ensemble is what the reported accuracy is based on — a single fold
    would silently underperform it)."""
    region_dir = Path(f"experiments/6_PointNet_Velocity_roi_{region_key}")
    if not region_dir.exists():
        return None, 0

    preds = []
    for fold_idx in range(N_FOLDS):
        model_path = region_dir / f"fold{fold_idx}_of{N_FOLDS}_model.pth"
        if not model_path.exists():
            continue
        model = AortaPointNetVelocity(input_channels=3).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred = model(points_tensor)[0].cpu().numpy()  # (4096,), scaled [0,1]
        preds.append(pred)

    if not preds:
        return None, 0
    return np.mean(preds, axis=0), len(preds)


def predict_wss_composite(points_3d, device='cpu'):
    """Runs all four region specialists and stitches their own columns of the
    32x128 surface grid into one composite full-vessel WSS prediction."""
    device = torch.device(device)
    points_tensor = torch.tensor(points_3d, dtype=torch.float32, device=device).unsqueeze(0)

    composite_grid = np.zeros((32, 128), dtype=np.float32)
    loaded_regions = 0

    print("\nRunning inference with region specialists...\n")
    for region_key, col_slice in REGIONS.items():
        ensemble_pred, n_folds_loaded = _predict_region_ensemble(region_key, points_tensor, device)
        if ensemble_pred is None:
            print(f"  \u26a0 No checkpoints found for '{region_key}' under "
                  f"experiments/6_PointNet_Velocity_roi_{region_key}/")
            continue

        ensemble_grid = ensemble_pred.reshape(32, 128)
        composite_grid[:, col_slice] = ensemble_grid[:, col_slice]
        print(f"  \u2713 {region_key}: ensembled {n_folds_loaded}/{N_FOLDS} folds "
              f"(columns {col_slice.start}\u2013{col_slice.stop})")
        loaded_regions += 1

    if loaded_regions == 0:
        raise FileNotFoundError(
            "No region specialist checkpoints found under experiments/6_PointNet_Velocity_roi_*/. "
            "Run this script from the project root (the directory containing experiments/)."
        )
    if loaded_regions < 4:
        print(f"\n\u26a0 WARNING: only {loaded_regions}/4 regions loaded — the missing "
              f"region(s) default to 0 in the composite output.")

    composite_scaled = composite_grid.reshape(4096)

    scaling = _get_y_scaling()
    if scaling is None:
        print(f"\n\u26a0 {DATA_PATH} not found — output is left in the model's scaled "
              f"[0,1] range, NOT real WSS units.")
        return composite_scaled

    y_min, y_max = scaling
    return composite_scaled * (y_max - y_min) + y_min


def visualize_wss(points_3d, wss_pred, output_file="wss_prediction.png", elev=20, azim=-60, interactive=False):
    """Visualize WSS predictions on the vessel surface mesh."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import art3d

        # Auto-orient: rotate points so the vessel's own long axis (PCA
        # principal component) becomes the plot's X axis, so the camera below
        # reliably gets a side view regardless of the scan's native
        # orientation -- without this, a vessel whose long axis happens to
        # point at the camera renders as concentric rings, not a tube.
        centered = points_3d - points_3d.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        rotation = eigvecs[:, [2, 1, 0]]  # columns: long axis, 2nd axis, 3rd axis
        points_view = centered @ rotation

        e1 = np.zeros((31, 127), dtype=int)
        e2 = np.zeros((31, 127), dtype=int)
        e3 = np.zeros((31, 127), dtype=int)
        e4 = np.zeros((31, 127), dtype=int)

        for rr in range(31):
            for ll in range(127):
                e1[rr, ll] = rr * 128 + ll
                e4[rr, ll] = (rr * 128) + (ll + 1)
                e2[rr, ll] = ((rr + 1) * 128) + ll
                e3[rr, ll] = ((rr + 1) * 128) + (ll + 1)

        faces = np.hstack((e1.reshape(-1, 1), e2.reshape(-1, 1), e3.reshape(-1, 1), e4.reshape(-1, 1)))
        vecs = points_view[faces, :]

        # Poly3DCollection needs one color per face, not per point -- average
        # each face's 4 corner-point predictions.
        face_wss = wss_pred[faces].mean(axis=1)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        cmap_obj = plt.cm.jet
        vmax = face_wss.max() if face_wss.max() > face_wss.min() else face_wss.min() + 1
        norm = plt.Normalize(vmin=face_wss.min(), vmax=vmax)
        colors = cmap_obj(norm(face_wss))

        ax.add_collection3d(art3d.Poly3DCollection(vecs, facecolors=colors, edgecolor="none", alpha=0.95))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(points_view[:, 0].min() - 5, points_view[:, 0].max() + 5)
        ax.set_ylim(points_view[:, 1].min() - 5, points_view[:, 1].max() + 5)
        ax.set_zlim(points_view[:, 2].min() - 5, points_view[:, 2].max() + 5)
        ax.set_box_aspect([
            np.ptp(points_view[:, 0]), np.ptp(points_view[:, 1]), np.ptp(points_view[:, 2])
        ])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title("WSS Prediction", fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n\u2713 Saved visualization to: {output_file}")

        if interactive:
            print("\nOpening interactive 3D view — click and drag to rotate.")
            print("Close the window when you've found an angle you like.")
            plt.show()
            print(f"\nFinal view angle: elev={ax.elev:.1f}, azim={ax.azim:.1f}")
            print("Pass these back in with --elev / --azim to reproduce this exact view "
                  "next time without rotating manually.")
        plt.close()
    except Exception as e:
        print(f"\n\u26a0 Visualization error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", default="wss_prediction.npy")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=-60)
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print("WSS PREDICTION — composite regional-specialist inference")
    print("=" * 70)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    points_3d = load_pointcloud(args.pointcloud)

    try:
        wss_pred = predict_wss_composite(points_3d, device=args.device)
    except FileNotFoundError as e:
        print(f"\n\u2717 {e}")
        sys.exit(1)

    np.save(args.output, wss_pred)
    print(f"\n\u2713 Saved to: {args.output}")
    print(f"\nWSS Statistics:")
    print(f"  Min:  {wss_pred.min():.4f}")
    print(f"  Max:  {wss_pred.max():.4f}")
    print(f"  Mean: {wss_pred.mean():.4f}")
    print(f"  Std:  {wss_pred.std():.4f}")

    if args.visualize:
        visualize_wss(points_3d, wss_pred, args.output.replace('.npy', '.png'),
                       elev=args.elev, azim=args.azim, interactive=args.interactive)


if __name__ == "__main__":
    main()