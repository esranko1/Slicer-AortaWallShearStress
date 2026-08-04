"""
Inference pipeline for new patients: runs the four region-specialist model ensembles
(pasc/arch/desc/abda, each an ensemble of its N_FOLDS per-fold checkpoints) and stitches
their outputs into one full-vessel WSS prediction, using each specialist only for the
longitudinal columns it was trained on.

IMPORTANT — new patient input requirements:
  - Point cloud must have exactly 4096 points, in the same 32 (circumferential) x 128
    (longitudinal) grid ordering as the training data (same mesh parameterization
    convention as 1.Data_preprocessing/1.3.convert_mat_to_npz.py produces). The model
    has no way to detect a different point ordering — predictions would be silently
    wrong, not just less accurate, if this isn't matched exactly.
  - Points must be centered on the patient's own centroid (mean-subtracted), matching
    1.3.convert_mat_to_npz.py's centering step.
  - Only xyz (3 input channels) is supported here, matching USE_GEOMETRIC_FEATURES=False
    used to train all four specialists (geometric features were confirmed not to help).

Usage:
    from predict_new_patient import predict_wss
    wss = predict_wss(points_xyz)  # points_xyz: (4096, 3) numpy array, in mm (or
                                    # whatever units the training data's b/WSS array
                                    # used) -> returns (4096,) WSS prediction.
"""

import os
import numpy as np
import torch

from main import AortaPointNetVelocity, ROI_REGIONS, N_FOLDS, DATA_PATH

REGION_RESULTS_DIRS = {
    "pasc": "../experiments/6_PointNet_Velocity_roi_pasc",
    "arch": "../experiments/6_PointNet_Velocity_roi_arch",
    "desc": "../experiments/6_PointNet_Velocity_roi_desc",
    "abda": "../experiments/6_PointNet_Velocity_roi_abda",
}


def _get_y_scaling():
    """
    Recomputes the [0, 1] target scaling used at training time, from the same source
    data file main.py trains on. Must match exactly to inverse-scale predictions back
    to real WSS units — not saved anywhere during training, so it's recomputed here
    the same way main() derives it.
    """
    data = np.load(DATA_PATH)
    Y = data['b'][:, :, 0].astype(np.float32)
    return float(Y.min()), float(Y.max())


def _load_region_ensemble(region_dir, device, input_channels=3):
    """
    Loads all N_FOLDS per-fold checkpoints for one region. Averaging their predictions
    is a standard ensemble over independently-trained folds — more robust for a new,
    unseen patient than picking any single fold's model.
    """
    models = []
    for fold_idx in range(N_FOLDS):
        path = os.path.join(region_dir, f"fold{fold_idx}_of{N_FOLDS}_model.pth")
        model = AortaPointNetVelocity(input_channels=input_channels).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


def predict_wss(points_xyz, device=None):
    """
    points_xyz: (4096, 3) numpy array, centered on the patient's own centroid, in the
    same 32x128 grid point ordering as training data.
    Returns: (4096,) numpy array of predicted WSS in the original (unscaled) units.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.shape != (4096, 3):
        raise ValueError(f"Expected points_xyz shape (4096, 3), got {points_xyz.shape}")

    y_min, y_max = _get_y_scaling()
    x = torch.tensor(points_xyz, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4096, 3)

    composite_grid = np.zeros((32, 128), dtype=np.float32)

    for region, col_slice in ROI_REGIONS.items():
        models = _load_region_ensemble(REGION_RESULTS_DIRS[region], device)
        with torch.no_grad():
            preds = [model(x)[0].cpu().numpy()[0] for model in models]  # each (4096,)
        ensemble_pred = np.mean(preds, axis=0)  # (4096,) scaled [0,1] prediction
        ensemble_grid = ensemble_pred.reshape(32, 128)
        composite_grid[:, col_slice] = ensemble_grid[:, col_slice]

    composite_scaled = composite_grid.reshape(4096)
    wss = composite_scaled * (y_max - y_min) + y_min
    return wss


if __name__ == "__main__":
    # Smoke test: run on training patient 0 just to confirm the pipeline runs end to
    # end and produces sane-looking values. NOT an accuracy check — patient 0 was in
    # the training set for most fold-models per region, so this only verifies plumbing.
    data = np.load(DATA_PATH)
    points_xyz = data['a'][0, :, :3].astype(np.float32)

    wss_pred = predict_wss(points_xyz)
    print(f"Predicted WSS shape: {wss_pred.shape}")
    print(f"Predicted WSS range: [{wss_pred.min():.4f}, {wss_pred.max():.4f}]")
    print(f"Predicted WSS mean: {wss_pred.mean():.4f}")

    true_wss = data['b'][0, :, 0]
    print(f"\nTrue WSS range (patient 0, plumbing check only): [{true_wss.min():.4f}, {true_wss.max():.4f}]")
