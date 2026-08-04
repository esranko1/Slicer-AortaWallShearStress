"""
Combines the four region-specialist models (pasc/arch/desc/abda, each trained with
USE_ROI_LOSS on its own region) into one composite full-vessel prediction, by taking
each specialist's own predictions only for the longitudinal columns it was trained on.

Requires all four specialist runs (main.py with ROI_NAME set to each region in turn)
to have already completed and saved their results.npz files. All four must have used
the identical KFold split (same SEED, same N_FOLDS, same X) for this to be valid —
this script verifies that before stitching anything together.
"""

import numpy as np
from scipy.stats import spearmanr

REGIONS = {
    "pasc": (slice(0, 36), "../experiments/6_PointNet_Velocity_roi_pasc/results.npz"),
    "arch": (slice(36, 60), "../experiments/6_PointNet_Velocity_roi_arch/results.npz"),
    "desc": (slice(60, 96), "../experiments/6_PointNet_Velocity_roi_desc/results.npz"),
    "abda": (slice(96, 128), "../experiments/6_PointNet_Velocity_roi_abda/results.npz"),
}


def concordance_correlation_coefficient(y_true, y_pred):
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    sd_true, sd_pred = np.std(y_true), np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def main():
    data = {name: np.load(path) for name, (_, path) in REGIONS.items()}

    all_true = data["pasc"]["all_true"]
    n_total = all_true.shape[0]

    for name, d in data.items():
        if not np.allclose(d["all_true"], all_true):
            raise ValueError(
                f"all_true mismatch in '{name}' results — fold splits differ across runs, "
                f"stitching would mix predictions from different train/test partitions. "
                f"Check that all four runs used the same SEED/N_FOLDS/X (USE_GEOMETRIC_FEATURES off)."
            )
    print(f"Fold-split consistency check passed across all 4 regions (N={n_total} patients).")

    true_grid = all_true.reshape(n_total, 32, 128)
    composite_pred_grid = np.zeros_like(true_grid)

    for name, (col_slice, _) in REGIONS.items():
        pred_grid = data[name]["all_pred"].reshape(n_total, 32, 128)
        composite_pred_grid[:, :, col_slice] = pred_grid[:, :, col_slice]

    composite_pred = composite_pred_grid.reshape(n_total, -1)
    all_true_flat = true_grid.reshape(n_total, -1)

    patient_true = np.median(all_true_flat, axis=1)
    patient_pred = np.median(composite_pred, axis=1)
    pearson_r = np.corrcoef(patient_true, patient_pred)[0, 1]
    spearman_r, spearman_p = spearmanr(patient_true, patient_pred)
    ccc = concordance_correlation_coefficient(patient_true, patient_pred)
    slope, intercept = np.polyfit(patient_true, patient_pred, 1)

    print("\n=== Composite (stitched) whole-vessel results ===")
    print(f"Pearson correlation:  {pearson_r:.3f}   (single whole-vessel baseline: 0.782)")
    print(f"Spearman correlation: {spearman_r:.3f} {spearman_p:.3g}")
    print(f"CCC: {ccc:.3f}")
    print(f"Slope, Intercept: {slope:.3f} {intercept:.3f}")

    print("\nPer-region Pearson within the composite (should match each specialist's own number):")
    for name, (col_slice, _) in REGIONS.items():
        t = np.median(true_grid[:, :, col_slice].reshape(n_total, -1), axis=1)
        p = np.median(composite_pred_grid[:, :, col_slice].reshape(n_total, -1), axis=1)
        r = np.corrcoef(t, p)[0, 1]
        print(f"  {name}: {r:.3f}")


if __name__ == "__main__":
    main()
