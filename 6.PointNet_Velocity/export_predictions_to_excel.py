"""
Exports the held-out (test-set, never-trained-on) composite WSS predictions to an
Excel file in the same shape as the original data: two sheets, each (100, 4096) —
rows = patients, columns = points — matching the .mat file's SWSS array exactly.
A third sheet summarizes how well the predictions did, per region and overall.

Requires all four region specialists (pasc/arch/desc/abda) to have completed their
10-fold runs, since it stitches them via stitch_specialists.build_composite().

results.npz stores predictions pooled in KFold-iteration order, not original patient
index order — this reorders rows back to match the original data's patient order by
replicating the identical KFold split (same SEED/N_FOLDS) each region was trained
with, so row i here lines up with row i of the original SWSS matrix / patient
identifiers array 'c'.

Usage (run from 6.PointNet_Velocity/, after all 4 regions have results.npz):
    python export_predictions_to_excel.py
Requires: pandas, openpyxl (pip install pandas openpyxl)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from stitch_specialists import build_composite, REGIONS, concordance_correlation_coefficient
from main import DATA_PATH, N_FOLDS, SEED, calculate_r2

OUTPUT_PATH = "../experiments/predicted_wss.xlsx"


def region_metrics(true_grid, pred_grid, col_slice):
    """
    true_grid/pred_grid: (n_patients, 32, 128). Computes the same metrics main.py and
    stitch_specialists.py print to console — point-level R2/MAE/RMSE over the region's
    own points, and patient-level (median-per-patient) Pearson/Spearman/CCC — so the
    exported file is self-contained without needing the training run's console log.
    """
    true_region = true_grid[:, :, col_slice]
    pred_region = pred_grid[:, :, col_slice]

    true_flat = true_region.reshape(-1)
    pred_flat = pred_region.reshape(-1)
    mae = float(np.mean(np.abs(true_flat - pred_flat)))
    rmse = float(np.sqrt(np.mean((true_flat - pred_flat) ** 2)))
    r2 = calculate_r2(true_flat, pred_flat)

    patient_true = np.median(true_region.reshape(true_region.shape[0], -1), axis=1)
    patient_pred = np.median(pred_region.reshape(pred_region.shape[0], -1), axis=1)
    pearson_r = float(np.corrcoef(patient_true, patient_pred)[0, 1])
    spearman_r, spearman_p = spearmanr(patient_true, patient_pred)
    ccc = float(concordance_correlation_coefficient(patient_true, patient_pred))

    return {
        "Pearson_r (patient-level)": pearson_r,
        "Spearman_r (patient-level)": float(spearman_r),
        "Spearman_p": float(spearman_p),
        "CCC (patient-level)": ccc,
        "R2 (point-level)": float(r2),
        "MAE (point-level)": mae,
        "RMSE (point-level)": rmse,
    }


def build_summary(true_ordered, pred_ordered):
    n_total = true_ordered.shape[0]
    true_grid = true_ordered.reshape(n_total, 32, 128)
    pred_grid = pred_ordered.reshape(n_total, 32, 128)

    region_labels = {
        "pasc": "Proximal Ascending",
        "arch": "Thoracic Arch",
        "desc": "Descending Aorta",
        "abda": "Abdominal Aorta",
    }
    rows = {"Whole_Vessel_Composite": region_metrics(true_grid, pred_grid, slice(0, 128))}
    for name, (col_slice, _) in REGIONS.items():
        rows[region_labels[name]] = region_metrics(true_grid, pred_grid, col_slice)

    summary_df = pd.DataFrame(rows).T
    summary_df.index.name = "Region"
    return summary_df


def main():
    all_true, composite_pred = build_composite()
    n_total = all_true.shape[0]

    data = np.load(DATA_PATH)
    patient_ids = data['c']
    assert len(patient_ids) == n_total, (
        f"Patient identifier count ({len(patient_ids)}) doesn't match results ({n_total}) "
        "— DATA_PATH may not be the same file the specialists were trained on."
    )

    # Undo KFold pooling: recompute the exact same split used at training time (same
    # SEED/N_FOLDS/n_total) to map each pooled row back to its original patient index.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pooled_to_original = []
    for _, test_ix in kf.split(np.arange(n_total)):
        pooled_to_original.extend(test_ix.tolist())
    original_to_pooled = np.empty(n_total, dtype=int)
    for pooled_idx, orig_idx in enumerate(pooled_to_original):
        original_to_pooled[orig_idx] = pooled_idx

    true_ordered = all_true[original_to_pooled]
    pred_ordered = composite_pred[original_to_pooled]

    point_cols = [f"point_{i}" for i in range(true_ordered.shape[1])]
    true_df = pd.DataFrame(true_ordered, columns=point_cols)
    true_df.insert(0, "patient_id", patient_ids)
    pred_df = pd.DataFrame(pred_ordered, columns=point_cols)
    pred_df.insert(0, "patient_id", patient_ids)

    summary_df = build_summary(true_ordered, pred_ordered)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Performance_Summary")
        true_df.to_excel(writer, sheet_name="True_WSS", index=False)
        pred_df.to_excel(writer, sheet_name="Predicted_WSS", index=False)

    print(f"Saved {OUTPUT_PATH}: {true_ordered.shape[0]} patients x {true_ordered.shape[1]} points")
    print("\n" + summary_df.to_string())


if __name__ == "__main__":
    main()
