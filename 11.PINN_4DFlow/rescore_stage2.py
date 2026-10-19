"""
Rescores the saved Stage 2 predictions with the metric used for the
region-specialist PointNet and the PI's baseline, so the two-stage PINN can go
in the same results table. No retraining.

Metric ("level r"): for each patient, the median WSS over the wall points of a
region at peak systole; then the Pearson correlation of those medians across
patients, predicted vs label. Regions are the same longitudinal ranges as the
PointNet models: pasc 0-35, arch 36-59, desc 60-95, abda 96-127 (of 128).

Peak systole = the label frame with the highest mean WSS, per patient.
Points here are stored axial-major (index = axial * 32 + circumferential), as
written by 1.Data_preprocessing/1.5.process_aggregated_mat.py.

Note for the table: the PINN's target is WSS_Norm3D from the AggregatedData
files; the PointNet models' target is SWSSreal from dataForPy9.mat.

Usage (from this folder):
    python rescore_stage2.py                 # Stage 2 with PINN flow inputs
    python rescore_stage2.py --drop-flow     # the ablation run
"""

import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_RESULTS_DIR

N_AXIAL, N_CIRC = 128, 32
REGIONS = {
    "Overall (whole vessel)": slice(0, 128),
    "Proximal ascending": slice(0, 36),
    "Thoracic arch": slice(36, 60),
    "Descending aorta": slice(60, 96),
    "Abdominal aorta": slice(96, 128),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--drop-flow", action="store_true", help="score the --drop-flow Stage 2 run instead")
    args = ap.parse_args()

    pred_dir = Path(args.results) / ("stage2_predictions_noflow" if args.drop_flow else "stage2_predictions")
    files = sorted(pred_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No predictions in {pred_dir}")

    level = {name: {"label": [], "stage2": [], "pinn_raw": []} for name in REGIONS}
    for f in files:
        with np.load(f) as d:
            label, pred, raw = d["label"], d["pred"], d["pinn_raw"]  # each (4096, T)
        peak = int(label.mean(axis=0).argmax())
        grids = {k: v[:, peak].reshape(N_AXIAL, N_CIRC) for k, v in
                 (("label", label), ("stage2", pred), ("pinn_raw", raw))}
        for name, col in REGIONS.items():
            for k, g in grids.items():
                level[name][k].append(np.median(g[col]))

    print(f"{pred_dir}: {len(files)} patients, peak-systolic frame per patient\n")
    print(f"{'Region':24s} {'Stage 2 (PINN + MLP)':>21s} {'Raw PINN WSS':>13s}")
    for name in REGIONS:
        t = np.array(level[name]["label"])
        r2 = np.corrcoef(t, level[name]["stage2"])[0, 1]
        rr = np.corrcoef(t, level[name]["pinn_raw"])[0, 1]
        print(f"{name:24s} {r2:21.3f} {rr:13.3f}")
    print("\nlevel r = Pearson across patients of each patient's median regional WSS "
          "(same metric as the PointNet table).")


if __name__ == "__main__":
    main()
