"""
Exports the actual 3D surface mesh (points + quad faces) for a handful of example
patients, with per-point scalars: true WSS, predicted WSS, and a region label
(pasc/arch/desc/abda) — so the region boundaries and prediction quality can be
inspected spatially in ParaView (or reopened with pyvista), not just as numbers.

Quad connectivity matches infer_wss_new_aorta.py's visualize_wss(): the 4096 points
are a 32 (circumferential) x 128 (longitudinal) grid, point i -> row i//128, col
i%128, same convention main.py uses for pred.reshape(batch, 1, 32, 128) and for the
ROI_REGIONS longitudinal column slices. The circumferential axis is left open (no
wrap-around face connecting row 31 back to row 0), matching that same reference
implementation.

Requires all four region specialists (pasc/arch/desc/abda) to have completed their
10-fold runs (same requirement as stitch_specialists.build_composite()), plus the xyz
coordinates in DATA_PATH's 'a' array.

Usage (run from 6.PointNet_Velocity/):
    python export_region_mesh.py
Requires: pyvista (pip install pyvista)
"""

import os
import numpy as np
import pyvista as pv
from sklearn.model_selection import KFold

from stitch_specialists import build_composite, REGIONS
from main import DATA_PATH, N_FOLDS, SEED

OUTPUT_DIR = "../experiments/region_meshes"
N_EXAMPLES_EACH = 2  # best N and worst N patients by composite whole-vessel Pearson r

REGION_CODES = {"pasc": 0, "arch": 1, "desc": 2, "abda": 3}
REGION_CODE_LEGEND = ", ".join(f"{v}={k}" for k, v in REGION_CODES.items())


def build_region_labels():
    """Per-point region code (0-3), from each point's longitudinal column (i % 128)."""
    labels = np.zeros(4096, dtype=np.int32)
    grid_labels = labels.reshape(32, 128)
    for name, (col_slice, _) in REGIONS.items():
        grid_labels[:, col_slice] = REGION_CODES[name]
    return labels


def build_quad_faces():
    """
    Same 31x127 quad grid as infer_wss_new_aorta.py's visualize_wss(): four corner
    point-indices per face, in pyvista's flat face-array format (leading count of 4
    per face).
    """
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

    quads = np.stack([e1, e2, e3, e4], axis=-1).reshape(-1, 4)
    counts = np.full((quads.shape[0], 1), 4, dtype=int)
    return np.hstack([counts, quads]).flatten()


def main():
    all_true, composite_pred = build_composite(verbose=False)
    n_total = all_true.shape[0]

    data = np.load(DATA_PATH)
    xyz = data['a'][:, :, :3].astype(np.float64)  # (n_total, 4096, 3), original patient order
    patient_ids = data['c']

    # Undo KFold pooling, same as export_predictions_to_excel.py, so row i matches
    # xyz[i] / patient_ids[i] (original patient order).
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pooled_to_original = []
    for _, test_ix in kf.split(np.arange(n_total)):
        pooled_to_original.extend(test_ix.tolist())
    original_to_pooled = np.empty(n_total, dtype=int)
    for pooled_idx, orig_idx in enumerate(pooled_to_original):
        original_to_pooled[orig_idx] = pooled_idx

    true_ordered = all_true[original_to_pooled]
    pred_ordered = composite_pred[original_to_pooled]

    patient_pearson = np.array([
        np.corrcoef(true_ordered[i], pred_ordered[i])[0, 1] for i in range(n_total)
    ])
    ranked = np.argsort(patient_pearson)
    worst_ix = ranked[:N_EXAMPLES_EACH]
    best_ix = ranked[-N_EXAMPLES_EACH:][::-1]

    faces = build_quad_faces()
    region_labels = build_region_labels()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Region codes: {REGION_CODE_LEGEND}\n")

    for label, idx_list in [("best", best_ix), ("worst", worst_ix)]:
        for rank, idx in enumerate(idx_list, start=1):
            pid = patient_ids[idx]
            r = patient_pearson[idx]
            mesh = pv.PolyData(xyz[idx], faces)
            mesh["WSS_true"] = true_ordered[idx]
            mesh["WSS_pred"] = pred_ordered[idx]
            mesh["region"] = region_labels
            mesh["abs_error"] = np.abs(true_ordered[idx] - pred_ordered[idx])

            out_path = os.path.join(OUTPUT_DIR, f"patient{pid}_{label}{rank}_r{r:.3f}.vtp")
            mesh.save(out_path)
            print(f"  {label} #{rank}: patient {pid} (composite Pearson r={r:.3f}) -> {out_path}")

    print(f"\nSaved {2 * N_EXAMPLES_EACH} meshes to {OUTPUT_DIR}/")
    print("Open in ParaView, or in Python: pv.read(path).plot(scalars='WSS_pred')")


if __name__ == "__main__":
    main()
