"""
Interim results from the folds finished so far, while a run is still going.
Reads only the saved fold files, so it is safe to run alongside training.

    python peek.py <region> [--cv 10|loo] [--no-flow]
    e.g. python peek.py arch --cv loo --no-flow

Prints the same metrics main.py reports at the end, on the patients done so
far: per-region level r (each patient's median WSS correlated across
patients), median spatial r, and PWV r / MAE. With most folds done these are
close to the final numbers; with few folds done they are noisy.
"""

import argparse
import glob
import os
import re

import numpy as np

from runs import REGIONS, WHOLE, r_level, r_spatial_per_patient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=["pasc", "arch", "desc", "abda", "whole"])
    ap.add_argument("--cv", choices=["10", "loo"], default="10")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--physics", choices=["none", "input", "ratio"], default="none")
    args = ap.parse_args()

    folder = (f"../experiments/14_Final_{args.region}_{'loo' if args.cv == 'loo' else 'cv10'}_"
              f"{'noflow' if args.no_flow else 'flow'}" + (f"_seed{args.seed}" if args.seed != 2024 else "")
              + ("_aligned" if args.aligned else "")
              + ("" if args.physics == "none" else f"_phys-{args.physics}"))
    files = sorted(glob.glob(os.path.join(folder, "fold*_of*_results.npz")),
                   key=lambda f: int(re.search(r"fold(\d+)_", f).group(1)))
    if not files:
        raise SystemExit(f"No finished folds in {folder}")
    n_folds = int(re.search(r"_of(\d+)_", files[0]).group(1))

    true, pred, pwv_t, pwv_p = [], [], [], []
    for f in files:
        d = np.load(f)
        true.append(d["true"])
        pred.append(d["pred"])
        if "pwv_pred" in d.files:
            pwv_t.append(np.atleast_1d(d["pwv_true"]))
            pwv_p.append(np.atleast_1d(d["pwv_pred"]))
    true = np.concatenate(true).reshape(-1, 32, 128)
    pred = np.concatenate(pred).reshape(-1, 32, 128)
    print(f"{folder}: {len(files)}/{n_folds} folds done, {true.shape[0]} held-out patients\n")

    print(f"{'region':8s} {'level r':>8s} {'spatial r (median)':>19s}")
    for name, col in [("whole", WHOLE)] + list(REGIONS.items()):
        mark = "  <-- this specialist's region" if name == args.region else ""
        print(f"{name:8s} {r_level(true, pred, col):8.3f} "
              f"{np.median(r_spatial_per_patient(true, pred, col)):19.3f}{mark}")
    if pwv_p:
        t, p = np.concatenate(pwv_t), np.concatenate(pwv_p)
        print(f"\nPWV: r {np.corrcoef(t, p)[0, 1]:.3f}   MAE {np.abs(p - t).mean():.2f} m/s")


if __name__ == "__main__":
    main()
