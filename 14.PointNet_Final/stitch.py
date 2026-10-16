"""
Results table for one setting: the four-region composite (and the whole-vessel
baseline, if that run exists), with both Pearson metrics per region.

    python stitch.py [--cv 10|loo] [--no-flow] [--seed N]

Metrics:
  level r    correlation across patients of each patient's median WSS (what
             6/13's stitch_specialists.py report; says whether the model ranks
             patients' overall WSS correctly)
  spatial r  each patient's Pearson across the wall points, median over
             patients (says whether the pattern on the wall is right)
Plus, for the whole vessel: Spearman, CCC, slope/intercept of the level values,
and pulse wave velocity (r, MAE, Bland-Altman) when the run predicted it (for
the composite: the mean of the four specialists' PWV predictions).
"""

import argparse
import os

import numpy as np
from scipy.stats import spearmanr

from runs import REGIONS, WHOLE, level_values, load_run, r_level, r_spatial_per_patient, results_path


def ccc(t, p):
    cor = np.corrcoef(t, p)[0, 1]
    return 2 * cor * t.std() * p.std() / (t.var() + p.var() + (t.mean() - p.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", choices=["10", "loo"], default="10")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    flow = "noflow" if args.no_flow else "flow"
    args.cv = "cv10" if args.cv == "10" else "loo"
    tail = f"{args.cv}:{flow}:{args.seed}"

    specs = [f"composite:{tail}"]
    if os.path.exists(results_path("whole", args.cv, flow, args.seed)):
        specs.append(f"whole:{tail}")

    for spec in specs:
        true, pred, pwv_true, pwv_pred = load_run(spec)
        n = true.shape[0]
        t, p = level_values(true, pred, WHOLE)
        slope, intercept = np.polyfit(t, p, 1)
        print(f"\n=== {spec}  (N={n} patients) ===")
        print(f"Whole vessel: level r {r_level(true, pred, WHOLE):.3f}   Spearman {spearmanr(t, p)[0]:.3f}   "
              f"CCC {ccc(t, p):.3f}   slope/intercept {slope:.3f}/{intercept:.3f}")
        print(f"{'region':8s} {'level r':>8s} {'spatial r (median)':>19s}")
        for name, col in [("whole", WHOLE)] + list(REGIONS.items()):
            print(f"{name:8s} {r_level(true, pred, col):8.3f} {np.median(r_spatial_per_patient(true, pred, col)):19.3f}")
        if pwv_pred is not None:
            d = pwv_pred - pwv_true
            print(f"PWV: r {np.corrcoef(pwv_true, pwv_pred)[0, 1]:.3f}   MAE {np.abs(d).mean():.2f} m/s   "
                  f"bias {d.mean():+.2f} m/s, limits of agreement "
                  f"[{d.mean() - 1.96 * d.std():+.2f}, {d.mean() + 1.96 * d.std():+.2f}] m/s")


if __name__ == "__main__":
    main()
