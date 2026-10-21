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
  MAE        mean absolute error over the region's wall points, in label units
             (Pa), median over patients [IQR]
  NMAE       MAE / that patient's true WSS range in the region, median [IQR]
  pooled rho Spearman over all wall points of all patients (the PI notebook's
             regional metric, for comparison with its baseline)
  level BA   Bland-Altman of each patient's median regional WSS (Pa)
Plus, for the whole vessel: Spearman, CCC, slope/intercept of the level values,
and pulse wave velocity (r, MAE, Bland-Altman) when the run predicted it (for
the composite: the mean of the four specialists' PWV predictions).
"""

import argparse
import os

import numpy as np
from scipy.stats import spearmanr

from runs import (REGIONS, WHOLE, level_bland_altman, level_values, load_run, mae_per_patient,
                  nmae_per_patient, pooled_spearman, r_level, r_spatial_per_patient, results_path)


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
        rows = [("whole", WHOLE)] + list(REGIONS.items())
        q = lambda v: f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}]"  # noqa: E731

        print("\nPattern and error on the wall (per patient, median [IQR] over patients):")
        print(f"{'region':8s} {'spatial r':>22s} {'MAE (Pa)':>22s} {'NMAE':>22s} {'pooled rho':>11s}")
        for name, col in rows:
            print(f"{name:8s} {q(r_spatial_per_patient(true, pred, col)):>22s} {q(mae_per_patient(true, pred, col)):>22s} "
                  f"{q(nmae_per_patient(true, pred, col)):>22s} {pooled_spearman(true, pred, col):11.3f}")

        print("\nPatient level (each patient's median regional WSS, across patients):")
        print(f"{'region':8s} {'level r':>8s} {'bias (Pa)':>10s} {'95% limits of agreement (Pa)':>30s}")
        for name, col in rows:
            bias, lo, hi = level_bland_altman(true, pred, col)
            print(f"{name:8s} {r_level(true, pred, col):8.3f} {bias:+10.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>30s}")
        print()
        if pwv_pred is not None:
            d = pwv_pred - pwv_true
            print(f"PWV: r {np.corrcoef(pwv_true, pwv_pred)[0, 1]:.3f}   MAE {np.abs(d).mean():.2f} m/s   "
                  f"bias {d.mean():+.2f} m/s, limits of agreement "
                  f"[{d.mean() - 1.96 * d.std():+.2f}, {d.mean() + 1.96 * d.std():+.2f}] m/s")


if __name__ == "__main__":
    main()
