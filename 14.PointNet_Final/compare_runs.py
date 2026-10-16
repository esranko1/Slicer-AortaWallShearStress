"""
Paired bootstrap between any two runs (same 100 patients), for both metrics,
whole vessel and per region.

    python compare_runs.py <run A> <run B>

Run spec: "<composite|whole>:<cv10|loo>:<flow|noflow>[:seed]" (see runs.py).
Examples:
    python compare_runs.py whole:cv10:flow composite:cv10:flow      # specialists vs baseline
    python compare_runs.py composite:cv10:noflow composite:cv10:flow  # flow ablation
    python compare_runs.py composite:cv10:flow composite:loo:flow     # 10-fold vs leave-one-out

Patients are resampled with replacement; on each resample both runs' metrics
are recomputed on the same patients and the difference B - A recorded. The
95% interval of the difference says whether it could be noise.
"""

import argparse

import numpy as np

from runs import REGIONS, WHOLE, level_values, load_run, r_spatial_per_patient

N_BOOTSTRAP = 10000
SEED = 2024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()

    true_a, pred_a, pwv_true_a, pwv_a = load_run(args.a)
    true_b, pred_b, pwv_true_b, pwv_b = load_run(args.b)
    if not np.allclose(true_a, true_b):
        raise SystemExit("The two runs do not contain the same patients; cannot pair them.")
    n = true_a.shape[0]
    rng = np.random.default_rng(SEED)
    boot = [rng.integers(0, n, size=n) for _ in range(N_BOOTSTRAP)]

    print(f"A = {args.a}\nB = {args.b}\nPaired bootstrap, N={n} patients, {N_BOOTSTRAP} resamples\n")
    print(f"{'metric':10s} {'region':7s} {'A':>7s} {'B':>7s} {'B - A':>7s}   {'95% CI':>17s}   verdict")
    for name, col in [("whole", WHOLE)] + list(REGIONS.items()):
        t, a = level_values(true_a, pred_a, col)
        _, b = level_values(true_b, pred_b, col)
        sa, sb = r_spatial_per_patient(true_a, pred_a, col), r_spatial_per_patient(true_b, pred_b, col)
        metrics = {
            "level r": (lambda i: np.corrcoef(t[i], a[i])[0, 1], lambda i: np.corrcoef(t[i], b[i])[0, 1]),
            "spatial r": (lambda i: np.median(sa[i]), lambda i: np.median(sb[i])),
        }
        for mname, (fa, fb) in metrics.items():
            full = np.arange(n)
            va, vb = fa(full), fb(full)
            diffs = np.array([fb(i) - fa(i) for i in boot])
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])  # a resample can be degenerate (all-identical values)
            verdict = "B higher" if lo > 0 else "A higher" if hi < 0 else "no detectable difference"
            print(f"{mname:10s} {name:7s} {va:7.3f} {vb:7.3f} {vb - va:+7.3f}   [{lo:+.3f}, {hi:+.3f}]   {verdict}")

    if pwv_a is not None and pwv_b is not None:
        t = pwv_true_a
        metrics = {
            "PWV r": (lambda i: np.corrcoef(t[i], pwv_a[i])[0, 1], lambda i: np.corrcoef(t[i], pwv_b[i])[0, 1]),
            "PWV MAE": (lambda i: np.abs(pwv_a[i] - t[i]).mean(), lambda i: np.abs(pwv_b[i] - t[i]).mean()),
        }
        for mname, (fa, fb) in metrics.items():
            full = np.arange(n)
            va, vb = fa(full), fb(full)
            lo, hi = np.nanpercentile(np.array([fb(i) - fa(i) for i in boot]), [2.5, 97.5])
            verdict = "B higher" if lo > 0 else "A higher" if hi < 0 else "no detectable difference"
            print(f"{mname:10s} {'-':7s} {va:7.3f} {vb:7.3f} {vb - va:+7.3f}   [{lo:+.3f}, {hi:+.3f}]   {verdict}")
        print("(PWV MAE in m/s: lower is better)")

    print("\nNote: several comparisons; one interval that just excludes 0 can be chance.")


if __name__ == "__main__":
    main()
