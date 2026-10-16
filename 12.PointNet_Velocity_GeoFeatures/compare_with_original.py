"""
Do the geometric features (12_GeoFeatures) change the region-specialist
composite (6_PointNet_Velocity_roi_*) by more than noise?

Same method as 6.PointNet_Velocity/check_significance.py: a paired bootstrap
over patients. Both models were scored on the same 100 patients in the same
KFold order (same SEED, N_FOLDS and data), so for every resample of patients
both Pearson correlations are recomputed on that resample and their difference
recorded. The 95% interval of those differences says whether the change could
be noise.

Metric: as in stitch_specialists.py, each patient's median WSS (whole vessel or
one region), correlated across patients.

Requires all four region runs of BOTH models to have finished.
Usage (from this folder):  python compare_with_original.py
"""

import numpy as np

REGION_SLICES = {
    "pasc": slice(0, 36),
    "arch": slice(36, 60),
    "desc": slice(60, 96),
    "abda": slice(96, 128),
}
MODELS = {
    "original": "../experiments/6_PointNet_Velocity_roi_{}/results.npz",
    "geo": "../experiments/12_GeoFeatures_roi_{}/results.npz",
}
N_BOOTSTRAP = 10000
SEED = 2024


def build_composite(path_pattern):
    """Stitch the four specialists' predictions, each over its own region."""
    data = {name: np.load(path_pattern.format(name)) for name in REGION_SLICES}
    all_true = data["pasc"]["all_true"]
    for name, d in data.items():
        if not np.allclose(d["all_true"], all_true):
            raise ValueError(f"{path_pattern.format(name)}: fold split differs from the pasc run")
    n = all_true.shape[0]
    pred_grid = np.zeros((n, 32, 128))
    for name, col in REGION_SLICES.items():
        pred_grid[:, :, col] = data[name]["all_pred"].reshape(n, 32, 128)[:, :, col]
    return all_true.reshape(n, 32, 128), pred_grid


def bootstrap_diff(t, a, b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Pearson(t, b) - Pearson(t, a) on paired bootstrap resamples of patients."""
    rng = np.random.default_rng(seed)
    n = len(t)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.corrcoef(t[idx], b[idx])[0, 1] - np.corrcoef(t[idx], a[idx])[0, 1]
    return diffs


def main():
    true_o, pred_o = build_composite(MODELS["original"])
    true_g, pred_g = build_composite(MODELS["geo"])
    if not np.allclose(true_o, true_g):
        raise ValueError("The two models' results are not in the same patient/fold order; "
                         "they must share SEED, N_FOLDS and Rpt0_N4096.npz for a paired test.")
    n = true_o.shape[0]
    print(f"Paired comparison on N={n} patients, {N_BOOTSTRAP} bootstrap resamples\n")
    print(f"{'region':8s} {'original':>9s} {'geo':>7s} {'change':>8s}   {'95% CI of change':>20s}   verdict")

    rows = [("whole", slice(0, 128))] + list(REGION_SLICES.items())
    for name, col in rows:
        t = np.median(true_o[:, :, col].reshape(n, -1), axis=1)
        a = np.median(pred_o[:, :, col].reshape(n, -1), axis=1)
        b = np.median(pred_g[:, :, col].reshape(n, -1), axis=1)
        r_a, r_b = np.corrcoef(t, a)[0, 1], np.corrcoef(t, b)[0, 1]
        lo, hi = np.percentile(bootstrap_diff(t, a, b), [2.5, 97.5])
        verdict = ("geo better" if lo > 0 else "original better" if hi < 0 else "no detectable difference")
        print(f"{name:8s} {r_a:9.3f} {r_b:7.3f} {r_b - r_a:+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {verdict}")

    print("\nNote: 5 comparisons; a single interval that just excludes 0 can be chance "
          "(about a 1 in 4 chance that at least one of five does).")


if __name__ == "__main__":
    main()
