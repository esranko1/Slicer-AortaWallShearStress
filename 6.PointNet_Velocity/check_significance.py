"""
Is the composite specialist result (0.816 Pearson) actually better than the single
whole-vessel baseline (0.782), or could that +0.034 gap just be noise from having only
~100 patients?

Both numbers come from the *same* patients in the *same* KFold order (same SEED,
N_FOLDS, and X — only USE_ROI_LOSS/USE_GEOMETRIC_FEATURES differ), so this is a paired
comparison: for every patient we have both a baseline prediction and a composite
prediction against the same true value. That pairing is what makes a paired bootstrap
the right tool here, rather than treating the two Pearson numbers as independent.

Method: resample patients with replacement B times. On each resample, recompute both
Pearson correlations from scratch on that resampled set, and record the difference
(composite - baseline). Doing both computations on the *same* resample in each
iteration preserves whatever correlation exists between the two estimates (they're
scored against the same patients, so they tend to move together) — an unpaired
comparison would overstate the uncertainty. The resulting distribution of differences
gives a 95% CI: if it excludes 0, the improvement is unlikely to be noise at this
sample size; if it includes 0, the data can't rule out "no real difference."

Requires: baseline's results.npz (main.py with USE_ROI_LOSS=False, USE_GEOMETRIC_FEATURES=False)
and all four specialist results.npz (main.py with USE_ROI_LOSS=True, one ROI_NAME each)
to have already completed.
"""

import numpy as np

from stitch_specialists import build_composite, REGIONS

BASELINE_RESULTS_PATH = "../experiments/6_PointNet_Velocity/results.npz"

N_BOOTSTRAP = 10000
SEED = 2024


def patient_level(all_true, all_pred):
    return np.median(all_true, axis=1), np.median(all_pred, axis=1)


def bootstrap_pearson_diff(true_a, pred_a, true_b, pred_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    true_a/pred_a and true_b/pred_b must be patient-aligned (same order, same patients).
    Returns array of shape (n_boot,): bootstrap Pearson(b) - Pearson(a) per resample.
    """
    assert len(true_a) == len(true_b) == len(pred_a) == len(pred_b)
    n = len(true_a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_a = np.corrcoef(true_a[idx], pred_a[idx])[0, 1]
        r_b = np.corrcoef(true_b[idx], pred_b[idx])[0, 1]
        diffs[i] = r_b - r_a
    return diffs


def report(name, diffs, point_a, point_b):
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    frac_le_zero = np.mean(diffs <= 0)
    verdict = "REAL (CI excludes 0)" if ci_lo > 0 else "NOT DISTINGUISHABLE FROM NOISE (CI includes 0)"
    print(f"\n{name}")
    print(f"  baseline={point_a:.3f}  composite={point_b:.3f}  observed gain={point_b - point_a:+.3f}")
    print(f"  bootstrap 95% CI on gain: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  P(bootstrap gain <= 0) = {frac_le_zero:.3f}   ({N_BOOTSTRAP} resamples)")
    print(f"  Verdict: {verdict}")


def main():
    baseline = np.load(BASELINE_RESULTS_PATH)
    baseline_true, baseline_pred = baseline["all_true"], baseline["all_pred"]

    composite_true, composite_pred = build_composite(verbose=False)

    if not np.allclose(baseline_true, composite_true):
        raise ValueError(
            "baseline results.npz and the stitched specialists don't share the same "
            "patient/fold order — baseline must have used the identical SEED/N_FOLDS/X "
            "(USE_ROI_LOSS=False, USE_GEOMETRIC_FEATURES=False) for this comparison to "
            "be valid. Re-run main.py with those settings if this fires."
        )
    print(f"Fold-order consistency check passed (N={baseline_true.shape[0]} patients).")

    # --- Whole-vessel composite vs baseline ---
    p_true, p_base = patient_level(baseline_true, baseline_pred)
    _, p_comp = patient_level(composite_true, composite_pred)
    r_base = np.corrcoef(p_true, p_base)[0, 1]
    r_comp = np.corrcoef(p_true, p_comp)[0, 1]
    diffs = bootstrap_pearson_diff(p_true, p_base, p_true, p_comp)
    report("=== Whole-vessel: composite vs single baseline ===", diffs, r_base, r_comp)

    # --- Per-region: specialist (within composite) vs the baseline model's own accuracy
    #     on that same region (baseline was never optimized for any one region, so this
    #     shows what each specialist actually bought you, region by region). ---
    n_total = baseline_true.shape[0]
    true_grid = baseline_true.reshape(n_total, 32, 128)
    base_grid = baseline_pred.reshape(n_total, 32, 128)
    comp_grid = composite_pred.reshape(n_total, 32, 128)

    print("\nPer-region: specialist (in composite) vs baseline's own accuracy in that region")
    for name, (col_slice, _) in REGIONS.items():
        t = np.median(true_grid[:, :, col_slice].reshape(n_total, -1), axis=1)
        b = np.median(base_grid[:, :, col_slice].reshape(n_total, -1), axis=1)
        c = np.median(comp_grid[:, :, col_slice].reshape(n_total, -1), axis=1)
        r_b = np.corrcoef(t, b)[0, 1]
        r_c = np.corrcoef(t, c)[0, 1]
        diffs = bootstrap_pearson_diff(t, b, t, c)
        report(f"  region={name}", diffs, r_b, r_c)


if __name__ == "__main__":
    main()
