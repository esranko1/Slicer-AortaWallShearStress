"""
Does conditioning the Thoracic Arch specialist on Proximal Ascending's upstream
predictions (main.py's USE_UPSTREAM_CONDITIONING) actually help, or is any apparent
gain just noise?

Both runs must come from the identical KFold split (same SEED/N_FOLDS/n_total) — this
holds automatically as long as both used the same DATA_PATH and neither changed
N_FOLDS, since main.py's KFold call is deterministic given those. Verified below
before computing anything, same pattern as check_significance.py.

Requires:
  - Plain Arch specialist: main.py with ROI_NAME="arch", USE_UPSTREAM_CONDITIONING=False
  - Conditioned Arch specialist: main.py with ROI_NAME="arch", USE_UPSTREAM_CONDITIONING=True,
    UPSTREAM_REGION="pasc" (needs the pasc specialist's results.npz to already exist)
"""

import numpy as np

from check_significance import bootstrap_pearson_diff, report

ARCH_PLAIN_PATH = "../experiments/6_PointNet_Velocity_roi_arch/results.npz"
CONDITIONING_MODE = "rich"  # "boundary" (already tested) or "rich" (new)
ARCH_CONDITIONED_PATH = (
    "../experiments/6_PointNet_Velocity_roi_arch_upstream_pasc"
    + ("" if CONDITIONING_MODE == "boundary" else f"_{CONDITIONING_MODE}")
    + "/results.npz"
)
ARCH_COL_SLICE = slice(36, 60)


def main():
    plain = np.load(ARCH_PLAIN_PATH)
    conditioned = np.load(ARCH_CONDITIONED_PATH)

    if not np.allclose(plain["all_true"], conditioned["all_true"]):
        raise ValueError(
            "Plain and upstream-conditioned Arch runs don't share the same patient/fold "
            "order — both must use the identical SEED/N_FOLDS/n_total for this comparison "
            "to be valid."
        )
    n_total = plain["all_true"].shape[0]
    print(f"Fold-order consistency check passed (N={n_total} patients).")

    true_grid = plain["all_true"].reshape(n_total, 32, 128)
    plain_grid = plain["all_pred"].reshape(n_total, 32, 128)
    cond_grid = conditioned["all_pred"].reshape(n_total, 32, 128)

    t = np.median(true_grid[:, :, ARCH_COL_SLICE].reshape(n_total, -1), axis=1)
    p_plain = np.median(plain_grid[:, :, ARCH_COL_SLICE].reshape(n_total, -1), axis=1)
    p_cond = np.median(cond_grid[:, :, ARCH_COL_SLICE].reshape(n_total, -1), axis=1)

    r_plain = np.corrcoef(t, p_plain)[0, 1]
    r_cond = np.corrcoef(t, p_cond)[0, 1]
    diffs = bootstrap_pearson_diff(t, p_plain, t, p_cond)
    report("=== Thoracic Arch: upstream-conditioned vs plain specialist ===", diffs, r_plain, r_cond)


if __name__ == "__main__":
    main()
