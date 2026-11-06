"""
WSS in quartiles (or deciles): how well does the model identify high and low
WSS, rather than how well it correlates? Reads saved results only (no
training; seconds on CPU).

    python quantiles.py whole:loo:flow:aligned:phys-input                   # quartiles
    python quantiles.py whole:loo:flow:aligned:phys-input --bins 10         # deciles
    python quantiles.py whole:cv10:flow:aligned:phys-input meanmap:cv10:flow  # vs the average map

Run specs as in compare_runs.py (see runs.py). Bins are by rank, so each bin
holds the same number of patients (or wall points), separately for the true
and the predicted values; this asks "is it in the right group", not "is the
value exactly right" (predictions are compressed toward the mean, so cut-offs
taken from the true values would push predictions into the middle bins).

1. Patients: each patient's median WSS in the region is put in a quartile of
   the cohort. Exact = right quartile; within 1 = right or neighbouring
   quartile; weighted kappa (quadratic; 0 = chance, 1 = perfect); AUC for
   finding the patients in the top / bottom quartile from the predicted level.
   95% CIs from 2000 patient resamples (bins kept fixed).
2. Wall areas within each patient: the region's wall points are put in
   quartiles of that patient's own WSS. Exact, within 1, weighted kappa, and
   overlap of the true and predicted highest / lowest quartile areas (Dice;
   both areas are the same size, so this is the share of the true area that
   is found). Median [IQR] over patients.
3. Areas of high / low WSS compared with the cohort (as in 4D-flow WSS "heat
   map" studies): at every wall point, a patient's WSS is high if it is in the
   top quartile of all patients at that same point, low if in the bottom one.
   Overlap = share of the true high (low) points also predicted high (low),
   pooled over patients; area r = correlation across patients of the share of
   the region that is high (low), true vs predicted. Cut-offs come from the
   evaluated patients themselves (descriptive, not a fitted threshold).
"""

import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from runs import REGIONS, WHOLE, level_values, load_run, region_points

N_BOOTSTRAP = 2000
SEED = 2024
REGION_ROWS = [("whole", WHOLE)] + list(REGIONS.items())


def rank_bins(x, n_bins, axis=0):
    """Equal-sized bins by rank along axis: 0 = lowest, n_bins - 1 = highest. Tied
    values share a bin (patients are stored sorted by true WSS, so breaking ties
    by position would quietly use the true order)."""
    r = rankdata(x, method="average", axis=axis) - 1
    return np.floor(r * n_bins / x.shape[axis]).astype(int)


def weighted_kappa(a, b, n_bins):
    """Quadratic-weighted Cohen's kappa between two bin labelings."""
    O = np.zeros((n_bins, n_bins))
    np.add.at(O, (a, b), 1)
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / O.sum()
    i, j = np.indices(O.shape)
    w = (i - j) ** 2
    return 1 - (w * O).sum() / (w * E).sum()


def ci(values):
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return f"[{lo:.2f}-{hi:.2f}]"


def med_iqr(x, pct=False):
    q1, m, q3 = np.percentile(x, [25, 50, 75]) * (100 if pct else 1)
    return f"{m:.0f}% [{q1:.0f}-{q3:.0f}]" if pct else f"{m:.2f} [{q1:.2f}-{q3:.2f}]"


def patients(true, pred, B, rng):
    print(f"\n1) Patients: right {B}-group of regional WSS level?")
    print(f"   chance: exact {1 / B:.0%}, within 1 {(3 * B - 2) / B ** 2:.0%}, kappa 0, AUC 0.5")
    print(f"   {'region':6s} {'exact':>6s} {'within 1':>9s}   {'weighted kappa':>18s}   "
          f"{'AUC top':>16s}   {'AUC bottom':>16s}")
    boot = [rng.integers(0, true.shape[0], true.shape[0]) for _ in range(N_BOOTSTRAP)]
    for name, col in REGION_ROWS:
        t, p = level_values(true, pred, col)
        bt, bp = rank_bins(t, B), rank_bins(p, B)
        top, bottom = bt == B - 1, bt == 0
        kap = weighted_kappa(bt, bp, B)
        a_top, a_bot = roc_auc_score(top, p), roc_auc_score(bottom, -p)
        bk, btop, bbot = [], [], []
        for i in boot:
            bk.append(weighted_kappa(bt[i], bp[i], B))
            btop.append(roc_auc_score(top[i], p[i]) if 0 < top[i].sum() < len(i) else np.nan)
            bbot.append(roc_auc_score(bottom[i], -p[i]) if 0 < bottom[i].sum() < len(i) else np.nan)
        print(f"   {name:6s} {np.mean(bt == bp):6.0%} {np.mean(np.abs(bt - bp) <= 1):9.0%}   "
              f"{kap:.2f} {ci(bk):>13s}   {a_top:.2f} {ci(btop):>11s}   {a_bot:.2f} {ci(bbot):>11s}")
        if name == "whole":
            whole_bins = (bt, bp)
    bt, bp = whole_bins
    print(f"\n   Whole vessel, number of patients (rows = true group, columns = predicted; 1 = lowest WSS):")
    print("        " + "".join(f"{k + 1:>5d}" for k in range(B)))
    for k in range(B):
        print(f"   {k + 1:>4d} " + "".join(f"{np.sum((bt == k) & (bp == m)):>5d}" for m in range(B)))


def wall_areas(true, pred, B):
    print(f"\n2) Wall areas within each patient: right {B}-group of that patient's own WSS?")
    print(f"   median [IQR] over patients; chance: exact {1 / B:.0%}, within 1 {(3 * B - 2) / B ** 2:.0%}, "
          f"kappa 0, overlap {1 / B:.2f}")
    print(f"   {'region':6s} {'exact':>17s} {'within 1':>17s} {'weighted kappa':>17s} "
          f"{'overlap highest':>17s} {'overlap lowest':>17s}")
    for name, col in REGION_ROWS:
        t, p = region_points(true, pred, col)
        bt, bp = rank_bins(t, B, axis=1), rank_bins(p, B, axis=1)
        exact = np.mean(bt == bp, axis=1)
        within = np.mean(np.abs(bt - bp) <= 1, axis=1)
        kap = np.array([weighted_kappa(a, b, B) for a, b in zip(bt, bp)])
        high = ((bt == B - 1) & (bp == B - 1)).sum(axis=1) / (bt == B - 1).sum(axis=1)
        low = ((bt == 0) & (bp == 0)).sum(axis=1) / (bt == 0).sum(axis=1)
        print(f"   {name:6s} {med_iqr(exact, True):>17s} {med_iqr(within, True):>17s} {med_iqr(kap):>17s} "
              f"{med_iqr(high):>17s} {med_iqr(low):>17s}")


def cohort_areas(true, pred, B):
    print(f"\n3) High / low WSS compared with the cohort at the same wall point "
          f"(top / bottom {1 / B:.0%} of patients there)")
    print(f"   overlap: share of true high (low) points predicted high (low), chance {1 / B:.2f}; "
          f"area r: share of the region high (low), true vs predicted")
    print(f"   {'region':6s} {'high: overlap':>14s} {'area r':>7s}   {'low: overlap':>13s} {'area r':>7s}")
    for name, col in REGION_ROWS:
        t, p = region_points(true, pred, col)
        bt, bp = rank_bins(t, B, axis=0), rank_bins(p, B, axis=0)
        row = []
        for k in (B - 1, 0):
            st, sp = bt == k, bp == k
            a, b = st.mean(axis=1), sp.mean(axis=1)
            row += [(st & sp).sum() / st.sum(), np.corrcoef(a, b)[0, 1] if b.std() > 0 else np.nan]
        print(f"   {name:6s} {row[0]:14.2f} {row[1]:7.2f}   {row[2]:13.2f} {row[3]:7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run specs, e.g. whole:loo:flow:aligned:phys-input")
    ap.add_argument("--bins", type=int, default=4, help="4 = quartiles, 10 = deciles")
    args = ap.parse_args()
    for spec in args.runs:
        true, pred, _, _ = load_run(spec)
        print(f"\n=== {spec}: {args.bins} groups, N={true.shape[0]} patients ===")
        patients(true, pred, args.bins, np.random.default_rng(SEED))
        wall_areas(true, pred, args.bins)
        cohort_areas(true, pred, args.bins)


if __name__ == "__main__":
    main()
