"""
Independent check of the paper numbers (python figures.py --paper) before they
are sent out. Reads the saved results.npz files directly with its own code
(not runs.py / figures.py) and checks:

  1. every run file exists and holds 100 patients, each exactly once, no NaN
  2. every run contains the SAME 100 patients with the SAME measured WSS (and PWV)
     as the final model, so all methods are compared on identical data
  3. predictions are real predictions (not a copy of the measured values)
  4. the level r per region and the PWV r / MAE recomputed here with scipy match
     table2_paper.csv to the 3rd decimal
  5. those numbers match the values reported earlier by compare_runs.py (so the
     files have not changed since), and the * marks match its verdicts

    python verify_paper_results.py                       # reads ../experiments/figures_paper/table2_paper.csv
    python verify_paper_results.py --table path/to/table2_paper.csv

Prints PASS / FAIL per check and a summary at the end. Nothing is written.
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import pearsonr

# run folders, written out in full (not built by runs.py, so a wrong mapping there would show up here)
RUNS = {
    "Final (LOO)": "../experiments/14_Final_whole_loo_flow_aligned_phys-input/results.npz",
    "Final (10-fold)": "../experiments/14_Final_whole_cv10_flow_aligned_phys-input/results.npz",
    "Average map": "../experiments/15_Baselines_cv10/meanmap/results.npz",
    "Poiseuille only": "../experiments/15_Baselines_cv10/poiseuille/results.npz",
    "Shape modes + ridge": "../experiments/15_Baselines_cv10/shapemodes/results.npz",
    "WSS modes + GP": "../experiments/15_Baselines_cv10/modes-gp-nophys/results.npz",
    "PointNet (basic)": "../experiments/14_Final_whole_cv10_flow/results.npz",
    "2D CNN": "../experiments/14_Final_whole_cv10_flow_aligned_phys-input_cnn/results.npz",
    "DGCNN": "../experiments/14_Final_whole_cv10_flow_aligned_phys-input_dgcnn/results.npz",
    "Geometry only (no flow)": "../experiments/14_Final_whole_cv10_noflow_aligned/results.npz",
}
REFERENCE = "Final (10-fold)"
N_PATIENTS = 100
# wall map: point index = around * 128 + along; regions by cross-section along the centreline
REGIONS = {"Whole aorta": (0, 128), "Prox. ascending": (0, 36), "Arch": (36, 60),
           "Descending": (60, 96), "Abdominal": (96, 128)}

# Values reported earlier by compare_runs.py / stitch.py output (level r per region, then PWV r, PWV MAE).
# None = not reported / not defined.
EXPECTED = {
    "Final (LOO)": (0.757, 0.780, 0.764, 0.677, 0.626, 0.565, 2.73),
    "Final (10-fold)": (0.779, 0.821, 0.762, 0.711, 0.634, 0.556, 2.857),
    "Average map": (None,) * 7,
    "Poiseuille only": (0.543, 0.776, 0.459, 0.572, 0.162, None, None),
    "Shape modes + ridge": (0.632, 0.718, 0.654, 0.563, 0.486, None, None),
    "WSS modes + GP": (0.726, 0.762, 0.705, 0.696, 0.557, None, None),
    "PointNet (basic)": (0.715, 0.764, 0.648, 0.630, 0.545, 0.570, 2.596),
    "2D CNN": (0.737, 0.866, 0.756, 0.703, 0.531, 0.522, 2.81),
    "DGCNN": (0.712, 0.766, 0.696, 0.658, 0.504, 0.515, 2.83),
    "Geometry only (no flow)": (0.661, 0.652, 0.630, 0.583, 0.577, 0.567, 2.784),
}
# level-r regions where compare_runs.py found the method significantly LOWER than Final (10-fold)
EXPECTED_STARS = {
    "Poiseuille only": {"Whole aorta", "Arch", "Abdominal"},
    "Shape modes + ridge": {"Whole aorta", "Prox. ascending", "Arch"},
    "PointNet (basic)": {"Arch"},
    "Geometry only (no flow)": {"Whole aorta", "Prox. ascending", "Arch", "Descending"},
}

results = []


def check(ok, what, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""))
    return ok


def load(path):
    d = np.load(path)
    true, pred = d["all_true"].reshape(-1, 32 * 128), d["all_pred"].reshape(-1, 32 * 128)
    pwv = (d["all_pwv_true"].ravel(), d["all_pwv_pred"].ravel()) if "all_pwv_pred" in d.files else None
    return true, pred, pwv


def match_rows(true, ref_true):
    """Index of each reference patient in this run, by identical measured WSS map."""
    key = {np.round(row, 6).tobytes(): i for i, row in enumerate(true)}
    return np.array([key.get(np.round(row, 6).tobytes(), -1) for row in ref_true])


def medians(maps, lo, hi):
    return np.median(maps.reshape(-1, 32, 128)[:, :, lo:hi].reshape(len(maps), -1), axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="../experiments/figures_paper/table2_paper.csv")
    args = ap.parse_args()

    print("1) Files")
    data = {}
    for label, path in RUNS.items():
        if not check(os.path.exists(path), label, path):
            continue
        true, pred, pwv = load(path)
        n_unique = len({np.round(r, 6).tobytes() for r in true})
        check(true.shape[0] == N_PATIENTS and n_unique == N_PATIENTS,
              f"{label}: {N_PATIENTS} patients, each once", f"{true.shape[0]} rows, {n_unique} distinct")
        check(np.isfinite(true).all() and np.isfinite(pred).all(), f"{label}: no NaN / inf")
        data[label] = (true, pred, pwv)
    if REFERENCE not in data:
        raise SystemExit(f"\n{REFERENCE} missing; cannot continue.")

    print(f"\n2) Same patients and measured values as {REFERENCE}")
    ref_true, _, ref_pwv = data[REFERENCE]
    aligned = {}
    for label, (true, pred, pwv) in data.items():
        idx = match_rows(true, ref_true)
        if not check((idx >= 0).all(), f"{label}: all {N_PATIENTS} measured WSS maps identical",
                     f"{(idx < 0).sum()} not found"):
            continue
        pwv_a = None
        if pwv is not None:
            pwv_a = (pwv[0][idx], pwv[1][idx])
            check(np.allclose(pwv_a[0], ref_pwv[0]), f"{label}: measured PWV identical")
        aligned[label] = (true[idx], pred[idx], pwv_a)

    print("\n3) Predictions are not the measured values")
    for label, (true, pred, _) in aligned.items():
        check(not np.allclose(true, pred) and np.abs(pred - true).mean() > 1e-3, label,
              f"mean |pred - true| = {np.abs(pred - true).mean():.3f} Pa")

    print("\n4) Recomputed here vs table2_paper.csv")
    mine = {}
    for label, (true, pred, pwv) in aligned.items():
        vals = []
        for lo, hi in REGIONS.values():
            t, p = medians(true, lo, hi), medians(pred, lo, hi)
            vals.append(np.nan if label == "Average map" else pearsonr(t, p)[0])
        vals += [pearsonr(*pwv)[0], np.abs(pwv[1] - pwv[0]).mean()] if pwv is not None else [np.nan, np.nan]
        mine[label] = vals
    table, stars = {}, {}
    if os.path.exists(args.table):
        with open(args.table, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        for row in rows[1:]:
            if len(row) < len(header) or row[1] not in RUNS:
                continue
            cells = row[2:2 + len(REGIONS) + 2]
            num = lambda c: np.nan if c.strip("*+") in ("—", "") else float(c.strip("*+"))  # noqa: E731
            table[row[1]] = [num(c) for c in cells]
            stars[row[1]] = {name for name, c in zip(REGIONS, cells) if c.endswith("*")}
        for label, vals in mine.items():
            if label not in table:
                check(False, f"{label}: row in table")
                continue
            tol = [0.0006] * len(REGIONS) + [0.0006, 0.006]
            diffs = [abs(a - b) for a, b in zip(vals, table[label]) if not (np.isnan(a) and np.isnan(b))]
            same_nan = all(np.isnan(a) == np.isnan(b) for a, b in zip(vals, table[label]))
            ok = same_nan and all(d <= t for d, t in zip(diffs, tol))
            check(ok, label, "max difference {:.4f}".format(max(diffs)) if diffs else "all undefined")
    else:
        check(False, "table found", f"{args.table} missing: run python figures.py --paper first")

    print("\n5) Same as the numbers reported earlier (compare_runs.py), and the * marks")
    for label, exp in EXPECTED.items():
        if label not in mine:
            continue
        bad = []
        for name, e, v in zip(list(REGIONS) + ["PWV r", "PWV MAE"], exp, mine[label]):
            if e is None:
                continue
            tol = 0.006 if name == "PWV MAE" and e == round(e, 2) else 0.0006
            if abs(e - v) > tol:
                bad.append(f"{name}: now {v:.3f}, earlier {e}")
        check(not bad, label, "; ".join(bad))
        if label in stars:
            want = EXPECTED_STARS.get(label, set())
            check(stars[label] == want, f"{label}: * marks", f"table {sorted(stars[label])}, expected {sorted(want)}")

    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail} of {len(results)} checks passed" + ("" if not n_fail else f"; {n_fail} FAILED"))


if __name__ == "__main__":
    main()
