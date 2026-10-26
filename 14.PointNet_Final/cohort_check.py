"""
Is a run's score partly just the two cohorts (MC vs other: different scanners
and sites) differing from each other? Re-scores saved held-out predictions
within each cohort. No training.

    python cohort_check.py whole:cv10:flow:aligned:phys-input [more run specs]

Needs ../Data/Processed4DFlow/patient_matching.csv (1.Data_preprocessing/
1.7.match_patients.py), which gives each training row its cohort. Patients in
a run's results are identified by their true WSS map.

Per region:
  level r all        the usual level r (each patient's median WSS, across patients)
  within MC / other  the same, inside one cohort only
  adjusted           level r after removing each cohort's mean (true and
                     predicted separately): the patient ranking that is left
                     once the cohort difference itself is taken out
  true / pred gap    MC minus other mean level, measured and predicted (Pa):
                     does the model reproduce the cohort difference?
If "adjusted" and the within-cohort values are close to "all", the score is not
driven by the cohort difference.
"""

import argparse
import csv

import numpy as np

from runs import REGIONS, WHOLE, level_values, load_run, r_spatial_per_patient

DATA_PATH = "../Data/Sampled/Rpt0_N4096.npz"
MATCHING_PATH = "../Data/Processed4DFlow/patient_matching.csv"


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--matching", default=MATCHING_PATH)
    args = ap.parse_args()

    with open(args.matching) as f:
        row_cohort = {r["row"]: r["cohort"] for r in csv.DictReader(f)}
    d = np.load(args.data)
    labels = d["b"][:, :, 0].astype(np.float64)
    rows = [str(c) for c in d["c"]]

    for spec in args.specs:
        true, pred, _, _ = load_run(spec)
        flat = true.reshape(true.shape[0], -1)
        # nearest training row for every held-out patient (float32 vs float64 tolerant)
        ix = np.array([np.abs(labels - t).sum(axis=1).argmin() for t in flat])
        if len(set(ix)) != len(ix):
            raise SystemExit(f"{spec}: could not identify patients uniquely from their WSS maps")
        groups = np.array([row_cohort.get(rows[i], "unknown") for i in ix])
        mc, ot = groups == "MC", groups == "other"
        print(f"\n=== {spec}  (MC {mc.sum()}, other {ot.sum()}, unmatched {(~mc & ~ot).sum()}) ===")
        print(f"{'region':8s} {'level r all':>11s} {'within MC':>10s} {'within other':>13s} {'adjusted':>9s} "
              f"{'true gap':>9s} {'pred gap':>9s}   {'spatial r MC / other':>20s}")
        for name, col in [("whole", WHOLE)] + list(REGIONS.items()):
            t, p = level_values(true, pred, col)
            ta, pa = t.copy(), p.copy()
            for m in (mc, ot):
                ta[m] -= t[m].mean()
                pa[m] -= p[m].mean()
            sp = r_spatial_per_patient(true, pred, col)
            print(f"{name:8s} {corr(t, p):11.3f} {corr(t[mc], p[mc]):10.3f} {corr(t[ot], p[ot]):13.3f} "
                  f"{corr(ta, pa):9.3f} {t[mc].mean() - t[ot].mean():+9.3f} {p[mc].mean() - p[ot].mean():+9.3f}   "
                  f"{np.median(sp[mc]):9.3f} / {np.median(sp[ot]):.3f}")


if __name__ == "__main__":
    main()
