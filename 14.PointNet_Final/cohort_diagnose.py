"""
Why is the patient ranking (level r) poor within the MC cohort and good within
the other cohort? Tests the candidate explanations directly on the data, no
training.

    python cohort_diagnose.py

Needs Rpt0_N4096.npz (training data), the 4D-flow legacy_Rpt0_N4096.npz and
patient_matching.csv (1.Data_preprocessing/1.5 and 1.7).

  A. Units      per patient, 4D-flow radius (mm) / training radius. If it
                varies from patient to patient within a cohort, the training
                coordinates have a per-patient scale error, which scrambles
                the size -> WSS relation for that cohort.
  B. Spread     SD of the true WSS level per region. A narrow spread in one
                cohort lowers r there even at equal error (and RMSE of the
                level is printed for that reason).
  C. Physics    within each cohort, r between the true WSS level and the
                Poiseuille level (4 mu Q / pi r^3, Q = peak inlet velocity x
                inlet area), from (i) the training geometry and Zreal and
                (ii) the 4D-flow geometry (mm) and measured inlet velocity.
                If (ii) is much better than (i) in MC, the training inputs
                are the problem and can be replaced. If both are poor, the
                MC WSS itself relates less to shape and flow.
  D. Drivers    within each cohort, r of the true WSS level with mean radius
                and with peak inlet velocity, separately.
"""

import argparse
import csv

import numpy as np

from runs import REGIONS, WHOLE

N_CIRC, N_AXIAL = 32, 128
TRAIN_PATH = "../Data/Sampled/Rpt0_N4096.npz"
AGG_PATH = "../Data/Processed4DFlow/legacy_Rpt0_N4096.npz"
MATCHING_PATH = "../Data/Processed4DFlow/patient_matching.csv"


def grids(path, axial_major):
    """-> xyz (n, 32, 128, 3), WSS (n, 32, 128), inlet curve (n, 50), ids"""
    d = np.load(path)
    n = d["a"].shape[0]
    if axial_major:
        xyz = d["a"][:, :, :3].reshape(n, N_AXIAL, N_CIRC, 3).transpose(0, 2, 1, 3)
        wss = d["b"][:, :, 0].reshape(n, N_AXIAL, N_CIRC).transpose(0, 2, 1)
    else:
        xyz = d["a"][:, :, :3].reshape(n, N_CIRC, N_AXIAL, 3)
        wss = d["b"][:, :, 0].reshape(n, N_CIRC, N_AXIAL)
    return (xyz.astype(np.float64), wss.astype(np.float64), d["a"][:, 0, 3:].astype(np.float64),
            [str(c) for c in d["c"]])


def ring_radius(xyz):
    return np.linalg.norm(xyz - xyz.mean(axis=1, keepdims=True), axis=3).mean(axis=1)   # (n, 128)


def poiseuille_level(xyz, inlet, col):
    r = ring_radius(xyz)
    Q = np.abs(inlet).max(axis=1) * np.pi * r[:, 0] ** 2
    return np.median(4 * Q[:, None] / (np.pi * r[:, col] ** 3), axis=1)


def level(wss, col):
    return np.median(wss[:, :, col].reshape(wss.shape[0], -1), axis=1)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=TRAIN_PATH)
    ap.add_argument("--agg", default=AGG_PATH)
    ap.add_argument("--matching", default=MATCHING_PATH)
    args = ap.parse_args()

    xyz_t, wss_t, Z_t, rows = grids(args.train, axial_major=False)
    xyz_a, _, v_a, ids = grids(args.agg, axial_major=True)
    with open(args.matching) as f:
        match = {r["row"]: r for r in csv.DictReader(f)}
    agg_ix = {p: i for i, p in enumerate(ids)}
    keep = [i for i, r in enumerate(rows) if r in match and match[r]["patient_id"] in agg_ix]
    j = np.array([agg_ix[match[rows[i]]["patient_id"]] for i in keep])
    keep = np.array(keep)
    xyz_t, wss_t, Z_t = xyz_t[keep], wss_t[keep], Z_t[keep]
    xyz_a, v_a = xyz_a[j], v_a[j]
    cohort = np.array([match[rows[i]]["cohort"] for i in keep])
    groups = [(g, cohort == g) for g in ("MC", "other") if (cohort == g).any()]
    regions = [("whole", WHOLE)] + list(REGIONS.items())
    q = lambda v: f"{np.median(v):.3g} [{np.percentile(v, 25):.3g}-{np.percentile(v, 75):.3g}]"  # noqa: E731
    print(f"{len(keep)} matched patients: " + ", ".join(f"{g} {m.sum()}" for g, m in groups))

    r_t, r_a = ring_radius(xyz_t).mean(axis=1), ring_radius(xyz_a).mean(axis=1)
    ratio = r_a / r_t
    print("\nA. Units: mean radius, median [IQR]")
    print(f"{'cohort':7s} {'training units':>20s} {'4D-flow mm':>20s} {'ratio':>20s} {'ratio CV':>9s}")
    for g, m in groups:
        print(f"{g:7s} {q(r_t[m]):>20s} {q(r_a[m]):>20s} {q(ratio[m]):>20s} {ratio[m].std() / ratio[m].mean():9.3f}")
    print("Ratio CV near 0 = one fixed scale per cohort (harmless within a cohort). Clearly above 0 = "
          "per-patient scale differences in the training coordinates.")

    print("\nB. Spread of the true WSS level (Pa): SD per cohort")
    print(f"{'region':8s} " + " ".join(f"{g + ' mean':>10s} {g + ' SD':>9s}" for g, _ in groups))
    for name, col in regions:
        lv = level(wss_t, col)
        print(f"{name:8s} " + " ".join(f"{lv[m].mean():10.3f} {lv[m].std():9.3f}" for _, m in groups))

    print("\nC. Physics within cohort: r(true level, Poiseuille level)")
    print(f"{'region':8s} " + " ".join(f"{g + ' training':>13s} {g + ' 4D-flow':>12s}" for g, _ in groups))
    for name, col in regions:
        lv = level(wss_t, col)
        pt, pa = poiseuille_level(xyz_t, Z_t, col), poiseuille_level(xyz_a, v_a, col)
        print(f"{name:8s} " + " ".join(f"{corr(lv[m], pt[m]):13.3f} {corr(lv[m], pa[m]):12.3f}" for _, m in groups))
    print("training = training geometry + Zreal; 4D-flow = 4D-flow geometry (mm) + measured inlet velocity.")

    print("\nD. Drivers within cohort: r(true whole-vessel level, ...)")
    lv = level(wss_t, WHOLE)
    peak_z, peak_v = np.abs(Z_t).max(axis=1), np.abs(v_a).max(axis=1)
    print(f"{'cohort':7s} {'radius (train)':>15s} {'radius (4D mm)':>15s} {'Zreal peak':>11s} {'4D inlet v':>11s}")
    for g, m in groups:
        print(f"{g:7s} {corr(lv[m], r_t[m]):15.3f} {corr(lv[m], r_a[m]):15.3f} "
              f"{corr(lv[m], peak_z[m]):11.3f} {corr(lv[m], peak_v[m]):11.3f}")
    print(f"\nWith 38 patients, r values within about +-0.3 of each other are not clearly different.")


if __name__ == "__main__":
    main()
