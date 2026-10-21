"""
Simple, non-deep-learning baselines for WSS, scored with the same 10-fold
cross-validation and the same metrics as the PointNet runs (stitch.py), so a
reviewer can see what the network adds beyond typical anatomy and basic physics.

    python baselines.py              # 10-fold CV, seed 2024 (same folds as main.py --cv 10)

Baselines (all fitted on the training folds only):
  meanmap     every test patient gets the training patients' average WSS map
              (works because grid point i is the same anatomical spot in
              every patient)
  poiseuille  per ring of 32 points: tau = 4 mu Q / (pi r^3), with r the mean
              distance of the ring's points to their centroid and
              Q = peak inlet velocity x inlet area (first ring). Its unknown
              scale is calibrated per region by a straight-line fit
              (WSS = a * tau + b) on the training patients.
  flowradius  the same ingredients, but the exponents are learned: per region,
              log(WSS) = c0 + c1 log(Q) + c2 log(r), fitted on the training
              patients (Poiseuille is the special case c1 = 1, c2 = -3).

Assumptions to check: the 50-point inlet curve (Zreal) is a mean velocity
(so Q needs the inlet area; use --z-is-flow if it is already a flow rate), and
the coordinates are in the same units (mm) for every patient.

Writes ../experiments/15_Baselines_cv10/<baseline>/results.npz (same layout as
main.py's results), so they work with compare_runs.py, e.g.
    python compare_runs.py poiseuille:cv10:flow composite:loo:flow
and prints the same per-region table as stitch.py.
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import KFold

from runs import (REGIONS, WHOLE, level_bland_altman, mae_per_patient, nmae_per_patient,
                  pooled_spearman, r_level, r_spatial_per_patient)

DATA_PATH = "../Data/Sampled/Rpt0_N4096.npz"
N_CIRC, N_AXIAL = 32, 128
EPS = 1e-3  # keeps log() finite where WSS is ~0


def ring_geometry(X, Z, z_is_flow):
    """
    Per patient and ring: radius r (n, 128) and the patient's flow Q (n,).
    Points are circumferential-major (index = c * 128 + a), as everywhere else.
    """
    grid = X.reshape(X.shape[0], N_CIRC, N_AXIAL, 3)
    centroid = grid.mean(axis=1, keepdims=True)                     # (n, 1, 128, 3)
    r = np.linalg.norm(grid - centroid, axis=3).mean(axis=1)          # (n, 128)
    peak = np.abs(Z).max(axis=1)                                     # (n,)
    Q = peak if z_is_flow else peak * np.pi * r[:, 0] ** 2
    return r, Q


def poiseuille_tau(r, Q):
    """4 mu Q / (pi r^3) per ring with mu = 1 (scale is calibrated later): (n, 128)."""
    return 4.0 * Q[:, None] / (np.pi * r ** 3)


def fit_predict(name, tr, te, Y, r, Q):
    """Fit one baseline on training patients tr, predict test patients te: (len(te), 32, 128)."""
    Yg = Y.reshape(Y.shape[0], N_CIRC, N_AXIAL)
    pred = np.zeros((len(te), N_CIRC, N_AXIAL))
    if name == "meanmap":
        pred[:] = Yg[tr].mean(axis=0)
        return pred

    tau = poiseuille_tau(r, Q)                                       # (n, 128)
    for col in REGIONS.values():
        y_tr = Yg[tr][:, :, col]                                     # (ntr, 32, k)
        if name == "poiseuille":
            x_tr = np.broadcast_to(tau[tr][:, None, col], y_tr.shape)
            a, b = np.polyfit(x_tr.ravel(), y_tr.ravel(), 1)
            pred[:, :, col] = (a * tau[te][:, None, col] + b)
        elif name == "flowradius":
            design = lambda idx: np.stack([np.ones_like(r[idx][:, col]),             # noqa: E731
                                           np.broadcast_to(np.log(Q[idx])[:, None], r[idx][:, col].shape),
                                           np.log(r[idx][:, col])], axis=-1)        # (m, k, 3)
            A = np.broadcast_to(design(tr)[:, None], y_tr.shape + (3,)).reshape(-1, 3)
            coef, *_ = np.linalg.lstsq(A, np.log(y_tr.ravel() + EPS), rcond=None)
            pred[:, :, col] = np.exp(design(te) @ coef)[:, None, :] - EPS
    return pred


def print_table(name, true, pred):
    rows = [("whole", WHOLE)] + list(REGIONS.items())
    q = lambda v: f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}]"  # noqa: E731
    print(f"\n=== {name} (10-fold CV, N={true.shape[0]} patients) ===")
    print(f"{'region':8s} {'spatial r':>22s} {'MAE (Pa)':>22s} {'NMAE':>22s} {'pooled rho':>11s} "
          f"{'level r':>8s} {'BA bias':>8s}")
    for rname, col in rows:
        bias, _, _ = level_bland_altman(true, pred, col)
        # meanmap has no patient-specific input: any level r would only reflect
        # fold-to-fold differences in the training average, so it is not reported
        level = "n/a" if name == "meanmap" else f"{r_level(true, pred, col):.3f}"
        with np.errstate(invalid="ignore", divide="ignore"):
            sp = r_spatial_per_patient(true, pred, col)
        print(f"{rname:8s} {q(sp):>22s} {q(mae_per_patient(true, pred, col)):>22s} "
              f"{q(nmae_per_patient(true, pred, col)):>22s} {pooled_spearman(true, pred, col):11.3f} "
              f"{level:>8s} {bias:+8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--z-is-flow", action="store_true", help="the inlet curve is already a flow rate")
    args = ap.parse_args()

    d = np.load(args.data)
    X = d["a"][:, :, :3].astype(np.float64)
    Z = d["a"][:, 0, 3:].astype(np.float64)
    Y = d["b"][:, :, 0].astype(np.float64)
    r, Q = ring_geometry(X, Z, args.z_is_flow)
    print(f"{X.shape[0]} patients. Ring radius {r.min():.2f}-{r.max():.2f} (input units), "
          f"Q {Q.min():.3g}-{Q.max():.3g}; Q = peak inlet {'flow' if args.z_is_flow else 'velocity x inlet area'}")

    # Same folds as main.py --cv 10 (KFold, shuffled, same seed)
    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X))
    for name in ("meanmap", "poiseuille", "flowradius"):
        all_true, all_pred = [], []
        for tr, te in folds:
            all_pred.append(fit_predict(name, tr, te, Y, r, Q).reshape(len(te), -1))
            all_true.append(Y[te])
        true, pred = np.concatenate(all_true), np.concatenate(all_pred)
        out = f"../experiments/15_Baselines_cv{args.folds}/{name}"
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "results.npz"), all_true=true, all_pred=pred)
        print_table(name, true.reshape(-1, N_CIRC, N_AXIAL), pred.reshape(-1, N_CIRC, N_AXIAL))

    print("\nmeanmap gives every patient the same map, so it cannot rank patients (level r n/a); "
          "its spatial r shows how much of the WSS pattern is just typical anatomy.")


if __name__ == "__main__":
    main()
