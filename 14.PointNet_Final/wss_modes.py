"""
WSS-modes model: instead of predicting all 4096 wall values, predict the first
N_OUT principal components ("modes") of the WSS maps, with one Gaussian process
per mode, from a short list of patient features. With 100 patients, ~20
outputs and ~15 inputs is a much easier problem than 4096 outputs, and the
Gaussian process gives every prediction an uncertainty.

    python wss_modes.py                 # 10-fold CV, seed 2024 (same folds as main.py --cv 10); ~10 min on CPU
    python wss_modes.py --folds 100     # leave-one-out; ~1 h on CPU

Inputs per patient (no labels; fitted on the training patients of each fold):
  shape    first k shape modes of the Procrustes-aligned geometry (as
           shape_modes.py)
  flow     log peak inlet flow Q and 3 inlet waveform modes
  physics  log Poiseuille level (4 mu Q / pi r^3, median over the region's
           rings) for the whole vessel and each of the 4 regions
Outputs: the first N_OUT PCA coefficients of the training patients' WSS maps;
the predicted map is the training mean map + predicted coefficients x modes.

Variants (each written to ../experiments/15_Baselines_<cv>/<variant>/results.npz,
so compare_runs.py works, e.g.
    python compare_runs.py modes-gp:cv10:flow whole:cv10:flow:aligned:phys-input):
  modes-gp         Gaussian process (smooth non-linear + linear kernel, one
                   length scale per input so unhelpful inputs are switched off)
  modes-gp-nophys  the same without the physics inputs: what the physics adds
  modes-ridge      linear ridge regression on the same inputs: what the
                   non-linearity adds
k is chosen by an inner 5-fold CV on the training patients (lowest squared WSS
error), separately per variant. --k fixes it for ALL variants (quick tests
only: the variants need different k, e.g. without physics inputs k=2 is far
too few).

Also prints, for modes-gp, how often the true regional mean WSS falls inside
the model's 95% interval (should be close to 95% if the uncertainty is honest).
"""

import argparse
import os
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from baselines import DATA_PATH, N_AXIAL, N_CIRC, poiseuille_tau, print_table, ring_geometry
from runs import REGIONS, WHOLE
from shape_modes import Features, align_shapes, procrustes_template

K_GRID = (2, 3, 5, 8)
N_OUT = 20
INNER_FOLDS = 5
VARIANTS = {"modes-gp": ("gp", True), "modes-gp-nophys": ("gp", False), "modes-ridge": ("ridge", True)}
REGION_ROWS = [("whole", WHOLE)] + list(REGIONS.items())


def physics_features(r, Q):
    """log Poiseuille level per region (whole + 4): (n, 5). Shape and inlet only."""
    tau = poiseuille_tau(r, Q)
    return np.log(np.stack([np.median(tau[:, col], axis=1) for _, col in REGION_ROWS], axis=1))


def region_mask(col):
    m = np.zeros((N_CIRC, N_AXIAL), dtype=bool)
    m[:, col] = True
    return m.ravel()


def gp_kernel(d):
    return (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.full(d, 3.0), (1e-1, 1e3))
            + ConstantKernel(0.1, (1e-4, 1e1)) * DotProduct(1.0, (1e-2, 1e2))
            + WhiteKernel(0.5, (1e-3, 1e1)))


def fit_predict(kind, F_tr, Y_tr, F_te):
    """-> predicted maps (m, 4096), std of each predicted mode coefficient (m, N_OUT) or None,
    and the WSS modes (N_OUT, 4096)."""
    mean = Y_tr.mean(axis=0)
    modes = np.linalg.svd(Y_tr - mean, full_matrices=False)[2][:N_OUT]
    C = (Y_tr - mean) @ modes.T
    xs, cs = StandardScaler().fit(F_tr), StandardScaler().fit(C)
    A, B, Cs = xs.transform(F_tr), xs.transform(F_te), cs.transform(C)
    if kind == "ridge":
        P, S = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(A, Cs).predict(B), None
    else:
        P, S = np.zeros((len(B), Cs.shape[1])), np.zeros((len(B), Cs.shape[1]))
        for j in range(Cs.shape[1]):
            gp = GaussianProcessRegressor(gp_kernel(A.shape[1]), random_state=0).fit(A, Cs[:, j])
            P[:, j], S[:, j] = gp.predict(B, return_std=True)
        S = S * cs.scale_
    return mean + cs.inverse_transform(P) @ modes, S, modes


def features(Xa, Z, Q, phys, tr, k, use_phys):
    f = Features(Xa[tr], Z[tr], use_flow=True)
    F = f(Xa, Z, Q, k)
    return np.hstack([F, phys]) if use_phys else F


def choose_k(kind, use_phys, tr, Xa, Z, Q, phys, Y, seed):
    err = np.zeros(len(K_GRID))
    for itr, iva in KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed).split(tr):
        a, b = tr[itr], tr[iva]
        for i, k in enumerate(K_GRID):
            F = features(Xa, Z, Q, phys, a, k, use_phys)
            err[i] += ((fit_predict(kind, F[a], Y[a], F[b])[0] - Y[b]) ** 2).sum()
    return K_GRID[int(np.argmin(err))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--folds", type=int, default=10, help="10, or the number of patients for leave-one-out")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--k", type=int, help="fix the number of shape modes for all variants (quick tests only)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--z-is-flow", action="store_true", help="the inlet curve is already a flow rate")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    d = np.load(args.data)
    X = d["a"][:, :, :3].astype(np.float64)
    Z = d["a"][:, 0, 3:].astype(np.float64)
    Y = d["b"][:, :, 0].astype(np.float64)
    r, Q = ring_geometry(X, Z, args.z_is_flow)
    phys = physics_features(r, Q)
    n = len(X)
    cv = "loo" if args.folds == n else f"cv{args.folds}"
    print(f"{n} patients, {cv}. Outputs: first {N_OUT} WSS modes. Shape modes k "
          f"{args.k if args.k else f'from {K_GRID} by inner {INNER_FOLDS}-fold CV'}.")

    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X))
    pred = {v: np.zeros_like(Y) for v in args.variants}
    coef_sd = {v: np.zeros((n, N_OUT)) for v in args.variants}
    level_sd = {v: np.full((n, len(REGION_ROWS)), np.nan) for v in args.variants}
    chosen = {v: [] for v in args.variants}
    masks = [region_mask(col) for _, col in REGION_ROWS]
    t0 = time.time()
    for f, (tr, te) in enumerate(folds):
        Xa = align_shapes(X, procrustes_template(X[tr]))
        for v in args.variants:
            kind, use_phys = VARIANTS[v]
            k = args.k or choose_k(kind, use_phys, tr, Xa, Z, Q, phys, Y, args.seed + f)
            F = features(Xa, Z, Q, phys, tr, k, use_phys)
            pred[v][te], S, modes = fit_predict(kind, F[tr], Y[tr], F[te])
            chosen[v].append(k)
            if S is not None:
                # regional mean WSS is linear in the coefficients: its sd from the per-mode sds
                coef_sd[v][te] = S
                w = np.stack([modes[:, m].mean(axis=1) for m in masks], axis=1)   # (N_OUT, regions)
                level_sd[v][te] = np.sqrt((S ** 2) @ (w ** 2))
        if f == 0 or (f + 1) % max(1, len(folds) // 10) == 0:
            print(f"fold {f + 1}/{len(folds)} ({time.time() - t0:.0f} s): k "
                  + ", ".join(f"{v} {chosen[v][-1]}" for v in args.variants))

    grid = lambda a: a.reshape(-1, N_CIRC, N_AXIAL)  # noqa: E731
    for v in args.variants:
        out = os.path.join("../experiments", f"15_Baselines_{cv}", v)
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "results.npz"), all_true=Y, all_pred=pred[v],
                            chosen_k=np.array(chosen[v]), coef_sd=coef_sd[v], level_sd=level_sd[v],
                            level_sd_regions=np.array([name for name, _ in REGION_ROWS]))
        print_table(f"{v} ({cv})", grid(Y), grid(pred[v]))
        print(f"chosen k per fold: {np.bincount(chosen[v]).nonzero()[0].tolist()} "
              f"(counts {np.bincount(chosen[v])[np.bincount(chosen[v]) > 0].tolist()})")

    if "modes-gp" in args.variants:
        print("\nmodes-gp uncertainty: share of patients whose true regional MEAN WSS lies in the 95% interval")
        for i, (name, _) in enumerate(REGION_ROWS):
            t, p = Y[:, masks[i]].mean(axis=1), pred["modes-gp"][:, masks[i]].mean(axis=1)
            sd = level_sd["modes-gp"][:, i]
            print(f"  {name:6s} {np.mean(np.abs(t - p) <= 1.96 * sd):6.1%}   "
                  f"(median interval half-width {1.96 * np.median(sd):.3f} Pa)")


if __name__ == "__main__":
    main()
