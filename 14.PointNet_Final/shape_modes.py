"""
Shape modes + ridge regression: how much of each patient's WSS map is a linear
function of a handful of main shape modes (and the inlet flow)? A statistical
shape model baseline, scored with the same 10-fold CV and metrics as the
PointNet runs, so it answers "how does shape relate to WSS" in an
interpretable way: if a few modes come close to the network, WSS is mostly a
simple function of global shape; if not, the network captures something
non-linear or local.

    python shape_modes.py               # 10-fold CV, seed 2024 (same folds as main.py --cv 10)
    python shape_modes.py --z-is-flow   # if the inlet curve is already a flow rate

Per outer fold, using the training patients only:
  1. Procrustes-align all shapes to a template built from the training shapes
     (as main.py --align; label-free).
  2. PCA of the aligned training shapes (all 4096 x 3 coordinates) -> shape
     modes; each patient is described by its first k mode scores.
  3. Flow features (the "with flow" variant): log peak inlet flow Q (as in the
     Poiseuille baseline) and the first 3 modes of the peak-normalised inlet
     waveform.
  4. Standardise the features and fit ridge regression to the whole 4096-point
     WSS map.
  k and the ridge strength are chosen by an inner 5-fold CV on the training
  patients (lowest squared WSS error), so the test fold never picks them.

Variants: shapemodes (shape + flow) and shapemodes-noflow (shape only).
Writes ../experiments/15_Baselines_cv10/<variant>/results.npz, so they work
with compare_runs.py, e.g.
    python compare_runs.py shapemodes:cv10:flow whole:cv10:flow:aligned:phys-input

Also prints a sensitivity table: the test-fold scores for every fixed k. It
shows how the scores grow with the number of modes; it is for description
only, the headline numbers use the inner-CV choice.
"""

import argparse
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import DATA_PATH, N_AXIAL, N_CIRC, print_table, ring_geometry
from runs import REGIONS, WHOLE, r_level, r_spatial_per_patient

K_GRID = (1, 2, 3, 5, 8, 12, 20, 30, 50)
ALPHAS = np.logspace(-3, 3, 13)
N_FLOW_MODES = 3
INNER_FOLDS = 5


# Same Procrustes alignment as main.py --align (copied: importing main.py would
# run its argument parser).
def best_rotation(A, B):
    """Rotation matrix R (A @ R) taking centred shape A (N, 3) onto centred shape B."""
    u, _, vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt


def procrustes_template(Xs, iterations=5):
    """Generalised Procrustes mean (rotation only) of centred shapes Xs (n, N, 3)."""
    Xc = Xs - Xs.mean(axis=1, keepdims=True)
    mean = Xc[0]
    for _ in range(iterations):
        mean = np.mean([x @ best_rotation(x, mean) for x in Xc], axis=0)
    return mean


def align_shapes(X, template):
    """Rotate every centred shape onto the template: (n, N, 3)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    return np.stack([x @ best_rotation(x, template) for x in Xc])


class Features:
    """Shape (and flow) modes fitted on one set of training patients; no labels used."""

    def __init__(self, Xa, Z, use_flow):
        flat = Xa.reshape(len(Xa), -1)
        self.shape_mean = flat.mean(axis=0)
        _, s, vt = np.linalg.svd(flat - self.shape_mean, full_matrices=False)
        self.shape_modes = vt
        self.shape_var = np.cumsum(s ** 2) / (s ** 2).sum()   # variance explained by the first k modes
        self.use_flow = use_flow
        if use_flow:
            w = self._waveform(Z)
            self.wave_mean = w.mean(axis=0)
            self.wave_modes = np.linalg.svd(w - self.wave_mean, full_matrices=False)[2][:N_FLOW_MODES]

    @staticmethod
    def _waveform(Z):
        return Z / np.abs(Z).max(axis=1, keepdims=True)

    def __call__(self, Xa, Z, Q, k):
        S = (Xa.reshape(len(Xa), -1) - self.shape_mean) @ self.shape_modes[:k].T
        if not self.use_flow:
            return S
        W = (self._waveform(Z) - self.wave_mean) @ self.wave_modes.T
        return np.hstack([S, np.log(Q)[:, None], W])


def fit_predict(feats, k, alpha, tr, te, Xa, Z, Q, Y):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(feats(Xa[tr], Z[tr], Q[tr], k), Y[tr])
    return model.predict(feats(Xa[te], Z[te], Q[te], k))


def inner_cv_error(tr, Xa, Z, Q, Y, use_flow, seed):
    """Squared WSS error for every (k, alpha), by 5-fold CV inside the training patients."""
    err = np.zeros((len(K_GRID), len(ALPHAS)))
    for itr, iva in KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed).split(tr):
        a, b = tr[itr], tr[iva]
        feats = Features(Xa[a], Z[a], use_flow)
        for i, k in enumerate(K_GRID):
            for j, alpha in enumerate(ALPHAS):
                err[i, j] += ((fit_predict(feats, k, alpha, a, b, Xa, Z, Q, Y) - Y[b]) ** 2).sum()
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--z-is-flow", action="store_true", help="the inlet curve is already a flow rate")
    ap.add_argument("--out", default="../experiments", help="parent folder of 15_Baselines_cv10/")
    args = ap.parse_args()

    d = np.load(args.data)
    X = d["a"][:, :, :3].astype(np.float64)
    Z = d["a"][:, 0, 3:].astype(np.float64)
    Y = d["b"][:, :, 0].astype(np.float64)
    _, Q = ring_geometry(X, Z, args.z_is_flow)
    n = X.shape[0]
    k_grid = [k for k in K_GRID if k < n * (1 - 1 / args.folds) * (1 - 1 / INNER_FOLDS)]
    print(f"{n} patients. Shape modes k from {k_grid}, ridge alpha {ALPHAS[0]:g}-{ALPHAS[-1]:g}, "
          f"chosen by inner {INNER_FOLDS}-fold CV on the training patients.")

    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X))
    variants = {"shapemodes": True, "shapemodes-noflow": False}
    pred = {v: np.zeros_like(Y) for v in variants}
    pred_k = {v: np.zeros((len(k_grid),) + Y.shape) for v in variants}
    chosen = {v: [] for v in variants}
    shape_var = []

    for f, (tr, te) in enumerate(folds):
        Xa = align_shapes(X, procrustes_template(X[tr]))
        for v, use_flow in variants.items():
            err = inner_cv_error(tr, Xa, Z, Q, Y, use_flow, args.seed + f)[:len(k_grid)]
            i, j = np.unravel_index(np.argmin(err), err.shape)
            feats = Features(Xa[tr], Z[tr], use_flow)
            pred[v][te] = fit_predict(feats, k_grid[i], ALPHAS[j], tr, te, Xa, Z, Q, Y)
            chosen[v].append((k_grid[i], ALPHAS[j]))
            for ik, k in enumerate(k_grid):
                pred_k[v][ik, te] = fit_predict(feats, k, ALPHAS[np.argmin(err[ik])], tr, te, Xa, Z, Q, Y)
            if use_flow:
                shape_var.append(feats.shape_var[np.array(k_grid) - 1])
        print(f"fold {f + 1}/{args.folds}: chosen (k, alpha) "
              + ", ".join(f"{v} ({k}, {a:g})" for v, (k, a) in ((v, chosen[v][-1]) for v in variants)))

    grid = lambda a: a.reshape(-1, N_CIRC, N_AXIAL)  # noqa: E731
    for v in variants:
        out = os.path.join(args.out, f"15_Baselines_cv{args.folds}", v)
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, "results.npz"), all_true=Y, all_pred=pred[v],
                            chosen_k=np.array([c[0] for c in chosen[v]]),
                            chosen_alpha=np.array([c[1] for c in chosen[v]]))
        print_table(v, grid(Y), grid(pred[v]))

    shape_var = np.mean(shape_var, axis=0)
    print("\nSensitivity: test-fold scores for each fixed number of shape modes k (alpha still from inner CV).")
    print("For description only - the tables above use the inner-CV choice of k.")
    print(f"{'k':>3s} {'shape var':>9s}  " + "  ".join(f"{v:>40s}" for v in variants))
    print(f"{'':>3s} {'explained':>9s}  " + "  ".join(f"{'level r whole/pasc/arch/desc/abda  spat r':>40s}"
                                                       for _ in variants))
    for ik, k in enumerate(k_grid):
        cells = []
        for v in variants:
            t, p = grid(Y), grid(pred_k[v][ik])
            levels = "/".join(f"{r_level(t, p, c):.2f}" for c in [WHOLE] + list(REGIONS.values()))
            cells.append(f"{levels}  {np.median(r_spatial_per_patient(t, p, WHOLE)):6.3f}")
        print(f"{k:3d} {shape_var[ik]:9.1%}  " + "  ".join(f"{c:>40s}" for c in cells))


if __name__ == "__main__":
    main()
