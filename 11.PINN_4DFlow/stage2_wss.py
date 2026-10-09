"""
Stage 2: WSS prediction from the Stage 1 PINN flow field, with 10-fold
patient-level cross-validation.

Per surface point and label frame, features are:
  PINN WSS magnitude, its axial and circumferential components, PINN
  near-wall tangential speed, inlet mean velocity at that time, sin/cos of
  cardiac phase, local radius, axial position along the aorta, cohort flag.
Target: the labelled WSS magnitude (WSS_Norm3D) at that frame.

Each test patient is scored three ways, so you can see what Stage 2 adds:
  pinn_raw    - the Stage 1 PINN WSS on its own (Pearson is scale-free; its
                MAE uses a linear calibration fit on the training folds)
  stage2      - the MLP on the features above
Metrics per patient: Pearson at the peak-systolic frame (comparable to the old
SWSS target), Pearson of time-averaged WSS (TAWSS), Pearson over all
points x frames, and MAE in Pa. Reported overall and per cohort, on the whole
wall and on the part covered by the velocity planes (within COVERED_MM).

Requires stage1_flow_pinn.py to have finished for the patients used.

Usage:
    python stage2_wss.py
    python stage2_wss.py --folds 5 --epochs 40
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COVERED_MM, DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, N_AXIAL, cohort_of,  # noqa: E402
                    load_patient, pearson)

logging.basicConfig(level=logging.INFO, format='%(message)s')

FEATURES = ['pinn_wss', 'pinn_wss_axial', 'pinn_wss_circ', 'near_wall_speed', 'inlet_velocity',
            'phase_sin', 'phase_cos', 'local_radius_mm', 'axial_position', 'dist_to_plane_mm', 'cohort_other']


def patient_features(data_dir, stage1_dir, pid):
    """X: (4096, T, F) features, y: (4096, T) labelled WSS, pinn: (4096, T),
    covered: (4096,) surface points within COVERED_MM of a velocity plane."""
    d = load_patient(data_dir, pid)
    with np.load(stage1_dir / f"{pid}.npz") as z:
        s1 = {k: z[k] for k in z.files}
    y = d['wss_mag'].astype(np.float32)
    N, T = y.shape

    vec = s1['pinn_wss_vec']
    t = np.arange(T) / T
    inlet = d['vel_through_plane'][d['vel_plane_id'] == 0].mean(axis=0)  # (P,) m/s
    inlet_t = np.interp(t, d['t_vel'], inlet, period=1.0)

    per_point_frame = [
        s1['pinn_wss_mag'],
        np.einsum('ntc,nc->nt', vec, s1['e_axial']),
        np.einsum('ntc,nc->nt', vec, s1['e_circ']),
        s1['near_wall_speed'],
    ]
    per_frame = [inlet_t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    # dist_to_plane_mm lets the model learn to trust the PINN less where the
    # measured planes do not reach (the aortic ends).
    per_point = [s1['local_radius_mm'], d['surface_grid_idx'][:, 0] / (N_AXIAL - 1), s1['dist_to_plane_mm']]
    per_patient = [float(cohort_of(pid) == 'other')]

    X = np.concatenate([
        np.stack(per_point_frame, axis=-1),
        np.broadcast_to(np.stack(per_frame, axis=-1)[None], (N, T, len(per_frame))),
        np.broadcast_to(np.stack(per_point, axis=-1)[:, None], (N, T, len(per_point))),
        np.full((N, T, len(per_patient)), per_patient, dtype=np.float32),
    ], axis=-1).astype(np.float32)
    return X, y, s1['pinn_wss_mag'].astype(np.float32), s1['dist_to_plane_mm'] < COVERED_MM


class WSSNet(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def score(pred, y):
    peak = int(y.mean(axis=0).argmax())
    return dict(
        r_peak=pearson(pred[:, peak], y[:, peak]),
        r_tawss=pearson(pred.mean(axis=1), y.mean(axis=1)),
        r_all=pearson(pred, y),
        mae_pa=float(np.abs(pred - y).mean()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(DEFAULT_DATA_DIR))
    ap.add_argument('--results', default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--pool-per-patient', type=int, default=20000,
                    help='random (point, frame) samples kept per training patient')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--steps', type=int, default=200, help='gradient steps per epoch')
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    results = Path(args.results)
    stage1_dir = results / "stage1"
    pred_dir = results / "stage2_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    patients = sorted(p.stem for p in stage1_dir.glob("*.npz"))
    if len(patients) < args.folds:
        raise SystemExit(f"Only {len(patients)} patients have Stage 1 results in {stage1_dir}; "
                         f"need at least --folds={args.folds}")
    logging.info(f"{len(patients)} patients with Stage 1 results, {args.folds}-fold CV, device {device}")

    # One pass over all patients to build a fixed random training pool each;
    # full feature tensors are recomputed per test patient at evaluation time.
    pools = {}
    for pid in patients:
        X, y, _, _ = patient_features(args.data, stage1_dir, pid)
        flat_X, flat_y = X.reshape(-1, X.shape[-1]), y.reshape(-1)
        idx = rng.choice(len(flat_y), min(args.pool_per_patient, len(flat_y)), replace=False)
        pools[pid] = (flat_X[idx], flat_y[idx])

    rows = []
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr, te) in enumerate(kf.split(patients)):
        train_ids = [patients[i] for i in tr]
        test_ids = [patients[i] for i in te]
        val_ids = list(rng.choice(train_ids, max(1, len(train_ids) // 10), replace=False))
        fit_ids = [p for p in train_ids if p not in val_ids]

        Xtr = np.concatenate([pools[p][0] for p in fit_ids])
        ytr = np.concatenate([pools[p][1] for p in fit_ids])
        Xva = np.concatenate([pools[p][0] for p in val_ids])
        yva = np.concatenate([pools[p][1] for p in val_ids])
        # A feature that is constant in this fold's training set (e.g. the
        # cohort flag when every training patient is from one cohort) keeps
        # sd=1, otherwise test patients from the other cohort explode.
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd = np.where(sd < 1e-3, 1.0, sd)
        y_mu, y_sd = ytr.mean(), ytr.std() + 1e-6

        # Linear calibration of the raw PINN WSS (feature 0), for the baseline MAE
        a, b = np.polyfit(Xtr[:, 0], ytr, 1)

        to = lambda arr: torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)  # noqa: E731
        Xtr_t, ytr_t = to((Xtr - mu) / sd), to((ytr - y_mu) / y_sd)
        Xva_t, yva_t = to((Xva - mu) / sd), to((yva - y_mu) / y_sd)

        model = WSSNet(Xtr.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(args.epochs):
            model.train()
            for _ in range(args.steps):
                idx = torch.randint(0, len(ytr_t), (args.batch,), device=device)
                loss = ((model(Xtr_t[idx]) - ytr_t[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val = ((model(Xva_t) - yva_t) ** 2).mean().item()
            if val < best:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        logging.info(f"Fold {fold}: {len(fit_ids)} train / {len(val_ids)} val / {len(test_ids)} test, "
                     f"stopped at epoch {epoch}, best val MSE (std units) {best:.4f}")

        for pid in test_ids:
            X, y, pinn, covered = patient_features(args.data, stage1_dir, pid)
            with torch.no_grad():
                pred = model(to((X.reshape(-1, X.shape[-1]) - mu) / sd)).cpu().numpy()
            pred = (pred * y_sd + y_mu).reshape(y.shape)
            np.savez_compressed(pred_dir / f"{pid}.npz", pred=pred.astype(np.float32), label=y,
                                pinn_raw=pinn, fold=fold)
            for method, p in (('pinn_raw', a * pinn + b), ('stage2', pred)):
                rows.append(dict(patient_id=pid, cohort=cohort_of(pid), fold=fold, method=method,
                                 **score(p, y), **{f"{k}_covered": v for k, v in score(p[covered], y[covered]).items()}))

    out_csv = results / "stage2_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    logging.info(f"\n{'=' * 72}\nPer-patient medians (Pearson r, MAE in Pa)\n{'=' * 72}")
    for cohort in ('all', 'MC', 'other'):
        for method in ('pinn_raw', 'stage2'):
            sel = [r for r in rows if r['method'] == method and (cohort == 'all' or r['cohort'] == cohort)]
            if not sel:
                continue
            med = {k: np.nanmedian([r[k] for r in sel]) for k in sel[0] if k.startswith(('r_', 'mae'))}
            logging.info(f"{cohort:6s} {method:9s} n={len(sel):3d}  r_peak={med['r_peak']:.3f}  "
                         f"r_tawss={med['r_tawss']:.3f}  r_all={med['r_all']:.3f}  MAE={med['mae_pa']:.3f}  | "
                         f"covered wall: r_peak={med['r_peak_covered']:.3f}  r_tawss={med['r_tawss_covered']:.3f}")
    fold_med = [np.nanmedian([r['r_peak'] for r in rows if r['method'] == 'stage2' and r['fold'] == k])
                for k in range(args.folds)]
    logging.info(f"Stage 2 r_peak, mean of fold medians: {np.mean(fold_med):.4f} +/- {np.std(fold_med):.4f} "
                 f"(comparable to 10.PINN_TwoStage's summary)")
    logging.info(f"Per-patient results: {out_csv}")


if __name__ == '__main__':
    main()
