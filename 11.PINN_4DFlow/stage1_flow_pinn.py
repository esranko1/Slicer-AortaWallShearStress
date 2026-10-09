"""
Stage 1: per-patient data-assimilation PINN for the full aortic flow field.

For each patient, fits u(x, y, z, t), v, w, p over one cardiac cycle to:
  - Data:       measured through-plane velocity on the 36 4D-flow planes
                (prediction projected on the plane normal, all phases)
  - No-slip:    u = 0 on the segmented wall (4096 surface points)
  - Physics:    incompressible Navier-Stokes momentum + continuity at points
                sampled inside the lumen mask, with real blood properties
  - Periodicity in time (built into the network's time encoding)

Unlike 10.PINN_TwoStage there is no Poiseuille guess, no PCA flow axis and no
inlet heuristic: the measured planes cover the aorta from the ascending to the
descending segment. The WSS labels are NOT used here, so Stage 1 can be run on
every patient (test patients included) without leaking labels into Stage 2.

Non-dimensionalisation (per patient):
  x* = (x - centre) / L        L = max distance of the wall from the lumen centroid
  u* = u / U                   U = 99th percentile of |measured velocity|
  t* = t / T_cycle             T_cycle = --cycle-s (NOT stored in the data; default 1 s)
  p* = p / (rho U^2)
  St * du*/dt* + (u*.grad*)u* + grad* p* - (1/Re) lap* u* = 0,  div* u* = 0
  Re = rho U L / mu,  St = L / (U T_cycle)

Validation: whole planes are held out of the data loss (every 6th plane by
default) and the relative error on them measures how well data + physics
reconstruct flow where there was no measurement.

Outputs, per patient, in <results>/stage1/:
  <id>.npz   PINN WSS vector/magnitude (Pa) on the 4096 surface points at
             every WSS label frame, near-wall speed, surface frames, metrics
  <id>.pt    network weights + normalisation constants
and <results>/stage1_summary.csv with one row per patient.

Usage:
    python stage1_flow_pinn.py                          # all patients
    python stage1_flow_pinn.py --patients 10MC0756 RAT_4858 --iters 2000
    python stage1_flow_pinn.py --overwrite              # redo finished patients
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COVERED_MM, DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, MU, RHO, cohort_of,  # noqa: E402
                    distance_to_planes, list_patients, load_patient, local_radius, pearson,
                    surface_frames)

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ============================================================================
# Network
# ============================================================================

class FlowNet(nn.Module):
    """
    (x*, y*, z*, t*) -> (u*, v*, w*, p*).
    Space: raw coordinates + Gaussian Fourier features (helps resolve the thin
    near-wall layer). Time: harmonics sin/cos(2 pi k t*), so the field is
    exactly periodic over the cardiac cycle.
    """

    def __init__(self, hidden=128, layers=6, n_fourier=64, sigma=2.0, n_harmonics=6):
        super().__init__()
        self.register_buffer('B', torch.randn(3, n_fourier) * sigma)
        self.register_buffer('k', torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        in_dim = 3 + 2 * n_fourier + 2 * n_harmonics
        mods = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods.append(nn.Linear(hidden, 4))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        xyz, t = x[:, :3], x[:, 3:4]
        proj = 2 * np.pi * xyz @ self.B
        wt = 2 * np.pi * t * self.k
        feats = torch.cat([xyz, torch.sin(proj), torch.cos(proj), torch.sin(wt), torch.cos(wt)], dim=1)
        return self.net(feats)


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def physics_residuals(model, x, Re, St):
    """Dimensionless NS momentum (N, 3) and continuity (N,) residuals."""
    x = x.requires_grad_(True)
    out = model(x)
    vel, p = out[:, :3], out[:, 3:4]
    J = [grad(vel[:, i:i + 1], x) for i in range(3)]   # each (N, 4): d/dx, d/dy, d/dz, d/dt
    gp = grad(p, x)
    mom = []
    for i in range(3):
        dt = J[i][:, 3]
        conv = (vel * J[i][:, :3]).sum(dim=1)
        lap = sum(grad(J[i][:, j:j + 1], x)[:, j] for j in range(3))
        mom.append(St * dt + conv + gp[:, i] - lap / Re)
    cont = J[0][:, 0] + J[1][:, 1] + J[2][:, 2]
    return torch.stack(mom, dim=1), cont


def wall_shear(model, x, normals, chunk=4096):
    """
    Dimensionless wall traction: tangential part of (grad u + grad u^T) n.
    Multiply by mu U / L to get Pa. x: (N, 4), normals: (N, 3).
    """
    out = []
    for s in range(0, len(x), chunk):
        xc = x[s:s + chunk].clone().requires_grad_(True)
        n = normals[s:s + chunk]
        vel = model(xc)[:, :3]
        J = torch.stack([torch.autograd.grad(vel[:, i].sum(), xc, retain_graph=True)[0][:, :3]
                         for i in range(3)], dim=1)  # (N, 3, 3), J[:, i, j] = du_i/dx_j
        traction = torch.einsum('bij,bj->bi', J + J.transpose(1, 2), n)
        tangential = traction - (traction * n).sum(dim=1, keepdim=True) * n
        out.append(tangential.detach())
    return torch.cat(out)


# ============================================================================
# Per-patient fit
# ============================================================================

def prepare(d, args):
    """Normalisation constants and the sampled point sets, as float32 arrays."""
    surface = d['surface_xyz'].astype(np.float64)
    centre = d['lumen_xyz'].mean(axis=0)
    L_mm = float(np.linalg.norm(surface - centre, axis=1).max())
    v_meas = d['vel_through_plane']  # (Npx, P) m/s
    U = float(np.percentile(np.abs(v_meas), 99))
    L = L_mm * 1e-3
    Re = RHO * U * L / MU
    St = L / (U * args.cycle_s)

    def nd(xyz_mm):
        return ((xyz_mm - centre) / L_mm).astype(np.float32)

    normal, e_axial, e_circ = surface_frames(d['surface_xyz'], d['centerline'])
    plane_id = d['vel_plane_id']
    if args.holdout_every > 0:
        val_planes = np.arange(args.holdout_every // 2, plane_id.max() + 1, args.holdout_every)
    else:
        val_planes = np.array([], dtype=int)
    is_val = np.isin(plane_id, val_planes)

    return dict(
        centre=centre, L_mm=L_mm, U=U, Re=Re, St=St,
        surface=nd(surface), normal=normal, e_axial=e_axial, e_circ=e_circ,
        lumen=nd(d['lumen_xyz']), voxel_nd=float(d['mm_per_voxel']) / L_mm,
        vel_xyz=nd(d['vel_xyz']), vel_n=d['vel_normal'].astype(np.float32),
        vel=(v_meas / U).astype(np.float32), is_val=is_val, val_planes=val_planes,
        n_phases=v_meas.shape[1], n_wss=d['wss_mag'].shape[1],
    )


def sample_data(prep, idx_pool, n, rng):
    px = rng.choice(idx_pool, n)
    ph = rng.integers(0, prep['n_phases'], n)
    x = np.concatenate([prep['vel_xyz'][px], (ph / prep['n_phases'])[:, None]], axis=1)
    return x.astype(np.float32), prep['vel_n'][px], prep['vel'][px, ph]


def through_plane_error(model, prep, mask, device, max_pts=20000, rng=None):
    """Relative L2 error of predicted vs measured through-plane velocity."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.nan
    rng = rng or np.random.default_rng(0)
    x, n, v = sample_data(prep, idx, min(max_pts, len(idx) * prep['n_phases']), rng)
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device))[:, :3].cpu().numpy()
    vn = (pred * n).sum(axis=1)
    return float(np.linalg.norm(vn - v) / (np.linalg.norm(v) + 1e-12))


def fit_patient(pid, d, args, device):
    prep = prepare(d, args)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = FlowNet(args.hidden, args.layers, args.fourier, args.sigma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(args.lr_final / args.lr) ** (1 / args.iters))

    train_idx = np.flatnonzero(~prep['is_val'])
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    jitter = prep['voxel_nd'] / 2
    t0 = time.time()

    for it in range(1, args.iters + 1):
        x_d, n_d, v_d = sample_data(prep, train_idx, args.n_data, rng)
        pred = model(to(x_d))[:, :3]
        loss_data = (((pred * to(n_d)).sum(dim=1) - to(v_d)) ** 2).mean()

        w_idx = rng.integers(0, len(prep['surface']), args.n_wall)
        x_w = np.concatenate([prep['surface'][w_idx], rng.random((args.n_wall, 1))], axis=1).astype(np.float32)
        loss_wall = (model(to(x_w))[:, :3] ** 2).sum(dim=1).mean()

        c_idx = rng.integers(0, len(prep['lumen']), args.n_colloc)
        x_c = prep['lumen'][c_idx] + rng.uniform(-jitter, jitter, (args.n_colloc, 3))  # stays in its voxel
        x_c = np.concatenate([x_c, rng.random((args.n_colloc, 1))], axis=1).astype(np.float32)
        mom, cont = physics_residuals(model, to(x_c), prep['Re'], prep['St'])
        loss_mom = (mom ** 2).sum(dim=1).mean()
        loss_cont = (cont ** 2).mean()

        # Physics is ramped in over the first part of training so the network
        # first finds the measured flow, then is pulled toward NS-consistency.
        ramp = min(1.0, it / max(1, args.physics_ramp))
        loss = (args.w_data * loss_data + args.w_wall * loss_wall
                + ramp * (args.w_mom * loss_mom + args.w_cont * loss_cont))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % args.log_every == 0 or it == args.iters:
            val_err = through_plane_error(model, prep, prep['is_val'], device, max_pts=5000)
            logging.info(f"  [{pid}] it {it:5d}  data={loss_data.item():.4f} wall={loss_wall.item():.4f} "
                         f"mom={loss_mom.item():.4f} cont={loss_cont.item():.4f}  "
                         f"val_plane_relerr={val_err:.3f}  ({time.time() - t0:.0f}s)")

    model.eval()
    return model, prep, time.time() - t0


def export_patient(pid, d, model, prep, fit_s, args, device, out_dir):
    """Evaluate the trained PINN at the WSS label frames and save features."""
    n_wss = prep['n_wss']
    N = len(prep['surface'])
    surf = torch.from_numpy(prep['surface']).to(device)
    normals = torch.from_numpy(prep['normal']).to(device)
    depth_nd = args.near_wall_mm / prep['L_mm']
    scale_pa = MU * prep['U'] / (prep['L_mm'] * 1e-3)

    wss_vec = np.zeros((N, n_wss, 3), dtype=np.float32)
    near_speed = np.zeros((N, n_wss), dtype=np.float32)
    for k in range(n_wss):
        t = torch.full((N, 1), k / n_wss, device=device)
        x = torch.cat([surf, t], dim=1)
        wss_vec[:, k] = (wall_shear(model, x, normals) * scale_pa).cpu().numpy()
        # Tangential speed a fixed depth inside the wall: a second, gradient-
        # free shear proxy that is less sensitive to how sharply the network
        # resolves the wall layer.
        x_in = torch.cat([surf - depth_nd * normals, t], dim=1)
        with torch.no_grad():
            u = model(x_in)[:, :3]
        u_t = u - (u * normals).sum(dim=1, keepdim=True) * normals
        near_speed[:, k] = (u_t.norm(dim=1) * prep['U']).cpu().numpy()
    wss_mag = np.linalg.norm(wss_vec, axis=2)

    label = d['wss_mag']
    peak = int(label.mean(axis=0).argmax())
    # The planes do not reach the ends of the segmented aorta; there the PINN
    # has no measured flow to go on, so metrics are also reported on the
    # covered part of the wall only.
    dist_mm = distance_to_planes(d)
    cov = dist_mm < COVERED_MM
    metrics = dict(
        patient_id=pid,
        cohort=cohort_of(pid),
        n_vel_phases=prep['n_phases'],
        n_wss_frames=n_wss,
        Re=round(prep['Re'], 1),
        St=round(prep['St'], 5),
        train_plane_relerr=round(through_plane_error(model, prep, ~prep['is_val'], device), 4),
        val_plane_relerr=round(through_plane_error(model, prep, prep['is_val'], device), 4),
        r_wss_all=round(pearson(wss_mag, label), 4),
        r_wss_peak=round(pearson(wss_mag[:, peak], label[:, peak]), 4),
        r_tawss=round(pearson(wss_mag.mean(axis=1), label.mean(axis=1)), 4),
        r_nearwall_peak=round(pearson(near_speed[:, peak], label[:, peak]), 4),
        covered_frac=round(float(cov.mean()), 3),
        r_wss_peak_covered=round(pearson(wss_mag[cov, peak], label[cov, peak]), 4),
        r_tawss_covered=round(pearson(wss_mag[cov].mean(axis=1), label[cov].mean(axis=1)), 4),
        wss_ratio_median=round(float(np.median(wss_mag[:, peak]) / (np.median(label[:, peak]) + 1e-12)), 4),
        fit_seconds=round(fit_s, 1),
    )

    np.savez_compressed(
        out_dir / f"{pid}.npz",
        pinn_wss_vec=wss_vec, pinn_wss_mag=wss_mag.astype(np.float32), near_wall_speed=near_speed,
        normal=prep['normal'], e_axial=prep['e_axial'], e_circ=prep['e_circ'],
        local_radius_mm=local_radius(d['surface_xyz'], d['centerline']),
        dist_to_plane_mm=dist_mm,
        val_planes=prep['val_planes'], **{f"m_{k}": v for k, v in metrics.items()},
    )
    torch.save({'state_dict': model.state_dict(),
                'arch': dict(hidden=args.hidden, layers=args.layers, n_fourier=args.fourier, sigma=args.sigma),
                'centre_mm': prep['centre'], 'L_mm': prep['L_mm'], 'U': prep['U'],
                'Re': prep['Re'], 'St': prep['St'], 'cycle_s': args.cycle_s},
               out_dir / f"{pid}.pt")
    return metrics


# ============================================================================
# Main
# ============================================================================

def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    # MPS lacks some double-backward kernels needed for the NS residual, so
    # Apple machines fall back to CPU unless --device mps is given explicitly.
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(DEFAULT_DATA_DIR))
    ap.add_argument('--results', default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument('--patients', nargs='*', help='only these patient IDs')
    ap.add_argument('--skip-check', action='store_true', help="skip patients the manifest marks 'check'")
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=2024)
    # physics / data
    ap.add_argument('--cycle-s', type=float, default=1.0, help='cardiac cycle length in s (not in the data)')
    ap.add_argument('--holdout-every', type=int, default=6, help='hold out every Nth plane for validation (0 = none)')
    ap.add_argument('--near-wall-mm', type=float, default=1.0)
    # network
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--fourier', type=int, default=64)
    ap.add_argument('--sigma', type=float, default=2.0)
    # optimisation
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-final', type=float, default=2e-5)
    ap.add_argument('--n-data', type=int, default=4096)
    ap.add_argument('--n-wall', type=int, default=2048)
    ap.add_argument('--n-colloc', type=int, default=2048)
    ap.add_argument('--w-data', type=float, default=1.0)
    ap.add_argument('--w-wall', type=float, default=1.0)
    ap.add_argument('--w-mom', type=float, default=0.1)
    ap.add_argument('--w-cont', type=float, default=0.1)
    ap.add_argument('--physics-ramp', type=int, default=1000)
    ap.add_argument('--log-every', type=int, default=500)
    args = ap.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.results) / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    patients = list_patients(args.data, args.skip_check, args.patients)
    logging.info(f"Device: {device}. {len(patients)} patients. Results: {out_dir}")

    summary_path = Path(args.results) / "stage1_summary.csv"
    rows = {}
    if summary_path.exists():
        with open(summary_path, newline='') as f:
            rows = {r['patient_id']: r for r in csv.DictReader(f)}

    for i, pid in enumerate(patients, 1):
        if (out_dir / f"{pid}.npz").exists() and not args.overwrite:
            logging.info(f"[{i}/{len(patients)}] {pid}: done already, skipping")
            continue
        logging.info(f"[{i}/{len(patients)}] {pid}")
        try:
            d = load_patient(args.data, pid)
            model, prep, fit_s = fit_patient(pid, d, args, device)
            m = export_patient(pid, d, model, prep, fit_s, args, device, out_dir)
            logging.info(f"  -> plane relerr train={m['train_plane_relerr']} val={m['val_plane_relerr']}  "
                         f"r(WSS peak)={m['r_wss_peak']} [covered {m['r_wss_peak_covered']}]  "
                         f"r(TAWSS)={m['r_tawss']} [covered {m['r_tawss_covered']}]  "
                         f"PINN/label WSS={m['wss_ratio_median']}")
            rows[pid] = m
        except Exception as e:  # keep the batch going; the error lands in the summary
            logging.exception(f"  FAILED {pid}: {e}")
            rows[pid] = {'patient_id': pid, 'error': f"{type(e).__name__}: {e}"}

        fields = []
        for r in rows.values():
            fields += [k for k in r if k not in fields]
        with open(summary_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows.values())

    done = [r for r in rows.values() if r.get('val_plane_relerr') not in (None, '')]
    if done:
        for key in ('val_plane_relerr', 'r_wss_peak', 'r_wss_peak_covered', 'r_tawss', 'r_tawss_covered',
                    'covered_frac', 'wss_ratio_median'):
            vals = np.array([float(r[key]) for r in done], dtype=float)
            logging.info(f"median {key}: {np.nanmedian(vals):.3f}")
    logging.info(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
