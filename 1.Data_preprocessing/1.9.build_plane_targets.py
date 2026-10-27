"""
Builds Data/Processed4DFlow/plane_targets.npz: per training patient, a set of
numbers describing the measured 4D-flow velocity on the 36 planes along the
aorta. main.py --planes W adds them as extra training targets (the network
predicts them from the same global feature as PWV); they are never inputs, so
a new patient still only needs shape + inlet curve.

    python 1.9.build_plane_targets.py

Needs the per-patient files from 1.5.process_aggregated_mat.py
(Data/Processed4DFlow/patients/<ID>.npz) and patient_matching.npz
(1.7.match_patients.py). Output rows follow Rpt0_N4096.npz exactly (`c` is
stored so main.py can check).

The in-plane image orientation is only estimated in 1.5, so every feature is
orientation-free (it does not depend on where the plane image starts or which
way it is mirrored). Per plane, at peak systole (the phase of the largest mean
inlet velocity, the same phase for every plane):
  mean_v      mean through-plane velocity (m/s)
  p95_v       95th percentile velocity (m/s): the jet speed
  ecc         distance from the lumen centre to the velocity-weighted centre,
              over the lumen radius: how off-centre (skewed) the jet is
  inner/mid/outer
              mean velocity in the inner / middle / outer third of the lumen
              (by distance from the centre, as a share of the radius), over
              mean_v: the shape of the velocity profile; "outer" is the
              near-wall flow that sets WSS
and over the whole cycle:
  cyc_mean_v  cycle-averaged mean velocity (m/s)
  cyc_ecc     eccentricity of the cycle-averaged velocity
  retro       share of pixel-phases with backward flow
Velocities are signed so that the inlet's systolic flow is positive. Planes
with fewer than MIN_PIXELS lumen pixels give NaN (masked out in training).

Also prints an "information check": how well these measured features predict
each patient's WSS level (10-fold ridge), next to how well the inlet curve +
size alone do. This is roughly the most an extra target could add.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_PLANES = 36
MIN_PIXELS = 10
PEAK_FEATURES = ("mean_v", "p95_v", "ecc", "inner", "mid", "outer")
CYCLE_FEATURES = ("cyc_mean_v", "cyc_ecc", "retro")
FEATURES = PEAK_FEATURES + CYCLE_FEATURES


def in_plane_radius(xyz, normal):
    """Distance of each pixel from the lumen centre, within the plane, and the
    in-plane offsets (n_px, 3)."""
    d = xyz - xyz.mean(axis=0)
    d = d - (d @ normal)[:, None] * normal[None, :]
    return np.linalg.norm(d, axis=1), d


def eccentricity(v, d, R):
    w = np.clip(v, 0, None)
    if w.sum() <= 0:
        return np.nan
    return float(np.linalg.norm((w[:, None] * d).sum(axis=0) / w.sum()) / R)


def plane_features(xyz, normal, v, peak):
    """v: (n_px, phases), already signed. -> one value per FEATURES entry."""
    r, d = in_plane_radius(xyz, normal)
    R = np.sqrt(2 * np.mean(r ** 2))           # equivalent radius (uniform disk: mean r^2 = R^2 / 2)
    vp, vc = v[:, peak], v.mean(axis=1)
    mean_v = vp.mean()
    third = np.minimum((r / R * 3).astype(int), 2)  # 0 inner, 1 middle, 2 outer (incl. beyond R)
    bands = [vp[third == k].mean() / mean_v if (third == k).any() and abs(mean_v) > 1e-6 else np.nan
             for k in range(3)]
    return [mean_v, np.percentile(vp, 95), eccentricity(vp, d, R), *bands,
            vc.mean(), eccentricity(vc, d, R), float((v < 0).mean())]


def patient_targets(p):
    v_all, pid_all = p["vel_through_plane"].astype(np.float64), p["vel_plane_id"]
    xyz_all, normals = p["vel_xyz"].astype(np.float64), p["plane_normal"].astype(np.float64)
    inlet = v_all[pid_all == 0].mean(axis=0)
    peak = int(np.argmax(np.abs(inlet)))
    sign = 1.0 if inlet[peak] >= 0 else -1.0
    out = np.full((N_PLANES, len(FEATURES)), np.nan)
    for j in range(min(N_PLANES, len(normals))):
        m = pid_all == j
        if m.sum() >= MIN_PIXELS:
            out[j] = plane_features(xyz_all[m], normals[j], sign * v_all[m], peak)
    return out, sign


def information_check(targets, wss, size, inlet):
    """10-fold ridge: WSS level from (i) inlet curve + size, (ii) + plane features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.median(wss, axis=1)
    flat = targets.reshape(len(targets), -1)
    flat = np.where(np.isfinite(flat), flat, np.nanmedian(flat, axis=0))
    base = np.column_stack([size, inlet])
    for name, Xf in (("inlet curve + size", base), ("+ measured plane features", np.column_stack([base, flat])),
                     ("plane features alone", flat)):
        pred = np.zeros_like(y)
        for tr, te in KFold(10, shuffle=True, random_state=2024).split(Xf):
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
            pred[te] = model.fit(Xf[tr], y[tr]).predict(Xf[te])
        print(f"  {name:28s} level r {np.corrcoef(y, pred)[0, 1]:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(ROOT / "Data" / "Sampled" / "Rpt0_N4096.npz"))
    ap.add_argument("--patients", default=str(ROOT / "Data" / "Processed4DFlow" / "patients"))
    ap.add_argument("--matching", default=str(ROOT / "Data" / "Processed4DFlow" / "patient_matching.npz"))
    ap.add_argument("--output", default=str(ROOT / "Data" / "Processed4DFlow" / "plane_targets.npz"))
    args = ap.parse_args()

    train, m = np.load(args.train), np.load(args.matching)
    rows = [str(c) for c in train["c"]]
    match = {str(r): str(p) for r, p in zip(m["row"], m["patient_id"])}
    cohort = {str(r): str(c) for r, c in zip(m["row"], m["cohort"])}
    missing = [r for r in rows if r not in match or not (Path(args.patients) / f"{match[r]}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} training rows have no per-patient 4D-flow file (e.g. {missing[:5]}).")

    targets, flipped = np.zeros((len(rows), N_PLANES, len(FEATURES))), 0
    for i, r in enumerate(rows):
        targets[i], sign = patient_targets(np.load(Path(args.patients) / f"{match[r]}.npz"))
        flipped += sign < 0
    np.savez_compressed(args.output, targets=targets.astype(np.float32), c=train["c"],
                        features=np.array(FEATURES), cohort=np.array([cohort[r] for r in rows]))

    n_nan = (~np.isfinite(targets)).any(axis=2).sum()
    print(f"Wrote {args.output}: {targets.shape} (patients, planes, features); "
          f"{n_nan} of {targets.shape[0] * N_PLANES} patient-planes incomplete (masked); "
          f"{flipped} patients had their velocity sign flipped.")
    print("\nMedian [IQR] over patients, planes 0 / 17 / 35:")
    for k, f in enumerate(FEATURES):
        cells = [targets[:, j, k] for j in (0, 17, 35)]
        print(f"  {f:11s} " + "   ".join(f"{np.nanmedian(c):7.3f} [{np.nanpercentile(c, 25):.3f}-"
                                         f"{np.nanpercentile(c, 75):.3f}]" for c in cells))

    print("\nInformation check: whole-vessel WSS level predicted from ... (10-fold ridge)")
    xyz = train["a"][:, :, :3].reshape(len(rows), 32, 128, 3)
    size = np.linalg.norm(xyz - xyz.mean(axis=1, keepdims=True), axis=3).mean(axis=(1, 2))
    information_check(targets, train["b"][:, :, 0], size, train["a"][:, 0, 3:])
    print("If '+ measured plane features' is not clearly above 'inlet curve + size', the planes carry little "
          "extra information about the WSS level and the multi-task run is unlikely to help it.")


if __name__ == "__main__":
    main()
