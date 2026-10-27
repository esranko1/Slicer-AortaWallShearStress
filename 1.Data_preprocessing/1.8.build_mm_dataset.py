"""
Builds Data/Sampled/Rpt0_N4096_mm.npz: the training data (Rpt0_N4096.npz) with
the geometry and inlet curve replaced by the 4D-flow versions from
1.5.process_aggregated_mat.py. The WSS labels, patient order and IDs are
unchanged, so results stay comparable (use main.py --mm).

Why: cohort_diagnose.py showed Xreal is not in mm. The 4D-flow / training size
ratio is ~2.0 in the MC cohort and ~1.6 in the other (matching their voxel
sizes), and varies ~10% between patients within each cohort. The 4D-flow
geometry (mm) and measured inlet velocity relate better to WSS in every region
and both cohorts, so they are taken as the more accurate inputs.

    python 1.8.build_mm_dataset.py

Needs Rpt0_N4096.npz, Data/Processed4DFlow/legacy_Rpt0_N4096.npz and
patient_matching.npz (1.7.match_patients.py). Each patient's 4D-flow grid is
re-indexed with the transform the matching found, so point i is the same wall
point as in the training data. Inlet curve: measured mean inlet velocity in
cm/s (Zreal is also a velocity; main.py rescales it anyway).
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_CIRC, N_AXIAL = 32, 128


def apply(grid, shift, rev_circ, rev_axial):
    """Same re-indexing as 1.7.match_patients.py, on (32, 128, ...) grids."""
    g = np.roll(grid, shift, axis=0)
    if rev_circ:
        g = g[::-1]
    if rev_axial:
        g = g[:, ::-1]
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(ROOT / "Data" / "Sampled" / "Rpt0_N4096.npz"))
    ap.add_argument("--agg", default=str(ROOT / "Data" / "Processed4DFlow" / "legacy_Rpt0_N4096.npz"))
    ap.add_argument("--matching", default=str(ROOT / "Data" / "Processed4DFlow" / "patient_matching.npz"))
    ap.add_argument("--output", default=str(ROOT / "Data" / "Sampled" / "Rpt0_N4096_mm.npz"))
    args = ap.parse_args()

    train, agg, m = np.load(args.train), np.load(args.agg), np.load(args.matching)
    rows = [str(c) for c in train["c"]]
    agg_ix = {str(p): i for i, p in enumerate(agg["c"])}
    match = {str(r): k for k, r in enumerate(m["row"])}
    missing = [r for r in rows if r not in match or str(m["patient_id"][match[r]]) not in agg_ix]
    if missing:
        raise SystemExit(f"{len(missing)} training rows have no matched 4D-flow patient (e.g. {missing[:5]}); "
                         "this file needs all of them.")

    n = len(rows)
    xyz = np.zeros((n, N_CIRC, N_AXIAL, 3))
    inlet = np.zeros((n, agg["a"].shape[2] - 3))
    wss_r = []
    for i, r in enumerate(rows):
        k = match[r]
        j = agg_ix[str(m["patient_id"][k])]
        tr = (int(m["circ_shift"][k]), bool(m["circ_reversed"][k]), bool(m["axial_reversed"][k]))
        xyz[i] = apply(agg["a"][j, :, :3].reshape(N_AXIAL, N_CIRC, 3).transpose(1, 0, 2), *tr)
        inlet[i] = agg["a"][j, 0, 3:] * 100.0                                  # m/s -> cm/s
        if "d" in train.files:                                                 # sanity: same wall points?
            w_a = apply(agg["d"][j, :, 0].reshape(N_AXIAL, N_CIRC).T, *tr).ravel()
            wss_r.append(np.corrcoef(w_a, train["d"][i, :, 0])[0, 1])
    xyz = xyz.reshape(n, -1, 3)
    xyz -= xyz.mean(axis=1, keepdims=True)                                     # centred, as in 1.3

    if wss_r:
        print(f"Point correspondence check (TAWSS r per patient): min {min(wss_r):.4f}, median {np.median(wss_r):.4f}")
        if min(wss_r) < 0.98:
            raise SystemExit("Some patients' grids do not line up with the training data; check patient_matching.")

    old = train["a"][:, :, :3].reshape(n, N_CIRC, N_AXIAL, 3)
    new = xyz.reshape(n, N_CIRC, N_AXIAL, 3)
    rad = lambda g: np.linalg.norm(g - g.mean(axis=1, keepdims=True), axis=3).mean(axis=(1, 2))  # noqa: E731
    cohort = {str(r): str(c) for r, c in zip(m["row"], m["cohort"])}
    for g in ("MC", "other"):
        sel = np.array([cohort[r] == g for r in rows])
        if sel.any():
            print(f"{g:6s} mean radius: training units {np.median(rad(old[sel])):.2f} -> mm {np.median(rad(new[sel])):.2f}")

    a = np.concatenate([xyz, np.repeat(inlet[:, None, :], xyz.shape[1], axis=1)], axis=2).astype(np.float32)
    out = {"a": a, "b": train["b"], "c": train["c"]}
    if "d" in train.files:
        out["d"] = train["d"]
    np.savez_compressed(args.output, **out)
    print(f"Wrote {args.output}: a {a.shape} (xyz in mm + inlet velocity cm/s), labels unchanged.")


if __name__ == "__main__":
    main()
