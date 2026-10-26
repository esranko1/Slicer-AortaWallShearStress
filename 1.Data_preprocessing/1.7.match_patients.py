"""
Matches the rows of the training data (Data/Sampled/Rpt0_N4096.npz, built from
dataForPy9.mat, rows "0".."99" with no patient names) to the AggregatedData
patients processed by 1.5.process_aggregated_mat.py (legacy_Rpt0_N4096.npz in
Data/Processed4DFlow, which keeps the patient IDs). Needed to use the 4D-flow
velocity planes per training patient, and gives each row its cohort.

    python 1.7.match_patients.py
    python 1.7.match_patients.py --train ..\\Data\\Sampled\\Rpt0_N4096.npz --agg ..\\Data\\Processed4DFlow\\legacy_Rpt0_N4096.npz

Cues, each scored for every (row, patient) pair:
  map      Pearson of the time-averaged WSS maps (TAWSS; does not depend on
           which frame counts as systole), maximised over the ways the two
           grids could be indexed differently: circumferential start point
           (32 shifts), circumferential direction, axial direction
  profile  Pearson of the TAWSS ring-mean profiles along the vessel (128 values)
  radius   Pearson of the ring radius profiles along the vessel (geometry only,
           independent of WSS)
The pairing maximises map r + radius r summed over pairs (Hungarian
assignment), so neither the WSS processing nor the geometry alone decides it.
A match is
  exact      map r > 0.98 (same data, possibly re-indexed)
  confident  the row's and the patient's best partner is each other, by the map
             AND by the radius profile, with a clear margin over the runner-up
  uncertain  anything else - check these before using them
Grid layouts: training data is circumferential-major (index = c*128 + a),
1.5's output axial-major (index = a*32 + c).

Also compares the inlet curves of matched pairs (training Zreal vs 1.5's
measured mean inlet velocity, m/s) to settle whether Zreal is a velocity or a
flow rate: whichever of (velocity) and (velocity x inlet area) Zreal is more
nearly proportional to across patients.

Writes Data/Processed4DFlow/patient_matching.csv and patient_matching.npz
(per training row: matched ID, cohort, scores, status and the grid transform
that maps the 4D-flow grid onto the training grid).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
N_CIRC, N_AXIAL = 32, 128
EXACT_R = 0.98
MARGIN = 0.05


def load(path, axial_major):
    """-> xyz (n, 32, 128, 3), TAWSS (n, 32, 128), inlet curve (n, 50), ids (n,)"""
    d = np.load(path)
    n = d["a"].shape[0]
    shape = (n, N_AXIAL, N_CIRC) if axial_major else (n, N_CIRC, N_AXIAL)
    grid = lambda v, *extra: v.reshape(shape + extra)  # noqa: E731
    xyz, tawss = grid(d["a"][:, :, :3], 3), grid(d["d"][:, :, 0])
    if axial_major:
        xyz, tawss = xyz.transpose(0, 2, 1, 3), tawss.transpose(0, 2, 1)
    return (xyz.astype(np.float64), tawss.astype(np.float64),
            d["a"][:, 0, 3:].astype(np.float64), np.array([str(c) for c in d["c"]]))


def zscore(v):
    v = v.reshape(v.shape[0], -1)
    v = v - v.mean(axis=1, keepdims=True)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def ring_radius(xyz):
    return np.linalg.norm(xyz - xyz.mean(axis=1, keepdims=True), axis=3).mean(axis=1)   # (n, 128)


def transforms():
    """All re-indexings of a (32, 128) grid: (shift, reverse circ, reverse axial)."""
    return [(s, rc, ra) for s in range(N_CIRC) for rc in (False, True) for ra in (False, True)]


def apply(grid, shift, rev_circ, rev_axial):
    """Re-index (..., 32, 128) grids; the same transform works on any trailing-dim-free map."""
    g = np.roll(grid, shift, axis=-2)
    if rev_circ:
        g = g[..., ::-1, :]
    if rev_axial:
        g = g[..., :, ::-1]
    return g


def best_map_scores(train_map, agg_map):
    """Map r for every pair, maximised over grid re-indexings: score (n, m), index of best transform (n, m)."""
    t = zscore(train_map)
    ts = transforms()
    scores = np.stack([t @ zscore(apply(agg_map, *tr)).T for tr in ts])   # (n_transforms, n, m)
    return scores.max(axis=0), scores.argmax(axis=0)


def profile_scores(train_prof, agg_prof):
    """Pearson of 1D profiles along the vessel, best of forward / reversed: (n, m)."""
    t = zscore(train_prof)
    return np.maximum(t @ zscore(agg_prof).T, t @ zscore(agg_prof[:, ::-1]).T)


def margin(scores, axis):
    """Best minus second best along an axis."""
    s = np.sort(scores, axis=axis)
    return np.take(s, -1, axis=axis) - np.take(s, -2, axis=axis)


def cohort(pid):
    return "MC" if "MC" in pid.upper() else "other"


def inlet_units(Z_train, v_agg, xyz_train):
    """
    Is Zreal proportional to the measured mean inlet velocity, or to velocity x
    inlet area? The area is the training grid's first ring (the inlet used by
    the Poiseuille input in 14.PointNet_Final), assumed in mm.
    """
    area = np.pi * ring_radius(xyz_train)[:, 0] ** 2
    z_pk, v_pk = np.abs(Z_train).max(axis=1), np.abs(v_agg).max(axis=1)   # Zreal units, m/s
    shape_r = []
    for z, v in zip(Z_train, v_agg):                          # waveform shape agreement, best circular lag
        zz, vv = zscore(z[None])[0], zscore(v[None])[0]
        shape_r.append(max(np.dot(zz, np.roll(vv, k)) for k in range(len(vv))))
    out = {}
    for name, ref in (("velocity", v_pk), ("flow = velocity x area", v_pk * area)):
        ratio = z_pk / ref
        out[name] = (np.median(ratio), np.std(ratio) / np.mean(ratio), np.corrcoef(z_pk, ref)[0, 1])
    return np.array(shape_r), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(ROOT / "Data" / "Sampled" / "Rpt0_N4096.npz"))
    ap.add_argument("--agg", default=str(ROOT / "Data" / "Processed4DFlow" / "legacy_Rpt0_N4096.npz"))
    ap.add_argument("--out", default=str(ROOT / "Data" / "Processed4DFlow"))
    args = ap.parse_args()

    xyz_t, ta_t, Z_t, rows = load(args.train, axial_major=False)
    xyz_a, ta_a, v_a, ids = load(args.agg, axial_major=True)
    print(f"Training rows: {len(rows)}. 4D-flow patients: {len(ids)} "
          f"({sum(cohort(p) == 'MC' for p in ids)} MC, {sum(cohort(p) == 'other' for p in ids)} other).")

    s_map, s_tr = best_map_scores(ta_t, ta_a)
    s_prof = profile_scores(ta_t.mean(axis=1), ta_a.mean(axis=1))
    s_rad = profile_scores(ring_radius(xyz_t), ring_radius(xyz_a))

    row_ix, col_ix = linear_sum_assignment(-(s_map + s_rad))
    ts = transforms()
    mutual_map = (s_map.argmax(axis=1)[row_ix] == col_ix) & (s_map.argmax(axis=0)[col_ix] == row_ix)
    mutual_rad = (s_rad.argmax(axis=1)[row_ix] == col_ix) & (s_rad.argmax(axis=0)[col_ix] == row_ix)
    m_map, m_rad = margin(s_map, 1)[row_ix], margin(s_rad, 1)[row_ix]

    records = []
    for k, (i, j) in enumerate(zip(row_ix, col_ix)):
        r = s_map[i, j]
        if r > EXACT_R:
            status = "exact"
        elif mutual_map[k] and mutual_rad[k] and m_map[k] > MARGIN:
            status = "confident"
        else:
            status = "uncertain"
        shift, rev_c, rev_a = ts[s_tr[i, j]]
        records.append(dict(row=rows[i], patient_id=ids[j], cohort=cohort(ids[j]), status=status,
                            map_r=round(r, 4), map_margin=round(m_map[k], 4), profile_r=round(s_prof[i, j], 4),
                            radius_r=round(s_rad[i, j], 4), radius_margin=round(m_rad[k], 4),
                            mutual_best_map=bool(mutual_map[k]), mutual_best_radius=bool(mutual_rad[k]),
                            circ_shift=shift, circ_reversed=rev_c, axial_reversed=rev_a))
    records.sort(key=lambda d: int(d["row"]) if d["row"].isdigit() else d["row"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "patient_matching.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    np.savez(out / "patient_matching.npz", **{k: np.array([d[k] for d in records]) for k in records[0]})

    counts = {s: sum(d["status"] == s for d in records) for s in ("exact", "confident", "uncertain")}
    print(f"\nMatched {len(records)} rows: {counts}")
    unmatched = len(rows) - len(records)
    if unmatched:
        print(f"{unmatched} training rows have no 4D-flow patient (fewer 4D-flow files than rows).")
    print(f"Map r of the matches: median {np.median(s_map[row_ix, col_ix]):.3f}, "
          f"min {s_map[row_ix, col_ix].min():.3f}. Radius-profile r: median {np.median(s_rad[row_ix, col_ix]):.3f}.")
    print(f"Agreement between independent cues: the WSS map and the geometry pick the same partner for "
          f"{np.mean(s_rad.argmax(axis=1)[row_ix] == col_ix):.0%} of rows.")
    tr_used = {}
    for d in records:
        key = (d["circ_reversed"], d["axial_reversed"])
        tr_used[key] = tr_used.get(key, 0) + 1
    print("Grid re-indexing needed (circ reversed, axial reversed): count  "
          + ", ".join(f"{k}: {v}" for k, v in sorted(tr_used.items())))
    print(f"Cohorts of matched rows: MC {sum(d['cohort'] == 'MC' for d in records)}, "
          f"other {sum(d['cohort'] == 'other' for d in records)}")
    worst = sorted(records, key=lambda d: d["map_r"])[:5]
    print("Weakest matches: " + "; ".join(f"row {d['row']}->{d['patient_id']} r={d['map_r']} ({d['status']})"
                                         for d in worst))

    good = np.flatnonzero(s_map[row_ix, col_ix] > 0.9)
    if len(good) == 0:
        good = np.arange(len(row_ix))
    scale = np.median(ring_radius(xyz_a)[col_ix].mean(axis=1) / ring_radius(xyz_t)[row_ix].mean(axis=1))
    print(f"Size check: 4D-flow radius / training radius, median {scale:.3f} (1.0 = same units).")

    # Per cohort: if the size or inlet ratio is constant within a cohort but differs between
    # cohorts, the training data has a cohort-dependent unit (e.g. voxels of different sizes).
    groups = np.array([cohort(ids[j]) for j in col_ix])
    r_t, r_a = ring_radius(xyz_t)[row_ix].mean(axis=1), ring_radius(xyz_a)[col_ix].mean(axis=1)
    inlet_ratio = np.abs(Z_t[row_ix]).max(axis=1) / np.abs(v_a[col_ix]).max(axis=1)
    q = lambda v: f"{np.median(v):7.3g} [{np.percentile(v, 25):.3g}-{np.percentile(v, 75):.3g}]"  # noqa: E731
    print(f"\nPer cohort, median [IQR]:\n{'cohort':7s} {'n':>3s} {'radius, training':>22s} "
          f"{'radius, 4D-flow (mm)':>22s} {'size ratio':>22s} {'Zreal / inlet v':>22s}")
    for g in ("MC", "other"):
        m = groups == g
        if m.any():
            print(f"{g:7s} {m.sum():3d} {q(r_t[m]):>22s} {q(r_a[m]):>22s} {q(r_a[m] / r_t[m]):>22s} "
                  f"{q(inlet_ratio[m]):>22s}")
    print("Aortic radius is ~10-15 mm. A size ratio that is tight within each cohort but differs between them "
          "means the training coordinates are in a per-cohort unit (e.g. voxels), not mm.")
    shape_r, units = inlet_units(Z_t[row_ix[good]], v_a[col_ix[good]], xyz_t[row_ix[good]])
    print(f"\nInlet curve: Zreal vs measured mean inlet velocity, waveform shape r median {np.median(shape_r):.3f} "
          f"(over {len(good)} well-matched pairs).")
    print(f"{'Zreal proportional to':26s} {'median ratio':>13s} {'ratio CV':>9s} {'r of peaks':>11s}")
    for name, (ratio, cv, r) in units.items():
        print(f"{name:26s} {ratio:13.3g} {cv:9.2f} {r:11.3f}")
    print("The row with the smaller ratio CV (and higher r) is what Zreal is. A ratio near 100 for velocity means "
          "cm/s; for flow, velocity [m/s] x area [mm^2] is mL/s, so a ratio near 1 means mL/s.")
    print(f"\nWrote {out / 'patient_matching.csv'} and patient_matching.npz")


if __name__ == "__main__":
    main()
