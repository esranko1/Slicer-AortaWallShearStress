"""
Are the patients aligned? CPU-only check; reads the data and one finished run's
saved predictions, changes nothing.

    python alignment_check.py                    # links to whole:loo:flow
    python alignment_check.py --run composite:loo:flow

Two separate questions:

1. Do the grid points line up around each ring? For every patient and region,
   roll the patient's true WSS around the ring (32 possible shifts) and find
   the shift that best matches the average map of the other patients (rebuilt
   from realigned patients over a few rounds, shifts relative to the most
   common one).
   Shifts near 0 mean ring index 0 sits on the same side of the wall in
   everyone. Reported in degrees (1 shift = 11.25 degrees).

2. Are the shapes oriented the same way in 3D? The conversion only subtracts
   each patient's centroid. Using the point correspondence, the best rotation
   (Procrustes, no scaling) between every pair of patients is computed; each
   patient's angle to the most typical patient (the medoid) says how far its
   orientation differs.

3. Does either explain where the model does badly? Per region, Spearman
   correlation of each patient's spatial r (from --run) with: the size of the
   ring shift, the rotation angle, the spread of the patient's true WSS in that
   region (flat maps give a noisy r), and tortuosity (centerline length /
   straight-line distance between its ends).

Writes alignment_check.csv (one row per patient) next to this script.
"""

import argparse
import csv

import numpy as np
from scipy.stats import spearmanr

from runs import REGIONS, load_run, r_spatial_per_patient

DATA_PATH = "../Data/Sampled/Rpt0_N4096.npz"
N_CIRC, N_AXIAL = 32, 128
DEG_PER_SHIFT = 360.0 / N_CIRC


def best_ring_shift(Yg, col, rounds=3):
    """
    Per patient: circular shift (grid steps, -16..15) of the region that best
    matches the average of the OTHER patients. The reference is rebuilt from the
    already-realigned patients for a few rounds, so a minority of misaligned
    patients cannot drag it, and shifts are reported relative to the most common
    shift (the reference's own rotation is arbitrary).
    """
    n = Yg.shape[0]
    maps = Yg[:, :, col]
    shifts = np.zeros(n, dtype=int)
    for _ in range(rounds):
        aligned = np.stack([np.roll(maps[i], shifts[i], axis=0) for i in range(n)])
        total = aligned.sum(axis=0)
        for i in range(n):
            ref = ((total - aligned[i]) / (n - 1)).ravel()
            rs = [np.corrcoef(np.roll(maps[i], s, axis=0).ravel(), ref)[0, 1] for s in range(N_CIRC)]
            shifts[i] = int(np.nanargmax(rs))
    mode = np.bincount(shifts % N_CIRC, minlength=N_CIRC).argmax()
    rel = (shifts - mode) % N_CIRC
    return np.where(rel >= N_CIRC // 2, rel - N_CIRC, rel)


def rotation_between(A, B):
    """Angle (degrees) of the best rotation taking centred shape A onto B."""
    u, _, vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1, 1, d]) @ vt
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def procrustes_angles(X):
    """
    Rotation angle (degrees) of each shape relative to the most typical patient
    (the medoid: smallest median rotation to everyone else). A blended average
    shape would sit between orientation groups and hide them.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    n = len(Xc)
    pair = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pair[i, j] = pair[j, i] = rotation_between(Xc[i], Xc[j])
    medoid = int(np.argmin(np.median(pair, axis=1)))
    return pair[medoid], medoid


def best_rotation(A, B):
    """Rotation matrix R (rows act on points: A @ R) taking centred shape A onto B."""
    u, _, vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt


def geometric_ring_check(X, medoid):
    """
    Does ring point 0 sit at the same anatomical spot in every patient? Uses
    geometry only (no WSS). Each shape is rotated onto the medoid shape; then,
    for every ring, the direction from the ring centre to point 0 is projected
    into the medoid's ring plane and its angle to the medoid's point-0
    direction is measured. Also checks the winding direction (point 0 -> 1
    clockwise or anticlockwise about the vessel), which flips if a patient's
    rings are numbered the other way round.
    returns angle (n, 128) in degrees (-180..180) and reversed (n,) bool
    """
    n = len(X)
    G = (X - X.mean(axis=1, keepdims=True)).reshape(n, N_CIRC, N_AXIAL, 3)
    ref = G[medoid]
    ref_c = ref.mean(axis=0)                                        # (128, 3)
    tang = np.gradient(ref_c, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    u_ref = ref[0] - ref_c
    u_ref -= (u_ref * tang).sum(axis=1, keepdims=True) * tang
    u_ref /= np.linalg.norm(u_ref, axis=1, keepdims=True)
    v_ref = np.cross(tang, u_ref)

    angle = np.zeros((n, N_AXIAL))
    winding = np.zeros(n)
    for i in range(n):
        # winding on the patient's own geometry: is point 1 anticlockwise from
        # point 0 about the patient's own centerline direction?
        own_c = G[i].mean(axis=0)
        own_t = np.gradient(own_c, axis=0)
        winding[i] = np.sign(np.mean(np.sign((np.cross(G[i][0] - own_c, G[i][1] - own_c) * own_t).sum(axis=1))))
        R = best_rotation(G[i].reshape(-1, 3), ref.reshape(-1, 3))
        g = (G[i].reshape(-1, 3) @ R).reshape(N_CIRC, N_AXIAL, 3)
        c = g.mean(axis=0)
        p0 = g[0] - c
        angle[i] = np.degrees(np.arctan2((p0 * v_ref).sum(axis=1), (p0 * u_ref).sum(axis=1)))
    majority = np.sign(winding.sum()) or 1.0
    reversed_ = winding != majority
    return angle, reversed_


def tortuosity(X):
    """Centerline (ring centroid) length / straight-line distance between its ends."""
    cent = X.reshape(len(X), N_CIRC, N_AXIAL, 3).mean(axis=1)
    length = np.linalg.norm(np.diff(cent, axis=1), axis=2).sum(axis=1)
    return length / np.linalg.norm(cent[:, -1] - cent[:, 0], axis=1)


def match_rows(Y, true):
    """Index into Y for each row of `true` (the run's patients, in its own order).
    Nearest row by distance: the run stored WSS in float32, the data is read as
    float64, so exact comparison fails."""
    Y32 = Y.astype(np.float32).astype(np.float64)
    idx = np.zeros(len(true), dtype=int)
    for k, row in enumerate(true.astype(np.float64)):
        dist = np.abs(Y32 - row).max(axis=1)
        idx[k] = int(np.argmin(dist))
        if dist[idx[k]] > 1e-4:
            raise SystemExit("Could not match the run's patients to the data file; is it the same data?")
    if len(set(idx)) != len(idx):
        raise SystemExit("Two of the run's patients matched the same data row; cannot pair them.")
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--run", default="whole:loo:flow", help="run whose spatial r is linked to the alignment")
    args = ap.parse_args()

    d = np.load(args.data)
    X = d["a"][:, :, :3].astype(np.float64)
    Y = d["b"][:, :, 0].astype(np.float64)
    n = len(X)
    Yg = Y.reshape(n, N_CIRC, N_AXIAL)

    angles, medoid = procrustes_angles(X)
    tort = tortuosity(X)
    print(f"{n} patients\n")
    print("3D orientation (rotation to the most typical patient):")
    print(f"  median {np.median(angles):.1f} deg, IQR {np.percentile(angles, 25):.1f}-{np.percentile(angles, 75):.1f}, "
          f"max {angles.max():.1f}; under 15 deg: {np.mean(angles < 15) * 100:.0f}% of patients")
    edges = [0, 15, 30, 45, 60, 90, 180]
    counts = np.histogram(angles, bins=edges)[0]
    print("  patients per angle band: " + ", ".join(f"{lo}-{hi} deg: {c}" for lo, hi, c in zip(edges, edges[1:], counts)))

    true, pred, _, _ = load_run(args.run)
    order = match_rows(Y, true.reshape(len(true), -1))   # run row k -> data row order[k]
    spatial = {name: np.full(n, np.nan) for name in REGIONS}
    for name, col in REGIONS.items():
        spatial[name][order] = r_spatial_per_patient(true, pred, col)

    geo_angle, reversed_ = geometric_ring_check(X, medoid)
    rows = {"patient": np.arange(n), "rotation_deg": angles, "tortuosity": tort,
            "ring_numbering_reversed": reversed_.astype(int)}
    for name, col in REGIONS.items():
        rows[f"{name}_point0_angle_deg"] = np.abs(geo_angle[:, col]).mean(axis=1)
    print("\nGrid alignment from geometry (direction of ring point 0, after rotating shapes onto the typical one):")
    print(f"  ring numbering reversed (other winding direction): {reversed_.sum()} patients")
    print(f"{'region':6s} {'median |angle|':>15s} {'within 11 deg':>14s} {'beyond 45 deg':>14s}")
    for name, col in REGIONS.items():
        a = np.abs(geo_angle[:, col]).mean(axis=1)
        print(f"{name:6s} {np.median(a):11.1f} deg {np.mean(a <= DEG_PER_SHIFT) * 100:13.0f}% "
              f"{np.mean(a > 45) * 100:13.0f}%")
    print("(If point 0 sits at the same anatomical spot, these angles are small. Large angles here mean the "
          "grid really is misaligned; small ones mean the WSS-based shifts below are mostly noise from flat maps.)")

    print("\nRing alignment (best shift of each patient's WSS onto the others' average map):")
    print(f"{'region':6s} {'median |shift|':>15s} {'within +-1 step':>16s} {'|shift| >= 4 (45 deg)':>22s}")
    shifts_by_region = {}
    for name, col in REGIONS.items():
        s = best_ring_shift(Yg, col)
        shifts_by_region[name] = s
        a = np.abs(s)
        print(f"{name:6s} {np.median(a) * DEG_PER_SHIFT:13.1f} deg {np.mean(a <= 1) * 100:15.0f}% "
              f"{np.mean(a >= 4) * 100:21.0f}%")
        rows[f"{name}_shift_deg"] = s * DEG_PER_SHIFT
        rows[f"{name}_wss_spread"] = Yg[:, :, col].reshape(n, -1).std(axis=1)
        rows[f"{name}_spatial_r"] = spatial[name]

    print(f"\nWhat goes with low spatial r ({args.run})? Spearman rho across patients:")
    print(f"{'region':6s} {'|ring shift|':>13s} {'rotation':>9s} {'WSS spread':>11s} {'tortuosity':>11s}")
    for name in REGIONS:
        r = rows[f"{name}_spatial_r"]
        ok = np.isfinite(r)
        c = lambda v: spearmanr(v[ok], r[ok])[0]  # noqa: E731
        print(f"{name:6s} {c(np.abs(shifts_by_region[name]).astype(float)):13.2f} {c(angles):9.2f} "
              f"{c(rows[f'{name}_wss_spread']):11.2f} {c(tort):11.2f}")
    print("\nNegative rho for shift/rotation = misaligned patients do worse (alignment problem).\n"
          "Positive rho for WSS spread = flat maps score worse (partly a metric artefact).")

    with open("alignment_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(rows)
        w.writerow(keys)
        for i in range(n):
            w.writerow([f"{rows[k][i]:.4f}" if isinstance(rows[k][i], (float, np.floating)) else rows[k][i]
                        for k in keys])
    print("Per-patient values: alignment_check.csv")


if __name__ == "__main__":
    main()
