"""
Geometric features for the region-specialist PointNet, computed from the
corrected point clouds in Data/Sampled/Rpt0_N4096.npz (Xreal, via
1.Data_preprocessing/1.3.convert_mat_to_npz.py) - never from the raw .mat, so
the old X / Xreal mix-up cannot come back.

Replaces preprocess_geometric_features.py, which (a) read the uncorrected 'X',
(b) measured "distance to inlet" from point 587, which sits in the descending
aorta, and (c) used k-NN PCA normals with an arbitrary sign per point.

The 4096 points are a structured 32 (circumferential) x 128 (longitudinal) grid,
flattened circumferential-major (index = c * 128 + a), the same layout the
models' reshape(32, 128) and the region slices (pasc = columns 0:36, ...) use.
The script checks this layout before computing anything.

Per point (7 channels, all O(1)):
    nx, ny, nz   outward unit normal from the grid (sign fixed: away from the
                 centroid of the point's own ring)
    s            position along the aorta, 0 at the first ring, 1 at the last,
                 by arc length of the ring-centroid centerline
    radius       distance to the ring centroid / cohort median radius
    curvature    centerline curvature x cohort median radius (dimensionless)
    side         -normal . (centerline curvature direction): +1 on the outer
                 wall of a bend, -1 on the inner wall, ~0 on straight segments

Usage (from this folder):
    python compute_geo_features.py
Writes ../Data/Sampled/geo_features_N4096.npz with
    features (100, 4096, 7), names (7,), c (patient ids, same order as Rpt0)
"""

import argparse
import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO, format='%(message)s')

N_CIRC, N_AXIAL = 32, 128
NAMES = np.array(['nx', 'ny', 'nz', 's', 'radius', 'curvature', 'side'])
CENTERLINE_SMOOTHING = 3.0  # rings; the curvature needs a second derivative


def to_grid(points):
    """(4096, 3) circumferential-major -> (32, 128, 3): [c, a]."""
    return points.reshape(N_CIRC, N_AXIAL, 3)


def ring_roundness(grid):
    """Mean (std / mean) of point distances to their ring centroid, over rings.
    Small for a real ring (points around one cross-section)."""
    centroids = grid.mean(axis=0, keepdims=True)
    d = np.linalg.norm(grid - centroids, axis=2)
    return float(np.mean(d.std(axis=0) / (d.mean(axis=0) + 1e-12)))


def check_layout(X):
    """Compare the assumed layout with the transposed one on a few patients; stop
    if the transposed one looks more like rings."""
    assumed = np.median([ring_roundness(to_grid(x)) for x in X[:10]])
    other = np.median([ring_roundness(x.reshape(N_AXIAL, N_CIRC, 3).transpose(1, 0, 2)) for x in X[:10]])
    logging.info(f"Grid layout check (ring roundness, lower = rings): "
                 f"circumferential-major {assumed:.3f}, longitudinal-major {other:.3f}")
    if not assumed < other:
        raise SystemExit("The points do not look circumferential-major (index = c * 128 + a). "
                         "The region slices and reshape(32, 128) in main.py assume that layout; "
                         "check the data before computing features.")


def patient_features(points):
    """(4096, 3) -> (4096, 7) raw features (radius and curvature in mm and 1/mm)."""
    grid = to_grid(points.astype(np.float64))            # [c, a, 3]
    centerline = grid.mean(axis=0)                       # (128, 3) ring centroids

    # Outward normals from the grid's two tangent directions
    d_axial = np.gradient(grid, axis=1)
    d_circ = np.roll(grid, -1, axis=0) - np.roll(grid, 1, axis=0)
    normal = np.cross(d_axial, d_circ)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-12
    outward = grid - centerline[None]
    normal[(normal * outward).sum(axis=2) < 0] *= -1

    # Arc length along the centerline
    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    s = arclen / (arclen[-1] + 1e-12)

    radius = np.linalg.norm(outward, axis=2)              # (32, 128) mm

    # Curvature of the smoothed centerline: kappa = |dT/ds|, direction N = dT/ds / kappa
    smooth = gaussian_filter1d(centerline, CENTERLINE_SMOOTHING, axis=0, mode='nearest')
    d1 = np.gradient(smooth, arclen, axis=0)
    tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-12)
    dT = np.gradient(tangent, arclen, axis=0)
    curvature = np.linalg.norm(dT, axis=1)                # (128,) 1/mm
    curve_dir = dT / (curvature[:, None] + 1e-12)
    side = -(normal * curve_dir[None]).sum(axis=2)        # +1 outer wall, -1 inner

    feats = np.concatenate([
        normal,
        np.broadcast_to(s[None, :, None], (N_CIRC, N_AXIAL, 1)),
        radius[..., None],
        np.broadcast_to(curvature[None, :, None], (N_CIRC, N_AXIAL, 1)),
        side[..., None],
    ], axis=2)
    return feats.reshape(N_CIRC * N_AXIAL, len(NAMES))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='../Data/Sampled/Rpt0_N4096.npz')
    ap.add_argument('--output', default='../Data/Sampled/geo_features_N4096.npz')
    args = ap.parse_args()

    with np.load(args.data) as z:
        X = z['a'][:, :, :3].astype(np.float64)
        ids = z['c']
    logging.info(f"Loaded {X.shape[0]} patients from {args.data}")
    check_layout(X)

    feats = np.stack([patient_features(x) for x in X]).astype(np.float64)

    # One cohort-wide scale for radius and curvature, so cross-patient size
    # differences are kept but the channels are O(1)
    r_scale = float(np.median(feats[:, :, 4]))
    feats[:, :, 4] /= r_scale
    feats[:, :, 5] *= r_scale
    logging.info(f"Median radius {r_scale:.3f} (input units) used to scale radius and curvature")

    for i, name in enumerate(NAMES):
        v = feats[:, :, i]
        logging.info(f"  {name:10s} min {v.min():+.3f}  median {np.median(v):+.3f}  max {v.max():+.3f}")
    np.savez_compressed(args.output, features=feats.astype(np.float32), names=NAMES, c=ids,
                        radius_scale=r_scale)
    logging.info(f"Saved {feats.shape} to {args.output}")


if __name__ == '__main__':
    main()
