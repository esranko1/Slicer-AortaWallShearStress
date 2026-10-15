"""
Shared helpers for the 4D-flow data-assimilation PINN (stage1_flow_pinn.py)
and the WSS model on top of it (stage2_wss.py).

Reads the per-patient npz files written by
1.Data_preprocessing/1.5.process_aggregated_mat.py.
"""

import csv
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_DIR / "Data" / "Processed4DFlow"
DEFAULT_RESULTS_DIR = REPO_DIR / "experiments" / "11_PINN_4DFlow"

N_AXIAL, N_CIRC = 128, 32

# Surface points farther than this from every velocity-plane pixel count as
# "not covered" by the measurement (the planes stop short of the aortic ends).
COVERED_MM = 5.0

# Blood
RHO = 1060.0   # kg/m^3
MU = 0.0035    # Pa.s


def list_patients(data_dir, skip_check=False, only=None):
    """Patient IDs with a processed npz, optionally dropping manifest rows
    marked 'check' and/or restricting to a given list."""
    data_dir = Path(data_dir)
    ids = sorted(p.stem for p in (data_dir / "patients").glob("*.npz"))
    manifest = data_dir / "manifest.csv"
    if skip_check and manifest.exists():
        with open(manifest, newline='') as f:
            flagged = {r['patient_id'] for r in csv.DictReader(f) if r['status'] != 'ok'}
        ids = [i for i in ids if i not in flagged]
    if only:
        wanted = set(only)
        ids = [i for i in ids if i in wanted]
    return ids


def load_patient(data_dir, pid):
    with np.load(Path(data_dir) / "patients" / f"{pid}.npz") as z:
        return {k: z[k] for k in z.files}


def cohort_of(pid):
    """The two acquisition cohorts differ in scanner/resolution - report them
    separately. '10MC0756'-style IDs vs everything else ('RAT_4858', ...)."""
    return 'MC' if re.fullmatch(r'\d+MC\d+(_\d+)?', pid) else 'other'


def surface_frames(surface_xyz, centerline):
    """
    Outward normals and local axial / circumferential unit vectors on the
    structured 128x32 surface grid (axial-major, circumferential index wraps).
    Much cleaner than k-NN PCA normals on an unstructured cloud.
    returns normal, e_axial, e_circ: each (4096, 3)
    """
    P = surface_xyz.reshape(N_AXIAL, N_CIRC, 3).astype(np.float64)
    d_axial = np.gradient(P, axis=0)
    d_circ = np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)
    normal = np.cross(d_axial, d_circ).reshape(-1, 3)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12

    # Orient outward: away from the nearest centerline point
    _, nearest = cKDTree(centerline).query(surface_xyz)
    outward = surface_xyz - centerline[nearest]
    normal[(normal * outward).sum(axis=1) < 0] *= -1

    e_axial = d_axial.reshape(-1, 3)
    e_axial -= (e_axial * normal).sum(axis=1, keepdims=True) * normal
    e_axial /= np.linalg.norm(e_axial, axis=1, keepdims=True) + 1e-12
    e_circ = np.cross(normal, e_axial)
    return normal.astype(np.float32), e_axial.astype(np.float32), e_circ.astype(np.float32)


def local_radius(surface_xyz, centerline):
    """Distance (mm) from each surface point to the nearest centerline point."""
    d, _ = cKDTree(centerline).query(surface_xyz)
    return d.astype(np.float32)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def distance_to_planes(d):
    """Distance (mm) from each surface point to the nearest velocity-plane pixel."""
    dist, _ = cKDTree(d['vel_xyz']).query(d['surface_xyz'])
    return dist.astype(np.float32)


def interp_frames(frames, f):
    """Centerline frame (origin, e0, e1, e2) at fractional index f of the
    (4, 4, K) orientationMatrix stack, re-orthonormalised. Same convention as
    1.5.process_aggregated_mat.py: column 2 = tangent, column 3 = origin."""
    K = frames.shape[2]
    f = float(np.clip(f, 0, K - 1))
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, K - 1)
    w = f - i0
    M = (1 - w) * frames[:, :, i0] + w * frames[:, :, i1]
    e2 = M[:3, 2] / np.linalg.norm(M[:3, 2])
    e0 = M[:3, 0] - (M[:3, 0] @ e2) * e2
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(e2, e0)
    if e1 @ M[:3, 1] < 0:
        e1 = -e1
    return M[:3, 3], e0, e1, e2


def neighbour_plane_baseline(d, val_planes):
    """
    Trivial reference for held-out planes: copy the two neighbouring planes'
    velocity (nearest pixel after aligning plane centroids) and average them.
    A flow reconstruction is only useful if it beats this.
    returns (Npx, P) prediction for pixels of val_planes (NaN elsewhere)
    """
    v, pid, xyz = d['vel_through_plane'], d['vel_plane_id'], d['vel_xyz']
    pred = np.full_like(v, np.nan)
    planes = set(np.unique(pid).tolist())
    for j in val_planes:
        m = pid == j
        X = xyz[m] - xyz[m].mean(axis=0)
        guesses = []
        for jj in (j - 1, j + 1):
            if jj in planes and jj not in set(val_planes):
                mm = pid == jj
                _, ix = cKDTree(xyz[mm] - xyz[mm].mean(axis=0)).query(X)
                guesses.append(v[mm][ix])
        if guesses:
            pred[m] = np.mean(guesses, axis=0)
    return pred
