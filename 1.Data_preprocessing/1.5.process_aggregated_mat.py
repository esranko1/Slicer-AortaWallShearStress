"""
Processes the per-patient 4D-flow .mat files in AggregatedData into a
consistent per-patient .npz format for the data-assimilation PINN
(10.PINN_TwoStage), plus a legacy a/b/c/d file matching Rpt0_N4096.npz.

Each .mat holds one struct (normally named `data_select`, but the variable
name is NOT relied on - any top-level struct with WSS / Vel / Ao_Seg fields
is accepted) containing:
    WSS.WSS_Norm3D        (128, 32, 50)     |WSS| on the surface grid, 50 frames
    WSS.WSS_Vect3D        (3, 128, 32, 50)  WSS vector
    Vel{36}.Velo2D        (ny, nx, P)       through-plane velocity on 36 planes,
                                            P cardiac phases (varies), 0 outside lumen
    Ao_Seg.pts            (3, 128, 32)      surface grid (= 4096 points)
    Ao_Seg.mask           (Nx, Ny, Nz)      binary lumen mask
    Ao_Seg.beas           centerline (s), 64 local frames (orientationMatrix),
                          cross-section radius rho(theta)
    Ao_Seg.lAortaMM       centerline length in mm

Filenames are inconsistent, so files are found recursively by extension and
identified by content. The patient ID is taken from the filename (a token like
"10MC0756" if one is present, otherwise the cleaned-up file stem).

Coordinate conventions (verified on 10MC0756.mat):
  - pts / axis / beas.s are in MATLAB 1-based voxel coordinates of `mask`
    (surface points sit ~0.5 voxel from the nearest mask voxel center).
  - orientationMatrix[:3, 2, k] is the centerline tangent (plane normal),
    columns 0 and 1 span the cross-section, column 3 is the origin.
  - Voxel -> mm uses ONE isotropic scale = lAortaMM / centerline arc length
    (assumes isotropic voxels - check this against the scan header if possible).

Velocity planes - what the .mat does NOT store, and what this script estimates:
  The plane positions along the centerline, the pixel size and the in-plane
  orientation of Velo2D are not saved. They are estimated per patient:
    1. Plane j -> fractional centerline frame f_j = a + j*(b-a)/35, with (a, b)
       chosen so lumen pixel area best correlates with the beas cross-section
       area pi*mean(rho^2).
    2. Pixel area from the mean (voxel area / pixel area) ratio.
    3. In-plane orientation (one of 8 flips/rotations of the image axes) and
       row/col pixel aspect ratio: the pair that best overlaps the lumen mask
       with the rho(theta) cross-section, shared by all planes of a patient.
  The fit quality (area correlation, mean IoU) is written to the manifest -
  review low values before training. If the real plane geometry becomes
  available, replace estimate_plane_geometry().

Units / timing assumptions (stored in each npz as `assumptions`):
  - Velo2D is in cm/s -> stored in m/s.  WSS assumed Pa.
  - Velocity phases and WSS frames both span exactly one cardiac cycle:
    t_vel = k/P, t_wss = k/T (P velocity phases, T WSS frames, read per file).

Usage (on the Windows machine):
    python 1.5.process_aggregated_mat.py
    python 1.5.process_aggregated_mat.py --input "C:\\path\\to\\AggregatedData" --output ..\\Data\\Processed4DFlow
"""

import argparse
import csv
import hashlib
import logging
import traceback
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format='%(message)s')

DEFAULT_INPUT = r"C:\Users\esranko1\Desktop\AggregatedData"
DEFAULT_OUTPUT = str(Path(__file__).resolve().parent.parent / "Data" / "Processed4DFlow")

N_AXIAL, N_CIRC = 128, 32
N_SURFACE = N_AXIAL * N_CIRC
N_PLANES = 36
# The number of velocity phases and WSS frames varies between patients, so both
# are read from each file rather than fixed here.
VEL_TO_MS = 0.01  # cm/s -> m/s

REQUIRED_FIELDS = ('WSS', 'Vel', 'Ao_Seg')
PATIENT_ID_RE = re.compile(r'\d{1,3}[A-Za-z]{1,4}\d{3,6}')

# Candidate row/col pixel-size ratios (log-spaced, includes 1.0)
PIXEL_ASPECTS = np.exp(np.linspace(np.log(0.4), np.log(2.5), 21))

# The 8 flips/rotations of a 2D image's (row, col) axes, as (swap, flip_row, flip_col)
DIHEDRAL = [(s, fr, fc) for s in (False, True) for fr in (False, True) for fc in (False, True)]


# ============================================================================
# Discovery / loading
# ============================================================================

def find_mat_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() == '.mat')


def patient_id_from_path(path):
    match = PATIENT_ID_RE.search(path.stem)
    if match:
        return match.group(0).upper()
    return re.sub(r'[^A-Za-z0-9]+', '_', path.stem).strip('_')


def file_hash(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def is_mat_struct(obj):
    return hasattr(obj, '_fieldnames')


def find_patient_struct(mat):
    """Return the struct holding WSS/Vel/Ao_Seg, whatever its variable name."""
    candidates = [v for k, v in mat.items() if not k.startswith('__')]
    for v in candidates:
        if is_mat_struct(v) and all(f in v._fieldnames for f in REQUIRED_FIELDS):
            return v
    # One level of nesting (e.g. the struct wrapped in another struct)
    for v in candidates:
        if is_mat_struct(v):
            for f in v._fieldnames:
                sub = getattr(v, f)
                if is_mat_struct(sub) and all(g in sub._fieldnames for g in REQUIRED_FIELDS):
                    return sub
    return None


def load_patient_struct(path):
    try:
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        raise ValueError("MATLAB v7.3 (HDF5) file - re-save in MATLAB with save(..., '-v7') "
                         "or add an h5py reader")
    struct = find_patient_struct(mat)
    if struct is None:
        names = [k for k in mat if not k.startswith('__')]
        raise ValueError(f"no struct with fields {REQUIRED_FIELDS} (variables: {names})")
    return struct


# ============================================================================
# Geometry helpers
# ============================================================================

def centerline_scale(seg):
    """mm per voxel, from the stored centerline length (isotropic assumption)."""
    s = np.asarray(seg.beas.s, dtype=np.float64).T  # (64, 3)
    arclen = np.linalg.norm(np.diff(s, axis=0), axis=1).sum()
    return float(seg.lAortaMM) / arclen


def interp_frames(frames, f):
    """Linearly interpolate (4, 4, K) centerline frames at fractional index f,
    re-orthonormalizing the axes."""
    K = frames.shape[2]
    f = float(np.clip(f, 0, K - 1))
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, K - 1)
    w = f - i0
    M = (1 - w) * frames[:, :, i0] + w * frames[:, :, i1]
    origin = M[:3, 3]
    e2 = M[:3, 2] / np.linalg.norm(M[:3, 2])
    e0 = M[:3, 0] - (M[:3, 0] @ e2) * e2
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(e2, e0)
    if e1 @ M[:3, 1] < 0:  # keep the stored handedness
        e1 = -e1
    return origin, e0, e1, e2


def interp_rho(rho, f):
    K = rho.shape[0]
    f = float(np.clip(f, 0, K - 1))
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, K - 1)
    w = f - i0
    return (1 - w) * rho[i0] + w * rho[i1]


def plane_pixel_offsets(shape, lumen, transform, row_size, col_size):
    """In-plane (u, v) offsets in voxels of every grid pixel from the lumen
    centroid, given the pixel size along image rows / columns (voxels), after
    applying one of the DIHEDRAL axis transforms."""
    swap, flip_r, flip_c = transform
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    r0, c0 = rr[lumen].mean(), cc[lumen].mean()
    dr, dc = (rr - r0) * row_size, (cc - c0) * col_size
    if flip_r:
        dr = -dr
    if flip_c:
        dc = -dc
    return (dc, dr) if swap else (dr, dc)


def rasterize_cross_section(u, v, rho_k, theta):
    """Mask of pixels inside the rho(theta) cross-section; u, v in voxels."""
    r = np.hypot(u, v)
    ang = np.mod(np.arctan2(v, u), 2 * np.pi)
    th = np.concatenate([theta, [2 * np.pi]])
    rh = np.concatenate([rho_k, rho_k[:1]])
    return r <= np.interp(ang, th, rh)


def estimate_plane_geometry(seg, lumens):
    """
    Estimate, for each of the 36 velocity planes, its fractional centerline
    frame index, plus the shared pixel size (voxels/pixel) and in-plane axis
    transform. See the module docstring for the method.
    """
    rho = np.asarray(seg.beas.rho, dtype=np.float64)
    theta = np.asarray(seg.beas.theta, dtype=np.float64)
    K = rho.shape[0]
    n = len(lumens)
    npx = np.array([l.sum() for l in lumens], dtype=np.float64)
    section_area = np.pi * (rho ** 2).mean(axis=1)  # voxel^2

    best = (-np.inf, 0.0, K - 1.0)
    for a in np.arange(0, K / 2, 0.5):
        for b in np.arange(a + n / 2, K - 0.5 + 1e-9, 0.5):
            f = np.linspace(a, b, n)
            areas = np.interp(f, np.arange(K), section_area)
            c = np.corrcoef(npx, areas)[0, 1]
            if np.isfinite(c) and c > best[0]:
                best = (c, a, b)
    area_corr, a, b = best
    frame_pos = np.linspace(a, b, n)
    areas = np.interp(frame_pos, np.arange(K), section_area)
    pixel_size = float(np.sqrt(np.mean(areas / npx)))  # voxels per pixel (geometric mean of row/col)

    # Pixels need not be square (the non-10MC cohort has row spacing ~0.6x the
    # column spacing), so fit the row/col aspect jointly with the transform,
    # keeping the pixel area fixed at pixel_size**2.
    sections = [interp_rho(rho, f) for f in frame_pos]
    results = []  # (mean IoU, aspect, transform)
    for transform in DIHEDRAL:
        for aspect in PIXEL_ASPECTS:
            row_size, col_size = pixel_size * np.sqrt(aspect), pixel_size / np.sqrt(aspect)
            ious = []
            for lumen, rho_k in zip(lumens, sections):
                u, v = plane_pixel_offsets(lumen.shape, lumen, transform, row_size, col_size)
                section = rasterize_cross_section(u, v, rho_k, theta)
                ious.append((section & lumen).sum() / max((section | lumen).sum(), 1))
            results.append((float(np.mean(ious)), float(aspect), transform))
    results.sort(key=lambda r: -r[0])
    mean_iou, aspect, transform = results[0]
    runner_up = next(r[0] for r in results if r[2] != transform)

    return {
        'frame_pos': frame_pos,
        'pixel_size_vox': pixel_size,
        'row_size_vox': pixel_size * np.sqrt(aspect),
        'col_size_vox': pixel_size / np.sqrt(aspect),
        'pixel_aspect': aspect,
        'transform': transform,
        'area_corr': float(area_corr),
        'mean_iou': mean_iou,
        'iou_margin': mean_iou - runner_up,
    }


# ============================================================================
# Per-patient processing
# ============================================================================

def process_patient(struct):
    seg, wss, vel = struct.Ao_Seg, struct.WSS, struct.Vel
    mm = centerline_scale(seg)

    # --- surface + WSS (fixed 128x32 grid -> 4096 points, axial-major) -----
    pts = np.asarray(seg.pts, dtype=np.float64)
    if pts.shape != (3, N_AXIAL, N_CIRC):
        raise ValueError(f"Ao_Seg.pts has shape {pts.shape}, expected (3, {N_AXIAL}, {N_CIRC})")
    surface_xyz = pts.reshape(3, -1).T * mm  # (4096, 3)
    wss_mag = np.asarray(wss.WSS_Norm3D, dtype=np.float32).reshape(N_SURFACE, -1)  # (4096, T)
    wss_vec = np.asarray(wss.WSS_Vect3D, dtype=np.float32).reshape(3, N_SURFACE, -1).transpose(1, 2, 0)
    surface_grid_idx = np.stack(np.meshgrid(np.arange(N_AXIAL), np.arange(N_CIRC), indexing='ij'),
                                axis=-1).reshape(-1, 2)

    # --- centerline + interior lumen points ---------------------------------
    frames = np.asarray(seg.beas.orientationMatrix, dtype=np.float64)  # (4, 4, 64)
    centerline = np.asarray(seg.beas.s, dtype=np.float64).T * mm
    mask = np.asarray(seg.mask).astype(bool)
    lumen_xyz = (np.argwhere(mask) + 1.0) * mm  # 1-based voxel centres, like pts

    # --- velocity planes ----------------------------------------------------
    planes = [np.asarray(c.Velo2D, dtype=np.float64) for c in np.atleast_1d(vel)]
    planes = [p[:, :, None] if p.ndim == 2 else p for p in planes]
    phase_counts = sorted({p.shape[2] for p in planes})
    if len(phase_counts) != 1:
        raise ValueError(f"velocity planes have different numbers of phases: {phase_counts}")
    n_vel_phases = phase_counts[0]
    if len(planes) != N_PLANES:
        logging.warning(f"    {len(planes)} velocity planes (expected {N_PLANES})")
    lumens = [np.isfinite(p).all(axis=2) & (p != 0).any(axis=2) for p in planes]
    geo = estimate_plane_geometry(seg, lumens)

    vel_xyz, vel_normal, vel_values, vel_plane_id = [], [], [], []
    plane_origin, plane_normal = [], []
    for j, (plane, lumen) in enumerate(zip(planes, lumens)):
        origin, e0, e1, e2 = interp_frames(frames, geo['frame_pos'][j])
        u, v = plane_pixel_offsets(lumen.shape, lumen, geo['transform'],
                                   geo['row_size_vox'], geo['col_size_vox'])
        u, v = u[lumen], v[lumen]
        xyz = origin[None, :] + u[:, None] * e0[None, :] + v[:, None] * e1[None, :]
        vel_xyz.append(xyz * mm)
        vel_normal.append(np.tile(e2, (len(u), 1)))
        vel_values.append(plane[lumen] * VEL_TO_MS)  # (n_px, P)
        vel_plane_id.append(np.full(len(u), j, dtype=np.int16))
        plane_origin.append(origin * mm)
        plane_normal.append(e2)

    vel_values = np.concatenate(vel_values).astype(np.float32)
    vel_plane_id = np.concatenate(vel_plane_id)

    # Positive through-plane velocity should be forward flow along the
    # centerline tangent; record whether mean systolic flow agrees.
    plane0 = vel_values[vel_plane_id == 0]
    peak_phase = int(np.argmax(np.abs(plane0.mean(axis=0))))
    forward_sign_ok = bool(plane0[:, peak_phase].mean() > 0)

    # Inlet waveform (mean through-plane velocity on plane 0), resampled to a
    # fixed 50 samples for the legacy `Z` input (phase counts vary by patient).
    t_vel = np.arange(n_vel_phases) / n_vel_phases
    t_wss = np.arange(wss_mag.shape[1]) / wss_mag.shape[1]
    inlet_mean = plane0.mean(axis=0)
    inlet_waveform_50 = np.interp(np.arange(50) / 50, t_vel, inlet_mean, period=1.0)
    pixel_area_m2 = (geo['pixel_size_vox'] * mm * 1e-3) ** 2
    inlet_flow_rate = plane0.sum(axis=0) * pixel_area_m2  # m^3/s, (P,)

    out = dict(
        surface_xyz=surface_xyz.astype(np.float32),
        surface_grid_idx=surface_grid_idx.astype(np.int16),
        wss_mag=wss_mag,
        wss_vec=wss_vec,
        t_wss=t_wss.astype(np.float32),
        centerline=centerline.astype(np.float32),
        centerline_frames=frames.astype(np.float32),
        beas_rho_mm=(np.asarray(seg.beas.rho) * mm).astype(np.float32),
        beas_theta=np.asarray(seg.beas.theta, dtype=np.float32),
        lumen_xyz=lumen_xyz.astype(np.float32),
        mask=mask,
        vel_xyz=np.concatenate(vel_xyz).astype(np.float32),
        vel_normal=np.concatenate(vel_normal).astype(np.float32),
        vel_through_plane=vel_values,
        vel_plane_id=vel_plane_id,
        t_vel=t_vel.astype(np.float32),
        plane_origin=np.array(plane_origin, dtype=np.float32),
        plane_normal=np.array(plane_normal, dtype=np.float32),
        plane_frame_pos=geo['frame_pos'].astype(np.float32),
        inlet_waveform_50=inlet_waveform_50.astype(np.float32),
        inlet_flow_rate=inlet_flow_rate.astype(np.float32),
        mm_per_voxel=np.float32(mm),
        landmarks_xyz=(np.asarray(seg.landmarks, dtype=np.float64) * mm).astype(np.float32),
        # typeL only exists in the 10MC-14MC cohort; the other cohort has
        # untyped landmarks (and a different count), so -1 marks "unknown".
        landmark_types=(np.asarray(seg.typeL) if hasattr(seg, 'typeL')
                        else np.full(len(np.atleast_2d(seg.landmarks)), -1)).astype(np.int16),
        assumptions=np.array([
            "units: xyz mm, velocity m/s (Velo2D assumed cm/s), WSS Pa",
            "isotropic voxels: mm_per_voxel = lAortaMM / centerline arc length",
            f"time: vel phase k -> t=k/{n_vel_phases}, WSS frame k -> t=k/{wss_mag.shape[1]}, one cardiac cycle",
            "velocity plane positions/pixel size/in-plane orientation ESTIMATED (see manifest)",
        ]),
    )
    qc = dict(
        n_lumen_voxels=int(mask.sum()),
        n_vel_pixels=len(vel_values),
        n_planes=len(planes),
        n_vel_phases=n_vel_phases,
        n_wss_frames=wss_mag.shape[1],
        mm_per_voxel=round(mm, 4),
        pixel_size_mm=round(geo['pixel_size_vox'] * mm, 4),
        pixel_row_mm=round(geo['row_size_vox'] * mm, 4),
        pixel_col_mm=round(geo['col_size_vox'] * mm, 4),
        plane_frame_start=round(float(geo['frame_pos'][0]), 2),
        plane_frame_end=round(float(geo['frame_pos'][-1]), 2),
        area_corr=round(geo['area_corr'], 3),
        mean_iou=round(geo['mean_iou'], 3),
        iou_margin=round(geo['iou_margin'], 3),
        inplane_transform=str(geo['transform']),
        forward_sign_ok=forward_sign_ok,
        peak_vel_ms=round(float(np.abs(vel_values).max()), 3),
        peak_wss_frame=int(wss_mag.mean(axis=0).argmax()),
        max_wss=round(float(wss_mag.max()), 3),
    )
    return out, qc


def surface_inside_check(out):
    """Fraction of velocity pixels lying within 1.5 voxels of the lumen mask -
    a sanity check on the estimated plane placement."""
    tree = cKDTree(out['lumen_xyz'])
    d, _ = tree.query(out['vel_xyz'])
    return float((d <= 1.5 * float(out['mm_per_voxel'])).mean())


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=DEFAULT_INPUT, help='folder with the .mat files (searched recursively)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='output folder')
    parser.add_argument('--overwrite', action='store_true', help='reprocess patients that already have an npz')
    args = parser.parse_args()

    out_dir = Path(args.output)
    patient_dir = out_dir / 'patients'
    patient_dir.mkdir(parents=True, exist_ok=True)

    files = find_mat_files(args.input)
    logging.info(f"Found {len(files)} .mat files under {args.input}")

    manifest, seen_ids, seen_hashes = [], {}, {}
    legacy = {'ids': [], 'xyz': [], 'z': [], 'swss': [], 'tawss': []}

    for path in files:
        row = {'file': str(path), 'patient_id': '', 'status': '', 'note': ''}
        try:
            digest = file_hash(path)
            if digest in seen_hashes:
                row.update(status='skipped', note=f'identical content to {seen_hashes[digest]}')
                manifest.append(row)
                logging.info(f"SKIP  {path.name}: duplicate of {seen_hashes[digest]}")
                continue
            seen_hashes[digest] = path.name

            pid = patient_id_from_path(path)
            if pid in seen_ids:
                n = 2
                while f"{pid}_{n}" in seen_ids:
                    n += 1
                row['note'] = f'ID collision with {seen_ids[pid]}; renamed'
                pid = f"{pid}_{n}"
            seen_ids[pid] = path.name
            row['patient_id'] = pid

            npz_path = patient_dir / f"{pid}.npz"
            struct = load_patient_struct(path)
            out, qc = process_patient(struct)
            qc['vel_px_in_lumen'] = round(surface_inside_check(out), 3)

            if args.overwrite or not npz_path.exists():
                np.savez_compressed(npz_path, patient_id=pid, source_file=path.name, **out)

            legacy['ids'].append(pid)
            legacy['xyz'].append(out['surface_xyz'])
            legacy['z'].append(out['inlet_waveform_50'])
            legacy['swss'].append(out['wss_mag'][:, qc['peak_wss_frame']])
            legacy['tawss'].append(out['wss_mag'].mean(axis=1))

            flags = []
            if qc['area_corr'] < 0.6:
                flags.append('low area_corr')
            if qc['mean_iou'] < 0.6:
                flags.append('low IoU')
            if qc['vel_px_in_lumen'] < 0.8:
                flags.append('planes outside lumen')
            if not qc['forward_sign_ok']:
                flags.append('inlet flow negative')
            row.update(status='ok' if not flags else 'check', **qc)
            row['note'] = '; '.join(filter(None, [row['note']] + flags))
            logging.info(f"{row['status'].upper():5s} {path.name} -> {pid}  "
                         f"area_corr={qc['area_corr']} IoU={qc['mean_iou']} "
                         f"in_lumen={qc['vel_px_in_lumen']} {row['note']}")
        except Exception as e:  # keep going; every failure ends up in the manifest
            where = traceback.extract_tb(e.__traceback__)[-1]
            row.update(status='failed', note=f'{type(e).__name__}: {e} (line {where.lineno}, {where.name})')
            logging.info(f"FAIL  {path.name}: {row['note']}")
        manifest.append(row)

    fieldnames = []
    for row in manifest:
        fieldnames += [k for k in row if k not in fieldnames]
    with open(out_dir / 'manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    # Legacy a/b/c/d file in the Rpt0_N4096.npz layout, for the existing models.
    # NOTE: b = WSS at each patient's peak-systolic frame, which may not be the
    # same phase the old SWSSreal was taken at.
    if legacy['ids']:
        X = np.stack(legacy['xyz'])
        X = X - X.mean(axis=1, keepdims=True)
        Z = np.stack(legacy['z'])
        a = np.concatenate([X, np.repeat(Z[:, None, :], N_SURFACE, axis=1)], axis=-1)
        np.savez_compressed(
            out_dir / f"legacy_Rpt0_N{N_SURFACE}.npz",
            a=a.astype(np.float32),
            b=np.stack(legacy['swss'])[:, :, None].astype(np.float32),
            c=np.array(legacy['ids']),
            d=np.stack(legacy['tawss'])[:, :, None].astype(np.float32),
        )

    counts = {s: sum(r['status'] == s for r in manifest) for s in ('ok', 'check', 'skipped', 'failed')}
    logging.info(f"\nDone: {counts}. Manifest: {out_dir / 'manifest.csv'}")


if __name__ == '__main__':
    main()
