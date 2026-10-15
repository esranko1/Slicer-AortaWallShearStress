"""
Checks which in-plane orientation of the Velo2D images is right, across all
patients.

1.5.process_aggregated_mat.py has to guess how each velocity image is
rotated/flipped relative to the centerline frame (one of 8 flips/rotations).
It picks the one whose lumen shape best matches the segmentation, but the
cross-sections are nearly circular, so the 8 options score almost the same
and the choice is close to a coin flip.

Here every orientation is scored a second way: with the planes placed in 3D
using that orientation, how well does the measured speed near the wall
(< 2 mm) correlate with the labelled WSS at the nearest wall point, at peak
systole? The right orientation should put fast flow next to high WSS; a wrong
one puts it next to the wrong part of the wall.

If one orientation wins for (nearly) every patient, that is the convention of
the code that extracted the planes, and it should be fixed in preprocessing
(a single global setting chosen from the data, not tuned per patient).

Writes plane_orientation_check.csv (one row per patient, r for each of the 8
orientations) next to the processed data, and prints a tally per cohort.

Usage:
    python 1.6.check_plane_orientation.py --input "C:\\Users\\esranko1\\Desktop\\AggregatedData"
"""

import argparse
import csv
import importlib.util
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format='%(message)s')

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("proc", HERE / "1.5.process_aggregated_mat.py")
proc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proc)

NEAR_WALL_MM = 2.0


def cohort_of(pid):
    return 'MC' if re.fullmatch(r'\d+MC\d+(_\d+)?', pid) else 'other'


def score_orientations(struct):
    """r(near-wall measured speed, label WSS at nearest wall point) at peak
    systole, for each of the 8 orientations; plus the one 1.5 chose."""
    seg = struct.Ao_Seg
    mm = proc.centerline_scale(seg)
    frames = np.asarray(seg.beas.orientationMatrix, dtype=np.float64)
    planes = [np.asarray(c.Velo2D, dtype=np.float64) for c in np.atleast_1d(struct.Vel)]
    planes = [p[:, :, None] if p.ndim == 2 else p for p in planes]
    lumens = [np.isfinite(p).all(axis=2) & (p != 0).any(axis=2) for p in planes]
    geo = proc.estimate_plane_geometry(seg, lumens)

    surface = np.asarray(seg.pts, dtype=np.float64).reshape(3, -1).T * mm
    wss = np.asarray(struct.WSS.WSS_Norm3D, dtype=np.float64).reshape(surface.shape[0], -1)
    n_phases, n_frames = planes[0].shape[2], wss.shape[1]
    speed = np.mean([np.abs(p[l]).mean(axis=0) for p, l in zip(planes, lumens)], axis=0)
    peak = int(speed.argmax())
    frame = int(round(peak / n_phases * n_frames)) % n_frames
    tree = cKDTree(surface)

    scores = {}
    for transform in proc.DIHEDRAL:
        xyz, vel = [], []
        for j, (p, lumen) in enumerate(zip(planes, lumens)):
            origin, e0, e1, _ = proc.interp_frames(frames, geo['frame_pos'][j])
            u, v = proc.plane_pixel_offsets(lumen.shape, lumen, transform, geo['row_size_vox'], geo['col_size_vox'])
            xyz.append((origin + u[lumen][:, None] * e0 + v[lumen][:, None] * e1) * mm)
            vel.append(np.abs(p[lumen][:, peak]))
        dist, nearest = tree.query(np.concatenate(xyz))
        near = dist < NEAR_WALL_MM
        scores[transform] = float(np.corrcoef(np.concatenate(vel)[near], wss[nearest[near], frame])[0, 1])
    return scores, geo['transform']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=proc.DEFAULT_INPUT, help='folder with the .mat files (searched recursively)')
    ap.add_argument('--output', default=proc.DEFAULT_OUTPUT, help='folder to write the CSV to')
    args = ap.parse_args()

    files = proc.find_mat_files(args.input)
    logging.info(f"Scoring 8 plane orientations for {len(files)} files")
    rows, seen = [], set()
    for path in files:
        pid = proc.patient_id_from_path(path)
        if pid in seen:
            continue
        seen.add(pid)
        try:
            scores, chosen = score_orientations(proc.load_patient_struct(path))
        except Exception as e:
            logging.info(f"FAIL  {path.name}: {type(e).__name__}: {e}")
            continue
        best = max(scores, key=scores.get)
        rows.append({'patient_id': pid, 'cohort': cohort_of(pid), 'best': str(best), 'chosen_by_1.5': str(chosen),
                     'r_best': round(scores[best], 3), 'r_chosen': round(scores[chosen], 3),
                     **{str(t): round(r, 3) for t, r in scores.items()}})
        logging.info(f"{pid:12s} best {str(best):22s} r={scores[best]:+.2f}   (1.5 chose {str(chosen)}: r={scores[chosen]:+.2f})")

    out = Path(args.output) / "plane_orientation_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    logging.info("\nWinning orientation (swap axes, flip rows, flip cols) per cohort:")
    for cohort in ('MC', 'other'):
        sel = [r for r in rows if r['cohort'] == cohort]
        if not sel:
            continue
        tally = Counter(r['best'] for r in sel)
        logging.info(f"  {cohort} ({len(sel)} patients) - median r per orientation, and how often it wins:")
        medians = {str(t): np.median([r[str(t)] for r in sel]) for t in proc.DIHEDRAL}
        for t, med in sorted(medians.items(), key=lambda kv: -kv[1]):
            logging.info(f"    {t:22s} median r {med:+.3f}   best for {tally.get(t, 0)}/{len(sel)}")
        logging.info(f"    1.5's per-patient choice:      median r {np.median([r['r_chosen'] for r in sel]):+.3f}")
    logging.info(f"\nPer-patient scores: {out}")


if __name__ == '__main__':
    main()
