"""
Re-evaluate already-trained Stage 1 PINNs on their held-out planes, without
retraining. Loads each <results>/stage1/<id>.pt, rebuilds the network,
predicts every held-out plane pixel at every phase and compares it with the
measurement - and with the neighbour-copy baseline (average of the two
neighbouring measured planes).

Metrics (see FLOW_METRICS in stage1_flow_pinn.py): relative error overall /
systole / diastole, RMSE in m/s, and Pearson r of the cross-section profile at
peak systole, of the plane-mean velocity waveform, and over all pixels x
phases.

Writes <results>/stage1_flow_eval.csv and prints a PINN-vs-baseline table,
overall and per cohort.

Usage:
    python evaluate_stage1.py
    python evaluate_stage1.py --patients 10MC0506 RAT_4858
"""

import argparse
import csv
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, cohort_of, load_patient,  # noqa: E402
                    neighbour_plane_baseline)
from stage1_flow_pinn import (FlowNet, flow_metrics, pick_device, plane_predictions,  # noqa: E402
                              prepare, print_flow_summary)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def evaluate(pid, data_dir, stage1_dir, device):
    d = load_patient(data_dir, pid)
    # Your own checkpoint (it holds numpy arrays, hence weights_only=False)
    ckpt = torch.load(stage1_dir / f"{pid}.pt", map_location=device, weights_only=False)
    with np.load(stage1_dir / f"{pid}.npz") as z:
        val_planes = z['val_planes']

    args = types.SimpleNamespace(cycle_s=ckpt['cycle_s'], holdout_every=ckpt.get('holdout_every', 6),
                                 wall_offset_mm=ckpt.get('wall_offset_mm', 0.0))
    prep = prepare(d, args)
    # Use exactly the planes that were held out in training
    prep['val_planes'] = val_planes
    prep['is_val'] = np.isin(d['vel_plane_id'], val_planes)

    model = FlowNet(prep['centerline'], **ckpt['arch']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    pred = plane_predictions(model, prep, device)
    base = neighbour_plane_baseline(d, val_planes) / prep['U']
    m = flow_metrics(pred, prep['vel'], base, prep, d['vel_plane_id'])
    return {'patient_id': pid, 'cohort': cohort_of(pid), **{k: round(v, 4) for k, v in m.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(DEFAULT_DATA_DIR))
    ap.add_argument('--results', default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument('--patients', nargs='*', help='only these patient IDs')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = pick_device(args.device)
    stage1_dir = Path(args.results) / "stage1"
    patients = sorted(p.stem for p in stage1_dir.glob("*.pt"))
    if args.patients:
        patients = [p for p in patients if p in set(args.patients)]
    logging.info(f"Evaluating {len(patients)} trained Stage 1 models on {device}")

    rows = []
    for i, pid in enumerate(patients, 1):
        try:
            rows.append(evaluate(pid, args.data, stage1_dir, device))
        except Exception as e:  # e.g. a checkpoint from the older (pre-centerline) network
            logging.info(f"  [{i}/{len(patients)}] {pid}: FAILED {type(e).__name__}: {e}")
            continue
        if i % 10 == 0 or i == len(patients):
            logging.info(f"  {i}/{len(patients)} done")

    if not rows:
        raise SystemExit("Nothing evaluated.")
    out_csv = Path(args.results) / "stage1_flow_eval.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print_flow_summary(rows)
    for cohort in ('MC', 'other'):
        sel = [r for r in rows if r['cohort'] == cohort]
        if sel and len(sel) < len(rows):
            logging.info(f"\n--- cohort {cohort} ---")
            print_flow_summary(sel)
    logging.info(f"\nPer-patient results: {out_csv}")


if __name__ == '__main__':
    main()
