"""
Runs K-fold cross-validation by invoking 3.PointNet/main.py once per fold
(sequentially, since each run trains a full model), then aggregates the
per-patient Pearson correlation across all folds into one overall summary.
Every patient lands in exactly one fold's validation set, so the aggregate
matches the "patient-level median Pearson" methodology used for the
0.782 baseline.

Run this from the same directory you've been running main.py from
(Slicer-AortaWallSheerStress/). Pass it the same flags you'd pass to
main.py itself, minus --fold (this script sets that per fold):

    python run_kfold.py --use_geometric_features --geometric_features_path X_with_features.npy \
        --input_components xyznd --output_components x --gpu 0 --N_iterations 200 --batch_size 32 -N_p 4096
"""

import argparse
import os
import subprocess
import sys
import numpy as np


def build_name(args, fold):
    split_label = f'fold{fold}of{args.n_folds}'
    return (f'PointNet_RUN{args.RUN}_D{args.N_samples}_N{args.N_pt}_input_{args.input_components}'
            f'_output_{args.output_components}_split_{split_label}_scaling_{args.scaling}')


def main():
    parser = argparse.ArgumentParser(description='Run K-fold CV by driving main.py once per fold.')
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--RUN', type=int, default=0)
    parser.add_argument('-N_d', '--N_samples', type=int, default=3000)
    parser.add_argument('-N_p', '--N_pt', type=int, default=4096)
    parser.add_argument('-ic', '--input_components', type=str, default='xyzmlc')
    parser.add_argument('-oc', '--output_components', type=str, default='xyzs')
    parser.add_argument('--scaling', type=float, default=0.53)
    parser.add_argument('--base_dir', type=str, default='../experiments')
    parser.add_argument('--start_fold', type=int, default=0, help='Resume from this fold if a previous run was interrupted')
    args, passthrough = parser.parse_known_args()

    main_py = os.path.join('3.PointNet', 'main.py')

    for fold in range(args.start_fold, args.n_folds):
        cmd = [sys.executable, main_py,
               '--fold', str(fold), '--n_folds', str(args.n_folds),
               '--RUN', str(args.RUN), '-N_d', str(args.N_samples), '-N_p', str(args.N_pt),
               '-ic', args.input_components, '-oc', args.output_components,
               '--scaling', str(args.scaling), '--base_dir', args.base_dir] + passthrough
        print(f"\n=== Fold {fold}/{args.n_folds - 1} ===")
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

    all_pearson = []
    all_ids = []
    for fold in range(args.n_folds):
        name = build_name(args, fold)
        exp_dir = os.path.join(args.base_dir, name)
        pearson = np.load(os.path.join(exp_dir, 'per_patient_pearson.npy'))
        ids = np.load(os.path.join(exp_dir, 'per_patient_ids.npy'), allow_pickle=True)
        all_pearson.append(pearson)
        all_ids.append(ids)

    all_pearson = np.concatenate(all_pearson, axis=0)
    all_ids = np.concatenate(all_ids, axis=0)

    median_pearson = np.median(all_pearson, axis=0)
    mean_pearson = np.mean(all_pearson, axis=0)

    print(f"\n=== {args.n_folds}-Fold CV Aggregate (N={all_pearson.shape[0]} patients) ===")
    print(f"Median Pearson: {median_pearson}")
    print(f"Mean Pearson: {mean_pearson}")

    tag = f'RUN{args.RUN}_{args.input_components}_{args.output_components}'
    summary_path = os.path.join(args.base_dir, f'kfold_summary_{tag}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"N patients: {all_pearson.shape[0]}\n")
        f.write(f"Median Pearson: {median_pearson}\n")
        f.write(f"Mean Pearson: {mean_pearson}\n")
    np.save(os.path.join(args.base_dir, f'kfold_all_pearson_{tag}.npy'), all_pearson)
    np.save(os.path.join(args.base_dir, f'kfold_all_ids_{tag}.npy'), all_ids)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
