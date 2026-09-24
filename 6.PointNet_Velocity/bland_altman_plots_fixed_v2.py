"""
Generate Bland-Altman plots from PointNet region-specialist results.

Workflow:
  1. Run each region model separately to generate results.npz
  2. This script loads all 4 regions and creates:
     - Individual Bland-Altman plots per region
     - Combined full-aorta plot

Run this AFTER all 4 region models finish:
    python bland_altman_plots_fixed.py

FIX (vs. original script): each patient's 4096 points are a flattened
32 (circumferential) x 128 (longitudinal) grid -- see stitch_specialists.py's
REGIONS dict and check_significance.py's `.reshape(n_total, 32, 128)` calls.
The region column-slice bounds (0-36, 36-60, 60-96, 96-128) are correct AS
LONGITUDINAL-AXIS bounds on that reshaped grid; they were never meant to
index the flat (100, 4096) array directly, which is what the original
script effectively did by never using grid_slice at all. This version
reshapes to (n, 32, 128) and slices the last axis, matching the convention
already used in stitch_specialists.py and check_significance.py, and reuses
build_composite() for the full-aorta stitching so there's only one
implementation of that logic in the codebase.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stitch_specialists import build_composite, REGIONS

N_CIRC = 32   # circumferential points
N_LONG = 128  # longitudinal points (32 * 128 = 4096)


def bland_altman_plot(true_vals, pred_vals, title, ax=None, region_name=""):
    """
    Create a Bland-Altman plot.

    Args:
        true_vals: ground truth values (1D array)
        pred_vals: predicted values (1D array)
        title: plot title
        ax: matplotlib axis (if None, creates new figure)
        region_name: name of region for legend

    Returns:
        ax: matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    mean_vals = (true_vals + pred_vals) / 2
    diff_vals = pred_vals - true_vals

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)

    ax.scatter(mean_vals, diff_vals, alpha=0.3, s=10, label=region_name if region_name else "Data")

    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean diff: {mean_diff:.4f}')

    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    ax.axhline(upper_limit, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'+1.96 SD: {upper_limit:.4f}')
    ax.axhline(lower_limit, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'-1.96 SD: {lower_limit:.4f}')

    ax.set_xlabel('Mean(True, Pred)', fontsize=12)
    ax.set_ylabel('Pred - True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    return ax


def main():
    output_dir = Path("experiments") / "bland_altman_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    region_stats = {}

    # =========================================================================
    # Process each region separately, using ONLY that region's own longitudinal
    # columns (grid-sliced, not flat-sliced).
    # =========================================================================
    for region_key, (col_slice, results_path) in REGIONS.items():
        region_name = region_key  # swap in your display names if you want, e.g. "Proximal Ascending"
        results_path = Path(results_path)

        print(f"\n{'='*70}")
        print(f"Processing: {region_name}")
        print(f"{'='*70}")

        if not results_path.exists():
            print(f"SKIP: {results_path} not found")
            continue

        print(f"Loading {results_path}...")
        data = np.load(results_path)
        n_total = data['all_true'].shape[0]

        true_grid = data['all_true'].reshape(n_total, N_CIRC, N_LONG)
        pred_grid = data['all_pred'].reshape(n_total, N_CIRC, N_LONG)

        # Region-masked: only this specialist's own longitudinal columns
        all_true = true_grid[:, :, col_slice].reshape(n_total, -1)
        all_pred = pred_grid[:, :, col_slice].reshape(n_total, -1)

        print(f"Loaded full: {data['all_true'].shape} -> region slice: {all_true.shape}")

        stored_pearson_key = f'pearson_{region_key}'
        stored_pearson = float(data[stored_pearson_key]) if stored_pearson_key in data else None

        # --- Bland-Altman: All points (region-masked)
        true_flat = all_true.flatten()
        pred_flat = all_pred.flatten()

        fig, ax = plt.subplots(figsize=(12, 7))
        bland_altman_plot(true_flat, pred_flat,
                         f"Bland-Altman: All Points ({region_name})",
                         ax=ax, region_name=region_name)
        plt.tight_layout()
        plot_name = f"01_bland_altman_all_points_{region_key}.png"
        plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_name}")
        plt.close()

        # --- Bland-Altman: Per-patient
        patient_true = np.median(all_true, axis=1)
        patient_pred = np.median(all_pred, axis=1)

        fig, ax = plt.subplots(figsize=(12, 7))
        bland_altman_plot(patient_true, patient_pred,
                         f"Bland-Altman: Per-Patient Median ({region_name})",
                         ax=ax, region_name=region_name)
        plt.tight_layout()
        plot_name = f"02_bland_altman_per_patient_{region_key}.png"
        plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_name}")
        plt.close()

        # --- Correlation plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(patient_true, patient_pred, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

        min_val = min(patient_true.min(), patient_pred.min())
        max_val = max(patient_true.max(), patient_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect agreement')

        corr = np.corrcoef(patient_true, patient_pred)[0, 1]
        title_r = stored_pearson if stored_pearson is not None else corr
        ax.set_xlabel('Ground Truth WSS', fontsize=12)
        ax.set_ylabel('Predicted WSS', fontsize=12)
        ax.set_title(f'Correlation Plot ({region_name}, Pearson r={title_r:.3f})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        plot_name = f"03_correlation_plot_{region_key}.png"
        plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_name}")
        plt.close()

        mean_all = np.mean(pred_flat - true_flat)
        std_all = np.std(pred_flat - true_flat)
        mean_patient = np.mean(patient_pred - patient_true)
        std_patient = np.std(patient_pred - patient_true)

        region_stats[region_key] = {
            'name': region_name,
            'mean_diff_all': mean_all,
            'std_all': std_all,
            'mean_diff_patient': mean_patient,
            'std_patient': std_patient,
            'pearson_r_recomputed': corr,
            'pearson_r_stored': stored_pearson,
        }

        print(f"\nStats for {region_name}:")
        print(f"  All points:  mean_diff={mean_all:.6f}, std={std_all:.6f}")
        print(f"  Per-patient: mean_diff={mean_patient:.6f}, std={std_patient:.6f}, "
              f"Pearson r (recomputed, region-masked)={corr:.4f}, "
              f"Pearson r (stored in npz)={stored_pearson}")

    # =========================================================================
    # Full aorta: reuse the already-verified stitching logic from
    # stitch_specialists.py instead of reimplementing it here.
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"STITCHING: Full Aorta (All Regions Combined)")
    print(f"{'='*70}")

    all_true_combined, all_pred_combined = build_composite(verbose=True)
    print(f"Combined shape: {all_true_combined.shape}")

    true_flat = all_true_combined.flatten()
    pred_flat = all_pred_combined.flatten()

    fig, ax = plt.subplots(figsize=(12, 7))
    bland_altman_plot(true_flat, pred_flat,
                     "Bland-Altman: All Points (Full Aorta - All Regions)",
                     ax=ax, region_name="All points")
    plt.tight_layout()
    plot_name = f"10_bland_altman_all_points_FULL_AORTA.png"
    plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_name}")
    plt.close()

    patient_true = np.median(all_true_combined, axis=1)
    patient_pred = np.median(all_pred_combined, axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))
    bland_altman_plot(patient_true, patient_pred,
                     "Bland-Altman: Per-Patient Median (Full Aorta - All Regions)",
                     ax=ax, region_name="Patient medians")
    plt.tight_layout()
    plot_name = f"11_bland_altman_per_patient_FULL_AORTA.png"
    plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_name}")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(patient_true, patient_pred, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

    min_val = min(patient_true.min(), patient_pred.min())
    max_val = max(patient_true.max(), patient_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect agreement')

    corr = np.corrcoef(patient_true, patient_pred)[0, 1]
    ax.set_xlabel('Ground Truth WSS', fontsize=12)
    ax.set_ylabel('Predicted WSS', fontsize=12)
    ax.set_title(f'Correlation Plot (Full Aorta, Pearson r={corr:.3f})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plot_name = f"12_correlation_plot_FULL_AORTA.png"
    plt.savefig(output_dir / plot_name, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_name}")
    plt.close()

    mean_all = np.mean(pred_flat - true_flat)
    std_all = np.std(pred_flat - true_flat)
    mean_patient = np.mean(patient_pred - patient_true)
    std_patient = np.std(patient_pred - patient_true)

    print(f"\nFull Aorta Summary:")
    print(f"  All points:  mean_diff={mean_all:.6f}, std={std_all:.6f}")
    print(f"  Per-patient: mean_diff={mean_patient:.6f}, std={std_patient:.6f}, Pearson r={corr:.4f}")

    # =========================================================================
    # Print summary table
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Region':<20} {'Pearson r (stored)':>18} {'Pearson r (recomp.)':>20} {'Mean Diff':>12} {'Std Dev':>12}")
    print("-" * 90)

    for region_key, stats in region_stats.items():
        stored = stats['pearson_r_stored']
        stored_str = f"{stored:.4f}" if stored is not None else "N/A"
        print(f"{stats['name']:<20} {stored_str:>18} {stats['pearson_r_recomputed']:>20.4f} "
              f"{stats['mean_diff_patient']:>12.6f} {stats['std_patient']:>12.6f}")

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()