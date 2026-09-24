"""
Find which patient has the best prediction performance on COMPOSITE (all 4 regions).
Loads all 4 region specialists, stitches them together, then analyzes.
"""

import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

def load_composite_results():
    """Load all 4 region results and stitch into composite"""
    regions = {
        'pasc': (0, 36*32),      # Proximal Ascending
        'arch': (36*32, 60*32),  # Thoracic Arch
        'desc': (60*32, 96*32),  # Descending
        'abda': (96*32, 128*32), # Abdominal
    }

    all_true_composite = None
    all_pred_composite = None

    print("Loading region-specialist results...")

    for region_key, (start_idx, end_idx) in regions.items():
        results_path = f"experiments/6_PointNet_Velocity_roi_{region_key}/results.npz"

        if not Path(results_path).exists():
            print(f"  ✗ Missing: {results_path}")
            return None, None

        data = np.load(results_path)
        all_true = data['all_true']  # (100, 4096)
        all_pred = data['all_pred']  # (100, 4096)

        print(f"  ✓ Loaded {region_key}: Pearson = {pearsonr(all_true[:, start_idx:end_idx].flatten(), all_pred[:, start_idx:end_idx].flatten())[0]:.4f}")

        if all_true_composite is None:
            all_true_composite = all_true.copy()
            all_pred_composite = all_pred.copy()
        else:
            # Take this region's predictions for its portion
            all_true_composite[:, start_idx:end_idx] = all_true[:, start_idx:end_idx]
            all_pred_composite[:, start_idx:end_idx] = all_pred[:, start_idx:end_idx]

    return all_true_composite, all_pred_composite


def compute_patient_metrics(Y_true, Y_pred):
    """Compute metrics per patient"""
    n_patients = Y_true.shape[0]

    metrics = []
    for i in range(n_patients):
        true_flat = Y_true[i, :].flatten()
        pred_flat = Y_pred[i, :].flatten()

        # Pearson correlation
        pearson_r = pearsonr(true_flat, pred_flat)[0]

        # MAE
        mae = np.mean(np.abs(true_flat - pred_flat))

        # RMSE
        rmse = np.sqrt(np.mean((true_flat - pred_flat) ** 2))

        # R2
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics.append({
            'patient': i,
            'pearson': pearson_r,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })

    return metrics


def main():
    print("="*80)
    print("PATIENT-LEVEL PERFORMANCE ANALYSIS (COMPOSITE - ALL 4 REGIONS)")
    print("="*80 + "\n")

    Y_true, Y_pred = load_composite_results()

    if Y_true is None:
        print("\n❌ Could not load all 4 region results. Make sure you've run:")
        print("  - experiments/6_PointNet_Velocity_roi_pasc/results.npz")
        print("  - experiments/6_PointNet_Velocity_roi_arch/results.npz")
        print("  - experiments/6_PointNet_Velocity_roi_desc/results.npz")
        print("  - experiments/6_PointNet_Velocity_roi_abda/results.npz")
        return

    metrics = compute_patient_metrics(Y_true, Y_pred)

    # Sort by Pearson correlation
    sorted_by_r = sorted(metrics, key=lambda x: x['pearson'], reverse=True)

    print("\n🏆 TOP 10 BEST PERFORMING PATIENTS (by Pearson r):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Patient':<8} {'Pearson r':<12} {'R2':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 80)
    for rank, m in enumerate(sorted_by_r[:10], 1):
        print(f"{rank:<6} {m['patient']:<8} {m['pearson']:<12.4f} {m['r2']:<10.4f} {m['mae']:<10.4f} {m['rmse']:<10.4f}")

    print("\n📊 BOTTOM 10 WORST PERFORMING PATIENTS (by Pearson r):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Patient':<8} {'Pearson r':<12} {'R2':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 80)
    for rank, m in enumerate(sorted_by_r[-10:], 1):
        print(f"{rank:<6} {m['patient']:<8} {m['pearson']:<12.4f} {m['r2']:<10.4f} {m['mae']:<10.4f} {m['rmse']:<10.4f}")

    # Statistics
    print("\n📈 OVERALL STATISTICS (FULL AORTA):")
    print("-" * 80)
    pearson_vals = [m['pearson'] for m in metrics]
    r2_vals = [m['r2'] for m in metrics]
    mae_vals = [m['mae'] for m in metrics]

    print(f"Pearson r:  mean={np.mean(pearson_vals):.4f}, std={np.std(pearson_vals):.4f}, min={np.min(pearson_vals):.4f}, max={np.max(pearson_vals):.4f}")
    print(f"R2:         mean={np.mean(r2_vals):.4f}, std={np.std(r2_vals):.4f}, min={np.min(r2_vals):.4f}, max={np.max(r2_vals):.4f}")
    print(f"MAE:        mean={np.mean(mae_vals):.4f}, std={np.std(mae_vals):.4f}, min={np.min(mae_vals):.4f}, max={np.max(mae_vals):.4f}")

    best = sorted_by_r[0]
    worst = sorted_by_r[-1]

    print("\n🎯 SUMMARY:")
    print(f"Best patient:  #{best['patient']:03d} with Pearson r = {best['pearson']:.4f}")
    print(f"Worst patient: #{worst['patient']:03d} with Pearson r = {worst['pearson']:.4f}")
    print(f"Range: {best['pearson'] - worst['pearson']:.4f}")


if __name__ == "__main__":
    main()
