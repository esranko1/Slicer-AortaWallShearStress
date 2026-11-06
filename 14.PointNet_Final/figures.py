"""
Paper figures and Table 2 for every method tried, by region. Reads saved
results only (no training; about a minute on CPU).

    python figures.py                                  # all methods in METHODS that have results
    python figures.py --only "Final (LOO)" "2D CNN"    # a subset, by label
    python figures.py --bins 10                        # Dice of the top / bottom 10% instead of 25%

Writes to ../experiments/figures/ (PNG at 300 dpi + PDF):
  box_spatial_r     per-patient spatial r (Pearson across the region's wall
                    points), box per method, one panel per region
  box_dice_high     per-patient Dice of the highest-WSS area: the region's top
                    1/bins of wall points by true vs by predicted WSS
  box_dice_low      the same for the lowest-WSS area. Dashed line = chance (1/bins)
  ba_<method>       Bland-Altman of each patient's median regional WSS, one
                    panel per region (axes shared across methods)
  ba_grid           all methods x regions in one figure (supplement)
  table2_wide.csv   level r per region + PWV r / MAE, one row per method
  table2_long.csv   every metric per method x region (level r, spatial r
                    median [IQR], Dice, MAE, pooled Spearman, Bland-Altman)
Methods whose results file is missing are skipped with a note.
"""

import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from quantiles import rank_bins  # noqa: E402
from runs import (REGIONS, WHOLE, level_bland_altman, level_values, load_run, mae_per_patient,  # noqa: E402
                  pooled_spearman, r_spatial_per_patient, region_points)

FINAL = "whole:cv10:flow:aligned:phys-input"
# (group, label, run spec), in plotting order (top to bottom)
METHODS = [
    ("Final", "Final (LOO)", "whole:loo:flow:aligned:phys-input"),
    ("Final", "Final (10-fold)", FINAL),
    ("Baseline", "Average map", "meanmap:cv10:flow"),
    ("Baseline", "Poiseuille only", "poiseuille:cv10:flow"),
    ("Baseline", "Flow + radius", "flowradius:cv10:flow"),
    ("Shape statistics", "Shape modes + ridge", "shapemodes:cv10:flow"),
    ("Shape statistics", "WSS modes + ridge", "modes-ridge:cv10:flow"),
    ("Shape statistics", "WSS modes + GP", "modes-gp-nophys:cv10:flow"),
    ("Architecture", "PointNet (basic)", "whole:cv10:flow"),
    ("Architecture", "2D CNN", FINAL + ":cnn"),
    ("Architecture", "DGCNN", FINAL + ":dgcnn"),
    ("Input", "PointNet + alignment", "whole:cv10:flow:aligned"),
    ("Input", "4D-flow mm geometry", FINAL + ":mm"),
    ("Target / training", "WSS / Poiseuille", "whole:cv10:flow:aligned:phys-ratio"),
    ("Target / training", "+ velocity planes", FINAL + ":planes0.1"),
    ("Target / training", "Hybrid (asc. specialist)", "hybrid:cv10:flow:aligned:phys-input"),
]
REGION_ROWS = [("whole", "Whole aorta", WHOLE), ("pasc", "Prox. ascending", REGIONS["pasc"]),
               ("arch", "Arch", REGIONS["arch"]), ("desc", "Descending", REGIONS["desc"]),
               ("abda", "Abdominal", REGIONS["abda"])]

# one accent for the final model, neutral for the rest
ACCENT, NEUTRAL, NEUTRAL_FILL, INK, MUTED = "#2a78d6", "#52514e", "#d9d8d3", "#0b0b0b", "#8c8b86"
# boxplot's vert= was replaced by orientation= in matplotlib 3.10
HORIZONTAL = ({"orientation": "horizontal"} if tuple(map(int, re.findall(r"\d+", matplotlib.__version__)[:2])) >= (3, 10)
              else {"vert": False})
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": NEUTRAL,
                     "ytick.color": NEUTRAL, "savefig.dpi": 300, "savefig.bbox": "tight"})


def dice(t, p, bins, top):
    """Per-patient Dice of the top (or bottom) 1/bins of wall points, true vs predicted."""
    k = bins - 1 if top else 0
    a, b = rank_bins(t, bins, axis=1) == k, rank_bins(p, bins, axis=1) == k
    return 2 * (a & b).sum(axis=1) / (a.sum(axis=1) + b.sum(axis=1))


def score(true, pred, pwv_true, pwv_pred, bins, ranks=True):
    """ranks=False for the average map: it gives every patient the training mean, so its level r is noise."""
    out = {}
    for key, _, col in REGION_ROWS:
        t, p = region_points(true, pred, col)
        lt, lp = level_values(true, pred, col)
        out[key] = {"level_r": float(np.corrcoef(lt, lp)[0, 1]) if ranks else np.nan,
                    "spatial": r_spatial_per_patient(true, pred, col),
                    "dice_high": dice(t, p, bins, True), "dice_low": dice(t, p, bins, False),
                    "mae": mae_per_patient(true, pred, col), "pooled_rho": pooled_spearman(true, pred, col),
                    "ba": level_bland_altman(true, pred, col), "level": (lt, lp)}
    if pwv_pred is not None:
        out["pwv"] = (float(np.corrcoef(pwv_true, pwv_pred)[0, 1]), float(np.abs(pwv_pred - pwv_true).mean()))
    return out


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def boxplots(res, out, metric, xlabel, name, chance=None):
    n = len(res)
    fig, axes = plt.subplots(1, len(REGION_ROWS), figsize=(13, 0.28 * n + 1.2), sharey=True)
    for ax, (key, title, _) in zip(axes, REGION_ROWS):
        data = [r[3][key][metric] for r in res][::-1]       # first method at the top
        bp = ax.boxplot(data, **HORIZONTAL, patch_artist=True, widths=0.6,
                        medianprops={"color": INK, "linewidth": 1.2},
                        whiskerprops={"color": NEUTRAL, "linewidth": 0.8},
                        capprops={"color": NEUTRAL, "linewidth": 0.8},
                        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": MUTED,
                                    "markeredgecolor": "none"})
        for patch, r in zip(bp["boxes"], res[::-1]):
            final = r[0] == "Final"
            patch.set_facecolor(ACCENT if final else NEUTRAL_FILL)
            patch.set_alpha(0.45 if final else 1.0)
            patch.set_edgecolor(ACCENT if final else NEUTRAL)
            patch.set_linewidth(0.8)
        groups = [r[0] for r in res][::-1]
        for i in range(1, n):                                 # thin line between method groups
            if groups[i] != groups[i - 1]:
                ax.axhline(i + 0.5, color=NEUTRAL_FILL, linewidth=0.8)
        if chance is not None:
            ax.axvline(chance, color=MUTED, linestyle="--", linewidth=0.8)
        ax.set_title(title, color=INK)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#ecebe8", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(1, n + 1), [r[1] for r in res][::-1])
    save(fig, out, name)


def ba_limits(res):
    lim = {}
    for key, _, _ in REGION_ROWS:
        m = np.concatenate([(r[3][key]["level"][0] + r[3][key]["level"][1]) / 2 for r in res])
        d = np.concatenate([r[3][key]["level"][1] - r[3][key]["level"][0] for r in res])
        pad_m, pad_d = 0.05 * np.ptp(m), 0.05 * np.ptp(d)
        lim[key] = ((m.min() - pad_m, m.max() + pad_m), (d.min() - pad_d, d.max() + pad_d))
    return lim


def ba_panel(ax, r, key, lim):
    t, p = r[3][key]["level"]
    bias, lo, hi = r[3][key]["ba"]
    ax.axhline(0, color=NEUTRAL_FILL, linewidth=0.8)
    ax.scatter((t + p) / 2, p - t, s=9, color=NEUTRAL, alpha=0.75, linewidths=0)
    ax.axhline(bias, color=ACCENT, linewidth=1.2)
    for v in (lo, hi):
        ax.axhline(v, color=MUTED, linestyle="--", linewidth=0.8)
    for v in (hi, bias, lo):                                  # values just outside the right edge
        ax.annotate(f"{v:+.2f}", (1, v), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.5,
                    color=ACCENT if v == bias else NEUTRAL, annotation_clip=False)
    ax.set_xlim(*lim[key][0])
    ax.set_ylim(*lim[key][1])


def bland_altman(res, out):
    lim = ba_limits(res)
    for r in res:
        fig, axes = plt.subplots(1, len(REGION_ROWS), figsize=(13, 2.6), gridspec_kw={"wspace": 0.45})
        for ax, (key, title, _) in zip(axes, REGION_ROWS):
            ba_panel(ax, r, key, lim)
            ax.set_title(title, color=INK)
        axes[0].set_ylabel("predicted - true (Pa)")
        fig.supxlabel("mean of true and predicted median WSS (Pa); blue = bias, dashed = 95% limits of agreement",
                      fontsize=8, color=INK, y=-0.08)
        fig.suptitle(r[1], x=0.01, ha="left", color=INK, fontsize=9)
        save(fig, out, f"ba_{slug(r[1])}")
    fig, axes = plt.subplots(len(res), len(REGION_ROWS), figsize=(13, 2.0 * len(res)), squeeze=False,
                             gridspec_kw={"wspace": 0.45})
    for i, r in enumerate(res):
        for j, (key, title, _) in enumerate(REGION_ROWS):
            ba_panel(axes[i, j], r, key, lim)
            if i == 0:
                axes[i, j].set_title(title, color=INK)
            if i == len(res) - 1:
                axes[i, j].set_xlabel("mean median WSS (Pa)")
        axes[i, 0].set_ylabel(f"{r[1]}\npred - true (Pa)")
    save(fig, out, "ba_grid")


def tables(res, out, bins):
    keys = [k for k, _, _ in REGION_ROWS]
    with open(os.path.join(out, "table2_wide.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "run"] + [f"level r {k}" for k in keys] + ["PWV r", "PWV MAE (m/s)"])
        for g, label, spec, s in res:
            pwv = s.get("pwv", (np.nan, np.nan))
            w.writerow([g, label, spec] + [f"{s[k]['level_r']:.3f}" for k in keys] + [f"{v:.3f}" for v in pwv])
    with open(os.path.join(out, "table2_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "run", "region", "level r", "spatial r median", "spatial r Q1",
                    "spatial r Q3", f"Dice top 1/{bins} median", f"Dice bottom 1/{bins} median",
                    "MAE median (Pa)", "pooled Spearman", "BA bias (Pa)", "BA LoA low", "BA LoA high"])
        for g, label, spec, s in res:
            for k in keys:
                q1, m, q3 = np.percentile(s[k]["spatial"], [25, 50, 75])
                w.writerow([g, label, spec, k] + [f"{v:.3f}" for v in (
                    s[k]["level_r"], m, q1, q3, np.median(s[k]["dice_high"]), np.median(s[k]["dice_low"]),
                    np.median(s[k]["mae"]), s[k]["pooled_rho"], *s[k]["ba"])])

    print(f"\nLevel r by region, PWV ({out}/table2_wide.csv)")
    print(f"{'method':26s}" + "".join(f"{k:>7s}" for k in keys) + f"{'PWV r':>8s}{'MAE':>7s}")
    for g, label, spec, s in res:
        pwv = s.get("pwv", (np.nan, np.nan))
        print(f"{label:26s}" + "".join(f"{s[k]['level_r']:7.3f}" for k in keys) + f"{pwv[0]:8.3f}{pwv[1]:7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="method labels to include (default: all in METHODS)")
    ap.add_argument("--bins", type=int, default=4, help="Dice area = top / bottom 1/bins of wall points")
    ap.add_argument("--out", default="../experiments/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    methods = [m for m in METHODS if not args.only or m[1] in args.only]
    res = []
    for g, label, spec in methods:
        try:
            run = load_run(spec)
        except (OSError, ValueError, SystemExit) as e:
            print(f"skipped {label} ({spec}): {e}")
            continue
        res.append((g, label, spec, score(*run, args.bins, ranks=not spec.startswith("meanmap"))))
        print(f"loaded {label}")
    if not res:
        raise SystemExit("No results found.")

    tables(res, args.out, args.bins)
    boxplots(res, args.out, "spatial", "spatial r (per patient)", "box_spatial_r")
    boxplots(res, args.out, "dice_high", f"Dice, highest {100 // args.bins}% WSS area", "box_dice_high",
             chance=1 / args.bins)
    boxplots(res, args.out, "dice_low", f"Dice, lowest {100 // args.bins}% WSS area", "box_dice_low",
             chance=1 / args.bins)
    bland_altman(res, args.out)
    print(f"\nFigures written to {args.out}")


if __name__ == "__main__":
    main()
