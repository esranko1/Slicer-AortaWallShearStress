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
                    panel per region
  ba_grid           all methods x regions in one figure, axes shared per region (supplement)
  table2_wide.csv   level r per region + PWV r / MAE, one row per method
  table2_long.csv   every metric per method x region (level r, spatial r
                    median [IQR], Dice, MAE, pooled Spearman, Bland-Altman)
  table2_paper.csv  Table 2 as in the paper: level r per region + PWV r / MAE,
                    * = worse than Final (10-fold), paired bootstrap 95% CI
                    excludes 0 (same resamples as compare_runs.py)
  table2_vs_final.csv  every difference from Final (10-fold) with its 95% CI
  forest_vs_final   difference from Final (10-fold) with 95% CI, whole aorta:
                    level r, spatial r, PWV MAE
Final model only (Final (LOO) if loaded, else Final (10-fold)):
  final_scatter     predicted vs true median WSS per patient, one panel per region
  final_pwv         predicted vs true PWV + Bland-Altman
  final_maps        true / predicted / difference wall maps for the patients at
                    the 10th, 50th and 90th percentile of whole-aorta spatial r
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
    ("Input", "Geometry only (no flow)", "whole:cv10:noflow:aligned"),
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
        out["pwv_values"] = (pwv_true, pwv_pred)
    return out


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def boxplots(res, out, metric, xlabel, name, chance=None):
    n = len(res)
    fig, axes = plt.subplots(1, len(REGION_ROWS), figsize=(13, 0.28 * n + 1.2), sharey=True, sharex=True)
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
    for r in res:                                             # own axes: shared ones squash the better methods
        fig, axes = plt.subplots(1, len(REGION_ROWS), figsize=(13, 2.6), gridspec_kw={"wspace": 0.45})
        for ax, (key, title, _) in zip(axes, REGION_ROWS):
            ba_panel(ax, r, key, ba_limits([r]))
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


# ---- paired bootstrap against the final model (as compare_runs.py, same resamples) ----
REFERENCE = "Final (10-fold)"
N_BOOTSTRAP, SEED = 10000, 2024


def boot_indices(n):
    rng = np.random.default_rng(SEED)
    return np.stack([rng.integers(0, n, size=n) for _ in range(N_BOOTSTRAP)])


def pearson_rows(a, b):
    """Pearson r along the last axis (one value per bootstrap resample)."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=-1) / np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))


def metric_fns(s):
    """{(metric, region): f(index array) -> value(s)}; lower is better only for PWV MAE."""
    fns = {}
    for key, _, _ in REGION_ROWS:
        t, p = s[key]["level"]
        if not np.isnan(s[key]["level_r"]):
            fns[("level r", key)] = lambda i, t=t, p=p: pearson_rows(t[i], p[i])
        fns[("spatial r", key)] = lambda i, sp=s[key]["spatial"]: np.median(sp[i], axis=-1)
    if "pwv_values" in s:
        t, p = s["pwv_values"]
        fns[("PWV r", "-")] = lambda i, t=t, p=p: pearson_rows(t[i], p[i])
        fns[("PWV MAE", "-")] = lambda i, t=t, p=p: np.abs(p[i] - t[i]).mean(axis=-1)
    return fns


def versus_final(res):
    """{label: {(metric, region): (value, difference from the final model, CI low, CI high, verdict)}}
    for every method outside the Final group. verdict: 'worse', 'better' or ''."""
    ref = next((r for r in res if r[1] == REFERENCE), None)
    if ref is None:
        print(f"\n{REFERENCE} not loaded: no comparison against the final model")
        return {}
    t_ref = ref[3]["whole"]["level"][0]
    idx, full = boot_indices(len(t_ref)), np.arange(len(t_ref))
    ref_fns = metric_fns(ref[3])
    out = {}
    for g, label, spec, s in res:
        if g == "Final":
            continue
        if not np.allclose(s["whole"]["level"][0], t_ref):
            print(f"{label}: different patients from {REFERENCE}; not compared")
            continue
        out[label] = {}
        for k, fb in metric_fns(s).items():
            if k not in ref_fns:
                continue
            fa = ref_fns[k]
            diffs = fb(idx) - fa(idx)
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            lower_better = k[0] == "PWV MAE"
            worse = (lo > 0) if lower_better else (hi < 0)
            better = (hi < 0) if lower_better else (lo > 0)
            out[label][k] = (float(fb(full)), float(fb(full) - fa(full)), lo, hi,
                             "worse" if worse else "better" if better else "")
    return out


def paper_table(res, cmp, out):
    """Table 2 for the paper; * worse / + better than the final model (10-fold)."""
    keys = [k for k, _, _ in REGION_ROWS]
    mark = lambda label, k: {"worse": "*", "better": "+"}.get(cmp.get(label, {}).get(k, (0,) * 5)[4], "")
    fmt = lambda v, d=3: "—" if v is None or np.isnan(v) else f"{v:.{d}f}"
    with open(os.path.join(out, "table2_paper.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Method"] + [title for _, title, _ in REGION_ROWS] + ["PWV r", "PWV MAE (m/s)"])
        for g, label, spec, s in res:
            pwv = s.get("pwv", (np.nan, np.nan))
            w.writerow([g, label] + [fmt(s[k]["level_r"]) + mark(label, ("level r", k)) for k in keys]
                       + [fmt(pwv[0]) + mark(label, ("PWV r", "-")), fmt(pwv[1], 2) + mark(label, ("PWV MAE", "-"))])
        w.writerow([])
        w.writerow(["Level r: Pearson r across patients of the median regional WSS. * lower (PWV MAE: higher) "
                    f"than {REFERENCE}, + the opposite: paired bootstrap 95% CI of the difference "
                    f"({N_BOOTSTRAP} resamples) excludes 0; not adjusted for multiple comparisons. "
                    "—: not defined (the average map predicts the same level for every patient) "
                    "or not predicted (PWV)."])
    with open(os.path.join(out, "table2_vs_final.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "region", "value", f"minus {REFERENCE}", "CI low", "CI high", "verdict"])
        for label, d in cmp.items():
            for (m, k), (v, diff, lo, hi, verdict) in d.items():
                w.writerow([label, m, k, f"{v:.3f}", f"{diff:+.3f}", f"{lo:+.3f}", f"{hi:+.3f}", verdict])
    print(f"\nTable 2 with significance marks: {out}/table2_paper.csv (all differences: table2_vs_final.csv)")


def forest(res, cmp, out):
    """Difference from the final model with 95% CI, whole aorta."""
    panels = [(("level r", "whole"), "level r, whole aorta"), (("spatial r", "whole"), "spatial r (median), whole aorta"),
              (("PWV MAE", "-"), "PWV MAE (m/s); lower is better")]
    rows = [r for r in res if r[1] in cmp]
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 0.28 * n + 1.2), sharey=True)
    for ax, (k, title) in zip(axes, panels):
        for y, r in enumerate(rows[::-1]):
            if k not in cmp[r[1]]:
                continue
            _, diff, lo, hi, verdict = cmp[r[1]][k]
            sig = verdict != ""
            ax.plot([lo, hi], [y, y], color=NEUTRAL if sig else MUTED, linewidth=1.2, solid_capstyle="round")
            ax.plot(diff, y, "o", markersize=4.5, color=NEUTRAL if sig else "white",
                    markeredgecolor=NEUTRAL if sig else MUTED, markeredgewidth=1)
        groups = [r[0] for r in rows][::-1]
        for i in range(1, n):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color=NEUTRAL_FILL, linewidth=0.8)
        ax.axvline(0, color=ACCENT, linewidth=1)
        ax.set_title(title, color=INK)
        ax.set_xlabel(f"method minus {REFERENCE}")
        ax.grid(axis="x", color="#ecebe8", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(n), [r[1] for r in rows][::-1])
    axes[0].set_ylim(-0.6, n - 0.4)
    fig.text(0.01, -0.02, "Dot = difference, line = paired bootstrap 95% CI; filled = CI excludes 0. "
             "Blue line = no difference from the final model.", fontsize=7, color=NEUTRAL)
    save(fig, out, "forest_vs_final")


def final_figures(label, s, run, out):
    """Scatter, PWV and example maps for the final model."""
    true, pred, pwv_true, pwv_pred = run
    # predicted vs true median WSS
    fig, axes = plt.subplots(1, len(REGION_ROWS), figsize=(13, 2.9), gridspec_kw={"wspace": 0.35})
    for ax, (key, title, _) in zip(axes, REGION_ROWS):
        t, p = s[key]["level"]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linestyle="--", linewidth=0.8)
        ax.scatter(t, p, s=10, color=ACCENT, alpha=0.75, linewidths=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_title(title, color=INK)
        ax.text(0.04, 0.96, f"r = {s[key]['level_r']:.2f}", transform=ax.transAxes, va="top", color=INK)
        ax.set_xlabel("measured median WSS (Pa)")
    axes[0].set_ylabel("predicted median WSS (Pa)")
    fig.suptitle(f"{label}: one dot per patient; dashed = identity", x=0.01, ha="left", color=INK, fontsize=9)
    save(fig, out, "final_scatter")

    if pwv_pred is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"wspace": 0.4})
        lo, hi = min(pwv_true.min(), pwv_pred.min()), max(pwv_true.max(), pwv_pred.max())
        a1.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=0.8)
        a1.scatter(pwv_true, pwv_pred, s=10, color=ACCENT, alpha=0.75, linewidths=0)
        a1.set_aspect("equal")
        a1.set_xlabel("measured PWV (m/s)")
        a1.set_ylabel("predicted PWV (m/s)")
        a1.text(0.04, 0.96, f"r = {s['pwv'][0]:.2f}\nMAE = {s['pwv'][1]:.2f} m/s", transform=a1.transAxes,
                va="top", color=INK)
        d = pwv_pred - pwv_true
        bias, sd = d.mean(), d.std()
        a2.axhline(0, color=NEUTRAL_FILL, linewidth=0.8)
        a2.scatter((pwv_true + pwv_pred) / 2, d, s=10, color=NEUTRAL, alpha=0.75, linewidths=0)
        a2.axhline(bias, color=ACCENT, linewidth=1.2)
        for v in (bias - 1.96 * sd, bias + 1.96 * sd):
            a2.axhline(v, color=MUTED, linestyle="--", linewidth=0.8)
        for v in (bias + 1.96 * sd, bias, bias - 1.96 * sd):
            a2.annotate(f"{v:+.2f}", (1, v), xycoords=("axes fraction", "data"), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=6.5,
                        color=ACCENT if v == bias else NEUTRAL, annotation_clip=False)
        a2.set_xlabel("mean of measured and predicted PWV (m/s)")
        a2.set_ylabel("predicted - measured (m/s)")
        fig.suptitle(f"{label}: pulse wave velocity", x=0.01, ha="left", color=INK, fontsize=9)
        save(fig, out, "final_pwv")

    # example maps: patients at the 10th / 50th / 90th percentile of whole-aorta spatial r
    sp = s["whole"]["spatial"]
    order = np.argsort(sp)
    picks = [(q, order[int(round(q / 100 * (len(sp) - 1)))]) for q in (90, 50, 10)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 5.6), gridspec_kw={"wspace": 0.12, "hspace": 0.55})
    bounds = [REGIONS[k].stop + 0.5 for k in ("pasc", "arch", "desc")]
    for row, (q, i) in enumerate(picks):
        vmax = np.percentile(np.concatenate([true[i].ravel(), pred[i].ravel()]), 99)
        dmax = np.percentile(np.abs(pred[i] - true[i]), 99)
        for col, (img, name, cmap, lim) in enumerate([
                (true[i], "measured", "viridis", (0, vmax)), (pred[i], "predicted", "viridis", (0, vmax)),
                (pred[i] - true[i], "predicted - measured", "RdBu_r", (-dmax, dmax))]):
            ax = axes[row, col]
            im = ax.imshow(img, aspect="auto", origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1],
                           extent=(0.5, 128.5, 0.5, 32.5), interpolation="nearest")
            for b in bounds:
                ax.axvline(b, color="white" if cmap == "viridis" else MUTED, linewidth=0.8)
            ax.set_yticks([1, 16, 32])
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("around (point)")
            ax.set_title(f"{name}" + (f"  —  {q}th percentile of spatial r (r = {sp[i]:.2f})" if col == 0 else ""),
                         loc="left", color=INK, fontsize=8)
            ax.set_xticks([(c.start + c.stop) / 2 + 0.5 for _, _, c in REGION_ROWS[1:]],
                          ["Prox. asc.", "Arch", "Desc.", "Abdominal"])
            ax.tick_params(axis="x", length=0)
            if row == 2:
                ax.set_xlabel("along the centreline, root to distal")
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("Pa", fontsize=6)
    fig.suptitle(f"{label}: WSS wall maps (unwrapped; lines = region boundaries)", x=0.01, ha="left",
                 color=INK, fontsize=9, y=1.0)
    save(fig, out, "final_maps")
    print("Final-model figures: final_scatter, final_pwv, final_maps "
          f"(patients at the 90th / 50th / 10th percentile of spatial r: rows {[int(i) for _, i in picks]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="method labels to include (default: all in METHODS)")
    ap.add_argument("--bins", type=int, default=4, help="Dice area = top / bottom 1/bins of wall points")
    ap.add_argument("--out", default="../experiments/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    methods = [m for m in METHODS if not args.only or m[1] in args.only]
    res, final_run = [], None
    for g, label, spec in methods:
        try:
            run = load_run(spec)
        except (OSError, ValueError, SystemExit) as e:
            print(f"skipped {label} ({spec}): {e}")
            continue
        res.append((g, label, spec, score(*run, args.bins, ranks=not spec.startswith("meanmap"))))
        if g == "Final" and (final_run is None or label == "Final (LOO)"):
            final_run = (label, res[-1][3], run)
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
    cmp = versus_final(res)
    paper_table(res, cmp, args.out)
    forest(res, cmp, args.out)
    if final_run is not None:
        final_figures(*final_run, args.out)
    print(f"\nFigures written to {args.out}")


if __name__ == "__main__":
    main()
