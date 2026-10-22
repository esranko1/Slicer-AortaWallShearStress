"""
Shared helpers for stitch.py and compare_runs.py: load a run's held-out
predictions, build the four-region composite, and score both Pearson metrics.

A run is named by a spec string  "<model>:<cv>:<flow>[:seed]"
    model  composite (the four specialists stitched) | whole (single model)
           | hybrid (the pasc specialist on the proximal ascending aorta, the
             whole-vessel model everywhere else; PWV from the whole model)
           | meanmap | poiseuille | flowradius (baselines.py; cv10 only, flow ignored)
    cv     cv10 | loo
    flow   flow | noflow
    seed   optional, default 2024
e.g. "composite:loo:flow", "whole:cv10:noflow", "composite:cv10:flow:7".

Patients are put in one canonical order (sorted by their true WSS), so runs
with different CV schemes - whose pooled arrays come out in different patient
orders - can still be paired patient by patient.
"""

import numpy as np

REGIONS = {
    "pasc": slice(0, 36),
    "arch": slice(36, 60),
    "desc": slice(60, 96),
    "abda": slice(96, 128),
}
WHOLE = slice(0, 128)


BASELINES = ("meanmap", "poiseuille", "flowradius")


def results_path(region, cv, flow, seed):
    if region in BASELINES:
        return f"../experiments/15_Baselines_{cv}/{region}/results.npz"
    suffix = f"_seed{seed}" if seed != 2024 else ""
    return f"../experiments/14_Final_{region}_{cv}_{flow}{suffix}/results.npz"


def parse_spec(spec):
    parts = spec.split(":")
    if len(parts) not in (3, 4) or parts[0] not in ("composite", "whole", "hybrid") + BASELINES \
            or parts[1] not in ("cv10", "loo") or parts[2] not in ("flow", "noflow"):
        raise SystemExit(f"Bad run spec '{spec}'; expected <composite|whole>:<cv10|loo>:<flow|noflow>[:seed]")
    return parts[0], parts[1], parts[2], int(parts[3]) if len(parts) == 4 else 2024


def _order(true):
    """Patient order that sorts by true WSS, so different runs line up."""
    return np.lexsort(true[:, :8].T[::-1])


def _pwv(d, order):
    if "all_pwv_true" not in d.files:
        return None, None
    return d["all_pwv_true"][order], d["all_pwv_pred"][order]


def load_run(spec):
    """
    -> (true, pred, pwv_true, pwv_pred): WSS (n_patients, 32, 128) and PWV (n,)
    in canonical patient order. PWV is None if the run had no PWV output; for the
    composite it is the mean of the four specialists' PWV predictions.
    """
    model, cv, flow, seed = parse_spec(spec)
    if model == "whole" or model in BASELINES:
        d = np.load(results_path(model, cv, flow, seed))
        o = _order(d["all_true"])
        pt, pp = _pwv(d, o)
        return d["all_true"][o].reshape(-1, 32, 128), d["all_pred"][o].reshape(-1, 32, 128), pt, pp

    if model == "hybrid":
        w = np.load(results_path("whole", cv, flow, seed))
        p = np.load(results_path("pasc", cv, flow, seed))
        ow, op = _order(w["all_true"]), _order(p["all_true"])
        true = w["all_true"][ow].reshape(-1, 32, 128)
        if not np.allclose(p["all_true"][op].reshape(-1, 32, 128), true):
            raise ValueError(f"{spec}: pasc and whole runs have different patients")
        pred = w["all_pred"][ow].reshape(-1, 32, 128).copy()
        pred[:, :, REGIONS["pasc"]] = p["all_pred"][op].reshape(-1, 32, 128)[:, :, REGIONS["pasc"]]
        pt, pp = _pwv(w, ow)
        return true, pred, pt, pp

    data = {r: np.load(results_path(r, cv, flow, seed)) for r in REGIONS}
    true_ref = data["pasc"]["all_true"][_order(data["pasc"]["all_true"])]
    pred = np.zeros((true_ref.shape[0], 32, 128))
    pwv_true, pwv_preds = None, []
    for r, col in REGIONS.items():
        o = _order(data[r]["all_true"])
        if not np.allclose(data[r]["all_true"][o], true_ref):
            raise ValueError(f"{spec}: region {r} has different patients from pasc")
        pred[:, :, col] = data[r]["all_pred"][o].reshape(-1, 32, 128)[:, :, col]
        pt, pp = _pwv(data[r], o)
        if pp is not None:
            pwv_true = pt
            pwv_preds.append(pp)
    pwv_pred = np.mean(pwv_preds, axis=0) if len(pwv_preds) == len(REGIONS) else None
    return true_ref.reshape(-1, 32, 128), pred, (pwv_true if pwv_pred is not None else None), pwv_pred


def level_values(true, pred, col):
    """Each patient's median WSS in the region: (n,), (n,)."""
    n = true.shape[0]
    return (np.median(true[:, :, col].reshape(n, -1), axis=1),
            np.median(pred[:, :, col].reshape(n, -1), axis=1))


def r_level(true, pred, col):
    """Metric used so far: correlation across patients of each patient's median WSS."""
    t, p = level_values(true, pred, col)
    return float(np.corrcoef(t, p)[0, 1])


def r_spatial_per_patient(true, pred, col):
    """Per-patient Pearson across the wall points of the region: (n,)."""
    n = true.shape[0]
    t = true[:, :, col].reshape(n, -1)
    p = pred[:, :, col].reshape(n, -1)
    t = t - t.mean(axis=1, keepdims=True)
    p = p - p.mean(axis=1, keepdims=True)
    return (t * p).sum(axis=1) / (np.sqrt((t ** 2).sum(axis=1) * (p ** 2).sum(axis=1)) + 1e-12)


def region_points(true, pred, col):
    """(n, k) true and predicted WSS at the region's wall points."""
    n = true.shape[0]
    return true[:, :, col].reshape(n, -1), pred[:, :, col].reshape(n, -1)


def mae_per_patient(true, pred, col):
    """Mean absolute error over the region's wall points, per patient: (n,), in label units (Pa)."""
    t, p = region_points(true, pred, col)
    return np.abs(p - t).mean(axis=1)


def nmae_per_patient(true, pred, col):
    """MAE divided by that patient's true WSS range in the region (max - min): (n,), a fraction."""
    t, p = region_points(true, pred, col)
    return np.abs(p - t).mean(axis=1) / (t.max(axis=1) - t.min(axis=1) + 1e-12)


def pooled_spearman(true, pred, col):
    """The PI notebook's regional metric: Spearman over all wall points of all patients pooled."""
    from scipy.stats import spearmanr
    t, p = region_points(true, pred, col)
    return float(spearmanr(t.ravel(), p.ravel())[0])


def level_bland_altman(true, pred, col):
    """Bias and 95% limits of agreement of each patient's median regional WSS (pred - true)."""
    t, p = level_values(true, pred, col)
    d = p - t
    return float(d.mean()), float(d.mean() - 1.96 * d.std()), float(d.mean() + 1.96 * d.std())
