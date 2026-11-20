"""Statistics for the validation study. Everything quoted in the documents comes from here.

The same statistics are computed twice: on all cells as planned, and on the cells measured
while the machine ran at full speed (see ``measure.clean_view``).

Intervals are 95% percentile bootstrap intervals that resample workloads, not cells: the
twelve cells of one workload share a model and are not independent observations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .evaluate import DESIGNS, METHODS, load_predictions

ALL_METHODS = METHODS + ("sim_full",)
BOOT = 2000
BOOT_RANK = 500
EXTRAPOLATION_BINS = ((1.0, 3.0), (3.0, 10.0), (10.0, float("inf")))


def ape(pred: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return np.abs(pred / measured - 1.0)


def mape(pred, measured) -> float:
    return float(np.mean(ape(np.asarray(pred, float), np.asarray(measured, float))))


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    rho = spearmanr(a, b).statistic
    return float(rho)


def _interval(samples) -> list[float]:
    lo, hi = np.nanpercentile(np.asarray(samples, float), [2.5, 97.5])
    return [float(lo), float(hi)]


def _groups(rows: list[dict]) -> list[np.ndarray]:
    by: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["workload"]].append(i)
    return [np.array(v) for _, v in sorted(by.items())]


def cluster_bootstrap(rows: list[dict], stat, n: int = BOOT, seed: int = 0) -> list[float]:
    """Bootstrap ``stat(index array)`` by resampling whole workloads with replacement."""
    groups = _groups(rows)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(stat(np.concatenate([groups[g] for g in pick])))
    return out


def method_stats(rows: list[dict], method: str) -> dict:
    measured = np.array([r["measured"] for r in rows])
    pred = np.array([r[method] for r in rows])
    err = ape(pred, measured)
    out = {
        "mape": float(err.mean()),
        "mape_ci": _interval(cluster_bootstrap(rows, lambda i: err[i].mean())),
        "median_ape": float(np.median(err)),
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(pred / measured)))),
        "spearman": spearman(pred, measured),
        "spearman_ci": _interval(cluster_bootstrap(
            rows, lambda i: spearman(pred[i], measured[i]), BOOT_RANK)),
        "within_workload_spearman": within_workload_spearman(rows, method),
    }
    return out


def paired_difference(rows: list[dict], a: str, b: str) -> dict:
    """MAPE of ``a`` minus MAPE of ``b`` on the same cells. Negative favours ``a``."""
    measured = np.array([r["measured"] for r in rows])
    d = ape(np.array([r[a] for r in rows]), measured) - ape(
        np.array([r[b] for r in rows]), measured)
    return {"difference": float(d.mean()),
            "ci": _interval(cluster_bootstrap(rows, lambda i: d[i].mean()))}


def within_workload_spearman(rows: list[dict], method: str) -> float | None:
    """Mean rank correlation across device configurations, within workload and mode."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["workload"], r["mode"])].append(r)
    vals = []
    for group in by.values():
        if len(group) >= 4:
            rho = spearman([g[method] for g in group], [g["measured"] for g in group])
            if not np.isnan(rho):
                vals.append(rho)
    return float(np.mean(vals)) if vals else None


def decision_quality(rows: list[dict], method: str) -> dict | None:
    """Choose the thread count predicted fastest at float32; score it by measured time."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["dtype"] == "float32":
            by[(r["workload"], r["mode"])].append(r)
    regrets, top1 = [], []
    for group in by.values():
        if len(group) < 3:
            continue
        chosen = min(group, key=lambda g: g[method])
        best = min(group, key=lambda g: g["measured"])
        regrets.append(chosen["measured"] / best["measured"] - 1.0)
        top1.append(chosen["threads"] == best["threads"])
    if not regrets:
        return None
    regrets = np.array(regrets)
    return {"choices": int(len(regrets)), "top1": float(np.mean(top1)),
            "mean_regret": float(regrets.mean()), "max_regret": float(regrets.max()),
            "within_10pct": float(np.mean(regrets <= 0.10))}


def _subset_mape(rows: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    for value in sorted({str(r[key]) for r in rows}):
        sub = [r for r in rows if str(r[key]) == value]
        out[value] = {"cells": len(sub),
                      **{m: mape([r[m] for r in sub], [r["measured"] for r in sub])
                         for m in ALL_METHODS}}
    return out


def noise_floor(measurements: list[dict]) -> dict:
    main = {(r["workload"], r["mode"], r["dtype"], r["threads"]): r for r in measurements
            if r["session"] == "main"}
    rep = [r for r in measurements if r["session"] == "repeat"]
    diffs = np.array([abs(r["t"] / main[(r["workload"], r["mode"], r["dtype"], r["threads"])]
                          ["t"] - 1.0) for r in rep
                      if (r["workload"], r["mode"], r["dtype"], r["threads"]) in main])
    rows = list(main.values())
    iqr = np.array([(r["q3"] - r["q1"]) / r["t"] for r in rows])
    out = {
        "cells": len(rows),
        "repeat_cells": int(len(diffs)),
        "median_relative_iqr": float(np.median(iqr)),
        "cells_measured_while_busy": int(sum(r["other_cpus"] > 0.4 for r in rows)),
        "cells_retried": int(sum(r["attempts"] > 1 for r in rows)),
        "measurement_minutes": float(sum(r["wall"] for r in measurements) / 60.0),
        "fastest_cell_s": float(min(r["t"] for r in rows)),
        "slowest_cell_s": float(max(r["t"] for r in rows)),
    }
    if len(diffs):
        out["repeat_mape"] = float(diffs.mean())
        out["repeat_median_ape"] = float(np.median(diffs))
        out["repeat_p90_ape"] = float(np.percentile(diffs, 90))
        by_threads = defaultdict(list)
        for r in rep:
            key = (r["workload"], r["mode"], r["dtype"], r["threads"])
            if key in main:
                by_threads[r["threads"]].append(abs(r["t"] / main[key]["t"] - 1.0))
        out["repeat_mape_by_threads"] = {str(k): float(np.mean(v))
                                         for k, v in sorted(by_threads.items())}
    return out


def hypotheses(s: dict) -> dict:
    d = s["designs"]
    out: dict[str, dict] = {}

    def m(design, method):
        return d[design]["methods"][method]["mape"]

    diff = d["random"]["paired"]["sim_minus_gbm"]
    out["H1"] = {"sim_mape": m("random", "sim"), "gbm_mape": m("random", "gbm"),
                 "difference": diff, "supported": bool(diff["ci"][0] > 0)}
    diff = d["size"]["paired"]["sim_minus_gbm"]
    out["H2"] = {"sim_mape": m("size", "sim"), "gbm_mape": m("size", "gbm"),
                 "loglinear_mape": m("size", "loglinear"), "difference": diff,
                 "difference_vs_loglinear": d["size"]["paired"]["sim_minus_loglinear"],
                 "supported": bool(diff["ci"][1] < 0)}
    wins = [f for f, v in s["family_folds"].items() if v["sim"] < v["gbm"]]
    out["H3"] = {"families_where_sim_beats_gbm": wins, "supported": len(wins) >= 3}
    out["H4"] = {"sim_mape": m("threads", "sim"), "gbm_mape": m("threads", "gbm"),
                 "loglinear_mape": m("threads", "loglinear"),
                 "supported": bool(m("threads", "sim") < min(m("threads", "gbm"),
                                                           m("threads", "loglinear")))}
    dq = s["decision"]["random"]["sim"]
    out["H5"] = {"within_10pct": dq["within_10pct"], "choices": dq["choices"],
                 "supported": bool(dq["within_10pct"] >= 0.90)}
    worst = {design: max(METHODS, key=lambda k: m(design, k)) for design in DESIGNS}
    out["H6"] = {"worst_method_by_design": worst, "flops_mape_random": m("random", "flops"),
                 "supported": bool(all(v == "flops" for v in worst.values())
                                   and m("random", "flops") > 0.5)}
    out["H7"] = {"sim_mape_all_cells": m("random", "sim"),
                 "supported": bool(m("random", "sim") < 0.30)}
    return out


def clean_noise(pairs: list[tuple[dict, dict]], cells: int) -> dict:
    """Noise floor of the clean-window analysis, from cells measured in both windows."""
    d = np.array([abs(a["t"] / b["t"] - 1.0) for a, b in pairs])
    return {"cells": cells, "repeat_cells": int(len(d)), "repeat_mape": float(d.mean()),
            "repeat_median_ape": float(np.median(d)),
            "repeat_p90_ape": float(np.percentile(d, 90))}


def rescaled_simulator(predictions: Path, factor: float) -> dict:
    """Exploratory: the simulator with every prediction divided by one constant.

    The constant is how much slower operators ran during calibration than in a later
    recheck. It comes from operator timings alone, not from any measured workload.
    """
    rows = load_predictions(predictions)
    for r in rows:
        r["sim_rescaled"] = r["sim"] / factor
    out = {}
    for design in DESIGNS:
        sub = [r for r in rows if r["design"] == design]
        stats_ = method_stats(sub, "sim_rescaled")
        out[design] = {"mape": stats_["mape"], "mape_ci": stats_["mape_ci"],
                       "median_ape": stats_["median_ape"],
                       "geometric_mean_ratio": stats_["geometric_mean_ratio"],
                       "minus_gbm": paired_difference(sub, "sim_rescaled", "gbm"),
                       "minus_loglinear": paired_difference(sub, "sim_rescaled", "loglinear")}
    return out


def summarise(predictions: Path) -> dict:
    rows = load_predictions(predictions)
    s: dict = {"designs": {}, "decision": {}}
    for design in DESIGNS:
        sub = [r for r in rows if r["design"] == design]
        s["designs"][design] = {
            "test_cells": len(sub),
            "test_workloads": len({r["workload"] for r in sub}),
            "methods": {m: method_stats(sub, m) for m in ALL_METHODS},
            "paired": {
                "sim_minus_gbm": paired_difference(sub, "sim", "gbm"),
                "sim_minus_loglinear": paired_difference(sub, "sim", "loglinear"),
                "simfit_minus_gbm": paired_difference(sub, "simfit", "gbm"),
                "simfit_minus_sim": paired_difference(sub, "simfit", "sim"),
                "loglinear_minus_gbm": paired_difference(sub, "loglinear", "gbm"),
            },
        }
        if design != "threads":
            s["decision"][design] = {m: decision_quality(sub, m) for m in ALL_METHODS}
    fam = [r for r in rows if r["design"] == "family"]
    s["family_folds"] = _subset_mape(fam, "fold")
    rand = [r for r in rows if r["design"] == "random"]
    s["random_by_family"] = _subset_mape(rand, "family")
    s["random_by_mode"] = _subset_mape(rand, "mode")
    s["random_by_threads"] = _subset_mape(rand, "threads")
    s["random_by_dtype"] = _subset_mape(rand, "dtype")
    size = [r for r in rows if r["design"] == "size"]
    s["size_by_mode"] = _subset_mape(size, "mode")
    s["size_by_family"] = _subset_mape(size, "family")
    bins = {}
    for lo, hi in EXTRAPOLATION_BINS:
        sub = [r for r in size if lo <= r["extrapolation"] < hi]
        if sub:
            label = f"{lo:g}x to {hi:g}x" if hi != float("inf") else f"over {lo:g}x"
            bins[label] = {"cells": len(sub),
                           "workloads": len({r["workload"] for r in sub}),
                           **{m: mape([r[m] for r in sub], [r["measured"] for r in sub])
                              for m in ALL_METHODS}}
    s["size_by_extrapolation"] = bins
    s["size_extrapolation_range"] = [min(r["extrapolation"] for r in size),
                                     max(r["extrapolation"] for r in size)]
    s["hypotheses"] = hypotheses(s)
    return s


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
