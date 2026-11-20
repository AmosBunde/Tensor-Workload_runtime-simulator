"""Every figure in the README and the report, drawn from the committed result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluate import load_predictions  # noqa: E402

BLUE, RED, GREY = "#2F5D9B", "#BA3D1A", "#5A6678"
INK = "#1F2430"
# (key, label, colour, filled). The simulator appears twice in the same blue: hollow as
# calibrated from microbenchmarks alone, filled with per-class corrections fitted.
SERIES = (("sim", "Simulator, microbenchmarks only", BLUE, False),
          ("simfit", "Simulator, corrections fitted", BLUE, True),
          ("gbm", "Gradient boosting", RED, True),
          ("loglinear", "Log-linear regression", GREY, True))
DESIGN_LABELS = {"random": "Random\nworkloads", "size": "Small to\nlarge",
                 "family": "Unseen\narchitecture", "threads": "Unseen\nthread count"}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "text.color": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.color": "#D9DDE3", "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _grouped_bars(ax, groups: list[str], values: dict[str, list[float]],
                  intervals: dict[str, list[list[float]]] | None = None,
                  cap: float = 150.0) -> bool:
    """Grouped bars of percentages. Bars above ``cap`` are cut and labelled with their value.

    Returns whether any bar was cut, so the caller can say so in the caption.
    """
    width = 0.2
    x = np.arange(len(groups))
    top, cut = 0.0, False
    for k, (key, label, colour, filled) in enumerate(SERIES):
        v = np.array(values[key]) * 100
        shown = np.minimum(v, cap)
        pos = x + (k - 1.5) * width
        ax.bar(pos, shown, width * 0.86, color=colour if filled else "white",
               edgecolor=colour, linewidth=1.4, label=label)
        tops = shown.copy()
        if intervals:
            ci = np.minimum(np.array(intervals[key]) * 100, cap)
            for xi, vi, (lo, hi) in zip(pos, v, ci):
                if vi <= cap:
                    ax.plot([xi, xi], [lo, hi], color=INK, linewidth=0.9)
            tops = np.where(v <= cap, ci[:, 1], shown)
        for xi, vi, ti in zip(pos, v, tops):
            ax.text(xi, ti + 1.0, f"{vi:.0f}", ha="center", va="bottom", fontsize=7.5)
        cut = cut or bool(np.any(v > cap))
        top = max(top, float(np.max(tops)))
    ax.set_xticks(x, groups)
    ax.set_ylim(0, top * 1.36 + 2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean absolute percentage error, %")
    return cut


def fig_headline(summary: dict, out_dir: Path) -> Path:
    designs = list(DESIGN_LABELS)
    values = {k: [summary["designs"][d]["methods"][k]["mape"] for d in designs]
              for k, *_ in SERIES}
    cis = {k: [summary["designs"][d]["methods"][k]["mape_ci"] for d in designs]
           for k, *_ in SERIES}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    cut = _grouped_bars(ax, [DESIGN_LABELS[d] for d in designs], values, cis)
    ax.set_title("Prediction error under four ways of holding out test cells")
    ax.legend(loc="upper left", ncols=2, fontsize=8.5)
    note = ("Bars: error on test cells. Lines: 95% bootstrap interval over workloads. "
            "Lower is better.")
    if cut:
        note += " Bars above 150% are cut; the label gives the value."
    ax.text(0, -0.24, note, transform=ax.transAxes, fontsize=8, color=GREY)
    return _save(fig, out_dir, "headline.png")


def fig_family(summary: dict, out_dir: Path) -> Path:
    folds = summary["family_folds"]
    names = {"mlp": "MLP", "cnn": "Residual CNN", "transformer": "Transformer",
             "mixer": "MLP-Mixer"}
    fams = [f for f in names if f in folds]
    values = {k: [folds[f][k] for f in fams] for k, *_ in SERIES}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    cut = _grouped_bars(ax, [names[f] for f in fams], values)
    ax.set_title("Error on an architecture family never seen in fitting")
    ax.set_xlabel("Family held out" + (" (bars above 150% are cut; the label gives the "
                                       "value)" if cut else ""))
    ax.legend(loc="upper left", ncols=2, fontsize=8.5)
    return _save(fig, out_dir, "family.png")


def fig_scatter(predictions: list[dict], out_dir: Path) -> Path:
    rows = [r for r in predictions if r["design"] == "size"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True, sharey=True)
    lo = min(min(r["measured"], r["simfit"], r["gbm"]) for r in rows) * 0.6
    hi = max(max(r["measured"], r["simfit"], r["gbm"]) for r in rows) * 1.6
    for ax, (key, label) in zip(axes, (("simfit", "Simulator, corrections fitted"),
                                       ("gbm", "Gradient boosting"))):
        for mode, colour, name in (("forward", BLUE, "Forward pass"),
                                   ("train", RED, "Training step")):
            sub = [r for r in rows if r["mode"] == mode]
            ax.scatter([r["measured"] for r in sub], [r[key] for r in sub], s=9,
                       color=colour, alpha=0.55, linewidths=0, label=name)
        ax.plot([lo, hi], [lo, hi], color=GREY, linewidth=1, label="Perfect prediction")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(label)
        ax.set_xlabel("Measured time, seconds")
    axes[0].set_ylabel("Predicted time, seconds")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Large workloads predicted after fitting on small ones", x=0.07, ha="left",
                 fontsize=11, fontweight="bold", y=1.0)
    return _save(fig, out_dir, "scatter.png")


def fig_extrapolation(summary: dict, out_dir: Path) -> Path:
    bins = summary["size_by_extrapolation"]
    labels = list(bins)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(labels))
    top = 0.0
    for key, label, colour, filled in SERIES:
        v = [bins[b][key] * 100 for b in labels]
        ax.plot(x, v, marker="o", color=colour, label=label, linewidth=2,
                linestyle="-" if filled else "--", markerfacecolor=colour if filled else "white")
        top = max(top, max(v))
    ax.set_xticks(x, [f"{b}\n{bins[b]['workloads']} workloads" for b in labels])
    ax.set_xlim(-0.3, len(labels) - 0.7)
    ax.set_ylim(0, top * 1.45)
    ax.set_xlabel("Test workload FLOPs relative to the largest training workload of its family")
    ax.set_ylabel("Mean absolute percentage error, %")
    ax.set_title("Error against distance beyond the training sizes")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, out_dir, "extrapolation.png")


def _stacked(ax, labels: list[str], shares: list[dict], legend: bool = True) -> None:
    parts = (("Compute", ("compute",), BLUE), ("Memory", ("memory",), RED),
             ("Call overhead and threading loss", ("overhead", "threading"), GREY))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for name, keys, colour in parts:
        v = np.array([100 * sum(s[k] for k in keys) for s in shares])
        ax.barh(y, v, left=left, color=colour, label=name, height=0.62)
        for yi, vi, li in zip(y, v, left):
            if vi >= 9:
                ax.text(li + vi / 2, yi, f"{vi:.0f}%", ha="center", va="center",
                        color="white", fontsize=8)
        left += v
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of predicted time, %")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=9)


def fig_breakdown(whatif: dict, out_dir: Path) -> Path:
    data = whatif["breakdown"]
    labels = list(data["workloads"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    _stacked(ax, labels, [data["workloads"][w]["share"] for w in labels])
    ax.set_title(f"Predicted bottleneck: forward pass, float32, {data['threads']} threads")
    return _save(fig, out_dir, "breakdown.png")


def fig_whatif(whatif: dict, out_dir: Path) -> Path:
    series = (("2x memory bandwidth", BLUE), ("8 cores", RED), ("16 cores", GREY))
    names = {"mlp": "MLP", "cnn": "Residual CNN", "transformer": "Transformer",
             "mixer": "MLP-Mixer"}
    keys = [k for k in whatif["projection"]["workloads"] if k.endswith("/train")]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(keys))
    width = 0.26
    top = 1.0
    for k, (label, colour) in enumerate(series):
        v = [whatif["projection"]["workloads"][w]["scenarios"][label]["speedup_over_host"]
             for w in keys]
        pos = x + (k - 1) * width
        ax.bar(pos, v, width * 0.92, color=colour, label=label)
        for xi, vi in zip(pos, v):
            ax.text(xi, vi + 0.04, f"{vi:.2f}", ha="center", va="bottom", fontsize=8)
        top = max(top, max(v))
    ax.axhline(1.0, color=INK, linewidth=0.8)
    ax.set_xticks(x, [names[w.split("-")[0]] for w in keys])
    ax.set_ylim(0, top * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Projected speed-up over the measured host")
    ax.set_title("Unvalidated projection: training step on devices that were not measured")
    ax.legend(loc="upper left", ncols=3, fontsize=8.5)
    ax.text(0, -0.16, "Largest workload of each family, float32. Host: two cores, two "
            "threads. None of these devices was measured.", transform=ax.transAxes,
            fontsize=8, color=GREY)
    return _save(fig, out_dir, "whatif.png")


def fig_drift(full: dict, out_dir: Path) -> Path:
    table = full["drift"]
    cutoff = full["clean"]["cutoff_minutes"]
    x = [0.5 * (b["from_minute"] + b["to_minute"]) for b in table]
    y = [b["median_ratio"] for b in table]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axvspan(0, cutoff, color=BLUE, alpha=0.12, label="First-pass window kept")
    ax.plot(x, y, marker="o", color=RED, linewidth=2,
            label="Median of first-pass time over second-pass time")
    ax.axhline(1.0, color=GREY, linewidth=1, label="No change")
    ax.set_xlim(0, max(b["to_minute"] for b in table))
    ax.set_ylim(0.5, max(y) * 1.25)
    ax.set_xlabel("Minutes into the first measurement pass")
    ax.set_ylabel("Slowdown relative to the second pass")
    ax.set_title("The machine slowed down part-way through the first pass")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, out_dir, "drift.png")


def draw_all(results: Path, out_dir: Path, log=print) -> list[Path]:
    _style()
    full = json.loads((results / "summary.json").read_text())
    summary = full["clean"]
    predictions = load_predictions(results / "predictions_clean.csv")
    whatif = json.loads((results / "whatif.json").read_text())
    paths = [
        fig_headline(summary, out_dir),
        fig_family(summary, out_dir),
        fig_scatter(predictions, out_dir),
        fig_extrapolation(summary, out_dir),
        fig_breakdown(whatif, out_dir),
        fig_whatif(whatif, out_dir),
        fig_drift(full, out_dir),
    ]
    for p in paths:
        log(f"wrote {p}")
    return paths
