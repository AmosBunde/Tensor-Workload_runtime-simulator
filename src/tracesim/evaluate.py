"""The validation study: four test designs, five predictors, one table of predictions.

Each design splits the measured cells into folds of training and test cells. For every
fold the black-box baselines are fitted on the training cells, the simulator is calibrated
from the microbenchmarks it is allowed to see under that design, and all of them predict
the test cells. The output is one row per (design, test cell) holding the measured time
and every predictor's prediction. Statistics are computed later from that file alone.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .baselines import FlopsProportional, GradientBoosting, LogLinear, SimulatorCorrections
from .calibrate import fit_descriptor
from .device import DeviceDescriptor
from .microbench import load_points
from .simulator import class_times
from .trace import Trace, cached_trace
from .zoo import FAMILIES, grid

METHODS = ("sim", "simfit", "flops", "gbm", "loglinear")
DESIGNS = ("random", "size", "family", "threads")
COLUMNS = ["design", "fold", "workload", "family", "mode", "dtype", "threads", "flops",
           "extrapolation", "measured", "sim_full"] + list(METHODS)


def load_cells(rows: list[dict], cache_dir: Path | None, log=print) -> list[dict]:
    """Measured cells (one row each) joined with their traces and trace features."""
    workloads = {w.id: w for w in grid()}
    traces: dict[tuple, Trace] = {}
    cells = []
    for r in rows:
        key = (r["workload"], r["mode"], r["dtype"])
        if key not in traces:
            traces[key] = cached_trace(workloads[r["workload"]], r["mode"], r["dtype"],
                                       cache_dir)
            if len(traces) % 100 == 0:
                log(f"  {len(traces)} traces captured")
        cells.append({**r, "trace": traces[key], "features": traces[key].features()})
    sizes: dict[str, float] = {}
    for c in cells:
        # Size is the forward-pass FLOP count, whichever cells happen to be present.
        if c["workload"] not in sizes:
            sizes[c["workload"]] = cached_trace(workloads[c["workload"]], "forward",
                                                "float32", cache_dir).flops
        c["size"] = sizes[c["workload"]]
    return cells


def _workload_ids(cells: list[dict]) -> list[str]:
    return sorted({c["workload"] for c in cells})


def folds_random(cells: list[dict], k: int = 5, seed: int = 0):
    ids = _workload_ids(cells)
    random.Random(seed).shuffle(ids)
    for i in range(k):
        test = set(ids[i::k])
        yield (f"fold{i}", [c for c in cells if c["workload"] not in test],
               [c for c in cells if c["workload"] in test], {})


def size_split(cells: list[dict]) -> tuple[set[str], set[str]]:
    """Per family: the smaller half of workloads by forward FLOPs, and the larger half."""
    small: set[str] = set()
    large: set[str] = set()
    for fam in FAMILIES:
        sizes = sorted({(c["size"], c["workload"]) for c in cells if c["family"] == fam})
        half = len(sizes) // 2
        small |= {w for _, w in sizes[:half]}
        large |= {w for _, w in sizes[half:]}
    return small, large


def folds_size(cells: list[dict]):
    small, large = size_split(cells)
    train = [c for c in cells if c["workload"] in small]
    test = [c for c in cells if c["workload"] in large]
    cap: dict[str, float] = {}
    for c in train:
        for op in c["trace"].ops:
            cap[op.op_class] = max(cap.get(op.op_class, 0.0), op.work)
    yield "small-to-large", train, test, {"max_work": cap}


def folds_family(cells: list[dict]):
    for fam in FAMILIES:
        yield (fam, [c for c in cells if c["family"] != fam],
               [c for c in cells if c["family"] == fam], {})


def folds_threads(cells: list[dict], held_out: int = 4):
    yield (f"to-{held_out}-threads", [c for c in cells if c["threads"] < held_out],
           [c for c in cells if c["threads"] == held_out], {"max_threads": held_out // 2})


FOLDS = {"random": folds_random, "size": folds_size, "family": folds_family,
         "threads": folds_threads}


def _class_times(cells: list[dict], device: DeviceDescriptor) -> list[dict[str, float]]:
    return [class_times(c["trace"], device, c["threads"]) for c in cells]


def run_design(design: str, cells: list[dict], points: list[dict], full: DeviceDescriptor,
               log=print) -> list[dict]:
    rows: list[dict] = []
    # A restricted descriptor must describe the same cores as the full one. Reading the
    # topology of whatever process runs the analysis would count cores that the
    # measurements were never pinned to.
    info = {"cores": full.cores, "smt": full.smt, "llc": full.cache_bytes,
            "l1d": full.l1d_bytes, "l2": full.l2_bytes, "simd": full.simd, "cpu": full.notes}
    for fold, train, test, restriction in FOLDS[design](cells):
        device = fit_descriptor(points, info=info, **restriction) if restriction else full
        train_ct, test_ct = _class_times(train, device), _class_times(test, device)
        measured = [c["t"] for c in train]
        preds = {
            "sim": [sum(ct.values()) for ct in test_ct],
            "sim_full": [sum(ct.values()) for ct in _class_times(test, full)],
            "simfit": SimulatorCorrections().fit(train_ct, measured).predict(test_ct),
            "flops": FlopsProportional().fit(train).predict(test),
            "gbm": GradientBoosting().fit(train).predict(test),
            "loglinear": LogLinear().fit(train).predict(test),
        }
        largest: dict[str, float] = {}
        for c in train:
            largest[c["family"]] = max(largest.get(c["family"], 0.0), c["size"])
        for i, c in enumerate(test):
            row = {"design": design, "fold": fold, "workload": c["workload"],
                   "family": c["family"], "mode": c["mode"], "dtype": c["dtype"],
                   "threads": c["threads"], "flops": c["features"]["flops"],
                   "extrapolation": (c["size"] / largest[c["family"]]
                                     if c["family"] in largest else ""),
                   "measured": c["t"]}
            for m, p in preds.items():
                row[m] = float(p[i])
            rows.append(row)
        log(f"  {design}/{fold}: {len(train)} training cells, {len(test)} test cells")
    return rows


def run(measurements: list[dict], microbench: Path, out: Path, device_path: Path,
        cache_dir: Path | None, log=print) -> list[dict]:
    """Run every design on ``measurements`` (one row per cell) and write the predictions."""
    points = load_points(microbench)
    full = DeviceDescriptor.load(device_path)
    log(f"loading {len(measurements)} cells and their traces")
    cells = load_cells(measurements, cache_dir, log)
    rows: list[dict] = []
    for design in DESIGNS:
        rows += run_design(design, cells, points, full, log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                             for k, v in r.items()})
    log(f"wrote {len(rows)} rows to {out}")
    return rows


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            for k in ("flops", "measured", "sim_full") + METHODS:
                r[k] = float(r[k])
            r["threads"] = int(r["threads"])
            r["extrapolation"] = float(r["extrapolation"]) if r["extrapolation"] else None
            rows.append(r)
    return rows


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
