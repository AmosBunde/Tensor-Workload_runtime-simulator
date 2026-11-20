"""Measure real runtimes of the workload grid on this machine.

A cell is one workload in one mode (forward pass or training step) on one device
configuration (thread count and dtype). Cells are visited in a fixed random order, so
that slow drift in machine load or temperature is not confounded with workload size, and
results are appended to a JSON-lines file as they arrive so that a run can be stopped and
resumed. A second pass re-measures a random subset of cells; the disagreement between the
two passes is the noise floor reported with every prediction error. Each cell waits for
the rest of the machine to be quiet and is repeated if it was not (see ``quiet.py``).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from .quiet import pin_to_cores, run_quiet
from .reference import ReferenceLog
from .timing import Timing, time_callable
from .zoo import TrainStep, Workload, grid

MODES = ("forward", "train")
DTYPES = ("float32", "float64")
THREADS = (1, 2, 4)
QUIET_LIMIT = 0.4  # how many of our CPUs other processes may keep busy meanwhile
ATTEMPTS = 3
TIMER = dict(min_reps=5, max_reps=30, min_time=0.15, max_time=3.0, warmup=2, floor_reps=3)


def all_cells() -> list[tuple[str, str, str, int]]:
    return [(w.id, m, d, n) for w in grid() for m in MODES for d in DTYPES for n in THREADS]


def measure_cell(workload: Workload, mode: str, dtype: str, threads: int, timer: dict | None
                 = None) -> Timing:
    torch.set_num_threads(threads)
    torch.set_flush_denormal(True)
    td = getattr(torch, dtype)
    model = workload.build(td)
    x, y = workload.inputs(td)
    timer = timer or TIMER
    if mode == "forward":
        model.eval()
        with torch.no_grad():
            return time_callable(lambda: model(x), **timer)
    step_module = TrainStep(model).train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        step_module(x, y).backward()
        opt.step()

    return time_callable(step, **timer)


def load_measurements(path: Path, session: str | None = "main") -> list[dict]:
    if not Path(path).exists():
        return []
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if session is None or r["session"] == session]


def run(path: Path, repeat_fraction: float = 0.2, max_minutes: float | None = None,
        only: list[str] | None = None, seed: int = 0, log=print,
        reference: Path | None = None) -> None:
    """Measure every cell once, then a random subset a second time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    workloads = {w.id: w for w in grid()}
    cells = [c for c in all_cells() if only is None or c[0] in only]
    rng = random.Random(seed)
    main = cells[:]
    rng.shuffle(main)
    repeat = rng.sample(cells, int(round(repeat_fraction * len(cells))))
    plan = [("main", c) for c in main] + [("repeat", c) for c in repeat]
    done = {(r["session"], r["workload"], r["mode"], r["dtype"], r["threads"])
            for r in load_measurements(path, None)}
    todo = [(s, c) for s, c in plan if (s, *c) not in done]
    log(f"{len(todo)} cells to measure, {len(plan) - len(todo)} already done")
    log(f"pinned to CPUs {pin_to_cores()}")
    start = time.time()
    ref = ReferenceLog(reference, "measure") if reference and todo else None
    with path.open("a") as f:
        for i, (session, (wid, mode, dtype, threads)) in enumerate(todo):
            if ref:
                ref.maybe_record()
            if max_minutes is not None and time.time() - start > max_minutes * 60:
                log(f"stopping at the time limit with {len(todo) - i} cells left")
                break
            t0 = time.time()
            timing, load, attempts = run_quiet(
                lambda: measure_cell(workloads[wid], mode, dtype, threads), QUIET_LIMIT,
                attempts=ATTEMPTS)
            rec = {"session": session, "workload": wid, "family": workloads[wid].family,
                   "mode": mode, "dtype": dtype, "threads": threads, **timing.to_json(),
                   "other_cpus": round(load, 2), "attempts": attempts,
                   "wall": round(time.time() - t0, 3), "at": round(time.time(), 1)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(todo)} cells, {(time.time() - start) / 60:.1f} min")
    if ref:
        ref.record()
    log(f"finished in {(time.time() - start) / 60:.1f} min")


CLEAN_CUTOFF_MINUTES = 10.0


def _key(r: dict) -> tuple:
    return (r["workload"], r["mode"], r["dtype"], r["threads"])


def drift_table(rows: list[dict], bin_minutes: float = 3.0) -> list[dict]:
    """How much slower the first pass was than the second, by time into the first pass.

    Every cell of the second pass was also measured in the first. The ratio of the two
    times, grouped by when the first measurement was taken, shows when the machine as a
    whole was running slow.
    """
    main = {_key(r): r for r in rows if r["session"] == "main"}
    start = min(r["at"] for r in rows)
    bins: dict[int, list[float]] = {}
    for r in rows:
        if r["session"] == "repeat" and _key(r) in main:
            m = main[_key(r)]
            minute = (m["at"] - start) / 60.0
            bins.setdefault(int(minute // bin_minutes), []).append(m["t"] / r["t"])
    out = []
    for b in sorted(bins):
        v = sorted(bins[b])
        out.append({"from_minute": b * bin_minutes, "to_minute": (b + 1) * bin_minutes,
                    "pairs": len(v), "median_ratio": v[len(v) // 2] if len(v) % 2 else
                    0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])})
    return out


def clean_view(rows: list[dict], cutoff_minutes: float = CLEAN_CUTOFF_MINUTES) -> list[dict]:
    """One row per cell, using only the two windows in which the machine ran at full speed.

    Those are the first ``cutoff_minutes`` of the first pass and the whole second pass.
    A cell measured in both takes its second-pass time. Cells measured only in the slow
    stretch of the first pass are left out. Rows are relabelled as session "main" so the
    rest of the pipeline can treat them like any other set of measurements.
    """
    start = min(r["at"] for r in rows)
    chosen: dict[tuple, dict] = {}
    for r in rows:
        if r["session"] == "main" and (r["at"] - start) / 60.0 < cutoff_minutes:
            chosen[_key(r)] = r
    for r in rows:
        if r["session"] == "repeat":
            chosen[_key(r)] = r
    return [{**r, "session": "main"} for r in chosen.values()]


def clean_pairs(rows: list[dict], cutoff_minutes: float = CLEAN_CUTOFF_MINUTES
                ) -> list[tuple[dict, dict]]:
    """Cells measured in both clean windows: (first pass, second pass)."""
    start = min(r["at"] for r in rows)
    main = {_key(r): r for r in rows if r["session"] == "main"
            and (r["at"] - start) / 60.0 < cutoff_minutes}
    return [(main[_key(r)], r) for r in rows if r["session"] == "repeat" and _key(r) in main]
