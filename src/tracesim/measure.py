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
        only: list[str] | None = None, seed: int = 0, log=print) -> None:
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
    with path.open("a") as f:
        for i, (session, (wid, mode, dtype, threads)) in enumerate(todo):
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
    log(f"finished in {(time.time() - start) / 60:.1f} min")
