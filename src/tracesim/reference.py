"""A fixed small benchmark timed every few minutes, to measure drift in machine speed.

Workload timings can only be compared if the machine ran at the same speed throughout.
The first run of this study showed that it may not. These probes do the same work every
time, so any change in their timing is a change in the machine. They are a record and
are not used to correct anything.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from .quiet import cpu_mhz
from .timing import time_callable

# (name, threads). Three matrix products of the same size and one large addition.
PROBES = (("gemm-1-thread", 1), ("gemm-2-threads", 2), ("gemm-4-threads", 4),
          ("add-1-thread", 1))


class ReferenceLog:
    def __init__(self, path: Path, phase: str, every_seconds: float = 120.0):
        self.path, self.phase, self.every = Path(path), phase, every_seconds
        g = torch.Generator().manual_seed(0)
        self.a = torch.randn(512, 512, generator=g)
        self.b = torch.randn(2000, 2000, generator=g)
        self.last = 0.0

    def maybe_record(self) -> bool:
        """Record the probes if ``every_seconds`` have passed since the last time."""
        if time.time() - self.last < self.every:
            return False
        self.record()
        return True

    def record(self) -> None:
        self.last = time.time()
        rows = []
        with torch.no_grad():
            for name, threads in PROBES:
                torch.set_num_threads(threads)
                fn = (lambda: self.b + self.b) if name.startswith("add") else (
                    lambda: self.a @ self.a)
                t = time_callable(fn, min_reps=5, max_reps=15, min_time=0.03, max_time=0.5,
                                  warmup=1)
                rows.append({"at": round(time.time(), 1), "phase": self.phase, "probe": name,
                             "t": t.median, "mhz": cpu_mhz()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def load_reference(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def by_window(rows: list[dict], windows: dict[str, tuple[float, float]]) -> dict:
    """Probe slowdown and CPU frequency within named time windows.

    Slowdown is each reading over the fastest reading of its probe, averaged over the
    readings that fall in the window.
    """
    fastest = {name: min(r["t"] for r in rows if r["probe"] == name) for name, _ in PROBES
               if any(r["probe"] == name for r in rows)}
    out = {}
    for label, (lo, hi) in windows.items():
        sub = [r for r in rows if lo <= r["at"] <= hi]
        if not sub:
            continue
        mhz = [r["mhz"] for r in sub if r.get("mhz")]
        out[label] = {
            "readings": len(sub),
            "mean_slowdown_over_fastest": sum(r["t"] / fastest[r["probe"]] for r in sub)
            / len(sub),
            "median_cpu_mhz": statistics.median(mhz) if mhz else None,
            "lowest_cpu_mhz": min(mhz) if mhz else None,
        }
    return out


def summarise_reference(rows: list[dict]) -> dict:
    """Spread of each probe's timing over the whole run."""
    out: dict = {"probes": {}}
    for name, _ in PROBES:
        t = sorted(r["t"] for r in rows if r["probe"] == name)
        if len(t) < 2:
            continue
        med = statistics.median(t)
        q = statistics.quantiles(t, n=10, method="inclusive")
        out["probes"][name] = {
            "readings": len(t), "median_s": med, "slowest_over_fastest": t[-1] / t[0],
            "p90_over_p10": q[8] / q[0],
            "mean_abs_deviation_from_median": sum(abs(x / med - 1) for x in t) / len(t),
            "largest_deviation_from_median": max(abs(x / med - 1) for x in t),
        }
    mhz = sorted(r["mhz"] for r in rows if r.get("mhz"))
    if mhz:
        out["cpu_mhz"] = {"lowest": mhz[0], "median": statistics.median(mhz),
                          "highest": mhz[-1]}
    if rows:
        out["span_minutes"] = (max(r["at"] for r in rows) - min(r["at"] for r in rows)) / 60
    return out
