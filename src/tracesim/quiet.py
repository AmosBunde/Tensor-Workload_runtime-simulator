"""Guard timings against other work on the machine.

The machine is shared. When other processes keep the cores busy, a multi-threaded
operator has to wait for the scheduler to give all of its threads a core at once, and a
call that normally takes tens of microseconds can take milliseconds. Medians do not help
when the interference lasts longer than the measurement.

Two defences are used together. First, the process pins itself to both hardware threads
of a few physical cores, so the scheduler has every reason to place other work on the
remaining cores. Second, the guard reads from ``/proc/stat`` how busy the pinned CPUs
were and subtracts this process's own CPU time. What is left is the time other processes
spent on our CPUs. A measurement is repeated if that was high while it ran, and the
attempt with the least interference is kept.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

_HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_STAT = Path("/proc/stat")


def physical_cores(cpus: set[int] | None = None) -> dict[tuple[str, str], list[int]]:
    """Map each physical core to its logical CPUs, limited to ``cpus`` if given."""
    cores: dict[tuple[str, str], list[int]] = {}
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        cpu = int(path.name[3:])
        if cpus is not None and cpu not in cpus:
            continue
        try:
            key = ((path / "topology/physical_package_id").read_text().strip(),
                   (path / "topology/core_id").read_text().strip())
        except OSError:
            continue
        cores.setdefault(key, []).append(cpu)
    return {k: sorted(v) for k, v in sorted(cores.items())}


def pin_to_cores(n_cores: int = 2) -> list[int]:
    """Pin this process to every hardware thread of the last ``n_cores`` physical cores."""
    if not hasattr(os, "sched_setaffinity"):
        return []
    cores = list(physical_cores().values())
    if len(cores) < n_cores:
        return sorted(os.sched_getaffinity(0))
    cpus = sorted(c for group in cores[-n_cores:] for c in group)
    os.sched_setaffinity(0, cpus)
    return cpus


def _machine_busy_seconds() -> float | None:
    """CPU seconds spent so far, by anyone, on the CPUs this process may run on."""
    mine = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    total = 0
    try:
        for line in _STAT.read_text().splitlines():
            if not (line.startswith("cpu") and line[3:4].isdigit()):
                continue
            fields = line.split()
            if mine is not None and int(fields[0][3:]) not in mine:
                continue
            user, nice, system, _idle, _iowait, irq, softirq, steal = (
                int(x) for x in fields[1:9])
            total += user + nice + system + irq + softirq + steal
    except OSError:
        return None
    return total / _HZ


@dataclass
class Snapshot:
    wall: float
    machine: float | None
    own: float

    @classmethod
    def take(cls) -> "Snapshot":
        return cls(time.perf_counter(), _machine_busy_seconds(), time.process_time())


def other_cpus(since: Snapshot) -> float:
    """Average number of our CPUs that other processes kept busy since ``since``."""
    now = Snapshot.take()
    wall = now.wall - since.wall
    if since.machine is None or now.machine is None or wall <= 0:
        return 0.0
    return max(0.0, ((now.machine - since.machine) - (now.own - since.own)) / wall)


def run_quiet(fn, limit: float = 0.4, attempts: int = 4):
    """Call ``fn`` and repeat it if other processes were busy on our CPUs meanwhile.

    Returns ``(result, other_cpus, attempts_used)`` for the attempt that saw the least
    interference. The caller should record the load so that a reader can see which
    results were taken on a busy machine. There is no point waiting for quiet before
    starting: while this process sleeps, other work moves onto its CPUs, and moves off
    again once it starts computing.
    """
    best = None
    for attempt in range(1, attempts + 1):
        snap = Snapshot.take()
        result = fn()
        load = other_cpus(snap)
        if best is None or load < best[1]:
            best = (result, load, attempt)
        if load <= limit:
            break
    return best[0], best[1], attempt
