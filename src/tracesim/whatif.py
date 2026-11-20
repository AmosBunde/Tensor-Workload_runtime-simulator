"""Projections for devices that do not exist here.

The descriptor of the measured host is edited (more cores, more memory bandwidth) and the
same traces are replayed against the edited copy. Nothing in this module is checked
against a measurement, and every output says so. The projections lean on two
extrapolations that the validation study cannot test: that an operator class keeps the
serial fraction it showed between one and two threads, and that memory bandwidth and core
count can be changed independently of everything else.
"""

from __future__ import annotations

from .device import DeviceDescriptor
from .simulator import simulate
from .trace import capture_workload
from .zoo import by_id

WORKLOADS = ("mlp-w2048-d6-b512", "cnn-w64-b2-i64-b4", "transformer-d512-d4-s128-b8",
             "mixer-d512-d4-i64-b8")
NOTE = "Unvalidated projection: no such device was measured."


def scenarios(host: DeviceDescriptor) -> list[tuple[str, DeviceDescriptor, int]]:
    """(label, device, threads). The first entry is the measured host, one thread per core."""
    c = host.cores
    return [
        ("host", host, c),
        ("2x memory bandwidth", host.variant("host-2x-bandwidth", bandwidth_factor=2.0,
                                             notes=NOTE), c),
        ("8 cores", host.variant("host-8-cores", cores=8, notes=NOTE), 8),
        ("16 cores", host.variant("host-16-cores", cores=16, notes=NOTE), 16),
        ("8 cores, 2x memory bandwidth",
         host.variant("host-8-cores-2x-bandwidth", cores=8, bandwidth_factor=2.0, notes=NOTE),
         8),
    ]


def project(host: DeviceDescriptor, measured: dict | None = None) -> dict:
    """Predicted time, speed-up over the host and time split for each scenario."""
    out: dict = {"note": NOTE, "baseline": f"host with {host.cores} threads, one per core",
                 "workloads": {}}
    for wid in WORKLOADS:
        for mode in ("forward", "train"):
            trace = capture_workload(by_id(wid), mode, "float32")
            entry: dict = {"scenarios": {}}
            base = None
            for label, device, threads in scenarios(host):
                r = simulate(trace, device, threads)
                if base is None:
                    base = r.total
                total = sum(r.breakdown.values())
                entry["scenarios"][label] = {
                    "threads": threads,
                    "predicted_s": r.total,
                    "speedup_over_host": base / r.total,
                    "share": {k: v / total for k, v in r.breakdown.items()},
                    "validated": label == "host",
                }
            if measured:
                m = measured.get((wid, mode, "float32", host.cores))
                if m:
                    entry["host_measured_s"] = m
                    entry["host_prediction_error"] = (
                        entry["scenarios"]["host"]["predicted_s"] / m - 1.0)
            out["workloads"][f"{wid}/{mode}"] = entry
    return out


def breakdowns(host: DeviceDescriptor, workload_ids: list[str], threads: int) -> dict:
    """Share of predicted forward time by cause, for the bottleneck figure."""
    out = {}
    for wid in workload_ids:
        r = simulate(capture_workload(by_id(wid), "forward", "float32"), host, threads)
        total = sum(r.breakdown.values())
        out[wid] = {"predicted_s": r.total,
                    "share": {k: v / total for k, v in r.breakdown.items()}}
    return out
