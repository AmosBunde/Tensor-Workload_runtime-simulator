"""Fit a device descriptor to microbenchmark points.

Fitting is a pure function of the points, so the same code can produce a descriptor from
all of them or from a restricted set: only small operators, or only one and two threads.
The validation study uses the restricted fits to make sure the simulator is not shown
what it is then asked to predict.
"""

from __future__ import annotations

import math
import os
import platform
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

from .device import DeviceDescriptor, OpClassModel, amdahl
from .quiet import physical_cores
from .simulator import forward_cost
from .trace import FLOP_CLASSES, OP_CLASSES, Op

KEEP_SMALLEST = 4


def host_info() -> dict:
    """Cores and cache sizes of this machine, read from sysfs where available.

    Only the cores this process is allowed to run on are counted.
    """
    info = {"cores": os.cpu_count() or 1, "smt": 1, "l1d": 0, "l2": 0, "llc": 8 << 20, "simd": "",
            "cpu": platform.processor() or platform.machine()}
    cpu = Path("/sys/devices/system/cpu")
    try:
        mine = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        cores = physical_cores(mine)
        if cores:
            info["cores"] = len(cores)
            info["smt"] = max(len(v) for v in cores.values())
            info["cpus"] = sorted(c for v in cores.values() for c in v)
        for idx in (cpu / "cpu0/cache").glob("index*"):
            level = int((idx / "level").read_text())
            kind = (idx / "type").read_text().strip()
            text = (idx / "size").read_text().strip()
            size = int(text[:-1]) * {"K": 1 << 10, "M": 1 << 20}[text[-1]]
            if level == 1 and kind == "Data":
                info["l1d"] = size
            elif level == 2:
                info["l2"] = size
            elif level >= 3:
                info["llc"] = size
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                for name in ("avx512f", "avx2", "avx", "sse2"):
                    if name in flags:
                        info["simd"] = name
                        break
                break
    except (OSError, ValueError, KeyError):
        pass
    return info


def _op(p: dict) -> Op:
    return Op(name=p["id"], kind=p["kind"], op_class=p["op_class"], in_shapes=[], out_shape=[],
              dtype=p["dtype"], work=p["work"], items=p["items"], bytes_read=p["bytes_read"],
              bytes_written=p["bytes_written"])


def restrict(points: list[dict], max_threads: int | None = None,
             max_work: dict[str, float] | None = None) -> list[dict]:
    """Drop points the simulator is not allowed to see.

    ``max_work`` maps an operator class to the largest operator it may be calibrated on.
    The few smallest operators of a class are always kept so that every class can be
    fitted. The memory-bandwidth points (``bench == "stream"``) are a property of the
    device, not of an operator class, and are never dropped by size.
    """
    out = [p for p in points if max_threads is None or p["threads"] <= max_threads]
    if not max_work:
        return out
    kept: list[dict] = []
    for bench in sorted({p["bench"] for p in out}):
        group = [p for p in out if p["bench"] == bench]
        if bench == "stream":
            kept += group
            continue
        cap = max_work.get(group[0]["op_class"], 0.0)
        floor = sorted({p["work"] for p in group})[:KEEP_SMALLEST][-1]
        kept += [p for p in group if p["work"] <= max(cap, floor)]
    return kept


def _fit_bandwidth(points: list[dict], cache_bytes: int, threshold: int, cores: int
                   ) -> tuple[float, float, float]:
    """Memory bandwidth at one thread, its serial fraction, and the first-touch rate."""
    stream = [p for p in points if p["bench"] == "stream"]
    recycled = [p for p in stream if p["bytes_written"] <= threshold]
    if not recycled:
        raise ValueError("no memory-bandwidth points to calibrate from")
    by_n: dict[int, list[float]] = {}
    for p in [q for q in recycled if q["threads"] <= cores]:
        ws = p["bytes_read"] + p["bytes_written"]
        by_n.setdefault(p["threads"], []).append((ws - cache_bytes) / p["t"])
    bw = {n: statistics.median(v) for n, v in by_n.items()}
    base = bw[1]
    fractions = [(base / bw[n] - 1.0 / n) / (1.0 - 1.0 / n) for n in bw if n > 1]
    s = float(np.clip(np.mean(fractions), 0.0, 1.0)) if fractions else 1.0
    rates = []
    for p in stream:
        if p["bytes_written"] > threshold:
            ws = p["bytes_read"] + p["bytes_written"]
            extra = p["t"] - (ws - cache_bytes) / (base / amdahl(min(p["threads"], cores), s))
            if extra > 0:
                rates.append(p["bytes_written"] / extra)
    return base, s, (statistics.median(rates) if rates else 0.0)


def _log_residuals(model: OpClassModel, device: DeviceDescriptor, pts: list[dict]) -> np.ndarray:
    return np.array([
        math.log(forward_cost(_op(p), model, device, p["threads"]).total / p["t"]) for p in pts
    ])


def _fit_class(op_class: str, pts: list[dict], device: DeviceDescriptor) -> OpClassModel:
    unit = "flops" if op_class in FLOP_CLASSES else "elements"
    model = OpClassModel(unit=unit)
    single = [p for p in pts if p["threads"] == 1]
    if op_class == "free":
        model.unit = "none"
        model.call_overhead_s = statistics.median(p["t"] for p in single)
        ratios = [(p["t_fb"] - p["t"]) / p["t"] for p in single if p.get("t_fb")]
        model.backward_overhead_ratio = max(0.0, statistics.median(ratios)) if ratios else 1.0
        model.backward_ratio = 0.0
        return model

    dtypes = sorted({p["dtype"] for p in single})
    has_items = any(p["items"] > 1 for p in single)
    o0 = min(p["t"] for p in single)
    r0 = {d: max(p["work"] / p["t"] for p in single if p["dtype"] == d) for d in dtypes}

    def unpack(theta: np.ndarray) -> None:
        model.call_overhead_s = math.exp(theta[0])
        model.rate = {d: math.exp(theta[1 + i]) for i, d in enumerate(dtypes)}
        model.item_overhead_s = math.exp(theta[1 + len(dtypes)]) if has_items else 0.0

    theta0 = [math.log(o0)] + [math.log(r0[d]) for d in dtypes]
    if has_items:
        theta0.append(math.log(o0 / 10))

    def res1(theta):
        unpack(theta)
        return _log_residuals(model, device, single)

    unpack(least_squares(res1, np.array(theta0), loss="soft_l1", f_scale=0.3).x)

    multi = [p for p in pts if 1 < p["threads"] <= device.cores]
    if multi:
        def set_par(theta):
            model.serial_fraction = 1.0 / (1.0 + math.exp(-theta[0]))
            model.fork_overhead_s = math.exp(theta[1])

        def res2(theta):
            set_par(theta)
            return _log_residuals(model, device, multi)

        best = None
        for start in ((-2.0, math.log(1e-6)), (0.0, math.log(1e-5)), (-4.0, math.log(1e-4))):
            sol = least_squares(res2, np.array(start), loss="soft_l1", f_scale=0.3)
            if best is None or sol.cost < best.cost:
                best = sol
        set_par(best.x)

    hyper = [p for p in pts if p["threads"] > device.cores]
    if hyper:
        def res3(theta):
            model.smt_yield = float(theta[0])
            return _log_residuals(model, device, hyper)

        sol = least_squares(res3, np.array([0.2]), bounds=([-0.45], [1.0]), loss="soft_l1",
                            f_scale=0.3)
        model.smt_yield = float(sol.x[0])

    _fit_backward(model, single)
    return model


def _fit_backward(model: OpClassModel, single: list[dict]) -> None:
    """Fit backward cost as ``a * overhead + b * body`` of the forward cost."""
    o = model.call_overhead_s

    def rows(key: str):
        a, y = [], []
        for p in single:
            if p.get(key):
                body = max(p["t"] - o, 0.0)
                a.append([o / p[key], body / p[key]])
                y.append((p[key] - p["t"]) / p[key])
        return np.array(a), np.array(y)

    a, y = rows("t_fb")
    if len(y) >= 2:
        coef, _ = nnls(a, np.clip(y, 0.0, None))
        model.backward_overhead_ratio, model.backward_ratio = float(coef[0]), float(coef[1])
    a, y = rows("t_fb_noin")
    if len(y) >= 2:
        rest = np.clip(y, 0.0, None) - model.backward_overhead_ratio * a[:, 0]
        coef, _ = nnls(a[:, 1:2], np.clip(rest, 0.0, None))
        model.backward_ratio_no_input_grad = float(coef[0])
        if not any(p.get("t_fb") for p in single):
            coef, _ = nnls(a, np.clip(y, 0.0, None))
            model.backward_overhead_ratio = float(coef[0])
            model.backward_ratio = model.backward_ratio_no_input_grad = float(coef[1])


def fit_descriptor(points: list[dict], name: str = "host", info: dict | None = None,
                   max_threads: int | None = None, max_work: dict[str, float] | None = None,
                   ) -> DeviceDescriptor:
    info = info or host_info()
    pts = restrict(points, max_threads, max_work)
    threshold = DeviceDescriptor.fresh_alloc_threshold_bytes
    bw, s_mem, first_touch = _fit_bandwidth(pts, info["llc"], threshold, info["cores"])
    device = DeviceDescriptor(
        name=name, cores=info["cores"], cache_bytes=info["llc"],
        mem_bandwidth_bytes_per_s=bw, mem_serial_fraction=s_mem, op_classes={},
        first_touch_bytes_per_s=first_touch, smt=info.get("smt", 1),
        simd=info.get("simd", ""), l1d_bytes=info.get("l1d", 0), l2_bytes=info.get("l2", 0),
        notes=info.get("cpu", ""),
    )
    for op_class in OP_CLASSES:
        group = [p for p in pts if p["op_class"] == op_class and p["bench"] != "stream"]
        if group:
            device.op_classes[op_class] = _fit_class(op_class, group, device)
    device.calibration = {
        "points": len(pts),
        "max_threads": max(p["threads"] for p in pts),
        "restricted_by_size": bool(max_work),
    }
    return device


def fit_report(points: list[dict], device: DeviceDescriptor) -> dict[str, dict]:
    """How well the descriptor reproduces the microbenchmarks it was fitted to."""
    out: dict[str, dict] = {}
    for op_class, model in device.op_classes.items():
        pts = [p for p in points if p["op_class"] == op_class and p["bench"] != "stream"]
        if not pts:
            continue
        err = np.abs(np.exp(_log_residuals(model, device, pts)) - 1.0)
        out[op_class] = {"points": len(pts), "mape": float(err.mean()),
                         "median_ape": float(np.median(err))}
    return out
