"""Microbenchmarks of single operators.

Each case is a tiny module holding one priced operator. It is traced with the same code
that traces whole models, so the work and byte counts used for calibration are counted
exactly as they will be counted in a workload. Outputs are allocated afresh on every call,
as they are inside a real model, so allocation cost is part of what is measured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .quiet import pin_to_cores, run_quiet
from .reference import ReferenceLog
from .timing import time_callable
from .trace import Op, capture

DTYPES = ("float32", "float64")
THREADS = (1, 2, 4)
MAX_FLOPS = 6e9
QUIET_LIMIT = 0.4  # how many of our CPUs other processes may keep busy meanwhile


class _Add(nn.Module):
    def forward(self, a, b):
        return a + b


class _MatMul(nn.Module):
    def forward(self, a, b):
        return a @ b


class _Softmax(nn.Module):
    def forward(self, x):
        return x.softmax(dim=-1)


class _Mean(nn.Module):
    def forward(self, x):
        return x.mean(1)


class _MergeHeads(nn.Module):
    """The copy that attention makes when it merges heads back together."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.transpose(1, 2).reshape(x.size(0), -1, self.dim)


class _View(nn.Module):
    def forward(self, x):
        return x.view(-1)


@dataclass
class Case:
    bench: str
    label: str
    make: Callable[[torch.dtype], tuple[nn.Module, tuple]]

    @property
    def id(self) -> str:
        return f"{self.bench}:{self.label}"


def _rand(dtype, *shape):
    return torch.randn(*shape).to(dtype)


def cases() -> list[Case]:
    out: list[Case] = []

    def add(bench, label, make):
        out.append(Case(bench, label, make))

    for n in (1, 16, 128, 1024):
        for d in (32, 128, 512, 2048):
            if 2 * n * d * d <= MAX_FLOPS:
                add("gemm", f"n{n}-d{d}",
                    lambda t, n=n, d=d: (nn.Linear(d, d).to(t), (_rand(t, n, d),)))
    for n in (32, 1024):
        for d in (64, 256, 512):
            add("gemm", f"n{n}-d{d}x4",
                lambda t, n=n, d=d: (nn.Linear(d, 4 * d).to(t), (_rand(t, n, d),)))

    for batch in (1, 8, 64):
        for seq in (32, 128):
            add("bmm", f"qk-b{batch}-s{seq}", lambda t, b=batch, s=seq: (
                _MatMul(), (_rand(t, b, 4, s, 64), _rand(t, b, 4, 64, s))))
            add("bmm", f"av-b{batch}-s{seq}", lambda t, b=batch, s=seq: (
                _MatMul(), (_rand(t, b, 4, s, s), _rand(t, b, 4, s, 64))))

    for batch in (1, 4):
        for c in (16, 64, 256):
            for h in (8, 32, 64):
                if 2 * batch * c * c * 9 * h * h <= MAX_FLOPS:
                    add("conv_s1", f"3x3-b{batch}-c{c}-h{h}", lambda t, b=batch, c=c, h=h: (
                        nn.Conv2d(c, c, 3, 1, 1, bias=False).to(t), (_rand(t, b, c, h, h),)))
        for c in (16, 64, 128):
            for h in (16, 64):
                add("conv_s2", f"3x3s2-b{batch}-c{c}-h{h}", lambda t, b=batch, c=c, h=h: (
                    nn.Conv2d(c, 2 * c, 3, 2, 1, bias=False).to(t), (_rand(t, b, c, h, h),)))
                add("conv_k1", f"1x1s2-b{batch}-c{c}-h{h}", lambda t, b=batch, c=c, h=h: (
                    nn.Conv2d(c, 2 * c, 1, 2, 0, bias=False).to(t), (_rand(t, b, c, h, h),)))
        for c in (16, 64):
            for h in (32, 64):
                add("conv_thin", f"stem-b{batch}-c{c}-h{h}", lambda t, b=batch, c=c, h=h: (
                    nn.Conv2d(3, c, 3, 1, 1, bias=False).to(t), (_rand(t, b, 3, h, h),)))
    for batch in (1, 8):
        for c in (64, 512):
            for h in (32, 64):
                add("conv_thin", f"patch-b{batch}-c{c}-h{h}", lambda t, b=batch, c=c, h=h: (
                    nn.Conv2d(3, c, 4, 4).to(t), (_rand(t, b, 3, h, h),)))

    for n in (1_000, 10_000, 100_000, 1_000_000, 4_000_000, 16_000_000):
        add("ew_cheap", f"relu-{n}", lambda t, n=n: (nn.ReLU(), (_rand(t, n // 100, 100),)))
        add("ew_cheap", f"add-{n}", lambda t, n=n: (
            _Add(), (_rand(t, n // 100, 100), _rand(t, n // 100, 100))))
        add("ew_transcend", f"gelu-{n}", lambda t, n=n: (nn.GELU(), (_rand(t, n // 100, 100),)))
    # Large additions measure the machine, not the operator: the small ones give memory
    # bandwidth, the big ones add the cost of first touching freshly mapped output pages.
    for n in (2_000_000, 3_000_000, 16_000_000, 32_000_000):
        add("stream", f"add-{n}", lambda t, n=n: (
            _Add(), (_rand(t, n // 100, 100), _rand(t, n // 100, 100))))

    for d in (64, 512):
        for rows in (8, 64, 512, 4096, 16384):
            add("layernorm", f"d{d}-r{rows}",
                lambda t, d=d, r=rows: (nn.LayerNorm(d).to(t), (_rand(t, r, d),)))
    for b, c, h in ((1, 16, 8), (1, 64, 16), (1, 64, 32), (4, 64, 32), (4, 64, 64),
                    (4, 256, 64)):
        add("batchnorm_eval", f"b{b}-c{c}-h{h}", lambda t, b=b, c=c, h=h: (
            nn.BatchNorm2d(c).to(t).eval(), (_rand(t, b, c, h, h),)))
        add("batchnorm_train", f"b{b}-c{c}-h{h}", lambda t, b=b, c=c, h=h: (
            nn.BatchNorm2d(c).to(t).train(), (_rand(t, b, c, h, h),)))
        add("pool", f"b{b}-c{c}-h{h}", lambda t, b=b, c=c, h=h: (
            nn.AdaptiveAvgPool2d(1), (_rand(t, b, c, h, h),)))
    for rows in (8, 64, 512, 4096, 32768):
        add("softmax", f"r{rows}", lambda t, r=rows: (_Softmax(), (_rand(t, r, 128),)))
    for b, s, d in ((1, 32, 64), (8, 32, 64), (8, 128, 256), (8, 128, 512), (64, 128, 512)):
        add("reduce", f"b{b}-s{s}-d{d}", lambda t, b=b, s=s, d=d: (_Mean(), (_rand(t, b, s, d),)))
        add("embedding", f"b{b}-s{s}-d{d}", lambda t, b=b, s=s, d=d: (
            nn.Embedding(1000, d).to(t), (torch.randint(0, 1000, (b, s)),)))
        add("copy", f"b{b}-s{s}-d{d}", lambda t, b=b, s=s, d=d: (
            _MergeHeads(d), (_rand(t, b, 4, s, d // 4),)))
    add("free", "view", lambda t: (_View(), (_rand(t, 64, 64),)))
    return out


def _priced_op(module: nn.Module, inputs: tuple, bench: str) -> Op:
    trace = capture(module, inputs, set_mode=False)
    if bench == "free":
        return trace.ops[0]
    priced = [o for o in trace.ops if o.op_class != "free"]
    if len(priced) != 1:
        raise RuntimeError(f"expected one priced operator, found {len(priced)}")
    return priced[0]


def _time_backward(module: nn.Module, inputs: tuple, input_grad: bool, **timer) -> float | None:
    """Median time of a forward pass with gradients on plus the backward pass."""
    params = [p for p in module.parameters() if p.requires_grad]
    xs = tuple(
        t.clone().requires_grad_(True) if (input_grad and t.is_floating_point()) else t
        for t in inputs
    )
    wrt = params + [t for t in xs if t.requires_grad]
    if not wrt:
        return None
    with torch.enable_grad():
        seed = torch.ones_like(module(*xs))

        def step():
            torch.autograd.grad(module(*xs), wrt, grad_outputs=seed)

        return time_callable(step, **timer).median


def run_case(case: Case, dtype: str, threads: int) -> dict:
    start = time.perf_counter()
    rec, load, attempts = run_quiet(lambda: _run_case(case, dtype, threads), QUIET_LIMIT)
    rec["other_cpus"] = round(load, 2)
    rec["attempts"] = attempts
    rec["wall"] = round(time.perf_counter() - start, 3)
    return rec


def _run_case(case: Case, dtype: str, threads: int) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    module, inputs = case.make(getattr(torch, dtype))
    op = _priced_op(module, inputs, case.bench)
    timer = dict(min_reps=7, max_reps=40, min_time=0.04, max_time=1.5, warmup=2)
    with torch.no_grad():
        t = time_callable(lambda: module(*inputs), **timer)
    rec = {
        "id": case.id, "bench": "stream" if case.bench == "stream" else op.op_class, "op_class": op.op_class, "kind": op.kind,
        "dtype": dtype, "threads": threads, "work": op.work, "items": op.items,
        "bytes_read": op.bytes_read, "bytes_written": op.bytes_written, **t.to_json(),
    }
    if threads == 1 and t.median < 0.5:
        rec["t_fb"] = _time_backward(module, inputs, True, **timer)
        has_params = any(True for _ in module.parameters())
        rec["t_fb_noin"] = _time_backward(module, inputs, False, **timer) if has_params else None
    return rec


def load_points(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_all(path: Path, log=print, reference: Path | None = None) -> list[dict]:
    """Run every case on every configuration. Cases already in ``path`` are skipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(p["id"], p["dtype"], p["threads"]) for p in load_points(path)}
    todo = [(c, d, n) for c in cases() for d in DTYPES for n in THREADS
            if (c.id, d, n) not in done]
    log(f"{len(todo)} microbenchmarks to run, {len(done)} already done")
    log(f"pinned to CPUs {pin_to_cores()}")
    torch.set_flush_denormal(True)
    ref = ReferenceLog(reference, "microbench") if reference and todo else None
    with path.open("a") as f:
        for i, (case, dtype, threads) in enumerate(todo):
            if ref:
                ref.maybe_record()
            rec = run_case(case, dtype, threads)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(todo)}")
    if ref:
        ref.record()
    return load_points(path)


def recheck(points_path: Path, out: Path, every: int = 6, log=print) -> list[dict]:
    """Time a sample of the cases again and record old and new times side by side.

    A descriptor is only as good as the state the machine was in when it was calibrated.
    Running this later shows whether the machine has sped up or slowed down since.
    """
    old = {(p["id"], p["dtype"], p["threads"]): p for p in load_points(points_path)}
    pin_to_cores()
    torch.set_flush_denormal(True)
    rows = []
    for case in cases()[::every]:
        if case.bench == "stream":
            continue
        for threads in THREADS:
            key = (case.id, "float32", threads)
            if key not in old:
                continue
            new = run_case(case, "float32", threads)
            rows.append({"id": case.id, "op_class": new["op_class"], "threads": threads,
                         "t_calibration": old[key]["t"], "t_recheck": new["t"],
                         "other_cpus": new["other_cpus"]})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ratios = sorted(r["t_calibration"] / r["t_recheck"] for r in rows)
    log(f"{len(rows)} timings repeated; calibration time over recheck time, median "
        f"{ratios[len(ratios) // 2]:.2f}")
    return rows
