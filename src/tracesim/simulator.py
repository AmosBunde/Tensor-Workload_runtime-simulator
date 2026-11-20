"""Replay a trace against a device descriptor.

Pricing one operator
--------------------
An operator call costs a fixed overhead plus the larger of two times:

* a compute time, ``work / rate``, shortened by threads according to the class's serial
  fraction and lengthened by a fork cost when more than one thread is used;
* a memory time, the bytes the call must bring in from beyond the cache divided by the
  memory bandwidth at that thread count.

Taking the larger of the two is the roofline idea: a call is either limited by arithmetic
or by how fast memory can feed it. The share of the working set that misses the cache
grows smoothly from zero once the working set exceeds the cache size. An output too large
for the allocator to recycle is charged a first-touch cost on top.

In a training step each operator that a backward pass visits is charged again, scaled by
the class's measured backward-to-forward ratios.

Executing the trace
-------------------
The executor is a small discrete-event simulation. An operator becomes ready when the
operators it depends on have finished, and ready operators are started while a stream is
free. With one stream, which is how PyTorch runs a model on a CPU, this is a sequential
replay and the total is the sum of the operator times. More streams model running
independent branches side by side; that mode is not validated in this study.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .device import DeviceDescriptor, OpClassModel, amdahl
from .trace import Op, Trace

PARTS = ("overhead", "compute", "memory", "threading")


@dataclass
class OpCost:
    total: float
    overhead: float = 0.0
    compute: float = 0.0
    memory: float = 0.0
    threading: float = 0.0
    backward: float = 0.0
    bound: str = "overhead"

    def parts(self) -> dict[str, float]:
        return {p: getattr(self, p) for p in PARTS}


@dataclass
class SimResult:
    total: float
    op_costs: list[OpCost]
    finish_times: list[float]
    breakdown: dict[str, float] = field(default_factory=dict)
    by_class: dict[str, float] = field(default_factory=dict)
    backward: float = 0.0


def forward_cost(op: Op, model: OpClassModel, device: DeviceDescriptor, threads: int
                 ) -> OpCost:
    """Price one forward call of ``op``."""
    overhead = model.call_overhead_s
    if op.op_class == "free" or op.work <= 0:
        return OpCost(total=overhead, overhead=overhead)
    rate = model.rate.get(op.dtype) or next(iter(model.rate.values()))
    serial = op.work / rate + op.items * model.item_overhead_s
    n = min(threads, device.cores)
    body, ideal, parallel = serial, serial, False
    if n > 1:
        par = serial * amdahl(n, model.serial_fraction) + model.fork_overhead_s
        if par < serial:
            body, ideal, parallel = par, serial / n, True
    ws = op.working_set
    miss = max(0.0, 1.0 - device.cache_bytes / ws) if ws > 0 else 0.0
    mem = miss * ws / device.bandwidth(threads)
    alloc = device.alloc_time(op.bytes_written)
    if mem > body:
        return OpCost(total=overhead + mem + alloc, overhead=overhead, memory=mem + alloc,
                      bound="memory")
    cost = OpCost(total=overhead + body + alloc, overhead=overhead, compute=ideal,
                  memory=alloc, threading=body - ideal if parallel else 0.0, bound="compute")
    if overhead > body:
        cost.bound = "overhead"
    return cost


def op_cost(op: Op, device: DeviceDescriptor, threads: int, scale: dict[str, float] | None
            = None) -> OpCost:
    """Price ``op``, including its backward pass if it has one."""
    model = device.op_classes.get(op.op_class)
    if model is None:
        return OpCost(total=0.0)
    c = forward_cost(op, model, device, threads)
    if op.has_backward and op.phase == "forward":
        body_ratio = model.backward_ratio
        if not op.needs_input_grad and model.backward_ratio_no_input_grad is not None:
            body_ratio = model.backward_ratio_no_input_grad
        r_over, r_body = model.backward_overhead_ratio, body_ratio
        back = c.overhead * r_over + (c.total - c.overhead) * r_body
        c = OpCost(
            total=c.total + back,
            overhead=c.overhead * (1 + r_over),
            compute=c.compute * (1 + r_body),
            memory=c.memory * (1 + r_body),
            threading=c.threading * (1 + r_body),
            backward=back,
            bound=c.bound,
        )
    k = (scale or {}).get(op.op_class, 1.0)
    if k != 1.0:
        c = OpCost(c.total * k, c.overhead * k, c.compute * k, c.memory * k,
                   c.threading * k, c.backward * k, c.bound)
    return c


def simulate(trace: Trace, device: DeviceDescriptor, threads: int = 1, streams: int = 1,
             scale: dict[str, float] | None = None) -> SimResult:
    """Run the trace through the executor and return predicted times.

    ``threads`` is the number of threads each operator may use. ``streams`` is how many
    operators may run at once. If ``threads * streams`` exceeds the cores, each operator
    is given an equal share of the cores instead.
    """
    if streams < 1 or threads < 1:
        raise ValueError("threads and streams must be at least 1")
    per_op_threads = threads
    if threads * streams > device.cores:
        per_op_threads = max(1, device.cores // streams)
    costs = [op_cost(op, device, per_op_threads, scale) for op in trace.ops]

    n = len(trace.ops)
    waiting = [len(op.deps) for op in trace.ops]
    children: list[list[int]] = [[] for _ in range(n)]
    for i, op in enumerate(trace.ops):
        for d in op.deps:
            children[d].append(i)
    ready = [i for i in range(n) if waiting[i] == 0]
    heapq.heapify(ready)
    running: list[tuple[float, int]] = []  # (finish time, op index)
    finish = [0.0] * n
    clock, done = 0.0, 0
    while done < n:
        while ready and len(running) < streams:
            i = heapq.heappop(ready)
            heapq.heappush(running, (clock + costs[i].total, i))
        if not running:
            raise ValueError("trace has a dependency cycle")
        clock, i = heapq.heappop(running)
        finish[i] = clock
        done += 1
        for c in children[i]:
            waiting[c] -= 1
            if waiting[c] == 0:
                heapq.heappush(ready, c)

    breakdown = {p: sum(getattr(c, p) for c in costs) for p in PARTS}
    by_class: dict[str, float] = {}
    for op, c in zip(trace.ops, costs):
        by_class[op.op_class] = by_class.get(op.op_class, 0.0) + c.total
    return SimResult(clock, costs, finish, breakdown, by_class,
                     backward=sum(c.backward for c in costs))


def class_times(trace: Trace, device: DeviceDescriptor, threads: int) -> dict[str, float]:
    """Predicted seconds per operator class. Used when fitting per-class corrections."""
    out: dict[str, float] = {}
    for op in trace.ops:
        out[op.op_class] = out.get(op.op_class, 0.0) + op_cost(op, device, threads).total
    return out
