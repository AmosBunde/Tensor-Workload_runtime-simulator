"""A declarative description of a device.

The descriptor is a small JSON document. It holds what a datasheet would give (cores,
cache size, memory bandwidth) and, for each operator class, the few numbers the executor
needs to price a call. Nothing in it refers to a particular workload. Thread scaling is
expressed with a serial fraction, in the manner of Amdahl's law, so that a descriptor can
be asked about a thread count that was never benchmarked.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def amdahl(threads: float, serial_fraction: float) -> float:
    """Time at ``threads`` relative to time at one thread."""
    return serial_fraction + (1.0 - serial_fraction) / max(threads, 1.0)


@dataclass
class OpClassModel:
    """How one operator class is priced.

    ``rate`` is work per second on one thread, by dtype. Work is FLOPs for matrix
    products and convolutions and elements for everything else.
    """

    unit: str = "elements"
    rate: dict[str, float] = field(default_factory=dict)
    call_overhead_s: float = 0.0
    item_overhead_s: float = 0.0
    serial_fraction: float = 1.0
    fork_overhead_s: float = 0.0
    # Cost of the backward pass relative to the forward pass, split the same way as the
    # forward cost: one ratio for the call overhead and one for the body.
    backward_overhead_ratio: float = 1.0
    backward_ratio: float = 1.0
    backward_ratio_no_input_grad: float | None = None


@dataclass
class DeviceDescriptor:
    name: str
    cores: int
    cache_bytes: int
    mem_bandwidth_bytes_per_s: float
    mem_serial_fraction: float
    op_classes: dict[str, OpClassModel]
    # An output larger than the allocator's threshold is given fresh pages by the
    # operating system on every call, and touching them for the first time has a cost.
    # glibc's threshold tops out at 32 MiB. A first-touch rate of 0 turns the term off.
    fresh_alloc_threshold_bytes: int = 32 << 20
    first_touch_bytes_per_s: float = 0.0
    simd: str = ""
    l1d_bytes: int = 0
    l2_bytes: int = 0
    notes: str = ""
    calibration: dict = field(default_factory=dict)

    def bandwidth(self, threads: int) -> float:
        n = min(threads, self.cores)
        return self.mem_bandwidth_bytes_per_s / amdahl(n, self.mem_serial_fraction)

    def alloc_time(self, bytes_written: int) -> float:
        if self.first_touch_bytes_per_s <= 0 or bytes_written <= self.fresh_alloc_threshold_bytes:
            return 0.0
        return bytes_written / self.first_touch_bytes_per_s

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "DeviceDescriptor":
        d = dict(d)
        d["op_classes"] = {k: OpClassModel(**v) for k, v in d["op_classes"].items()}
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "DeviceDescriptor":
        return cls.from_json(json.loads(Path(path).read_text()))

    def variant(self, name: str, *, cores: int | None = None, bandwidth_factor: float = 1.0,
                cache_factor: float = 1.0, notes: str = "") -> "DeviceDescriptor":
        """A hypothetical device derived from this one. Used for what-if projections."""
        d = copy.deepcopy(self)
        d.name = name
        if cores is not None:
            d.cores = cores
        d.mem_bandwidth_bytes_per_s *= bandwidth_factor
        d.cache_bytes = int(d.cache_bytes * cache_factor)
        d.notes = notes
        d.calibration = {"derived_from": self.name, "validated": False}
        return d
