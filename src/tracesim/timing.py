"""Wall-clock timing that tries to be honest about noise."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass


@dataclass
class Timing:
    median: float
    q1: float
    q3: float
    minimum: float
    reps: int
    inner: int

    @property
    def rel_iqr(self) -> float:
        return (self.q3 - self.q1) / self.median if self.median > 0 else 0.0

    def to_json(self) -> dict:
        return {"t": self.median, "q1": self.q1, "q3": self.q3, "min": self.minimum,
                "reps": self.reps, "inner": self.inner}


def summarise_samples(samples: list[float], inner: int = 1) -> Timing:
    s = sorted(samples)
    if len(s) >= 4:
        q = statistics.quantiles(s, n=4, method="inclusive")
        q1, q3 = q[0], q[2]
    else:
        q1, q3 = s[0], s[-1]
    return Timing(statistics.median(s), q1, q3, s[0], len(s), inner)


def time_callable(fn, *, min_reps: int = 5, max_reps: int = 30, min_time: float = 0.1,
                  max_time: float = 3.0, warmup: int = 2, floor_reps: int = 3,
                  clock=time.perf_counter) -> Timing:
    """Time ``fn()``.

    Fast calls are grouped so that one sample lasts at least about 0.2 ms; the sample is
    the group time divided by the group size. Sampling stops once there are ``min_reps``
    samples and ``min_time`` seconds have been spent, or at ``max_reps`` samples, or when
    ``max_time`` is exceeded with at least ``floor_reps`` samples in hand.
    """
    t0 = clock()
    fn()
    first = clock() - t0
    for _ in range(max(0, warmup - 1)):
        if first > 1.0:
            break
        fn()
    inner = 1
    if first < 2e-4:
        t0 = clock()
        fn()
        again = max(clock() - t0, 1e-7)
        inner = max(1, min(1000, int(2e-4 / again)))
    samples: list[float] = []
    start = clock()
    while True:
        t0 = clock()
        for _ in range(inner):
            fn()
        samples.append((clock() - t0) / inner)
        spent = clock() - start
        if len(samples) >= max_reps:
            break
        if len(samples) >= min_reps and spent >= min_time:
            break
        if len(samples) >= floor_reps and spent >= max_time:
            break
    return summarise_samples(samples, inner)
