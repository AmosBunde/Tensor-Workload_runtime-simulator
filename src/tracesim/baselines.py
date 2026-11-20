"""Predictors that the simulator is compared against, and its fitted-correction variant.

All of them are fitted on the measured cells of a training split. The black-box ones see
a cell as a row of trace totals plus the device configuration; they know nothing about
the order of operators or about the machine.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .trace import OP_CLASSES


def feature_names(cells: list[dict]) -> list[str]:
    return sorted(cells[0]["features"])


AGGREGATES = ("flops", "bytes", "params", "n_ops", "max_working_set")


def design_matrix(cells: list[dict], interactions: bool = False,
                  names: tuple[str, ...] | None = None) -> np.ndarray:
    """Log-scaled trace totals plus threads, dtype and mode."""
    names = [n for n in (names or feature_names(cells)) if n in cells[0]["features"]]
    rows = []
    for c in cells:
        logs = [math.log1p(c["features"][k]) for k in names]
        lt = math.log2(c["threads"])
        f64 = 1.0 if c["dtype"] == "float64" else 0.0
        train = 1.0 if c["mode"] == "train" else 0.0
        row = logs + [lt, f64, train]
        if interactions:
            lf = math.log1p(c["features"]["flops"])
            row += [lt * lf, f64 * lf, train * lf, lt * train, lt * f64]
        rows.append(row)
    return np.array(rows)


class FlopsProportional:
    """time = constant * FLOPs, one constant per (threads, dtype, mode)."""

    name = "flops"

    def fit(self, cells: list[dict]) -> "FlopsProportional":
        groups: dict[tuple, list[float]] = {}
        for c in cells:
            key = (c["threads"], c["dtype"], c["mode"])
            groups.setdefault(key, []).append(math.log(c["t"] / c["features"]["flops"]))
        self.const = {k: math.exp(float(np.mean(v))) for k, v in groups.items()}
        return self

    def _constant(self, threads: int, dtype: str, mode: str) -> float:
        if (threads, dtype, mode) in self.const:
            return self.const[(threads, dtype, mode)]
        # An unseen thread count: take the nearest seen one and assume ideal scaling.
        seen = [n for (n, d, m) in self.const if d == dtype and m == mode]
        nearest = min(seen, key=lambda n: abs(math.log(n / threads)))
        return self.const[(nearest, dtype, mode)] * nearest / threads

    def predict(self, cells: list[dict]) -> np.ndarray:
        return np.array([
            self._constant(c["threads"], c["dtype"], c["mode"]) * c["features"]["flops"]
            for c in cells
        ])


class GradientBoosting:
    """Gradient-boosted trees on log time. Hyperparameters are fixed, not tuned."""

    name = "gbm"

    def fit(self, cells: list[dict]) -> "GradientBoosting":
        self.model = GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0
        )
        self.model.fit(design_matrix(cells), np.log([c["t"] for c in cells]))
        return self

    def predict(self, cells: list[dict]) -> np.ndarray:
        return np.exp(self.model.predict(design_matrix(cells)))


class LogLinear:
    """Ridge regression of log time on the logs of whole-trace totals.

    It deliberately leaves out the per-class totals that the trees get. A class that is
    absent from the training workloads has no information about its coefficient, and a
    linear model then extrapolates wildly when the class appears at test time.
    """

    name = "loglinear"

    def fit(self, cells: list[dict]) -> "LogLinear":
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self.model.fit(design_matrix(cells, True, AGGREGATES), np.log([c["t"] for c in cells]))
        return self

    def predict(self, cells: list[dict]) -> np.ndarray:
        return np.exp(self.model.predict(design_matrix(cells, True, AGGREGATES)))


class SimulatorCorrections:
    """One multiplicative correction per operator class, fitted on measured workloads.

    Each cell contributes the simulator's predicted seconds per class. The corrections
    minimise squared relative error of the corrected total, are kept non-negative, and
    are pulled towards 1 so that a class the training split barely exercises keeps the
    microbenchmark calibration.
    """

    name = "simfit"

    def __init__(self, prior_weight: float = 0.05):
        self.prior_weight = prior_weight
        self.scale: dict[str, float] = {}

    def fit(self, class_times: list[dict[str, float]], measured: list[float]
            ) -> "SimulatorCorrections":
        classes = list(OP_CLASSES)
        a = np.array([[ct.get(c, 0.0) / t for c in classes] for ct, t in
                      zip(class_times, measured)])
        w = math.sqrt(self.prior_weight * len(measured))
        a = np.vstack([a, w * np.eye(len(classes))])
        y = np.concatenate([np.ones(len(measured)), w * np.ones(len(classes))])
        coef, _ = nnls(a, y)
        self.scale = {c: float(k) for c, k in zip(classes, coef)}
        return self

    def predict(self, class_times: list[dict[str, float]]) -> np.ndarray:
        return np.array([sum(self.scale.get(c, 1.0) * v for c, v in ct.items())
                         for ct in class_times])
