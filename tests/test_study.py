import math
import random

import numpy as np
import pytest

from tracesim import evaluate, stats
from tracesim.baselines import (FlopsProportional, GradientBoosting, LogLinear,
                                SimulatorCorrections)
from tracesim.quiet import Snapshot, other_cpus, run_quiet
from tracesim.timing import summarise_samples, time_callable
from tracesim.zoo import FAMILIES


def fake_cells(n_per_family: int = 8, seed: int = 0) -> list[dict]:
    """Cells whose time is an exact power law of FLOPs, divided by the thread count."""
    rng = random.Random(seed)
    cells = []
    for fam in FAMILIES:
        for i in range(n_per_family):
            flops = 10 ** rng.uniform(5, 10)
            for mode in ("forward", "train"):
                for dtype in ("float32", "float64"):
                    for threads in (1, 2, 4):
                        t = 1e-10 * flops / threads
                        t *= (3 if mode == "train" else 1) * (2 if dtype == "float64" else 1)
                        cells.append({
                            "workload": f"{fam}-{i}", "family": fam, "mode": mode,
                            "dtype": dtype, "threads": threads, "t": t, "size": flops,
                            "features": {"flops": flops, "bytes": flops / 10, "n_ops": 20.0},
                        })
    return cells


def test_flops_baseline_is_exact_when_time_is_proportional_to_flops():
    cells = fake_cells()
    train = [c for c in cells if c["threads"] < 4]
    test = [c for c in cells if c["threads"] == 4]
    model = FlopsProportional().fit(train)
    assert stats.mape(model.predict(train), [c["t"] for c in train]) < 1e-9
    # An unseen thread count falls back on the nearest one with ideal scaling.
    assert stats.mape(model.predict(test), [c["t"] for c in test]) < 1e-9


def test_trees_cannot_extrapolate_but_the_log_linear_model_can():
    cells = fake_cells(n_per_family=12)
    cut = float(np.median([c["size"] for c in cells]))
    train = [c for c in cells if c["size"] <= cut]
    test = [c for c in cells if c["size"] > 10 * cut]
    assert test
    measured = [c["t"] for c in test]
    gbm = stats.mape(GradientBoosting().fit(train).predict(test), measured)
    lin = stats.mape(LogLinear().fit(train).predict(test), measured)
    assert lin < 0.2 < gbm


def test_class_corrections_recover_a_known_miscalibration():
    rng = random.Random(1)
    class_times = [{"gemm": rng.uniform(0.1, 1), "ew_cheap": rng.uniform(0.1, 1)}
                   for _ in range(200)]
    measured = [2.0 * ct["gemm"] + 0.5 * ct["ew_cheap"] for ct in class_times]
    fit = SimulatorCorrections(prior_weight=1e-6).fit(class_times, measured)
    assert fit.scale["gemm"] == pytest.approx(2.0, rel=0.01)
    assert fit.scale["ew_cheap"] == pytest.approx(0.5, rel=0.01)
    assert fit.scale["conv_s1"] == pytest.approx(1.0)  # never exercised, so left alone
    assert stats.mape(fit.predict(class_times), measured) < 0.01


def test_folds_keep_all_cells_of_a_workload_together():
    cells = fake_cells()
    seen: set[str] = set()
    for _, train, test, _ in evaluate.folds_random(cells):
        tr, te = {c["workload"] for c in train}, {c["workload"] for c in test}
        assert not tr & te
        assert not seen & te
        seen |= te
    assert seen == {c["workload"] for c in cells}


def test_size_split_trains_on_the_smaller_half_of_every_family():
    cells = fake_cells()
    small, large = evaluate.size_split(cells)
    assert len(small) == len(large) == 16
    for fam in FAMILIES:
        biggest_small = max(c["size"] for c in cells if c["workload"] in small
                            and c["family"] == fam)
        smallest_large = min(c["size"] for c in cells if c["workload"] in large
                             and c["family"] == fam)
        assert biggest_small < smallest_large


def test_family_and_thread_folds_hold_out_what_they_say():
    cells = fake_cells()
    for fam, train, test, _ in evaluate.folds_family(cells):
        assert {c["family"] for c in test} == {fam}
        assert fam not in {c["family"] for c in train}
    ((_, train, test, restriction),) = list(evaluate.folds_threads(cells))
    assert {c["threads"] for c in train} == {1, 2}
    assert {c["threads"] for c in test} == {4}
    assert restriction == {"max_threads": 2}


def test_mape_and_paired_difference():
    rows = [{"workload": f"w{i}", "measured": 1.0, "a": 1.1, "b": 1.3} for i in range(20)]
    assert stats.mape([r["a"] for r in rows], [1.0] * 20) == pytest.approx(0.1)
    d = stats.paired_difference(rows, "a", "b")
    assert d["difference"] == pytest.approx(-0.2)
    assert d["ci"][0] == pytest.approx(-0.2) and d["ci"][1] == pytest.approx(-0.2)


def test_cluster_bootstrap_resamples_workloads_not_cells():
    # Two workloads with very different errors and many cells each. Resampling cells
    # would give a narrow interval; resampling workloads must give a wide one.
    rows = ([{"workload": "a", "mode": "forward", "measured": 1.0 + i, "m": 1.0 + i}
             for i in range(50)]
            + [{"workload": "b", "mode": "forward", "measured": 1.0 + i, "m": 2.0 + 2 * i}
               for i in range(50)])
    s = stats.method_stats(rows, "m")
    assert s["mape"] == pytest.approx(0.5)
    assert s["mape_ci"][0] == pytest.approx(0.0) and s["mape_ci"][1] == pytest.approx(1.0)


def test_decision_quality_scores_the_chosen_thread_count_by_measured_time():
    rows = []
    for w, pred in (("good", (3.0, 2.0, 1.0)), ("bad", (1.0, 2.0, 3.0))):
        for threads, measured, p in zip((1, 2, 4), (4.0, 2.0, 1.0), pred):
            rows.append({"workload": w, "mode": "forward", "dtype": "float32",
                         "threads": threads, "measured": measured, "m": p})
    q = stats.decision_quality(rows, "m")
    assert q["choices"] == 2
    assert q["top1"] == 0.5
    assert q["mean_regret"] == pytest.approx((0.0 + 3.0) / 2)
    assert q["within_10pct"] == 0.5


def test_within_workload_rank_correlation():
    rows = [{"workload": "w", "mode": "forward", "threads": n, "measured": 1.0 / n,
             "m": 5.0 / n} for n in (1, 2, 3, 4)]
    assert stats.within_workload_spearman(rows, "m") == pytest.approx(1.0)


def test_timer_groups_fast_calls_and_reports_quartiles():
    ticks = iter(i * 1e-5 for i in range(10_000_000))
    timing = time_callable(lambda: None, clock=lambda: next(ticks), min_reps=5, min_time=0.0)
    assert timing.reps >= 5 and timing.inner > 1
    s = summarise_samples([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s.median == 3.0 and s.minimum == 1.0
    assert s.q1 == 2.0 and s.q3 == 4.0


def test_guard_keeps_the_attempt_with_the_least_interference(monkeypatch):
    loads = iter([2.0, 0.9, 1.5])
    monkeypatch.setattr("tracesim.quiet.other_cpus", lambda since: next(loads))
    calls = []
    result, load, attempts = run_quiet(lambda: calls.append(1) or len(calls), limit=0.4,
                                       attempts=3)
    assert (result, load, attempts) == (2, 0.9, 3)
    loads = iter([0.1])
    assert run_quiet(lambda: "x", limit=0.4)[1:] == (0.1, 1)
    assert other_cpus(Snapshot.take()) >= 0.0
    assert math.isfinite(other_cpus(Snapshot.take()))


def _row(session, cell, minute, t):
    return {"session": session, "workload": cell, "mode": "forward", "dtype": "float32",
            "threads": 1, "t": t, "at": 1000.0 + 60.0 * minute}


def test_clean_view_keeps_the_early_window_and_prefers_the_second_pass():
    from tracesim.measure import clean_pairs, clean_view, drift_table

    rows = [_row("main", "early", 1, 1.0), _row("main", "early-both", 2, 1.0),
            _row("main", "late", 20, 3.0), _row("main", "late-both", 21, 3.0),
            _row("repeat", "early-both", 50, 1.1), _row("repeat", "late-both", 51, 1.0)]
    view = {r["workload"]: r for r in clean_view(rows, cutoff_minutes=10)}
    assert set(view) == {"early", "early-both", "late-both"}
    assert view["early-both"]["t"] == 1.1 and view["late-both"]["t"] == 1.0
    assert all(r["session"] == "main" for r in view.values())
    pairs = clean_pairs(rows, cutoff_minutes=10)
    assert [(a["t"], b["t"]) for a, b in pairs] == [(1.0, 1.1)]
    # Minutes are counted from the first measurement, which was taken at minute 1.
    table = drift_table(rows, bin_minutes=3.0)
    assert [(b["from_minute"], b["pairs"]) for b in table] == [(0.0, 1), (18.0, 1)]
    assert table[1]["median_ratio"] == pytest.approx(3.0)
