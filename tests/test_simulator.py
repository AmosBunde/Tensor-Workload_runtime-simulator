import pytest

from tracesim.device import DeviceDescriptor, amdahl
from tracesim.simulator import op_cost, simulate
from tracesim.trace import Op, Trace, capture_workload
from tracesim.zoo import Workload


def make_op(op_class="gemm", work=1e9, read=1000, written=1000, deps=(), **kw) -> Op:
    return Op(name="op", kind="k", op_class=op_class, in_shapes=[], out_shape=[],
              dtype=kw.pop("dtype", "float32"), work=work, flops=work, bytes_read=read,
              bytes_written=written, deps=list(deps), **kw)


def trace_of(*ops: Op) -> Trace:
    return Trace("t", "forward", "float32", list(ops))


def test_compute_bound_cost_is_overhead_plus_work_over_rate(device):
    c = op_cost(make_op(work=1e9), device, threads=1)
    assert c.bound == "compute"
    assert c.total == pytest.approx(1e-5 + 1e9 / 1e10)
    assert c.memory == 0 and c.threading == 0


def test_dtype_selects_the_rate(device):
    slow = op_cost(make_op(work=1e9, dtype="float64"), device, 1)
    assert slow.total == pytest.approx(1e-5 + 1e9 / 5e9)


def test_large_working_set_becomes_memory_bound(device):
    big = 1 << 30
    c = op_cost(make_op("ew_cheap", work=1e6, read=big, written=big), device, 1)
    assert c.bound == "memory"
    miss = 1 - device.cache_bytes / (2 * big)
    assert c.total == pytest.approx(1e-5 + miss * 2 * big / 1e10)


def test_doubling_bandwidth_halves_memory_time_only(device):
    fast = device.variant("fast", bandwidth_factor=2.0)
    big = 1 << 30
    mem_op = make_op("ew_cheap", work=1e6, read=big, written=big)
    cpu_op = make_op(work=1e9)
    assert op_cost(mem_op, fast, 1).memory == pytest.approx(op_cost(mem_op, device, 1).memory / 2)
    assert op_cost(cpu_op, fast, 1).total == pytest.approx(op_cost(cpu_op, device, 1).total)
    assert fast.calibration["validated"] is False


def test_threads_follow_the_serial_fraction(device):
    one = op_cost(make_op(work=1e10), device, 1)
    four = op_cost(make_op(work=1e10), device, 4)
    assert four.total == pytest.approx(1e-5 + 1.0 * amdahl(4, 0.1) + 1e-5)
    assert four.compute == pytest.approx(0.25)
    assert four.threading == pytest.approx(four.total - 1e-5 - 0.25)
    assert four.total < one.total


def test_small_operators_stay_serial_when_forking_would_cost_more(device):
    tiny = make_op(work=1e3)
    assert op_cost(tiny, device, 4).total == pytest.approx(op_cost(tiny, device, 1).total)


def test_threads_beyond_the_cores_do_not_help(device):
    op = make_op(work=1e10)
    assert op_cost(op, device, 16).total == pytest.approx(op_cost(op, device, 4).total)
    more = device.variant("more", cores=16)
    assert op_cost(op, more, 16).total < op_cost(op, device, 16).total


def test_first_touch_cost_applies_only_above_the_threshold(device):
    device.first_touch_bytes_per_s = 1e9
    small = make_op("ew_cheap", work=10, written=1 << 20)
    large = make_op("ew_cheap", work=10, written=64 << 20)
    assert op_cost(small, device, 1).memory == 0
    assert op_cost(large, device, 1).memory >= (64 << 20) / 1e9


def test_backward_pass_uses_the_class_ratios(device):
    fwd = op_cost(make_op(work=1e9), device, 1).total
    full = op_cost(make_op(work=1e9, has_backward=True, needs_input_grad=True), device, 1)
    first = op_cost(make_op(work=1e9, has_backward=True, needs_input_grad=False), device, 1)
    assert full.backward == pytest.approx(1e-5 * 1.0 + 0.1 * 2.0)
    assert first.backward == pytest.approx(1e-5 * 1.0 + 0.1 * 1.0)
    assert full.total == pytest.approx(fwd + full.backward)


def test_sequential_replay_sums_the_operator_costs(device):
    w = Workload("transformer", 2, (("dim", 64), ("depth", 1), ("seq", 8)))
    trace = capture_workload(w, "train")
    result = simulate(trace, device, threads=2)
    assert result.total == pytest.approx(sum(c.total for c in result.op_costs))
    assert sum(result.breakdown.values()) == pytest.approx(result.total)
    assert sum(result.by_class.values()) == pytest.approx(result.total)
    assert result.finish_times == sorted(result.finish_times)


def test_two_streams_run_independent_branches_side_by_side(device):
    a = make_op(work=1e9)
    left = make_op(work=2e9, deps=[0])
    right = make_op(work=3e9, deps=[0])
    join = make_op(work=1e9, deps=[1, 2])
    trace = trace_of(a, left, right, join)
    one = simulate(trace, device, threads=1, streams=1)
    two = simulate(trace, device, threads=1, streams=2)
    cost = [c.total for c in one.op_costs]
    assert one.total == pytest.approx(sum(cost))
    assert two.total == pytest.approx(cost[0] + max(cost[1], cost[2]) + cost[3])


def test_streams_share_the_cores(device):
    ops = [make_op(work=1e10), make_op(work=1e10)]
    shared = simulate(trace_of(*ops), device, threads=4, streams=2)
    assert shared.total == pytest.approx(op_cost(ops[0], device, 2).total)


def test_class_corrections_scale_the_matching_operators(device):
    trace = trace_of(make_op("gemm", work=1e9), make_op("ew_cheap", work=1e6))
    base = simulate(trace, device, 1)
    scaled = simulate(trace, device, 1, scale={"gemm": 2.0})
    assert scaled.total == pytest.approx(base.total + base.by_class["gemm"])


def test_a_dependency_cycle_is_reported(device):
    with pytest.raises(ValueError):
        simulate(trace_of(make_op(deps=[1]), make_op(deps=[0])), device)


def test_descriptor_survives_a_json_round_trip(device, tmp_path):
    device.save(tmp_path / "d.json")
    again = DeviceDescriptor.load(tmp_path / "d.json")
    assert again.to_json() == device.to_json()
    assert again.bandwidth(4) == pytest.approx(1e10 / amdahl(4, 0.5))
