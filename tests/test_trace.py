import torch
from torch import nn

from tracesim.trace import Trace, capture, capture_workload, conv_class
from tracesim.zoo import FAMILIES, Attention, TrainStep, Workload, grid


def small(family: str) -> Workload:
    params = {
        "mlp": dict(width=32, depth=2),
        "cnn": dict(width=8, blocks=1, image=16),
        "transformer": dict(dim=64, depth=1, seq=8),
        "mixer": dict(dim=32, depth=1, image=16),
    }[family]
    return Workload(family, 2, tuple(params.items()))


def test_linear_flops_and_bytes_are_exact():
    trace = capture(nn.Linear(20, 30), (torch.randn(7, 20),))
    (op,) = trace.ops
    assert op.op_class == "gemm"
    assert op.flops == 2 * 7 * 20 * 30
    assert op.params == 20 * 30 + 30
    assert op.bytes_read == 4 * (7 * 20 + 20 * 30 + 30)
    assert op.bytes_written == 4 * 7 * 30


def test_linear_on_a_sequence_counts_every_token():
    trace = capture(nn.Linear(16, 16), (torch.randn(3, 5, 16),))
    assert trace.ops[0].flops == 2 * (3 * 5) * 16 * 16


def test_conv_flops_and_classes():
    conv = nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False)
    (op,) = capture(conv, (torch.randn(2, 8, 10, 10),)).ops
    assert op.out_shape == [2, 16, 5, 5]
    assert op.flops == 2 * (2 * 16 * 5 * 5) * 8 * 9
    assert op.op_class == "conv_s2"
    assert conv_class(8, (1, 1), (2, 2)) == "conv_k1"
    assert conv_class(3, (3, 3), (1, 1)) == "conv_thin"
    assert conv_class(8, (3, 3), (1, 1)) == "conv_s1"


def test_float64_doubles_the_bytes_but_not_the_work():
    a = capture(nn.Linear(8, 8), (torch.randn(4, 8),)).ops[0]
    b = capture(nn.Linear(8, 8).double(), (torch.randn(4, 8).double(),)).ops[0]
    assert b.dtype == "float64" and a.dtype == "float32"
    assert b.flops == a.flops
    assert b.bytes_written == 2 * a.bytes_written


def test_attention_products_are_batched_matmuls():
    attn = Attention(dim=64, heads=2, seq=8)
    trace = capture(attn, (torch.randn(3, 8, 64),))
    bmm = [o for o in trace.ops if o.op_class == "bmm"]
    assert len(bmm) == 2
    assert bmm[0].items == 3 * 2
    assert bmm[0].flops == 2 * (3 * 2) * 8 * 32 * 8


def test_reshape_of_a_transposed_tensor_is_priced_as_a_copy():
    class M(nn.Module):
        def forward(self, x):
            return x.transpose(1, 2).reshape(x.size(0), -1), x.view(x.size(0), -1)

    trace = capture(M(), (torch.randn(2, 3, 4),))
    by_kind = {o.kind: o.op_class for o in trace.ops}
    assert by_kind["reshape"] == "copy"
    assert by_kind["view"] == "free"
    assert by_kind["transpose"] == "free"


def test_every_family_traces_with_no_unknown_operator():
    for family in FAMILIES:
        for mode in ("forward", "train"):
            trace = capture_workload(small(family), mode)
            assert trace.unknown == [], (family, mode, trace.unknown)
            assert trace.flops > 0


def test_training_trace_adds_one_update_per_parameter_tensor():
    w = small("mlp")
    model = w.build()
    x, y = w.inputs()
    trace = capture(TrainStep(model), (x, y), mode="train")
    updates = [o for o in trace.ops if o.phase == "update"]
    assert len(updates) == len(list(model.parameters()))
    assert sum(o.work for o in updates) == sum(p.numel() for p in model.parameters())
    forward = [o for o in trace.ops if o.phase == "forward"]
    assert all(o.has_backward for o in forward if o.op_class != "free")
    # The first layer reads the raw input, so no gradient flows further back from it.
    assert forward[0].op_class == "gemm" and not forward[0].needs_input_grad
    assert forward[2].op_class == "gemm" and forward[2].needs_input_grad


def test_forward_trace_has_no_backward_work():
    trace = capture_workload(small("cnn"), "forward")
    assert not any(o.has_backward for o in trace.ops)
    assert {o.op_class for o in trace.ops} >= {"conv_thin", "conv_s1", "batchnorm_eval"}


def test_batchnorm_class_follows_the_mode():
    classes = {o.op_class for o in capture_workload(small("cnn"), "train").ops}
    assert "batchnorm_train" in classes and "batchnorm_eval" not in classes


def test_dependencies_point_backwards_and_cover_the_residual():
    trace = capture_workload(small("transformer"), "forward")
    for i, op in enumerate(trace.ops):
        assert all(d < i for d in op.deps)
    assert any(len(op.deps) == 2 for op in trace.ops if op.kind == "add")


def test_trace_survives_a_json_round_trip(tmp_path):
    trace = capture_workload(small("mixer"), "train")
    trace.save(tmp_path / "t.json")
    again = Trace.load(tmp_path / "t.json")
    assert again.to_json() == trace.to_json()
    assert again.features() == trace.features()


def test_grid_has_96_distinct_workloads_in_four_equal_families():
    ids = [w.id for w in grid()]
    assert len(ids) == len(set(ids)) == 96
    for family in FAMILIES:
        assert sum(w.family == family for w in grid()) == 24
