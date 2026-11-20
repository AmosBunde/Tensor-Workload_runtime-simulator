"""Turn a PyTorch module and an example input into an operator trace.

The module is traced symbolically with ``torch.fx`` and then run once through an
interpreter that records, for every node, what kind of operation it is, the shapes it saw
and how much arithmetic and memory traffic it implies. The trace says nothing about any
device, so it can be captured on one machine and replayed against the description of
another.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import fx, nn

# Operator classes that the device model prices. "work" is counted in FLOPs for matrix
# products and convolutions and in tensor elements for the rest. Convolutions are split
# four ways because their speed per FLOP differs severalfold between the cases: 1x1
# kernels, layers with very few input channels (the first layer of a network), and larger
# kernels with and without a stride.
FLOP_CLASSES = ("gemm", "bmm", "conv_k1", "conv_thin", "conv_s1", "conv_s2")
ELEMENT_CLASSES = (
    "ew_cheap",
    "ew_transcend",
    "layernorm",
    "batchnorm_eval",
    "batchnorm_train",
    "softmax",
    "pool",
    "embedding",
    "reduce",
    "copy",
)
OP_CLASSES = FLOP_CLASSES + ELEMENT_CLASSES + ("free",)

_EW_CHEAP = {"add", "sub", "mul", "truediv", "div", "relu", "neg", "iadd"}
_EW_TRANSCEND = {"gelu", "tanh", "sigmoid", "exp", "silu"}
_SOFTMAX = {"softmax", "log_softmax", "cross_entropy"}
_REDUCE = {"mean", "sum"}
_VIEW = {
    "view", "reshape", "flatten", "transpose", "permute", "contiguous", "unsqueeze",
    "squeeze", "size", "getitem", "expand", "getattr", "dropout", "identity",
}  # fmt: skip


@dataclass
class Op:
    """One operator call in a trace."""

    name: str
    kind: str
    op_class: str
    in_shapes: list[list[int]]
    out_shape: list[int]
    dtype: str
    params: int = 0
    flops: float = 0.0
    work: float = 0.0
    items: int = 1
    bytes_read: int = 0
    bytes_written: int = 0
    deps: list[int] = field(default_factory=list)
    has_backward: bool = False
    needs_input_grad: bool = False
    phase: str = "forward"

    @property
    def working_set(self) -> int:
        return self.bytes_read + self.bytes_written


@dataclass
class Trace:
    workload: str
    mode: str  # "forward" or "train"
    dtype: str
    ops: list[Op]
    unknown: list[str] = field(default_factory=list)

    @property
    def flops(self) -> float:
        return sum(o.flops for o in self.ops)

    @property
    def params(self) -> int:
        return sum(o.params for o in self.ops if o.phase == "forward")

    def by_class(self, attr: str = "work") -> dict[str, float]:
        out = {c: 0.0 for c in OP_CLASSES}
        for o in self.ops:
            out[o.op_class] = out.get(o.op_class, 0.0) + getattr(o, attr)
        return out

    def features(self) -> dict[str, float]:
        """Totals used by the black-box baselines. They see nothing else of the trace."""
        f: dict[str, float] = {
            "flops": self.flops,
            "bytes": float(sum(o.working_set for o in self.ops)),
            "params": float(self.params),
            "n_ops": float(len(self.ops)),
            "max_working_set": float(max(o.working_set for o in self.ops)),
            "max_op_flops": float(max(o.flops for o in self.ops)),
        }
        for c, v in self.by_class("work").items():
            if c != "free":
                f[f"work_{c}"] = v
        counts = {c: 0.0 for c in OP_CLASSES}
        for o in self.ops:
            counts[o.op_class] += 1
        f["n_free"] = counts["free"]
        f["n_priced"] = float(len(self.ops)) - counts["free"]
        return f

    def to_json(self) -> dict:
        return {
            "workload": self.workload,
            "mode": self.mode,
            "dtype": self.dtype,
            "unknown": self.unknown,
            "ops": [asdict(o) for o in self.ops],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Trace":
        return cls(d["workload"], d["mode"], d["dtype"], [Op(**o) for o in d["ops"]],
                   d.get("unknown", []))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path: Path) -> "Trace":
        return cls.from_json(json.loads(path.read_text()))


def conv_class(in_channels: int, kernel: tuple[int, int], stride: tuple[int, int]) -> str:
    if tuple(kernel) == (1, 1):
        return "conv_k1"
    if in_channels <= 4:
        return "conv_thin"
    return "conv_s1" if tuple(stride) == (1, 1) else "conv_s2"


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _tensors(args) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []

    def walk(a):
        if isinstance(a, torch.Tensor):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            for b in a:
                walk(b)
        elif isinstance(a, dict):
            for b in a.values():
                walk(b)

    walk(args)
    return out


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _kind_of(node: fx.Node, module: nn.Module | None) -> str:
    if node.op == "call_module":
        return type(module).__name__.lower()
    if node.op == "call_method":
        return str(node.target)
    name = getattr(node.target, "__name__", str(node.target))
    return name.lstrip("_")


class _Recorder(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, float_dtype: str):
        super().__init__(gm)
        self.ops: list[Op] = []
        self.unknown: list[str] = []
        self.index: dict[fx.Node, int] = {}
        self.float_dtype = float_dtype

    def run_node(self, n: fx.Node):
        result = super().run_node(n)
        if n.op not in ("call_module", "call_function", "call_method"):
            return result
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        module = self.fetch_attr(n.target) if n.op == "call_module" else None
        ins = _tensors(args) + _tensors(kwargs)
        op = self._describe(n, module, ins, result)
        deps = sorted({self.index[m] for m in self._priced_inputs(n)})
        op.deps = deps
        if isinstance(result, torch.Tensor):
            op.has_backward = bool(result.requires_grad)
            op.needs_input_grad = any(t.requires_grad for t in ins)
        self.index[n] = len(self.ops)
        self.ops.append(op)
        return result

    def _priced_inputs(self, n: fx.Node) -> list[fx.Node]:
        return [m for m in n.all_input_nodes if m in self.index]

    def _describe(self, n: fx.Node, module, ins, result) -> Op:
        kind = _kind_of(n, module)
        is_tensor = isinstance(result, torch.Tensor)
        out_shape = list(result.shape) if is_tensor else []
        out_elems = result.numel() if is_tensor else 0
        out_bytes = _nbytes(result) if is_tensor else 0
        in_bytes = sum(_nbytes(t) for t in ins)
        in_elems = max((t.numel() for t in ins), default=0)
        op = Op(
            name=n.name,
            kind=kind,
            op_class="free",
            in_shapes=[list(t.shape) for t in ins],
            out_shape=out_shape,
            dtype=self.float_dtype,
        )

        def priced(cls: str, work: float, flops: float, read: int, items: int = 1,
                   params: int = 0) -> Op:
            op.op_class, op.work, op.flops, op.items = cls, float(work), float(flops), items
            op.bytes_read, op.bytes_written, op.params = int(read), out_bytes, params
            return op

        if module is not None:
            n_params = sum(p.numel() for p in module.parameters())
            p_bytes = sum(_nbytes(p) for p in module.parameters())
            if isinstance(module, nn.Linear):
                rows = ins[0].numel() // module.in_features
                flops = 2.0 * rows * module.in_features * module.out_features
                return priced("gemm", flops, flops, in_bytes + p_bytes, params=n_params)
            if isinstance(module, nn.Conv2d):
                kh, kw = module.kernel_size
                flops = 2.0 * out_elems * (module.in_channels // module.groups) * kh * kw
                cls = conv_class(module.in_channels, (kh, kw), module.stride)
                return priced(cls, flops, flops, in_bytes + p_bytes, params=n_params)
            if isinstance(module, nn.LayerNorm):
                return priced("layernorm", out_elems, out_elems, in_bytes + p_bytes,
                              params=n_params)
            if isinstance(module, nn.BatchNorm2d):
                cls = "batchnorm_train" if module.training else "batchnorm_eval"
                return priced(cls, out_elems, out_elems, in_bytes + p_bytes, params=n_params)
            if isinstance(module, nn.Embedding):
                return priced("embedding", out_elems, 0.0, in_bytes + out_bytes,
                              params=n_params)
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                return priced("pool", in_elems, in_elems, in_bytes)
            if isinstance(module, nn.ReLU):
                return priced("ew_cheap", out_elems, out_elems, in_bytes)
            if isinstance(module, (nn.GELU, nn.Tanh, nn.Sigmoid, nn.SiLU)):
                return priced("ew_transcend", out_elems, out_elems, in_bytes)
            if isinstance(module, nn.Softmax):
                return priced("softmax", in_elems, in_elems, in_bytes)
            if isinstance(module, (nn.Flatten, nn.Identity)) or (
                isinstance(module, nn.Dropout) and (not module.training or module.p == 0)
            ):
                return self._view_or_copy(op, ins, result, priced)
            self.unknown.append(kind)
            return op

        if kind == "matmul":
            a, b = ins[0], ins[1]
            m, k, cols = a.shape[-2], a.shape[-1], b.shape[-1]
            if a.dim() >= 3 and b.dim() >= 3:
                items = out_elems // (m * cols)
                flops = 2.0 * items * m * k * cols
                return priced("bmm", flops, flops, in_bytes, items=items)
            flops = 2.0 * (a.numel() // k) * k * cols
            return priced("gemm", flops, flops, in_bytes)
        if kind in _EW_CHEAP:
            if not is_tensor:
                return op
            return priced("ew_cheap", out_elems, out_elems, in_bytes)
        if kind in _EW_TRANSCEND:
            return priced("ew_transcend", out_elems, out_elems, in_bytes)
        if kind in _SOFTMAX:
            return priced("softmax", in_elems, in_elems, in_bytes)
        if kind in _REDUCE:
            return priced("reduce", in_elems, in_elems, in_bytes)
        if kind == "cat":
            return priced("copy", out_elems, 0.0, in_bytes)
        if kind in _VIEW:
            return self._view_or_copy(op, ins, result, priced)
        self.unknown.append(kind)
        return op

    @staticmethod
    def _view_or_copy(op: Op, ins, result, priced) -> Op:
        """A reshape of a non-contiguous tensor silently copies it. Price that."""
        if isinstance(result, torch.Tensor) and ins and not _shares_storage(result, ins[0]):
            return priced("copy", result.numel(), 0.0, _nbytes(ins[0]))
        return op


def capture(module: nn.Module, inputs: tuple, workload: str = "adhoc", mode: str = "forward",
            set_mode: bool = True) -> Trace:
    """Trace ``module(*inputs)``.

    ``mode="forward"`` records an inference pass. ``mode="train"`` expects a module that
    returns a scalar loss; it records the forward pass with gradients enabled, so each
    operator knows whether a backward pass will visit it, and appends one update operator
    per parameter tensor for the SGD step. ``set_mode=False`` leaves the module's own
    train/eval flag alone, which matters for batch normalisation.
    """
    if mode not in ("forward", "train"):
        raise ValueError(mode)
    floats = [p.dtype for p in module.parameters()] + [
        t.dtype for t in _tensors(inputs) if t.is_floating_point()
    ]
    float_dtype = str(floats[0]).replace("torch.", "") if floats else "float32"
    if set_mode:
        module.train(mode == "train")
    if type(module).__module__.startswith("torch.nn.modules") and not isinstance(
        module, nn.Sequential
    ):
        # fx always looks inside the root module. Wrap a bare layer so that it is
        # recorded as one operator, the way it would be inside a model.
        training = module.training
        module = nn.Sequential(module)
        module.train(training)
    gm = fx.symbolic_trace(module)
    rec = _Recorder(gm, float_dtype)
    with torch.set_grad_enabled(mode == "train"):
        rec.run(*inputs)
    ops = rec.ops
    if mode == "train":
        every = list(range(len(ops)))
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            ops.append(Op(
                name=f"sgd.{name}", kind="sgd_update", op_class="ew_cheap",
                in_shapes=[list(p.shape), list(p.shape)], out_shape=list(p.shape),
                dtype=float_dtype, flops=float(p.numel()), work=float(p.numel()),
                bytes_read=2 * _nbytes(p), bytes_written=_nbytes(p), deps=every,
                phase="update",
            ))
    return Trace(workload, mode, float_dtype, ops, sorted(set(rec.unknown)))


def capture_workload(workload, mode: str = "forward", dtype: str = "float32") -> Trace:
    """Capture one workload of the study grid."""
    from .zoo import TrainStep

    td = getattr(torch, dtype)
    model = workload.build(td)
    x, y = workload.inputs(td)
    if mode == "train":
        return capture(TrainStep(model), (x, y), workload.id, "train")
    return capture(model, (x,), workload.id, "forward")


def cached_trace(workload, mode: str, dtype: str, cache_dir: Path | None) -> Trace:
    if cache_dir is None:
        return capture_workload(workload, mode, dtype)
    path = cache_dir / f"{workload.id}__{mode}__{dtype}.json"
    if path.exists():
        return Trace.load(path)
    trace = capture_workload(workload, mode, dtype)
    trace.save(path)
    return trace


def log_span(values) -> float:
    """Orders of magnitude between the smallest and largest positive value."""
    vals = [v for v in values if v > 0]
    return math.log10(max(vals) / min(vals)) if vals else 0.0
