"""The workloads: four architecture families written from primitive operators.

Every model here is built so that ``torch.fx`` can trace it and so that each tensor
operation shows up as its own node. Attention, for example, is written out as matrix
products and a softmax instead of calling ``nn.MultiheadAttention``, which would hide the
work inside one opaque module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

FAMILIES = ("mlp", "cnn", "transformer", "mixer")
N_CLASSES = 10
VOCAB = 1000


class MLP(nn.Module):
    def __init__(self, width: int, depth: int, in_dim: int = 512):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers.append(nn.Linear(d, N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        self.has_shortcut = stride != 1 or cin != cout
        if self.has_shortcut:
            self.short_conv = nn.Conv2d(cin, cout, 1, stride, 0, bias=False)
            self.short_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.has_shortcut:
            x = self.short_bn(self.short_conv(x))
        return self.relu(y + x)


class ResNet(nn.Module):
    """Three stages of basic residual blocks; widths w, 2w, 4w."""

    def __init__(self, width: int, blocks: int):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(width)
        self.relu = nn.ReLU()
        layers: list[nn.Module] = []
        cin = width
        for stage in range(3):
            cout = width * 2**stage
            for b in range(blocks):
                layers.append(BasicBlock(cin, cout, 2 if (b == 0 and stage > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(cin, N_CLASSES)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem(x)))
        x = self.layers(x)
        return self.head(self.flatten(self.pool(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, seq: int):
        super().__init__()
        self.heads, self.seq, self.dim = heads, seq, dim
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.qkv(x).reshape(-1, self.seq, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(-1, self.seq, self.dim)
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, seq: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, seq)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class Transformer(nn.Module):
    """Pre-norm transformer encoder over token ids, mean-pooled into a classifier."""

    def __init__(self, dim: int, depth: int, seq: int):
        super().__init__()
        heads = max(1, dim // 64)
        self.embed = nn.Embedding(VOCAB, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq, dim))
        self.blocks = nn.Sequential(*[EncoderBlock(dim, heads, seq) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_CLASSES)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos
        x = self.norm(self.blocks(x))
        return self.head(x.mean(1))


class MixerBlock(nn.Module):
    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.tok1 = nn.Linear(tokens, tokens)
        self.tok_act = nn.GELU()
        self.tok2 = nn.Linear(tokens, tokens)
        self.norm2 = nn.LayerNorm(dim)
        self.ch1 = nn.Linear(dim, 4 * dim)
        self.ch_act = nn.GELU()
        self.ch2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        y = self.tok2(self.tok_act(self.tok1(y))).transpose(1, 2)
        x = x + y
        return x + self.ch2(self.ch_act(self.ch1(self.norm2(x))))


class Mixer(nn.Module):
    """MLP-Mixer: a patch embedding, then blocks that mix across tokens and channels."""

    def __init__(self, dim: int, depth: int, image: int, patch: int = 4):
        super().__init__()
        tokens = (image // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.blocks = nn.Sequential(*[MixerBlock(dim, tokens) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_CLASSES)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.norm(self.blocks(x))
        return self.head(x.mean(1))


class TrainStep(nn.Module):
    """Wraps a model with its loss so that the loss is part of the traced graph."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, y):
        return F.cross_entropy(self.model(x), y)


@dataclass(frozen=True)
class Workload:
    family: str
    batch: int
    params: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        parts = [self.family] + [f"{k[0]}{v}" for k, v in self.params] + [f"b{self.batch}"]
        return "-".join(parts)

    @property
    def p(self) -> dict[str, int]:
        return dict(self.params)

    def build(self, dtype: torch.dtype = torch.float32, seed: int = 0) -> nn.Module:
        torch.manual_seed(seed)
        p = self.p
        if self.family == "mlp":
            model = MLP(p["width"], p["depth"])
        elif self.family == "cnn":
            model = ResNet(p["width"], p["blocks"])
        elif self.family == "transformer":
            model = Transformer(p["dim"], p["depth"], p["seq"])
        elif self.family == "mixer":
            model = Mixer(p["dim"], p["depth"], p["image"])
        else:
            raise ValueError(f"unknown family {self.family!r}")
        return model.to(dtype)

    def inputs(self, dtype: torch.dtype = torch.float32, seed: int = 0):
        """Return ``(x, y)``: one input batch and integer class targets."""
        g = torch.Generator().manual_seed(seed + 1)
        p = self.p
        if self.family == "mlp":
            x = torch.randn(self.batch, 512, generator=g).to(dtype)
        elif self.family == "cnn":
            x = torch.randn(self.batch, 3, p["image"], p["image"], generator=g).to(dtype)
        elif self.family == "transformer":
            x = torch.randint(0, VOCAB, (self.batch, p["seq"]), generator=g)
        else:
            x = torch.randn(self.batch, 3, p["image"], p["image"], generator=g).to(dtype)
        y = torch.randint(0, N_CLASSES, (self.batch,), generator=g)
        return x, y


def _w(family: str, batch: int, **params: int) -> Workload:
    return Workload(family, batch, tuple(params.items()))


def grid() -> list[Workload]:
    """The 96 workloads of the validation study, 24 per family."""
    out: list[Workload] = []
    for width in (64, 256, 1024, 2048):
        for depth in (2, 6):
            for batch in (1, 32, 512):
                out.append(_w("mlp", batch, width=width, depth=depth))
    for width in (16, 32, 64):
        for blocks in (1, 2):
            for image in (32, 64):
                for batch in (1, 4):
                    out.append(_w("cnn", batch, width=width, blocks=blocks, image=image))
    for dim in (64, 256, 512):
        for depth in (2, 4):
            for seq in (32, 128):
                for batch in (1, 8):
                    out.append(_w("transformer", batch, dim=dim, depth=depth, seq=seq))
    for dim in (64, 256, 512):
        for depth in (2, 4):
            for image in (32, 64):
                for batch in (1, 8):
                    out.append(_w("mixer", batch, dim=dim, depth=depth, image=image))
    return out


def by_id(workload_id: str) -> Workload:
    for w in grid():
        if w.id == workload_id:
            return w
    raise KeyError(workload_id)
