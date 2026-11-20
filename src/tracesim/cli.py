"""Command line entry point: one subcommand per step of the pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("results")
DEVICE = Path("devices/host.json")
TRACE_CACHE = Path("data/traces")


def _cmd_capture(args) -> int:
    from .trace import capture_workload
    from .zoo import by_id, grid

    if args.list:
        for w in grid():
            print(w.id)
        return 0
    trace = capture_workload(by_id(args.workload), args.mode, args.dtype)
    if args.out:
        trace.save(Path(args.out))
        print(f"wrote {len(trace.ops)} operators to {args.out}")
        return 0
    print(f"{trace.workload} ({trace.mode}, {trace.dtype}): {len(trace.ops)} operators, "
          f"{trace.flops:.3g} FLOPs, {trace.params:,} parameters")
    print(f"{'operator':32s} {'class':16s} {'output shape':22s} {'work':>10s} {'bytes':>10s}")
    for op in trace.ops:
        if op.op_class == "free" and not args.all:
            continue
        print(f"{op.name[:32]:32s} {op.op_class:16s} {str(op.out_shape):22s} "
              f"{op.work:10.3g} {op.working_set:10.3g}")
    return 0


def _cmd_predict(args) -> int:
    from .device import DeviceDescriptor
    from .simulator import simulate
    from .trace import Trace, capture_workload
    from .zoo import by_id

    device = DeviceDescriptor.load(Path(args.device))
    if args.trace:
        trace = Trace.load(Path(args.trace))
    else:
        trace = capture_workload(by_id(args.workload), args.mode, args.dtype)
    result = simulate(trace, device, args.threads, args.streams)
    if args.json:
        print(json.dumps({"total_s": result.total, "breakdown": result.breakdown,
                          "by_class": result.by_class}, indent=2))
        return 0
    print(f"{trace.workload} ({trace.mode}, {trace.dtype}) on {device.name}, "
          f"{args.threads} thread(s): predicted {result.total * 1e3:.3f} ms")
    total = sum(result.breakdown.values()) or 1.0
    print("where the time goes:")
    for part, seconds in result.breakdown.items():
        print(f"  {part:10s} {seconds * 1e3:10.3f} ms  {100 * seconds / total:5.1f}%")
    print("by operator class:")
    for cls, seconds in sorted(result.by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:16s} {seconds * 1e3:10.3f} ms  {100 * seconds / result.total:5.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracesim",
        description="Predict the runtime of tensor workloads from an operator trace.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def workload_args(p):
        p.add_argument("--workload", default="transformer-d256-d2-s128-b8",
                       help="a workload id; see `tracesim capture --list`")
        p.add_argument("--mode", choices=("forward", "train"), default="forward")
        p.add_argument("--dtype", choices=("float32", "float64"), default="float32")

    p = sub.add_parser("capture", help="turn a workload into an operator trace")
    workload_args(p)
    p.add_argument("--list", action="store_true", help="list the workload ids and stop")
    p.add_argument("--all", action="store_true", help="also print operators that cost nothing")
    p.add_argument("--out", help="write the trace as JSON instead of printing it")
    p.set_defaults(func=_cmd_capture)

    p = sub.add_parser("predict", help="replay a trace against a device descriptor")
    workload_args(p)
    p.add_argument("--trace", help="a trace JSON written by `capture --out`")
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--streams", type=int, default=1,
                   help="operators allowed to run at once (not validated above 1)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
