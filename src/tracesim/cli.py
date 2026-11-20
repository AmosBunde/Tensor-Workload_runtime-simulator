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


def _cmd_microbench(args) -> int:
    from .microbench import run_all

    points = run_all(Path(args.out))
    print(f"{len(points)} microbenchmark points in {args.out}")
    return 0


def _cmd_calibrate(args) -> int:
    from .calibrate import fit_descriptor, fit_report, host_info
    from .microbench import load_points
    from .quiet import pin_to_cores

    points = load_points(Path(args.points))
    if not points:
        print(f"no microbenchmark points in {args.points}; run `tracesim microbench` first")
        return 1
    # Describe the cores the microbenchmarks ran on, not the whole machine.
    pin_to_cores()
    device = fit_descriptor(points, args.name, host_info())
    device.save(Path(args.device))
    report = fit_report(points, device)
    busy = sum(p.get("other_cpus", 0) > 0.4 for p in points)
    doc = {"points": len(points), "points_measured_while_busy": busy, "fit": report,
           "measurement_minutes": sum(p.get("wall", 0) for p in points) / 60}
    (RESULTS / "calibration.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{device.name}: {device.cores} cores x {device.smt} threads, "
          f"{device.mem_bandwidth_bytes_per_s / 1e9:.1f} GB/s memory bandwidth")
    print(f"{'class':16s} {'points':>6s} {'fit MAPE':>9s}  rate (float32)   overhead")
    for cls, r in report.items():
        m = device.op_classes[cls]
        rate = m.rate.get("float32", 0.0)
        print(f"{cls:16s} {r['points']:6d} {100 * r['mape']:8.1f}%  {rate:12.3g}/s "
              f"{m.call_overhead_s * 1e6:7.1f} us")
    print(f"wrote {args.device} and {RESULTS / 'calibration.json'}")
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

    p = sub.add_parser("microbench", help="time single operators on this machine")
    p.add_argument("--out", default=str(RESULTS / "microbench.jsonl"))
    p.set_defaults(func=_cmd_microbench)

    p = sub.add_parser("calibrate", help="fit a device descriptor to the microbenchmarks")
    p.add_argument("--points", default=str(RESULTS / "microbench.jsonl"))
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--name", default="host")
    p.set_defaults(func=_cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
