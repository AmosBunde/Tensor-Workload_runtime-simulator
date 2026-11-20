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
    from .microbench import recheck, run_all

    if args.recheck:
        recheck(Path(args.out), RESULTS / "microbench_recheck.jsonl")
        return 0
    points = run_all(Path(args.out), reference=RESULTS / "reference.jsonl")
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


def _cmd_measure(args) -> int:
    from .measure import run

    only = args.only.split(",") if args.only else None
    run(Path(args.out), repeat_fraction=args.repeat_fraction, max_minutes=args.max_minutes,
        only=only, reference=RESULTS / "reference.jsonl")
    return 0


def _cmd_evaluate(args) -> int:
    from .evaluate import run
    from .measure import clean_view, load_measurements

    rows = load_measurements(Path(args.measurements), None)
    cache = None if args.no_cache else TRACE_CACHE
    out = Path(args.out)
    run([r for r in rows if r["session"] == "main"], Path(args.points), out,
        Path(args.device), cache)
    if args.clean_cutoff_minutes is not None:
        print("cells measured while the machine ran at full speed")
        run(clean_view(rows, args.clean_cutoff_minutes), Path(args.points),
            out.with_name(out.stem + "_clean.csv"), Path(args.device), cache)
    return 0


def _cmd_summarise(args) -> int:
    from .device import DeviceDescriptor
    from .measure import (CLEAN_CUTOFF_MINUTES, clean_pairs, clean_view, drift_table,
                          load_measurements)
    from .reference import load_reference, summarise_reference
    from .stats import (clean_noise, noise_floor, rescaled_simulator, summarise,
                        write_summary)

    results = Path(args.results)
    measurements = load_measurements(results / "measurements.jsonl", None)
    planned = summarise(results / "predictions.csv")
    planned["noise"] = noise_floor(measurements)
    summary = {"planned": planned, "drift": drift_table(measurements)}
    clean = None
    if (results / "predictions_clean.csv").exists():
        # Only the first run has this: an analysis of the cells measured at full speed.
        clean = summarise(results / "predictions_clean.csv")
        view = clean_view(measurements)
        clean["noise"] = clean_noise(clean_pairs(measurements), len(view))
        clean["cutoff_minutes"] = CLEAN_CUTOFF_MINUTES
        clean["workloads"] = len({r["workload"] for r in view})
        clean["cells_by_threads"] = {str(n): sum(r["threads"] == n for r in view)
                                     for n in (1, 2, 4)}
        summary["clean"] = clean
    reference = load_reference(results / "reference.jsonl")
    if reference:
        summary["reference"] = summarise_reference(reference)
    device = DeviceDescriptor.load(Path(args.device))
    summary["device"] = {
        "name": device.name, "cpu": device.notes, "cores": device.cores, "smt": device.smt,
        "cache_bytes": device.cache_bytes,
        "mem_bandwidth_gb_per_s": device.mem_bandwidth_bytes_per_s / 1e9,
        "mem_serial_fraction": device.mem_serial_fraction,
        "first_touch_gb_per_s": device.first_touch_bytes_per_s / 1e9,
        "op_classes": {k: {"rate_float32": v.rate.get("float32"),
                           "rate_float64": v.rate.get("float64"),
                           "call_overhead_us": v.call_overhead_s * 1e6,
                           "serial_fraction": v.serial_fraction,
                           "smt_yield": v.smt_yield,
                           "backward_ratio": v.backward_ratio}
                       for k, v in device.op_classes.items()},
    }
    recheck_path = results / "microbench_recheck.jsonl"
    if recheck_path.exists():
        rows = [json.loads(x) for x in recheck_path.read_text().splitlines() if x.strip()]
        ratios = sorted(r["t_calibration"] / r["t_recheck"] for r in rows)
        summary["calibration_recheck"] = {
            "timings": len(rows),
            "median_calibration_over_recheck": ratios[len(ratios) // 2],
            "quartiles": [ratios[len(ratios) // 4], ratios[3 * len(ratios) // 4]],
            "share_slower_at_calibration": sum(r > 1.0 for r in ratios) / len(ratios),
        }
        factor = ratios[len(ratios) // 2]
        if clean is not None:
            clean["exploratory_rescaled_simulator"] = rescaled_simulator(
                results / "predictions_clean.csv", factor)
        planned["exploratory_rescaled_simulator"] = rescaled_simulator(
            results / "predictions.csv", factor)
    for name in ("calibration", "whatif"):
        path = results / f"{name}.json"
        if path.exists():
            summary[name] = json.loads(path.read_text())
    summary["study"] = {
        "workloads": len({m["workload"] for m in measurements}),
        "families": len({m["family"] for m in measurements}),
        "cells": sum(m["session"] == "main" for m in measurements),
        "device_configurations": len({(m["threads"], m["dtype"]) for m in measurements}),
    }
    write_summary(summary, results / "summary.json")
    print(f"wrote {results / 'summary.json'}")
    methods = ("sim", "simfit", "flops", "gbm", "loglinear")
    blocks = [("all cells, as planned", planned)]
    if clean is not None:
        blocks.append(("clean windows", clean))
    for label, block in blocks:
        print(f"{label}: noise floor {100 * block['noise']['repeat_mape']:.1f}%")
        print(f"  {'design':10s} " + " ".join(f"{m:>10s}" for m in methods))
        for design, d in block["designs"].items():
            print(f"  {design:10s} " + " ".join(
                f"{100 * d['methods'][m]['mape']:9.1f}%" for m in methods))
        print("  " + ", ".join(f"{name}: {'supported' if h['supported'] else 'not supported'}"
                               for name, h in block["hypotheses"].items()))
    return 0


def _cmd_whatif(args) -> int:
    from .device import DeviceDescriptor
    from .measure import load_measurements
    from .whatif import NOTE, breakdowns, project

    host = DeviceDescriptor.load(Path(args.device))
    measured = {(m["workload"], m["mode"], m["dtype"], m["threads"]): m["t"]
                for m in load_measurements(Path(args.measurements), "main")}
    ids = ["mlp-w64-d2-b1", "mlp-w2048-d6-b512", "cnn-w16-b1-i32-b1", "cnn-w64-b2-i64-b4",
           "transformer-d64-d2-s32-b1", "transformer-d512-d4-s128-b8",
           "mixer-d64-d2-i32-b1", "mixer-d512-d4-i64-b8"]
    doc = {"projection": project(host, measured),
           "breakdown": {"threads": host.cores, "workloads": breakdowns(host, ids, host.cores)}}
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(NOTE)
    for key, entry in doc["projection"]["workloads"].items():
        line = ", ".join(f"{label}: {v['speedup_over_host']:.2f}x"
                         for label, v in entry["scenarios"].items() if label != "host")
        print(f"  {key}: {line}")
    print(f"wrote {args.out}")
    return 0


def _cmd_figures(args) -> int:
    from .figures import draw_all

    draw_all(Path(args.results), Path(args.out))
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
    p.add_argument("--recheck", action="store_true",
                   help="time a sample of cases again to see if the machine has changed")
    p.set_defaults(func=_cmd_microbench)

    p = sub.add_parser("calibrate", help="fit a device descriptor to the microbenchmarks")
    p.add_argument("--points", default=str(RESULTS / "microbench.jsonl"))
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--name", default="host")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("measure", help="time the workload grid on this machine (resumable)")
    p.add_argument("--out", default=str(RESULTS / "measurements.jsonl"))
    p.add_argument("--repeat-fraction", type=float, default=0.2,
                   help="share of cells measured a second time to estimate noise")
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--only", help="comma-separated workload ids, for a quick check")
    p.set_defaults(func=_cmd_measure)

    p = sub.add_parser("evaluate", help="run the four test designs and write predictions")
    p.add_argument("--measurements", default=str(RESULTS / "measurements.jsonl"))
    p.add_argument("--points", default=str(RESULTS / "microbench.jsonl"))
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--out", default=str(RESULTS / "predictions.csv"))
    p.add_argument("--no-cache", action="store_true", help="do not cache traces on disk")
    p.add_argument("--clean-cutoff-minutes", type=float, default=None,
                   help="also analyse only the cells measured before this minute of the "
                        "first pass or in the second pass (used for the first run: 10)")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("whatif", help="project runtimes on devices that were not measured")
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--measurements", default=str(RESULTS / "measurements.jsonl"))
    p.add_argument("--out", default=str(RESULTS / "whatif.json"))
    p.set_defaults(func=_cmd_whatif)

    p = sub.add_parser("summarise", help="compute every statistic quoted in the documents")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--device", default=str(DEVICE))
    p.set_defaults(func=_cmd_summarise)

    p = sub.add_parser("figures", help="draw the figures")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--out", default="docs/img")
    p.set_defaults(func=_cmd_figures)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
