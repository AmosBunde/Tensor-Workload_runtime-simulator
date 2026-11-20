"""Command line entry point. Subcommands are added as the parts are built."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tracesim",
        description="Predict the runtime of tensor workloads from an operator trace.",
    )
    parser.add_subparsers(dest="command")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
