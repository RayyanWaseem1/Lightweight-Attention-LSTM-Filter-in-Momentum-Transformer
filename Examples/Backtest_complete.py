"""Legacy entry point for the ensemble backtest.

Use ``Examples/run_backtest.py --model ensemble`` for the maintained
implementation. This wrapper keeps old commands working without executing the
stale duplicated backtest logic.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv=None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from .run_backtest import main as run_backtest_main
    except ImportError:
        from run_backtest import main as run_backtest_main

    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--model" or arg.startswith("--model=") for arg in args):
        args = ["--model", "ensemble"] + args
    return run_backtest_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
