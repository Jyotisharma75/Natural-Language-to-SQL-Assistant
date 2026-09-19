"""Run an evaluation dataset and write the report.

A thin wrapper around ``nl2sql evaluate`` so the suite can be run without
installing the package.

    python scripts/run_evaluation.py --dataset evaluation/datasets/sustainability_demo.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nl2sql.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
