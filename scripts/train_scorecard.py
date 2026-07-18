#!/usr/bin/env python3
"""Run the complete scorecard development workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lendingclub-matplotlib")
sys.path.insert(0, str(ROOT / "src"))

from credit_scorecard.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "model_config.json"))
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "LendingClub_2007_to_2018Q4.csv"))
    args = parser.parse_args()
    result = run_pipeline(args.config, args.data, ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
