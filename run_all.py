"""Single entry point for the pipeline. Usage: python run_all.py [--synthetic]

Real data is used only if every required file is in data/raw/. Otherwise the whole run
uses synthetic data, never a mix of the two, and writes to separate output directories.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config import REAL_PATHS, SYNTHETIC_PATHS, RunPaths
from src.data.features import build_and_write, build_daily_tables
from src.data.loaders import DataFileError, assert_no_synthetic_files, missing_required
from src.data.synthetic import generate_synthetic_raw
from src.eval.challengers import run_challengers
from src.eval.compare import run_comparison
from src.eval.evaluate import run_evaluation
from src.eval.export import run_export
from src.eval.figures import write_figures
from src.models.garch import run_garch
from src.models.lstm import run_lstm
from src.models.mlp import run_mlp
from src.models.train import run_training
from src.sentiment.pipeline import run_sentiment


def choose_run(force_synthetic: bool) -> RunPaths:
    missing = missing_required(REAL_PATHS.raw)
    if not force_synthetic and not missing:
        assert_no_synthetic_files(REAL_PATHS.raw)
        return REAL_PATHS

    if missing and not force_synthetic:
        print("Required data files are missing:")
        for item in missing:
            print(f"  - {item}")
    print("\n" + "=" * 72)
    print("SYNTHETIC RUN: all inputs are simulated. Results are NOT reportable.")
    print(f"Outputs go to {SYNTHETIC_PATHS.processed} and {SYNTHETIC_PATHS.artifacts}")
    print("=" * 72)
    if missing_required(SYNTHETIC_PATHS.raw):
        generate_synthetic_raw(SYNTHETIC_PATHS.raw)
    return SYNTHETIC_PATHS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data even if real files exist")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        paths = choose_run(args.synthetic)
        # Bars come first because the kept sessions are the calendar that headlines roll onto.
        daily = build_daily_tables(paths)
        run_sentiment(paths, daily.dates)
        build_and_write(paths, daily)
        run_training(paths)
        evaluation = run_evaluation(paths)  # the only place test forecasts are made, after the freeze
        write_figures(paths, evaluation)
        run_export(paths, evaluation)
        run_garch(paths)
        run_mlp(paths)
        run_lstm(paths)
        run_comparison(paths)  # the only place their test forecasts are made, after their freeze
        run_challengers(paths)  # no-news versions, scored on the later check and never on the test period
    except DataFileError as error:
        print(f"\nDATA ERROR: {error}", file=sys.stderr)
        return 1
    print(f"\nTotal time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
