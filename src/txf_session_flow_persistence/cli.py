"""Run from Colab or an authenticated local environment; no automated trading."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import DEFAULT_INSTRUMENT, RAW_FOLDER_ID, RESULTS_FOLDER_ID, ResearchConfig
from .runner import run
from .storage import LocalStore, GoogleDriveStore


def main(argv=None):
    parser = argparse.ArgumentParser(description="TXF institutional day/night trade-flow persistence research")
    parser.add_argument("--start", default="2023-10-16")
    parser.add_argument("--end", required=True, help="Last FINLAB trade date YYYY-MM-DD")
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="EXACT FinLab column label, e.g. 臺股期貨_外資及陸資")
    parser.add_argument("--backend", choices=["drive", "local"], default="drive")
    parser.add_argument("--local-root", type=Path, default=Path("./local_data"))
    parser.add_argument("--refresh", action="store_true", help="Opt in to new immutable raw revision; never overwrite existing")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    cfg = ResearchConfig(start=args.start, end=args.end, instrument=args.instrument).validate()
    if args.backend == "drive":
        store = GoogleDriveStore(RAW_FOLDER_ID, RESULTS_FOLDER_ID)
    else:
        store = LocalStore(args.local_root / "raw", args.local_root / "results")
    info, _ = run(store, cfg, refresh=args.refresh, compute_figure=not args.no_figure)
    print(json.dumps({k: info[k] for k in ("status", "run_name", "full_trading_dates", "valid_paired_dates", "unusable_dates")}, indent=2))


if __name__ == "__main__":
    main()
