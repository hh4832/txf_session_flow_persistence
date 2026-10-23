"""Orchestrate market-regime analysis using immutable v0.1.0 flow snapshots."""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import FULL_DATASET, NIGHT_DATASET, ResearchConfig
from .market_data import (
    TAIEX_KEY, fetch_taiex, load_taiex_snapshot, price_start,
    save_taiex_snapshot, taiex_snapshot_path, validate_price_frame,
)
from .regime_analysis import analyze_regimes
from .source import normalize_session
from .storage import checksum, parquet_bytes, read_snapshot, snapshot_path
from .validation import build_daily_flows


def repo_sha():
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _csv(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def run_regime(store, cfg: ResearchConfig, *, getter=None, refresh_price=False):
    """Reuse validated full/night flow snapshots; fetch only absent TAIEX prices."""
    cfg.validate()
    frames = {}
    flow_manifest = {}
    for session, source in (("full_session", FULL_DATASET), ("afterhours", NIGHT_DATASET)):
        path = snapshot_path(session, cfg.start, cfg.end)
        frame = read_snapshot(store, path)
        if frame is None:
            raise RuntimeError("Raw " + session + " is absent; run v0.1.0 with identical date range first")
        meta = json.loads(store.read_raw(path + "/manifest.json"))
        if meta.get("dataset") != source or meta.get("start") != cfg.start or meta.get("end") != cfg.end:
            raise ValueError("Flow snapshot source/range mismatch")
        frames[session] = normalize_session(frame, cfg.instrument)
        flow_manifest[session] = {"path": path, "source": source, "sha256": meta["files"]}

    canonical, qa = build_daily_flows(frames["full_session"], frames["afterhours"])
    now = datetime.now(timezone.utc)
    beginning = price_start(cfg.start)
    revision = now.strftime("%Y%m%dT%H%M%S%fZ") if refresh_price else None
    price_path = taiex_snapshot_path(beginning, cfg.end, revision)
    taiex, price_meta = load_taiex_snapshot(store, price_path)
    if taiex is None:
        taiex = fetch_taiex(beginning, cfg.end, getter=getter)
        price_meta = save_taiex_snapshot(store, price_path, taiex, start=beginning, end=cfg.end)
    else:
        taiex = validate_price_frame(taiex, beginning, cfg.end)
    analysis = analyze_regimes(canonical, taiex, min_n=cfg.min_n,
                               hac_lags=cfg.hac_lags, holdout_fraction=cfg.holdout_fraction)
    nowlocal = now.astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    sha = repo_sha()
    run_name = "runs/" + nowlocal + "_" + sha + "_" + uuid.uuid4().hex[:6]

    tests = analysis["interaction_tests"]
    primary = tests.loc[tests.lookback.eq(3) & tests.subset.eq("full_history")]
    holdout = tests.loc[tests.lookback.eq(3) & tests.subset.eq("chronological_holdout")]
    report = [
        "# TXF foreign-flow / market-regime conditional analysis v0.2.0",
        "", "Run: " + run_name, "Period: " + cfg.start + " to " + cfg.end,
        "Source: FinLab " + TAIEX_KEY, "",
        "## Primary 3-day interaction tests (H1 and H2)",
        "Interaction measures the buy-vs-sell effect change per training-SD of trailing TAIEX return.",
        "HAC with year fixed effects and 5-day realized volatility controls.",
        "BH-FDR across all six full-sample interactions (2 hypotheses x 3 return horizons).",
        "", "~~~", primary.to_string(index=False), "~~~",
        "", "## Chronological holdout",
        "~~~", holdout.to_string(index=False), "~~~", "",
        "## Interpretation limits",
        "H1 trends end at T-1 close; H2 trends end at T close.",
        "Up/flat/down boundaries are trained only in the earlier 70% and frozen on holdout.",
        "Within-regime contrasts are descriptive only, not separate significance tests.",
        "Effect modification does not prove causality, nor establish trading profitability.",
        "No contemporaneous publication-time validation or independent exchange reconciliation.",
        "Prior v0.1.0 snapshots/results are untouched.",
    ]
    artifacts = {
        "processed/market_features.parquet": parquet_bytes(analysis["market_features"]),
        "diagnostics/daily_qa.csv": _csv(qa.reset_index()),
        "tables/interaction_tests.csv": _csv(tests),
        "tables/regime_probabilities.csv": _csv(analysis["regime_probabilities"]),
        "tables/market_feature_coverage.csv": _csv(analysis["market_feature_coverage"]),
        "reports/summary.md": ("\n".join(report) + "\n").encode("utf-8"),
        "config/research.json": json.dumps({
            "flow_config": cfg.__dict__, "analysis_version": "0.2.0",
            "predeclared_lookbacks": [1, 3, 5], "primary_lookback": 3,
            "regimes": "training terciles", "controls": ["TAIEX_vol5", "calendar_year"],
            "six_test_FDR_family": True,
        }, indent=2, ensure_ascii=False).encode("utf-8"),
    }
    manifest = {
        "schema_version": "1.0", "status": "ANALYSIS_COMPLETE",
        "analysis_version": "0.2.0", "analysis_type": "MARKET_REGIME_EFFECT_MODIFICATION",
        "data_type": "LIVE_FINLAB" if getter is None else "INJECTED_TEST_DATA",
        "run_name": run_name, "git_sha": sha, "executed_at_utc": now.isoformat(),
        "flow_provenance": flow_manifest,
        "taiex_provenance": {"path": price_path, "source": TAIEX_KEY, "sha256": price_meta["sha256"]},
        "flow_qa_valid_days": int(qa.valid.sum()),
        "feature_coverage": analysis["market_feature_coverage"].to_dict("records"),
        "artifact_sha256": {name: checksum(blob) for name, blob in artifacts.items()},
        "scope": "STATISTICAL_ASSOCIATION_NOT_TRADING_SIGNAL",
    }
    for name, blob in artifacts.items():
        store.write_result(run_name + "/" + name, blob)
    store.write_result(run_name + "/run_info.json",
                       json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    return manifest, analysis
