"""Research orchestration: immutable raw -> canonical QA -> append-only results."""
from __future__ import annotations
import io
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from .config import ResearchConfig, FULL_DATASET, NIGHT_DATASET
from .source import fetch_finlab, normalize_session
from .storage import checksum, parquet_bytes, read_snapshot, save_snapshot, snapshot_path
from .validation import build_daily_flows
from .analysis import analyze


def _gitsha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _csv(df):
    return df.to_csv(index=True).encode("utf-8-sig")


def _figure(canonical):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    mask = canonical.valid
    ax.scatter(canonical.loc[mask, "night_flow_ratio"], canonical.loc[mask, "day_flow_ratio"],
               s=8, alpha=0.3)
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.set(xlabel="Night foreign net / gross volume", ylabel="Regular-day foreign net / gross volume",
           title="Same-trade-date night vs regular-day flow (descriptive)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def run(store, cfg: ResearchConfig, *, getter=None, refresh=False, compute_figure=True):
    cfg.validate()
    raw = {}
    manifests = {}
    now = datetime.now(timezone.utc)
    revision = now.strftime("%Y%m%dT%H%M%S%fZ") if refresh else None
    for session, dataset in (("full_session", FULL_DATASET), ("afterhours", NIGHT_DATASET)):
        path = snapshot_path(session, cfg.start, cfg.end, revision)
        frames = read_snapshot(store, path)
        if frames is None:
            frames = fetch_finlab(session, cfg.start, cfg.end, getter=getter)
            manifests[session] = save_snapshot(store, path, frames, dataset=dataset, start=cfg.start, end=cfg.end)
        else:
            manifests[session] = json.loads(store.read_raw(path + "/manifest.json"))
        raw[session] = frames
    full = normalize_session(raw["full_session"], cfg.instrument)
    night = normalize_session(raw["afterhours"], cfg.instrument)
    canonical, qa = build_daily_flows(full, night)
    results = analyze(canonical, min_n=cfg.min_n, hac_lags=cfg.hac_lags,
                      holdout_fraction=cfg.holdout_fraction)
    stamp = now.astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    run_name = f"runs/{stamp}_{_gitsha()}_{uuid.uuid4().hex[:6]}"
    raw_ids = {k: {"snapshot_path": snapshot_path(k, cfg.start, cfg.end, revision),
                   "field_sha256": manifests[k]["files"]} for k in manifests}
    manifest = {"schema_version": "1.0", "status": "ANALYSIS_COMPLETE", "data_type": "LIVE_FINLAB" if getter is None else "INJECTED_TEST_DATA",
                "run_name": run_name, "executed_at_utc": now.isoformat(), "git_sha": _gitsha(),
                "config": cfg.__dict__, "raw_snapshot_provenance": raw_ids,
                "full_trading_dates": int(len(canonical)), "valid_paired_dates": int(qa.valid.sum()),
                "unusable_dates": int((~qa.valid).sum()), "research_scope": "STATISTICAL_PERSISTENCE_ONLY_NOT_TRADING_BACKTEST"}
    artifacts = {"processed/canonical_flow.parquet": parquet_bytes(canonical),
                 "diagnostics/daily_qa.csv": _csv(qa),
                 "config/research.json": json.dumps(cfg.__dict__, indent=2, ensure_ascii=False).encode()}
    for name, table in results.items():
        artifacts[f"tables/{name}.csv"] = _csv(table)
    if compute_figure:
        artifacts["figures/day_night_flow_ratio.png"] = _figure(canonical)
    report = ["# TXF foreign-flow day/night persistence — v0.1.0", "",
              f"Run: `{run_name}`", f"Period: {cfg.start}–{cfg.end}",
              f"Instrument column (exact): `{cfg.instrument}`", "",
              f"Full trade dates: {len(canonical)}; QA-valid dates: {int(qa.valid.sum())}; excluded: {int((~qa.valid).sum())}.",
              "", "## Primary predeclared hypotheses", "",
              "H1 night[T] → day[T]; H2 day[T] → night[next trading date].", "",
              "The after-hours table's trade_date T refers to the session beginning on prior calendar trading day at 15:00.",
              "Missing/invalid rows are excluded; no zero-imputation or carry-forward.", "",
              "## Analysis results", "", "```", results["direction_summary"].to_string(index=False), "```", "",
              "## Interpretation limits", "",
              "Trade-flow net contracts are NOT open-interest changes; no OI is computed.",
              "The result measures persistence of investor trades, NOT future price returns, profit or a backtested strategy.",
              "HAC(5 default) linear-probability differences with BH-FDR across primary full-history H1/H2; holdout and annual results descriptive.",
              "The sampled flow at session end might not be published until later; no point-in-time availability guarantee.",
              "Vendor data QA is arithmetic/date-oriented, not an independent TAIFEX source-by-source audit."]
    artifacts["reports/summary.md"] = ("\n".join(report) + "\n").encode("utf-8")
    manifest["artifact_sha256"] = {path: checksum(blob) for path, blob in artifacts.items()}
    for path, blob in artifacts.items():
        store.write_result(f"{run_name}/{path}", blob)
    store.write_result(f"{run_name}/run_info.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode())
    return manifest, results
