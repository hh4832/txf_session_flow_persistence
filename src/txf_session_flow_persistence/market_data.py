"""TAIEX price data adapter for market-regime research (FinLab, price index)."""
from __future__ import annotations

import io
import json
import os
from datetime import date, datetime, timedelta, timezone
import pandas as pd

from .storage import checksum, parquet_bytes

TAIEX_KEY = "taiex_total_index:收盤指數"
WARMUP_CALENDAR_DAYS = 45


def price_start(start: str) -> str:
    return (date.fromisoformat(start) - timedelta(days=WARMUP_CALENDAR_DAYS)).isoformat()


def fetch_taiex(start: str, end: str, *, getter=None) -> pd.DataFrame:
    """Obtain unchanged vendor prices; no ffill, substitutes or next-day values."""
    if getter is None:
        token = os.environ.get("FINLAB_API_TOKEN")
        if not token:
            raise RuntimeError("FINLAB_API_TOKEN required for uncached TAIEX")
        try:
            import finlab
            from finlab import data
            finlab.login(token)
            getter = data.get
        except Exception:
            raise RuntimeError("FinLab authentication for TAIEX failed") from None
    try:
        frame = getter(TAIEX_KEY)
    except Exception:
        raise RuntimeError("FinLab TAIEX source download failed") from None
    return validate_price_frame(frame, start, end)


def validate_price_frame(frame, start, end):
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 1 or frame.empty:
        raise ValueError("TAIEX must be exactly one nonempty price column")
    idx = frame.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None
            or idx.has_duplicates or not idx.is_monotonic_increasing
            or not (idx == idx.normalize()).all()):
        raise ValueError("Invalid TAIEX trading-date index")
    frame = frame.loc[start:end].copy()
    if len(frame) < 10:
        raise ValueError("TAIEX history too short to compute preregistered returns")
    p = pd.to_numeric(frame.iloc[:, 0], errors="raise")
    if p.isna().any() or (p <= 0).any() or not pd.Series(p.to_numpy()).map(lambda v: bool(__import__("math").isfinite(v))).all():
        raise ValueError("TAIEX contains invalid/missing price; no forward fill")
    return frame


def taiex_snapshot_path(start: str, end: str, revision: str | None = None):
    name = f"taiex_index/{start}_{end}"
    return name if revision is None else f"{name}__revision_{revision}"


def load_taiex_snapshot(store, path):
    raw = store.read_raw(path + "/manifest.json")
    if raw is None:
        return None, None
    meta = json.loads(raw)
    if meta.get("schema_version") != "1.0" or meta.get("dataset") != TAIEX_KEY:
        raise ValueError("TAIEX snapshot manifest schema/source mismatch")
    parquet = store.read_raw(path + "/close.parquet")
    if parquet is None or checksum(parquet) != meta.get("sha256"):
        raise ValueError("TAIEX raw cache checksum mismatch")
    return pd.read_parquet(io.BytesIO(parquet)), meta


def save_taiex_snapshot(store, path, frame, *, start, end):
    if store.read_raw(path + "/manifest.json") is not None:
        raise FileExistsError("TAIEX snapshot already immutable")
    payload = parquet_bytes(frame)
    store.write_raw(path + "/close.parquet", payload)
    meta = {
        "schema_version": "1.0", "dataset": TAIEX_KEY,
        "start": start, "end": end, "sha256": checksum(payload),
        "source": "FinLab data.get", "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    store.write_raw(path + "/manifest.json",
                    json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    return meta
