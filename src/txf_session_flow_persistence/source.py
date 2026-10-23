"""FinLab adapter: preserve all 12 vendor-original wide tables as immutable Parquet."""
from __future__ import annotations
import os
from collections.abc import Callable
import pandas as pd
from .config import FIELDS, FULL_DATASET, NIGHT_DATASET


def fetch_finlab(session: str, start: str, end: str, *, getter: Callable | None = None) -> dict[str, pd.DataFrame]:
    """Fetch all 6 wide fields; strict errors, no fallback or synthetic trading data."""
    if session not in ("full_session", "afterhours"):
        raise ValueError("unknown trading session")
    dataset = FULL_DATASET if session == "full_session" else NIGHT_DATASET
    if getter is None:
        token = os.environ.get("FINLAB_API_TOKEN")
        if not token:
            raise RuntimeError("FINLAB_API_TOKEN missing; no live fetch attempted")
        try:
            import finlab
            from finlab import data
            finlab.login(token)
            getter = data.get
        except Exception:
            raise RuntimeError("FinLab login failed; credentials/provider details suppressed") from None
    frames = {}
    for field, suffix in FIELDS.items():
        key = f"{dataset}:{suffix}"
        try:
            df = getter(key)
        except Exception:
            raise RuntimeError(f"FinLab retrieval failed for {dataset}:{field}") from None
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"FinLab returned empty/non-table data for {session}:{field}")
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is not None:
            raise ValueError("FinLab calendar must be a timezone-naive DatetimeIndex")
        if df.index.has_duplicates or not df.index.is_monotonic_increasing:
            raise ValueError("FinLab date index duplicate/unsorted")
        if not (df.index == df.index.normalize()).all():
            raise ValueError("FinLab dates must be normalized exchange trade dates")
        if df.columns.has_duplicates:
            raise ValueError("FinLab duplicate instrument columns")
        # Retain vendor's untouched wide field, including other futures/investors.
        frames[field] = df.loc[start:end].copy()
        if frames[field].empty:
            raise ValueError(f"No data in requested period for {session}:{field}")
    return frames


def normalize_session(frames: dict[str, pd.DataFrame], instrument: str) -> pd.DataFrame:
    if set(frames) != set(FIELDS):
        raise ValueError("all six original FinLab fields are required")
    cols = {}
    for field, df in frames.items():
        if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"invalid raw field {field}")
        if df.index.has_duplicates or df.index.tz is not None:
            raise ValueError("duplicate or timezone-aware dates not accepted")
        if instrument not in df.columns:
            # Fail instead of guessing among TX, MTX, TMF or investor identities.
            raise ValueError(f"Column '{instrument}' not found in {field}; verify exact FinLab label")
        cols[field] = pd.to_numeric(df[instrument], errors="raise")
    table = pd.concat(cols, axis=1, join="outer").sort_index()
    table.index.name = "trade_date"
    return table
