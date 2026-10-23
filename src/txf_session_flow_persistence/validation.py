"""Calendar and additive trading-flow QA. OI is never derived from volume."""
from __future__ import annotations
import numpy as np
import pandas as pd

FLOW_COLS = ("long", "short", "net")
AMOUNT_COLS = ("long_value", "short_value", "net_value")


def build_daily_flows(full: pd.DataFrame, night: pd.DataFrame):
    """Return canonical frame + per-day QA; negative day buys/sells fail closed.

    The provider's 'night' T belongs to the preceding 15:00--next 05:00
    session; night[T] chronologically precedes regular-day[T].
    """
    if not isinstance(full.index, pd.DatetimeIndex) or not isinstance(night.index, pd.DatetimeIndex):
        raise ValueError("trade dates must use DatetimeIndex")
    for frame in (full, night):
        if not set(FLOW_COLS + AMOUNT_COLS).issubset(frame):
            raise ValueError("missing required trading-flow columns")
        if frame.index.has_duplicates or frame.index.tz is not None:
            raise ValueError("duplicate or timezone-aware trade dates not accepted")
    # Full-session trading dates define the exchange calendar. Never join using calendar-day +1.
    if not night.index.difference(full.index).empty:
        raise ValueError("night trade dates absent from full-session calendar")
    x = full.sort_index().add_prefix("full_").join(night.sort_index().add_prefix("night_"), how="left")
    x.index.name = "trade_date"
    qa = pd.DataFrame(index=x.index)
    qa["night_present"] = x[["night_" + c for c in FLOW_COLS]].notna().all(axis=1)
    qa["all_fields_present"] = x.notna().all(axis=1)
    for session in ("full", "night"):
        for c in FLOW_COLS + AMOUNT_COLS:
            x[session + "_" + c] = pd.to_numeric(x[session + "_" + c], errors="raise")
        valid = x[[session + "_" + c for c in FLOW_COLS]].notna().all(axis=1)
        longs, shorts, nets = (x[session + "_" + c] for c in FLOW_COLS)
        qa[session + "_counts_valid"] = valid & longs.ge(0) & shorts.ge(0) & np.isclose(nets, longs - shorts, atol=0, rtol=0)
        for c in ("long", "short", "net"):
            v = x[session + "_" + c].dropna()
            if not np.equal(v, np.floor(v)).all():
                raise ValueError("nonintegral contract counts")
    for c in ("long", "short", "net", "long_value", "short_value", "net_value"):
        x["day_" + c] = x["full_" + c] - x["night_" + c]
    qa["day_counts_valid"] = (
        qa["full_counts_valid"] & qa["night_counts_valid"] &
        x["day_long"].ge(0) & x["day_short"].ge(0) &
        x["day_net"].eq(x["day_long"] - x["day_short"])
    )
    qa["amounts_valid"] = x[[p + "_" + v for p in ("full", "night") for v in AMOUNT_COLS]].notna().all(axis=1)
    # Amount unit is thousand NTD. Tolerance one for rounding across separate reports.
    for s in ("full", "night"):
        qa["amounts_valid"] &= np.isclose(x[s + "_net_value"], x[s + "_long_value"] - x[s + "_short_value"], atol=1, rtol=0)
    qa["amounts_valid"] &= x["day_long_value"].ge(-1) & x["day_short_value"].ge(-1)
    qa["valid"] = qa["all_fields_present"] & qa["day_counts_valid"] & qa["amounts_valid"]
    counts_present = x[[s + "_" + c for s in ("full", "night") for c in FLOW_COLS]].notna().all(axis=1)
    if (~qa.loc[counts_present, "day_counts_valid"]).any():
        raise ValueError("BLOCKED: invalid full/night trade-flow subtraction (counts or net)")
    # Missing night days stay visible, with NaN (never coerced to zero).
    for s in ("full", "night", "day"):
        denominator = x[f"{s}_long"] + x[f"{s}_short"]
        x[f"{s}_flow_ratio"] = x[f"{s}_net"] / denominator.where(denominator > 0)
    x["valid"] = qa["valid"]
    x["next_trade_date"] = pd.Series(x.index, index=x.index).shift(-1)
    for c in ("night_net", "night_flow_ratio"):
        x["next_" + c] = x[c].shift(-1)
    x["next_night_valid"] = x["valid"].shift(-1).astype("boolean").fillna(False).astype(bool)
    return x, qa
