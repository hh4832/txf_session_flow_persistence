from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from txf_session_flow_persistence.config import ResearchConfig, DEFAULT_INSTRUMENT, FIELDS
from txf_session_flow_persistence.source import normalize_session
from txf_session_flow_persistence.validation import build_daily_flows
from txf_session_flow_persistence.analysis import make_pairs, analyze


def canonical(fake_vendor):
    return build_daily_flows(*(normalize_session(fake_vendor[key], DEFAULT_INSTRUMENT) for key in ("full_session", "afterhours")))


def test_difference_and_trade_date_alignment(fake_vendor):
    df, qa = canonical(fake_vendor)
    assert qa.valid.all()
    assert (df.day_long + df.night_long == df.full_long).all()
    assert (df.day_short + df.night_short == df.full_short).all()
    assert (df.day_net + df.night_net == df.full_net).all()
    assert df.iloc[0].next_trade_date == df.index[1]
    pairs = make_pairs(df, "H1_night_to_day")
    assert np.allclose(pairs.predictor_net, df.night_net)
    assert np.allclose(pairs.outcome_net, df.day_net)
    pairs2 = make_pairs(df, "H2_day_to_next_night")
    assert pairs2.iloc[0].outcome_date == df.index[1]
    assert pairs2.iloc[0].outcome_net == df.iloc[1].night_net


def test_no_calendar_plus_one_or_future_prediction(fake_vendor):
    df, _ = canonical(fake_vendor)
    fri = df.index[df.index.weekday == 4][0]
    next_date = df.loc[fri, "next_trade_date"]
    assert next_date.weekday() == 0
    pairs = make_pairs(df, "H2_day_to_next_night")
    assert pairs.loc[fri, "outcome_date"] == next_date


def test_missing_night_stays_nan_and_no_jump(fake_vendor):
    raw = deepcopy(fake_vendor)
    absent = next(iter(raw["afterhours"].values())).index[3]
    for key in FIELDS:
        raw["afterhours"][key] = raw["afterhours"][key].drop(index=absent)
    df, qa = build_daily_flows(normalize_session(raw["full_session"], DEFAULT_INSTRUMENT),
                               normalize_session(raw["afterhours"], DEFAULT_INSTRUMENT))
    assert not bool(qa.loc[absent, "valid"])
    assert pd.isna(df.loc[absent, "night_net"])
    p = make_pairs(df, "H2_day_to_next_night")
    assert df.index[2] not in p.index  # cannot jump over a missing night to day 4


def test_wrong_net_rejected(fake_vendor):
    raw = deepcopy(fake_vendor)
    f = raw["full_session"]["net"]
    f.loc[f.index[0], DEFAULT_INSTRUMENT] += 1
    with pytest.raises(ValueError, match="BLOCKED"):
        canonical(raw)


def test_night_more_than_full_rejected(fake_vendor):
    raw = deepcopy(fake_vendor)
    row = raw["afterhours"]["long"].index[0]
    raw["afterhours"]["long"].loc[row, DEFAULT_INSTRUMENT] += 20000
    raw["afterhours"]["net"].loc[row, DEFAULT_INSTRUMENT] += 20000
    raw["afterhours"]["long_value"].loc[row, DEFAULT_INSTRUMENT] += 2000000
    raw["afterhours"]["net_value"].loc[row, DEFAULT_INSTRUMENT] += 2000000
    with pytest.raises(ValueError, match="BLOCKED"):
        canonical(raw)


def test_refuse_ambiguous_or_missing_instrument(fake_vendor):
    with pytest.raises(ValueError, match="not found"):
        normalize_session(fake_vendor["full_session"], "TX_UNKNOWN")


def test_no_oi_columns(fake_vendor):
    df, _ = canonical(fake_vendor)
    assert not any("oi" in name.lower() or "未平倉" in name for name in df.columns)


def test_statistics_and_fdr(fake_vendor):
    df, _ = canonical(fake_vendor)
    out = analyze(df, min_n=5)
    assert set(out) == {"direction_summary", "annual_stability", "correlations", "flow_ratio_bands", "aligned_pairs"}
    summary = out["direction_summary"]
    assert set(summary.hypothesis) == {"H1_night_to_day", "H2_day_to_next_night"}
    assert len(summary) == 4
    assert summary.loc[summary.subset == "full_history", "q_bh_primary"].notna().all()
    assert (summary.n > 0).all()


def test_zero_net_excluded_not_implicit_sell(fake_vendor):
    raw = deepcopy(fake_vendor)
    row = raw["afterhours"]["net"].index[0]
    raw["afterhours"]["short"].loc[row, DEFAULT_INSTRUMENT] = raw["afterhours"]["long"].loc[row, DEFAULT_INSTRUMENT]
    raw["afterhours"]["net"].loc[row, DEFAULT_INSTRUMENT] = 0
    raw["afterhours"]["short_value"].loc[row, DEFAULT_INSTRUMENT] = raw["afterhours"]["long_value"].loc[row, DEFAULT_INSTRUMENT]
    raw["afterhours"]["net_value"].loc[row, DEFAULT_INSTRUMENT] = 0
    # Adjust full by same difference to preserve the independent day portion.
    orig_short = fake_vendor["afterhours"]["short"].loc[row, DEFAULT_INSTRUMENT]
    delta = raw["afterhours"]["short"].loc[row, DEFAULT_INSTRUMENT] - orig_short
    raw["full_session"]["short"].loc[row, DEFAULT_INSTRUMENT] += delta
    raw["full_session"]["net"].loc[row, DEFAULT_INSTRUMENT] -= delta
    raw["full_session"]["short_value"].loc[row, DEFAULT_INSTRUMENT] += delta*100
    raw["full_session"]["net_value"].loc[row, DEFAULT_INSTRUMENT] -= delta*100
    df, _ = canonical(raw)
    pairs = make_pairs(df, "H1_night_to_day")
    assert not bool(pairs.loc[row, "direction_eligible"])
    summary = analyze(df, min_n=5)["direction_summary"]
    assert summary.iloc[0].zeros_excluded >= 1


def test_explicit_validation_of_date_and_config(fake_vendor):
    with pytest.raises(ValueError):
        ResearchConfig("2023-10-15", "2026-10-22").validate()
    with pytest.raises(ValueError):
        ResearchConfig("2026-10-23", "2026-10-22").validate()
    raw = deepcopy(fake_vendor)
    raw["afterhours"]["net"].index = raw["afterhours"]["net"].index.tz_localize("Asia/Taipei")
    with pytest.raises(ValueError):
        normalize_session(raw["afterhours"], DEFAULT_INSTRUMENT)


def test_zero_target_is_not_buy_even_if_predictor_buy(fake_vendor):
    raw = deepcopy(fake_vendor)
    row = raw["full_session"]["net"].index[1]
    # Make independent DAY flow exactly zero; adjust full with correct values.
    day_long = raw["full_session"]["long"].loc[row, DEFAULT_INSTRUMENT] - raw["afterhours"]["long"].loc[row, DEFAULT_INSTRUMENT]
    day_short = raw["full_session"]["short"].loc[row, DEFAULT_INSTRUMENT] - raw["afterhours"]["short"].loc[row, DEFAULT_INSTRUMENT]
    raw["full_session"]["short"].loc[row, DEFAULT_INSTRUMENT] += day_long - day_short
    raw["full_session"]["short_value"].loc[row, DEFAULT_INSTRUMENT] += (day_long - day_short)*100
    raw["full_session"]["net"].loc[row, DEFAULT_INSTRUMENT] = raw["afterhours"]["net"].loc[row, DEFAULT_INSTRUMENT]
    raw["full_session"]["net_value"].loc[row, DEFAULT_INSTRUMENT] = raw["afterhours"]["net_value"].loc[row, DEFAULT_INSTRUMENT]
    df, qa = canonical(raw)
    assert qa.valid.all()
    pairs = make_pairs(df, "H1_night_to_day")
    assert pairs.loc[row, "direction_eligible"] == (pairs.loc[row, "predictor_net"] != 0)
    assert pairs.loc[row, "outcome_net"] == 0


def test_reject_orphan_night_trade_date(fake_vendor):
    raw = deepcopy(fake_vendor)
    d = pd.Timestamp("2027-10-20")
    for field in FIELDS:
        night = raw["afterhours"][field]
        raw["afterhours"][field] = pd.concat([night, pd.DataFrame({DEFAULT_INSTRUMENT:[0],
                          "小型臺指期貨_外資及陸資":[0]}, index=pd.DatetimeIndex([d]))])
    with pytest.raises(ValueError, match="absent from full"):
        canonical(raw)
