"""Offline regression tests for predeclared TAIEX return/regime analysis."""
import numpy as np
import pandas as pd
import pytest

from txf_session_flow_persistence.config import FIELDS, FULL_DATASET, NIGHT_DATASET, ResearchConfig
from txf_session_flow_persistence.market_data import (
    TAIEX_KEY, fetch_taiex, price_start, validate_price_frame,
)
from txf_session_flow_persistence.regime_analysis import build_features, analyze_regimes
from txf_session_flow_persistence.regime_runner import run_regime
from txf_session_flow_persistence.source import normalize_session
from txf_session_flow_persistence.storage import LocalStore, save_snapshot, snapshot_path
from txf_session_flow_persistence.validation import build_daily_flows


def sample_prices(first="2026-09-01", n=115):
    dates = pd.bdate_range(first, periods=n)
    rng = np.random.default_rng(919)
    close = 18000 * np.exp(np.cumsum(rng.normal(0.0008, 0.011, len(dates))))
    return pd.DataFrame({"TAIEX": close}, index=dates)


def test_features_use_only_price_known_before_each_session():
    prices = sample_prices(n=30)
    canonical = pd.DataFrame(index=prices.index[7:25])
    features = build_features(canonical, prices)
    p = prices.iloc[:, 0]
    t = canonical.index[8]
    i = prices.index.get_loc(t)
    assert features.loc[t, "H1_night_to_day_r1"] == pytest.approx(p.iloc[i-1]/p.iloc[i-2]-1)
    assert features.loc[t, "H1_night_to_day_r5"] == pytest.approx(p.iloc[i-1]/p.iloc[i-6]-1)
    assert features.loc[t, "H2_day_to_next_night_r1"] == pytest.approx(p.iloc[i]/p.iloc[i-1]-1)
    assert features.loc[t, "H2_day_to_next_night_r5"] == pytest.approx(p.iloc[i]/p.iloc[i-5]-1)
    # A future close spike must not affect T predictors.
    later = prices.copy()
    later.iloc[i+1, 0] *= 8
    future_feat = build_features(canonical, later)
    for key in ("H1_night_to_day_r3", "H2_day_to_next_night_r3",
                "H1_night_to_day_vol5", "H2_day_to_next_night_vol5"):
        assert future_feat.loc[t, key] == pytest.approx(features.loc[t, key])


def test_wrong_index_prices_and_invalid_date_fail_closed():
    f = sample_prices()
    f.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_price_frame(f, "2026-09-01", "2026-10-30")
    assert price_start("2023-10-16") < "2023-10-16"


def test_regime_analysis_is_predeclared_and_uses_holdout(fake_vendor):
    full = normalize_session(fake_vendor["full_session"], "臺股期貨_外資及陸資")
    night = normalize_session(fake_vendor["afterhours"], "臺股期貨_外資及陸資")
    canonical, qa = build_daily_flows(full, night)
    assert qa.valid.all()
    results = analyze_regimes(canonical, sample_prices(), min_n=8)
    tests = results["interaction_tests"]
    assert len(tests) == 2 * 3 * 3
    assert set(tests.lookback) == {1,3,5}
    assert set(tests.subset) == {"training","full_history","chronological_holdout"}
    assert set(results["regime_probabilities"].regime) == {"down","flat","up"}
    assert tests.loc[tests.subset.eq("full_history"), "q_bh_six_tests"].notna().sum() == 6
    assert tests.loc[tests.subset.eq("chronological_holdout"), "q_bh_six_tests"].isna().all()
    rows = results["market_feature_coverage"]
    assert len(rows) == 6 and (rows.usable > 0).all()


def test_regime_pipeline_reuses_original_snapshots_and_price_cache(tmp_path, fake_vendor):
    pytest.importorskip("pyarrow")
    store = LocalStore(tmp_path / "raw", tmp_path / "results")
    cfg = ResearchConfig(start="2026-10-19", end="2027-01-31", min_n=8)
    for sess, dataset in [("full_session", FULL_DATASET), ("afterhours", NIGHT_DATASET)]:
        save_snapshot(store, snapshot_path(sess, cfg.start, cfg.end),
                      fake_vendor[sess], dataset=dataset, start=cfg.start, end=cfg.end)
    calls = []
    def getter(key):
        calls.append(key)
        assert key == TAIEX_KEY
        return sample_prices(n=125)
    a, results = run_regime(store, cfg, getter=getter)
    assert calls == [TAIEX_KEY]
    assert a["status"] == "ANALYSIS_COMPLETE"
    assert a["data_type"] == "INJECTED_TEST_DATA"
    assert a["flow_qa_valid_days"] == 60
    assert a["git_sha"] != "UNKNOWN"
    assert (tmp_path / "results" / a["run_name"] / "tables/interaction_tests.csv").is_file()
    assert (tmp_path / "results" / a["run_name"] / "run_info.json").is_file()
    def fail_getter(_):
        raise AssertionError("cached TAIEX snapshot must be reused")
    b, _ = run_regime(store, cfg, getter=fail_getter)
    assert b["run_name"] != a["run_name"]
    assert calls == [TAIEX_KEY]
