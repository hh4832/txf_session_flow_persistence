"""Prespecified effect-modification tests: market returns x foreign-flow direction.

Six HAC interaction tests (H1,H2 x 1,3,5 trading days), corrected together
by BH-FDR. The 3-day test is primary, others sensitivity tests. Regime
subgroups are descriptive, not separately tested until a new preregistration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .analysis import HYPOTHESES, make_pairs

LOOKBACKS = (1, 3, 5)
REGIMES = ("down", "flat", "up")


def build_features(canonical: pd.DataFrame, taiex: pd.DataFrame):
    """H1 sees T-1 close; H2 sees T close. No forward filling or vendor inference."""
    idx = taiex.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.has_duplicates
            or idx.tz is not None or not idx.is_monotonic_increasing):
        raise ValueError("Unusable TAIEX calendar")
    close = pd.to_numeric(taiex.iloc[:, 0], errors="raise")
    if close.isna().any() or (close <= 0).any() or not np.isfinite(close.to_numpy(dtype=float)).all():
        raise ValueError("Invalid TAIEX close")
    daily = close.pct_change(fill_method=None)
    vol5 = daily.rolling(5, min_periods=5).std()
    features = pd.DataFrame(index=canonical.index)
    for h in HYPOTHESES:
        # Both align ON T, no shifting of trading-date labels after joining.
        shift_by = 1 if h == "H1_night_to_day" else 0
        for k in LOOKBACKS:
            returns = close.div(close.shift(k)).sub(1).shift(shift_by)
            features[f"{h}_r{k}"] = returns.reindex(features.index)
        features[f"{h}_vol5"] = vol5.shift(shift_by).reindex(features.index)
    features["taiex_date_observed"] = features.index.isin(taiex.index)
    return features


def _fit(x: pd.DataFrame, hac_lags: int, min_n: int):
    out = {"n": len(x), "status": "INSUFFICIENT_SAMPLE", "interaction": np.nan,
           "p_hac": np.nan, "ci_low": np.nan, "ci_high": np.nan, "buy_coef": np.nan}
    if len(x) < min_n or x.buy.nunique() != 2 or x.rstd.std() < 1e-12:
        return out
    dummies = pd.get_dummies(x.index.year.astype(str), prefix="year", drop_first=True, dtype=float)
    dummies.index = x.index
    X = pd.DataFrame(index=x.index)
    X["const"] = 1.0
    X["buy"] = x.buy.astype(float)
    X["return_std"] = x.rstd
    X["buy_x_return"] = X.buy * X.return_std
    X["vol5_std"] = x.volstd
    X = pd.concat([X, dummies], axis=1)
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        out["status"] = "SINGULAR_MODEL"
        return out
    fit = sm.OLS(x.y.astype(float).to_numpy(), X.to_numpy()).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    k = X.columns.get_loc("buy_x_return")
    out.update(status="OK", interaction=float(fit.params[k]),
               p_hac=float(fit.pvalues[k]), ci_low=float(fit.conf_int()[k, 0]),
               ci_high=float(fit.conf_int()[k, 1]),
               buy_coef=float(fit.params[X.columns.get_loc("buy")]))
    return out


def analyze_regimes(canonical, taiex, *, min_n=30, hac_lags=5,
                    holdout_fraction=0.30):
    features = build_features(canonical, taiex)
    tests, bins, coverage = [], [], []
    all_pairs = {}
    for h in HYPOTHESES:
        pairs = make_pairs(canonical, h)
        all_pairs[h] = pairs
        for k in LOOKBACKS:
            col = f"{h}_r{k}"
            vcol = f"{h}_vol5"
            z = pairs.loc[pairs.direction_eligible].join(features[[col, vcol]])
            missing = int(z[[col, vcol]].isna().any(axis=1).sum())
            z = z.dropna(subset=[col, vcol]).copy()
            coverage.append(dict(hypothesis=h, lookback=k, candidates=len(pairs),
                                 usable=len(z), missing_market_features=missing))
            # Chronological split remains fixed; all boundaries/scale parameters
            # learned from earlier observations only and then frozen on holdout.
            cut = int(len(z) * (1 - holdout_fraction))
            train = z.iloc[:cut]
            test = z.iloc[cut:]
            if len(train) < 2 or train[col].std() <= 0 or train[vcol].std() <= 0:
                tests.append(dict(hypothesis=h, lookback=k, subset="full_history",
                                  n=len(z), status="INSUFFICIENT_TRAINING"))
                continue
            mean_r = float(train[col].mean())
            sd_r = float(train[col].std())
            mean_v = float(train[vcol].mean())
            sd_v = float(train[vcol].std())
            lo, hi = train[col].quantile([1/3, 2/3]).tolist()
            z["buy"] = z.predictor_net.gt(0).astype(int)
            z["y"] = z.outcome_net.gt(0).astype(int)
            z["rstd"] = (z[col] - mean_r) / sd_r
            z["volstd"] = (z[vcol] - mean_v) / sd_v
            z["regime"] = pd.cut(z[col], bins=[-np.inf, lo, hi, np.inf],
                                 labels=REGIMES, include_lowest=True) if lo < hi else None
            for label, sub in (("full_history", z), ("training", z.iloc[:cut]),
                               ("chronological_holdout", z.iloc[cut:])):
                test_row = _fit(sub, hac_lags=max(hac_lags, k), min_n=min_n)
                tests.append(dict(hypothesis=h, lookback=k, subset=label,
                                  train_cutoff_date=str(train.index[-1].date()),
                                  regime_low=lo, regime_high=hi,
                                  return_training_mean=mean_r, return_training_sd=sd_r,
                                  **test_row))
                for regime in REGIMES:
                    chunk = sub.loc[sub.regime == regime] if lo < hi else sub.iloc[0:0]
                    pos = chunk.loc[chunk.buy == 1]
                    neg = chunk.loc[chunk.buy == 0]
                    p_pos = float(pos.y.mean()) if len(pos) else np.nan
                    p_neg = float(neg.y.mean()) if len(neg) else np.nan
                    bins.append(dict(hypothesis=h, lookback=k, subset=label, regime=regime,
                                     n=len(chunk), buy_condition_n=len(pos), sell_condition_n=len(neg),
                                     p_outcome_buy_after_buy=p_pos, p_outcome_buy_after_sell=p_neg,
                                     difference_pp=100 * (p_pos - p_neg)))
    t = pd.DataFrame(tests)
    t["q_bh_six_tests"] = np.nan
    if not t.empty:
        mask = t.subset.eq("full_history") & t.p_hac.notna()
        if mask.any():
            t.loc[mask, "q_bh_six_tests"] = multipletests(t.loc[mask, "p_hac"].to_numpy(),
                                                           method="fdr_bh")[1]
    return {
        "interaction_tests": t,
        "regime_probabilities": pd.DataFrame(bins),
        "market_feature_coverage": pd.DataFrame(coverage),
        "market_features": features,
    }
