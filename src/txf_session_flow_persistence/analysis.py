"""Predeclared descriptive/HAC analysis: two consecutive-session hypotheses."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

HYPOTHESES = {
    "H1_night_to_day": ("night", "day"),
    "H2_day_to_next_night": ("day", "next_night"),
}
# Predeclared absolute flow-ratio bands; purely exploratory description, no optimization.
BINS = [-1.000001, -0.02, -0.005, 0.0, 0.005, 0.02, 1.000001]


def make_pairs(canonical: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in HYPOTHESES:
        raise ValueError("unknown hypothesis")
    p, q = HYPOTHESES[name]
    good = canonical["valid"].copy()
    if q == "next_night":
        good &= canonical["next_night_valid"]
    out = pd.DataFrame(index=canonical.index)
    out["predictor_net"] = canonical[f"{p}_net"]
    out["outcome_net"] = canonical[f"{q}_net"]
    out["predictor_ratio"] = canonical[f"{p}_flow_ratio"]
    out["outcome_ratio"] = canonical[f"{q}_flow_ratio"]
    out["predictor_date"] = canonical.index
    out["outcome_date"] = canonical["next_trade_date"] if q == "next_night" else canonical.index
    out = out.loc[good].dropna(subset=["predictor_net", "outcome_net", "predictor_ratio", "outcome_ratio"])
    # Zero predictor contains no directional condition. Zero outcome remains a real
    # non-buy for the exact estimand P(outcome>0 | predictor>0); separately report flats.
    out["direction_eligible"] = out.predictor_net.ne(0)
    return out


def _wilson(n_success: int, n: int):
    if not n:
        return (np.nan, np.nan)
    return tuple(map(float, proportion_confint(n_success, n, alpha=0.05, method="wilson")))


def _direction_summary(pairs: pd.DataFrame, name: str, min_n: int, hac_lags: int, subset: str):
    x = pairs.loc[pairs.direction_eligible]
    n = len(x)
    result = {"hypothesis": name, "subset": subset, "n": n, "zeros_excluded": int((~pairs.direction_eligible).sum()),
              "outcome_zero_count": int(x.outcome_net.eq(0).sum()),
              "status": "OK" if n >= min_n else "INSUFFICIENT_SAMPLE", "p_hac": np.nan,
              "coef_lpm": np.nan, "lpm_ci_low": np.nan, "lpm_ci_high": np.nan,
              "baseline_positive": np.nan}
    if n == 0:
        return result
    y = x.outcome_net.gt(0).astype(int)
    predictor_positive = x.predictor_net.gt(0).astype(int)
    result["baseline_positive"] = float(y.mean())
    for label, condition in (("after_buy", predictor_positive.eq(1)), ("after_sell", predictor_positive.eq(0))):
        k = int(condition.sum())
        wins = int(y.loc[condition].sum())
        low, high = _wilson(wins, k)
        result[label + "_n"] = k
        result[label + "_outcome_buy_probability"] = wins / k if k else np.nan
        result[label + "_ci_low"] = low
        result[label + "_ci_high"] = high
    if n >= min_n and predictor_positive.nunique() == 2:
        X = sm.add_constant(predictor_positive.to_numpy(dtype=float), has_constant="add")
        model = sm.OLS(y.to_numpy(dtype=float), X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        result["coef_lpm"] = float(model.params[1])
        result["p_hac"] = float(model.pvalues[1])
        ci = model.conf_int(alpha=0.05)
        result["lpm_ci_low"], result["lpm_ci_high"] = float(ci[1][0]), float(ci[1][1])
    elif n >= min_n:
        result["status"] = "SINGLE_PREDICTOR_DIRECTION"
    return result


def _correlation(pairs: pd.DataFrame, name: str):
    output = {"hypothesis": name, "n": len(pairs)}
    for field in ("net", "ratio"):
        a, b = pairs["predictor_" + field].to_numpy(), pairs["outcome_" + field].to_numpy()
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            pear, spear = np.nan, np.nan
        else:
            pear = float(stats.pearsonr(a, b).statistic)
            spear = float(stats.spearmanr(a, b).statistic)
        output[f"pearson_{field}"] = pear
        output[f"spearman_{field}"] = spear
    return output


def analyze(canonical: pd.DataFrame, *, min_n: int = 30, hac_lags: int = 5,
            holdout_fraction: float = 0.30) -> dict[str, pd.DataFrame]:
    summary, by_year, correlations, bands, pairs_dump = [], [], [], [], []
    for name in HYPOTHESES:
        pairs = make_pairs(canonical, name)
        if not pairs.empty:
            tagged = pairs.reset_index(drop=True).assign(hypothesis=name)
            pairs_dump.append(tagged)
        summary.append(_direction_summary(pairs, name, min_n, hac_lags, "full_history"))
        # Strict chronological partition of already-paired events; no thresholds learned on holdout.
        cut = int(len(pairs) * (1 - holdout_fraction))
        if 0 < cut < len(pairs):
            summary.append(_direction_summary(pairs.iloc[cut:], name, min_n, hac_lags, "chronological_holdout"))
        else:
            summary.append(_direction_summary(pairs.iloc[0:0], name, min_n, hac_lags, "chronological_holdout"))
        for year, subset in pairs.groupby(pairs.index.year):
            by_year.append(dict(year=int(year), **_direction_summary(subset, name, min_n, hac_lags, "by_year")))
        correlations.append(_correlation(pairs, name))
        if not pairs.empty:
            bin_labels = ["<-2%", "[-2%,-0.5%)", "[-0.5%,0%)", "[0%,0.5%)", "[0.5%,2%)", ">=2%"]
            bucket = pd.cut(pairs.predictor_ratio, bins=BINS, labels=bin_labels, right=False)
            for label in bin_labels:
                group = pairs.loc[bucket == label]
                x = group.loc[group.direction_eligible]
                bands.append({"hypothesis": name, "predictor_ratio_band": label, "n": len(group),
                              "n_direction_eligible": len(x),
                              "outcome_buy_probability": float(x.outcome_net.gt(0).mean()) if len(x) else np.nan})
    s = pd.DataFrame(summary)
    # Both primary H1/H2: a single predeclared FDR family; holdout descriptive only.
    mask = (s.subset == "full_history") & s.p_hac.notna()
    s["q_bh_primary"] = np.nan
    if mask.any():
        s.loc[mask, "q_bh_primary"] = multipletests(s.loc[mask, "p_hac"], method="fdr_bh")[1]
    return {"direction_summary": s, "annual_stability": pd.DataFrame(by_year),
            "correlations": pd.DataFrame(correlations), "flow_ratio_bands": pd.DataFrame(bands),
            "aligned_pairs": pd.concat(pairs_dump, ignore_index=True) if pairs_dump else pd.DataFrame()}
