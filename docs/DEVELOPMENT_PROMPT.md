# Development Prompt — TXF Session Flow Persistence v0.1.0

## Scope
Implement an independent Python 3.11 repository `hh4832/txf_session_flow_persistence`. Its *only* research objective is whether foreign investors' buying/selling in Taiwan TX night and regular day sessions predicts the **next trading session's foreign trading flow**. Do not research open-interest changes, intraday trading strategies, profitability or price prediction in v0.1.0. Do not edit `txf_overnight_spot_open`, `taifex-night-flow` or other repositories.

## Immutable source and destination contracts

1. FinLab full-session dataset `futures_institutional_investors_trading_summary` and night-session dataset `futures_institutional_investors_trading_summary_after_hours`.
2. Request six count/contract-value fields from each table using exact `data.get(dataset:field)` identifiers. Retain vendor original-wide DataFrames as Parquet without renaming/overwriting.
3. `RAW_FOLDER_ID=1913HOEClHF_GTuI54usuuTKmIlQOzU83`; `RESULTS_FOLDER_ID=1Plm1g2DORO2QBVt6lpxbef2sorne89AN`.
4. Verify SHA256 and source/date/field manifest before using raw cache; never re-fetch when an identical valid snapshot is available. Provider refresh may create a new immutable revision only with explicit opt-in. Never substitute synthetic market data.
5. One uniquely named, append-only results run per invocation. Store canonical processed frame, QA, hypothesis statistics, plotted diagnostics, summary and provenance (date range, source hashes, code SHA, configuration). No overwrites.

## Statistics & alignment

1. `night[T]` belongs to TAIFEX exchange trade date T and precedes `day[T]`. The next evening after `day[T]` is `night[T+1 trading date]`. Preserve missing sessions and never shift by calendar day or skip across missing night sessions.
2. `day_buy = full_buy - night_buy`, likewise sell, net and traded contract amounts. Strictly verify buy>=0, sell>=0, net=buy-sell; reject impossible counts. Never equate net trades with change in open-interest position.
3. H1 `P(day[T] net>0 | night[T] net>0)` and same after selling; H2 `P(night[T+1] net>0 | day[T] net>0)` and same after selling. Zero net excluded from sign analysis and counted.
4. Report unconditional target-buy rate, conditional probabilities, Wilson 95% CI, Pearson/Spearman of net and normalized net-flow ratios, HAC binary contrast, BH-FDR for two primary hypotheses, annual stability, fixed predeclared flow-ratio bins, and descriptive chronological 70:30 holdout.
5. No false assertion of tradability: FinLab publishing times, realized fills and trading constraints must be independently verified before any strategies are proposed.

## Delivery and QA

- Python package, Colab notebook, README, AGENTS.md, requirements, pytest offline fixture suite and GitHub Actions test-only CI.
- No secrets in code, GitHub, Drive artifacts or logs; token comes from interactive Colab secret or environment.
- Test trading-date joins, weekend skips, missing nights, invalid counts, zero flows, cache immutability, exact column selection and statistical output schema.
- Test locally (without fake performance claims), commit to `main` and verify push SHA. Live FinLab/Google Drive tests are gated on credentials and cannot be declared successful without actually running.
