# TXF Session Flow Persistence — v0.1.0

Independent, reproducible research of **foreign investor day/night trading-flow persistence in Taiwan TX futures**. Statistics only; **not a trading strategy** and **not OI/positions research**.

## Research hypotheses (predeclared)

- H1: **night[T] -> day[T]**, where night[T] is the session beginning at 15:00 on the preceding trading day, ending at 05:00 on date T, then regular-day[T] 08:45–13:45.
- H2: **day[T] -> night[T+1]**, where T+1 is the **next actual trading day**. Never calendar T+1 and never jump a missing night observation.
- `full_session` includes day+night trades; `afterhours` means night trades only; `day = full - night` for **trading volume/contract amounts**, **not open interest**.

Outputs: buy/sell transition probabilities vs unconditional base rate; Wilson 95% CI, HAC linear-probability contrast, BH-FDR (H1+H2, full-sample only), yearly stability, fixed flow-ratio bands, Pearson/Spearman correlations, and chronologically held-out descriptive results. No price/return outcomes in v0.1.0.

## Data contract

FinLab dataset prefixes:

- `futures_institutional_investors_trading_summary` — *all-session*
- `futures_institutional_investors_trading_summary_after_hours` — *night-session*

For each source, six `data.get(...)` fields: `多方交易口數`, `空方交易口數`, `多空交易口數淨額`, `多方交易契約金額(千元)`, `空方交易契約金額(千元)`, `多空交易契約金額淨額(千元)`.

The default column name is **`臺股期貨_外資及陸資`**. Vendor field column spellings and identity naming must be validated on the first credentialed pilot. The program deliberately fails if exact configured column is absent; use `--instrument` to specify the actual field label, never fuzzy-match against mini/micro futures or other institutions.

Dates begin **2023-10-16**, when night session data became available. Date T on after-hours **already means the next regular trading date**. Contract amounts are thousands of NTD. Actual daily source publication timestamp is unverified and cannot establish a before-15:00 trading signal. FinLab historical data is not independently guaranteed to match TAIFEX for every day; official collector spot-check is still recommended.

## Storage separation (provided Google Drive folder IDs)

- **Immutable raw:** [`finlab_futures_afterhours`](https://drive.google.com/drive/folders/1913HOEClHF_GTuI54usuuTKmIlQOzU83) — ID `1913HOEClHF_GTuI54usuuTKmIlQOzU83`.
- **Append-only results:** [`txf_session_flow_persistence`](https://drive.google.com/drive/folders/1Plm1g2DORO2QBVt6lpxbef2sorne89AN) — ID `1Plm1g2DORO2QBVt6lpxbef2sorne89AN`.

Raw frames keep **all original vendor-wide columns** as per-field Parquet; field SHA256 hashes are verified before cache reuse. For each session, a date-window snapshot contains six files and a last-written `manifest.json`. No download occurs if both matching snapshots pass checks; explicit `--refresh` creates *new revision folders* without changing prior data. If a snapshot is incomplete/corrupt, execution fails rather than silently using bad data. Results are always unique `runs/YYYYMMDD_HHMMSS_shortSHA_uniqueID/` with processed Parquet, QA, tables, figure, summary and SHA256 `run_info.json` written last.

## Run from Google Colab

Use [`notebooks/run_txf_session_flow_persistence.ipynb`](notebooks/run_txf_session_flow_persistence.ipynb) (Colab). Connect FinLab with `FINLAB_API_TOKEN` from Colab Secrets. OAuth Drive access is interactive and confined to your authenticated account. The code **does not install any automatic daily schedule** and does not persist credential strings in Google Drive.

The Colab bootstrap **updates an existing clean checkout with a fast-forward-only Git fetch**, installs dependencies using the active notebook kernel interpreter, adds `src/` to that kernel's `sys.path`, clears stale project module entries and **verifies the actual imported `config.py` location** before the research cells run. This addresses the common `ModuleNotFoundError: No module named 'txf_session_flow_persistence.config'` caused by the repo directory being mistaken for a namespace package. It does not modify your Drive cache or stored results. If the existing checkout contains local uncommitted edits or cannot be fast-forwarded, it stops instead of overwriting files. Rerun the notebook **from the first cell (Runtime → Run all)** rather than just its import cell.

The runner authenticates Google Drive first, then reads both raw cache snapshots. If present, it **does not require the FinLab token**. If missing, a FinLab login is required and errors stop the run. The use of the full Google Drive OAuth scope is necessary for working with pre-existing folders the script did not create; grant only when you trust the local Colab session.

## Local validation and offline tests

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m txf_session_flow_persistence --help
```

For local **live** data fetch (`FINLAB_API_TOKEN` required):

```bash
export FINLAB_API_TOKEN='YOUR_TOKEN'
python -m txf_session_flow_persistence --backend local --local-root ./local_data --start 2023-10-16 --end 2026-10-22
```

This is a live FinLab run, **not** a mock dataset. Unit tests use injected synthetic fixtures; their outputs must never be misrepresented as empirical findings.

### Known limits / scope gates

- FinLab column and availability metadata come from documentation and user-confirmed screenshot; no credentialed FinLab fetch has been run by CI.
- Current code does not estimate true OI changes, execute trades, backtest P&L, use future ETF price, or adjust position sizes.
- Missing days remain missing. Any impossible full-minus-night contract counts abort the run. Partial coverage and amount anomalies are retained in `diagnostics/daily_qa.csv`; observations failing QA are excluded.
- Unverified provider revisions require explicit refresh. No backfilled data may be silently substituted for live point-in-time snapshots.
- No `CI` secrets required; `.github/workflows/ci.yml` only executes offline tests.
