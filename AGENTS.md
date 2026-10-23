# AGENTS.md — research safety & engineering contract

- Scope: `hh4832/txf_session_flow_persistence` only. Do not modify `txf_overnight_spot_open` or the TAIFEX collector.
- Research only: no brokerage trading, no profitability claims, no automated scheduled market downloads.
- `*_summary` = FinLab FULL day + night trades, `*_after_hours` = FinLab NIGHT only.
- Exchange trade_date T: night[T] precedes regular-day[T]. H1 is night[T] -> day[T], H2 is day[T] -> night[T+1 trading date].
- Net trading contracts != open interest / change in positions; never derive OI from net trading flow.
- `day_long/short/net = full_long/short/net - night_long/short/net`; QA all components before analysis.
- Use exactly configured FinLab instrument and 6 field keys; fail when absent or ambiguous. Never silently replace vendor data with synthetic data.
- Raw Google Drive ID `1913HOEClHF_GTuI54usuuTKmIlQOzU83`, results ID `1Plm1g2DORO2QBVt6lpxbef2sorne89AN`.
- Cached raw frames and completed run folders immutable. Explicit refresh means NEW revision, never overwrite.
- Do not commit secrets, caches, downloaded vendor data, or outputs. Tests should use injected synthetic fixtures only, always labeled.
- Exclude missing rows/zero direction from binary transition statistics; preserve QA diagnostic evidence.
- H1/H2 primary tests only. Use HAC differences and BH-FDR; holdout/annual/bins are descriptive and do not justify market signals.
- No GitHub Actions scheduled provider fetch. CI only runs offline tests.
