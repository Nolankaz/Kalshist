# Kalshist Architecture

This page maps the Kalshist research pipeline from data acquisition through model comparison. It points to the implementing scripts, local artifact layers, and chronological technical records. See the [README](README.md) for the research question, current findings, and limitations.

## Pipeline stages

Local research datasets and many derived artifacts are intentionally absent from the repository. The paths below identify artifact roles; they are not a rebuild sequence.

| Stage | Purpose | Main scripts | Key artifacts | Technical note |
| --- | --- | --- | --- | --- |
| Settled-market acquisition | Collect and inspect market metadata and official outcomes. | `scripts/fetch_settled_markets.py`, `sample_settled_markets.py`, `inspect_kalshi_market.py` | `data/kalshi_markets/` | [Schema](schema.md) |
| BTC exchange acquisition and settlement/reference proxy | Collect exchange trades and assess the external reference against settlements. | `scripts/backfill_bullish.py`, `backfill_kraken.py`, `backfill_crypto_com.py`, `basis_risk.py`, `analyze_basis_risk.py` | `data/btc_prices_1s/` | [Exchange notes](exchange_notes.md) |
| Kalshi historical and live quotes | Collect historical candles, check coverage, and record live top-of-book snapshots. | `scripts/backfill_kalshi_quotes.py`, `check_kalshi_quotes.py`, `record_kalshi_quotes.py` | `data/kalshi_quotes/`, `data/kalshi_quotes_live/` | [Quote notes](quote_notes.md) |
| Realized-volatility reference and batch engines | Compute and cross-check point-in-time volatility features. | `scripts/realized_vol.py`, `realized_vol_batch.py`, `build_realized_vol_sample.py`, `check_realized_vol.py`, `check_realized_vol_batch.py`, `check_vol_coverage.py`, `benchmark_realized_vol.py`, `benchmark_realized_vol_batch.py` | Volatility fields in `data/features/market_features.parquet` | [Feature notes](feature_notes.md) |
| Point-in-time feature table | Align market, quote, and BTC features at decision time and check leakage boundaries. | `scripts/build_market_features.py`, `check_market_features.py` | `data/features/market_features.parquet` | [Feature-table notes](feature_table_notes.md) |
| Evaluation design and Stage 0 | Freeze the chronological split and score the baseline probability model. | `scripts/evaluation_split.py`, `stage0_baseline.py`, `score_stage0.py`, `analyze_stage0_reliability.py` | `data/models/stage0_predictions.parquet` | [Evaluation notes](evaluation_notes.md) |
| Sigma selection, calibration, and basis sensitivity | Select the volatility input, calibrate probabilities, and assess reference-price sensitivity. | `scripts/select_sigma.py`, `analyze_calibration.py`, `analyze_basis_probability_risk.py`, `analyze_model_market_gap.py` | `data/models/stage0_sigma_selection.parquet`, `stage0_platt_parameters.parquet` | [Calibration notes](calibration_notes.md) |
| Fees, spreads, and edge threshold | Model quoted trading costs and freeze the candidate-signal threshold. | `scripts/fees.py`, `check_fees.py`, `analyze_spreads.py`, `build_edge_threshold.py`, `count_threshold_clearance.py` | `data/execution/spread_summary.parquet`, `edge_threshold.parquet` | [Execution notes](execution_notes.md) |
| Tier-2 Stage 0 backtest | Select decisions before outcomes, then perform historical settlement accounting. | `scripts/build_stage0_trades.py`, `backtest.py`, `analyze_backtest.py` | `data/backtest/stage0_trade_decisions.parquet`, `stage0_results_validation.parquet` | [Backtest notes](backtest_notes.md) |
| Logistic correction and walk-forward harness | Build model inputs, compare fitted probabilities and economics, and verify the frozen result. | `scripts/build_derived_features.py`, `walk_forward.py`, `model_logistic.py`, `compare_models.py`, `build_model_trades.py`, `backtest_model.py`, `finalize_day12.py` | `data/features/derived_features.parquet`, `data/models/model_comparison.parquet` | [Model notes](model_notes.md) |
| HAR-RV forecasting and probability comparison | Build a quarantined forward target, compare volatility forecasts, and test their probability feed-through. | `scripts/build_forward_vol_target.py`, `model_har_rv.py`, `compare_vol_forecasts.py`, `compare_har_probabilities.py`, `analyze_har_residuals.py` | `data/targets/forward_vol_target.parquet`, `data/models/har_vol_comparison.parquet`, `har_probability_comparison.parquet` | [Model notes](model_notes.md) |

## Data and artifact layers

- **Source data:** `data/kalshi_markets/`, `data/kalshi_quotes/`, `data/kalshi_quotes_live/`, and `data/btc_prices_1s/` hold local market, quote, snapshot, and exchange data.
- **Features and targets:** `data/features/` holds model inputs; `data/targets/` holds future-target material separately from those inputs.
- **Model outputs:** `data/models/` holds predictions, calibration records, fitted-model outputs, and comparison summaries.
- **Trading analysis:** `data/execution/` holds cost and threshold artifacts; `data/backtest/` holds decision lists and historical settlement accounting.
- **Plots:** Selected figures under the relevant `data/*/plots/` directories are tracked; most datasets and derived artifacts remain local. See [schema.md](schema.md) for artifact definitions.

## Integrity controls

- Feature windows use observations strictly before decision time; the reference and batch volatility paths are checked for equivalence. [Feature notes](feature_notes.md), `scripts/check_realized_vol.py`, `scripts/check_realized_vol_batch.py`
- Future-only fields are quarantined, and safe-column assertions guard model and trade-selection inputs. [Feature-table notes](feature_table_notes.md), `scripts/build_market_features.py`, `scripts/count_threshold_clearance.py`
- The split is chronological by `close_date`, with asserted row and market counts and non-overlapping split masks. The protocol requires ticker disjointness and reserves the locked test for one outcome-derived evaluation. `scripts/evaluation_split.py`, [evaluation notes](evaluation_notes.md)
- Model and selection rules are recorded before their permitted validation queries, with a validation-query ledger. [Evaluation notes](evaluation_notes.md)
- Trade selections are fingerprinted before settlement outcomes are joined. `scripts/build_stage0_trades.py`, `scripts/build_model_trades.py`, [backtest notes](backtest_notes.md)
- HAR target separation, baseline reproduction, forecast fingerprints, and deterministic rerun checks guard the later comparisons. `scripts/build_forward_vol_target.py`, `scripts/model_har_rv.py`, `scripts/compare_har_probabilities.py`, [model notes](model_notes.md)
- The read-only `scripts/finalize_day12.py` checks the frozen logistic-stage artifacts and deterministic fit. Its day-numbered name and similar note headings record development provenance; they are not separate production systems.

## Conventions

Timestamps are UTC-aware. `storage.py` provides the local daily Parquet helpers, with some derived tables stored as single files. Run scripts from the repository root as `python -m scripts.<name>`; data-fetching and builder scripts write local artifacts.
