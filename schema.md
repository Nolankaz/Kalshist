# Data Schema

Source datasets use daily Parquet files. Timestamps are UTC-aware unless a
field is explicitly documented as Unix seconds, and daily partitions use UTC
dates. The derived model-ready table documented below is a single-file
exception.

## Settled Kalshi Markets

Path: `data/kalshi_markets/YYYY-MM-DD.parquet`

Each row represents one settled KXBTC15M market. Files are partitioned by the
UTC date of `settlement_ts`.

Columns:

- `ticker`: Kalshi market ticker.
- `strike`: BTC strike value for the market.
- `open_time`: exact market open timestamp in UTC.
- `close_time`: exact market close timestamp in UTC.
- `settlement_result`: categorical Kalshi outcome, `yes` or `no`.
- `expiration_value`: BTC reference value used at settlement; compared with the
  strike to determine the market outcome.
- `settlement_value`: numeric settled payout, `1.0` for YES and `0.0` for NO.
- `volume`: total market volume reported by Kalshi.
- `settlement_ts`: timestamp when Kalshi recorded settlement, in UTC.

## Kalshi One-Minute Quotes

Path: `data/kalshi_quotes/YYYY-MM-DD.parquet`

Each row is one returned one-minute candlestick for one KXBTC15M market. Files
are partitioned by the market's UTC close date. Missing minutes are not filled
or synthesized.

Columns:

```text
ticker
close_date
period_end_ts
yes_bid_open
yes_bid_high
yes_bid_low
yes_bid_close
yes_ask_open
yes_ask_high
yes_ask_low
yes_ask_close
price_open
price_high
price_low
price_close
volume
open_interest
source_endpoint
```

- `close_date`: market close date as a `YYYY-MM-DD` UTC date string.
- `period_end_ts`: candle end time as Unix seconds in UTC.
- `yes_bid_*`, `yes_ask_*`: one-minute OHLC values for the YES top-of-book
  prices, normalized to `[0, 1]` dollar units.
- `price_*`: one-minute traded-price OHLC values in `[0, 1]` dollar units;
  values may be missing for valid no-trade minutes.
- `volume`, `open_interest`: Kalshi candle volume and open interest.
- `source_endpoint`: `historical` or `series`, identifying the source API.

## Live Kalshi Quote Snapshots

Path: `data/kalshi_quotes_live/YYYY-MM-DD.parquet`

Each row is a live snapshot of one open KXBTC15M market, normally sampled about
every 10 seconds. This is not a candlestick dataset. Files are partitioned by
the UTC date of `observed_ts`.

Columns:

```text
ticker
observed_ts
yes_bid
yes_ask
yes_bid_size
yes_ask_size
last_price
liquidity
volume
volume_24h
open_interest
```

- `observed_ts`: one UTC observation timestamp shared by all markets returned
  in the same API response.
- `yes_bid`, `yes_ask`, `last_price`: prices normalized to dollar units.
- `yes_bid_size`, `yes_ask_size`: true top-of-book sizes from Kalshi, retained
  for later execution and depth analysis.
- `liquidity`: Kalshi's reported aggregate liquidity value; it is not a quoted
  size or full order-book depth.
- `volume`, `volume_24h`, `open_interest`: aggregate Kalshi market fields; they
  are not bid or ask sizes.

## BTC One-Second Prices

Paths:

- `data/btc_prices_1s/bullish/YYYY-MM-DD.parquet`
- `data/btc_prices_1s/kraken/YYYY-MM-DD.parquet`
- `data/btc_prices_1s/crypto_com/YYYY-MM-DD.parquet`

Each row represents one UTC second with exchange-specific BTC trade data. Files
are partitioned by the UTC date of `timestamp`.

Columns:

- `timestamp`: start of the one-second UTC bucket.
- `exchange`: `bullish`, `kraken`, or `crypto_com`.
- `price`: volume-weighted average trade price for the second.
- `volume`: total BTC quantity traded during the second.
- `trade_count`: number of trades in the second.

## Model-Ready Market Features

Path: `data/features/market_features.parquet`

The table contains exactly 17,182 rows. Each row is uniquely identified by
`(ticker, horizon_minutes)`, with one T-10 row and one T-5 row for each of 8,591
eligible markets.

Columns, in the exact order defined by the builder constants:

```text
ticker
horizon_minutes
open_time
close_time
decision_time
close_date
spot
spot_age_seconds
strike
log_moneyness
T_years
hour_utc
day_of_week
quote_yes_bid
quote_yes_ask
quote_mid
quote_spread
quote_last_price
quote_volume
quote_open_interest
quote_period_end_ts
quote_time
quote_age_seconds
5min_vol
5min_ewma_vol
5min_n_obs
5min_coverage
15min_vol
15min_ewma_vol
15min_n_obs
15min_coverage
1hr_vol
1hr_ewma_vol
1hr_n_obs
1hr_coverage
4hr_vol
4hr_ewma_vol
4hr_n_obs
4hr_coverage
24hr_vol
24hr_ewma_vol
24hr_n_obs
24hr_coverage
y
settlement_result
settlement_value
expiration_value
y_from_expiration
target_agrees
fwd_log_return
```

Unlike the source datasets, this table deliberately uses one Parquet file
rather than daily partitions. It is a derived, model-ready artifact intended
to be consumed and validated as a whole.

## Stage 0 Predictions

Path: `data/models/stage0_predictions.parquet`

Join key: `ticker`, `horizon_minutes`

Columns:

- `ticker`
- `horizon_minutes`
- `close_date`
- `split`
- `log_moneyness`
- `T_years`
- `y`
- `p_<candidate>` for all 10 sigma candidates
- `z_<candidate>` for all 10 sigma candidates

The file contains exactly 17,182 rows and 27 columns. This is a deliberate
single-file derived artifact consumed as a whole, similar to
`market_features.parquet`.

## Day 9 Stage 0 Selection and Calibration Artifacts

Unless stated otherwise, these artifacts cover train and validation only. A Day 9 **common population** contains only rows where all ten Stage 0 `p_*` columns are non-null. Probability values and probability shifts are stored in raw probability units: for example, `0.04` represents 4 percentage points when interpreted as a shift. Basis quantities are stored in basis points (bps). The value `train_fitted_platt` means legitimate train-fitted calibration.

### Stage 0 Sigma Selection

Path: `data/models/stage0_sigma_selection.parquet`

Purpose: records the frozen validation/common Brier selection, paired 2-SE comparisons, candidate summaries, and final selection. Row grain depends on `record_type`: one row per `(horizon, candidate_a, candidate_b)` paired comparison, one row per candidate summary, plus one final selection row. Important columns include `record_type`, `horizon`, `candidate_a`, `candidate_b`, `mean_difference`, `standard_error`, `two_se`, `candidate_joint_brier`, `coverage_t10`, `coverage_t5`, `eligible`, `tie_set_member`, `final_selected_sigma`, and `selection_rule`. Candidate probabilities entering selection are raw, uncalibrated Stage 0 values; the artifact uses validation common populations and contains no test results.

### Validation Reliability

Path: `data/models/stage0_reliability_validation.parquet`

Purpose: records raw Stage 0 validation reliability by decile. The key is `(candidate, split, horizon_minutes, decile)`. Important columns are `n`, `mean_predicted`, `observed_yes_rate`, `wilson_low`, and `wilson_high`; all probability columns use raw probability units. This artifact contains the two validation common populations for the selected candidate and no train or test rows.

### Platt Parameters

Path: `data/models/stage0_platt_parameters.parquet`

Purpose: stores fitted Platt mappings of the form `sigmoid(a + b * logit(p))`. The key is `(candidate, horizon_minutes, fit_split, parameter_role)`. Important columns include `n`, `a`, `b`, `iterations`, `final_log_likelihood`, and `final_gradient_norm`. Rows with `fit_split == "train"` and `parameter_role == "legitimate_train_fit"` are the legitimate train-fitted calibration. Rows with `parameter_role == "validation_in_sample_ceiling"` are diagnostic ceilings only and must not be used for production calibration. The fits use train or validation common populations as identified by their roles; there are no test fits.

### Platt Validation Scores

Path: `data/models/stage0_platt_validation_scores.parquet`

Purpose: compares raw and legitimate train-fitted Platt performance on the validation common populations. The key is `(candidate, horizon_minutes)`. Important columns include `n`, `raw_brier`, `calibrated_brier`, `brier_improvement`, `raw_log_loss`, `calibrated_log_loss`, `log_loss_improvement`, `raw_auc`, and `calibrated_auc`. The `validation_fit_ceiling_*` columns are diagnostic in-sample ceiling metrics, not production-calibration results. This artifact is validation-only and contains no test scores.

### Basis-Probability Sensitivity

Path: `data/models/stage0_basis_probability_sensitivity.parquet`

Purpose: records row-level Channel A probability sensitivity under symmetric BTC basis perturbations. The key is `(ticker, split, horizon_minutes, basis_bps, probability_version)`. Important columns include `log_moneyness`, `T_years`, `sigma`, `basis_bps`, `probability_version`, `stage0_p_raw`, `stage0_p_plus`, `stage0_p_minus`, `p_raw`, `p_plus`, `p_minus`, `shift_plus`, `shift_minus`, `max_abs_shift`, `probability_span`, and `raw_model_market_abs_gap`. `basis_bps` is in bps; probabilities and shifts are in raw probability units. `probability_version` is either `raw` or `train_fitted_platt`. `raw_model_market_abs_gap` is the raw selected Stage 0 gap retained for Part 3.1 context regardless of `probability_version`; use the model-market gap artifacts for version-specific comparisons. The artifact covers train and validation common populations only.

### Basis-Probability Summary

Path: `data/models/stage0_basis_probability_summary.parquet`

Purpose: aggregates Channel A sensitivity distributions. The key is `(split, horizon_minutes, basis_bps, probability_version)`. Important columns include `n`, `mean_max_abs_shift`, `median_max_abs_shift`, `p75_max_abs_shift`, `p90_max_abs_shift`, `p95_max_abs_shift`, `max_max_abs_shift`, `median_probability_span`, `p95_probability_span`, and `median_raw_model_market_abs_gap`. Basis is in bps, while probability shifts, spans, and gaps use raw probability units. `median_raw_model_market_abs_gap` summarizes the raw selected Stage 0 gap and is not version-specific. Separate `raw` and `train_fitted_platt` sensitivity rows cover train and validation common populations; no test summaries are present.

### Settlement Proximity Summary

Path: `data/models/settlement_proximity_summary.parquet`

Purpose: records Channel B settlement proximity to strike, kept separate from probability sensitivity. The key is `split`, with rows for `train`, `validation`, and `train_validation`. Important columns include `n_markets`, `median_settlement_distance_bps`, the `p25`/`p75`/`p90`/`p95` settlement-distance columns, `share_within_1_2_bps`, and `share_within_5_0_bps`. Settlement distances are in bps and shares are stored as proportions. The artifact contains no synthetic label-flip probability and no test summary.

### Model-Market Gap

Path: `data/models/stage0_model_market_gap.parquet`

Purpose: records row-level disagreement between selected Stage 0 probabilities and market `quote_mid`. Each row is uniquely keyed by `(ticker, horizon_minutes)` and also carries `split`. Important columns are `market_probability`, `raw_model_probability`, `platt_model_probability`, `signed_gap_raw`, `abs_gap_raw`, `signed_gap_platt`, and `abs_gap_platt`. Probabilities and gaps use raw probability units; positive signed gap means the model assigns a higher YES probability than the market. The Platt values use legitimate train-fitted parameters. The artifact covers train and validation common populations only.

### Model-Market Gap Summary

Path: `data/models/stage0_model_market_gap_summary.parquet`

Purpose: aggregates model-market disagreement by population and probability version. The key is `(split, horizon_minutes, model_version)`, where `model_version` is `raw` or `train_fitted_platt`. Important columns include `n`, `mean_abs_gap`, `median_abs_gap`, `p75_abs_gap`, `p90_abs_gap`, `p95_abs_gap`, `max_abs_gap`, `mean_signed_gap`, `median_signed_gap`, and `spearman_model_market`. Gap columns use raw probability units; Spearman correlation is unitless. The artifact covers train and validation common populations only and contains no test rows.

## Day 10 Execution Artifacts

These schemas were inspected from the files on disk on 2026-09-14. The Parquet artifacts cover train and validation only. They contain no test results.

### Spread and Fee Summary

Path: `data/execution/spread_summary.parquet`

Format and size: Parquet, 256 rows and 67 columns. This is a summary-level artifact; it contains no row-level quote records.

Unique key: `(population, split, horizon_minutes, breakdown, breakdown_value)`.

- `population` is `all_rows` or `day9_common`.
- `split` is `train` or `validation`.
- `breakdown` is `overall`, `hour_utc` or `price_bucket`.
- `breakdown_value` is `all`, an hour label `00` through `23`, or one of `p00_10`, `p10_25`, `p25_40`, `p40_60`, `p60_75`, `p75_90`, `p90_100`.

Columns, in on-disk order:

```text
population
split
horizon_minutes
breakdown
breakdown_value
row_count
spread_count
spread_mean_mils
spread_median_mils
spread_p10_mils
spread_p25_mils
spread_p75_mils
spread_p90_mils
spread_p95_mils
spread_p99_mils
spread_max_mils
spread_exactly_1_mil_share
spread_exactly_10_mils_share
spread_above_20_mils_share
fee_yes_count
fee_yes_mean_dollars
fee_yes_median_dollars
fee_yes_p10_dollars
fee_yes_p25_dollars
fee_yes_p75_dollars
fee_yes_p90_dollars
fee_yes_p95_dollars
fee_yes_p99_dollars
fee_yes_min_dollars
fee_yes_max_dollars
fee_no_count
fee_no_mean_dollars
fee_no_median_dollars
fee_no_p10_dollars
fee_no_p25_dollars
fee_no_p75_dollars
fee_no_p90_dollars
fee_no_p95_dollars
fee_no_p99_dollars
fee_no_min_dollars
fee_no_max_dollars
mid_cost_yes_mean_probability_units
mid_cost_yes_median_probability_units
mid_cost_yes_p10_probability_units
mid_cost_yes_p25_probability_units
mid_cost_yes_p75_probability_units
mid_cost_yes_p90_probability_units
mid_cost_yes_p95_probability_units
mid_cost_yes_p99_probability_units
mid_cost_yes_min_probability_units
mid_cost_yes_max_probability_units
mid_cost_no_mean_probability_units
mid_cost_no_median_probability_units
mid_cost_no_p10_probability_units
mid_cost_no_p25_probability_units
mid_cost_no_p75_probability_units
mid_cost_no_p90_probability_units
mid_cost_no_p95_probability_units
mid_cost_no_p99_probability_units
mid_cost_no_min_probability_units
mid_cost_no_max_probability_units
bid_subcent_share
ask_subcent_share
stale_gt_10s_count
stale_gt_60s_count
stale_gt_10s_share
stale_gt_60s_share
```

Spread statistics use mils (`0.001` probability units). Fee statistics are one-contract Direct Member net cash fees in dollars. Midpoint-cost statistics are `half_spread + fee` in raw probability units. Shares are proportions in `[0, 1]`; count fields are row counts. The artifact's purpose is to summarize historical quoted execution-cost structure, actual fee exposure, tapered/sub-cent quoting and quote staleness by population and breakdown.

### Frozen Edge Threshold

Path: `data/execution/edge_threshold.parquet`

Format and size: Parquet, 28 rows and 24 columns. This is a summary-level parameter artifact, not a row-level signal file.

Unique key: `(horizon_minutes, basis_bps, price_bucket)`, representing horizon × basis setting × price bucket. There are seven buckets for each of two horizons and two basis settings. `basis_bps = 1.2` with `threshold_role = primary` is the primary threshold; `basis_bps = 5.0` with `threshold_role = conservative_sensitivity` is non-primary.

Columns, in on-disk order:

```text
horizon_minutes
basis_bps
threshold_role
price_bucket
bucket_low
bucket_high
basis_term
model_error_term
required_net_edge
basis_validation_row_count
ece_validation_row_count
basis_statistic
model_error_statistic
probability_version
fee_model_status
order_size
sigma_candidate
bucket_merge_fired
max_abs_decile_deviation
platt_reproduction_max_abs_diff
calibrated_auc
stored_calibrated_auc
auc_regression_abs_diff
rule_reference
```

`basis_bps` is in basis points. `bucket_low` and `bucket_high` are `quote_mid` probability bounds. `basis_term`, `model_error_term`, `required_net_edge`, `max_abs_decile_deviation` and Platt-regression differences are raw probability units. AUC fields are unitless. Row-count and order-size fields are counts. The artifact's purpose is to store the mechanically constructed Stage 0 net-executable-edge threshold under the pre-registered additive basis-plus-ECE rule.

### Threshold-Clearance Summary

Path: `data/execution/threshold_clearance_summary.parquet`

Format and size: Parquet, 168 rows and 28 columns. This is summary-only. No row-level candidate-signal artifact was created, and this file contains no outcomes, returns or P&L.

Unique key: `(split, horizon_minutes, basis_bps, probability_version, breakdown, breakdown_value)`.

- `breakdown` is `overall`, `price_bucket`, `side` or `market_summary`.
- `breakdown_value` is `all`, one of the seven price-bucket labels, `YES`, `NO` or `cross_horizon`.
- Cross-horizon `market_summary` rows use `horizon_minutes = 0` as the documented sentinel because they summarize T-10 and T-5 together.
- `probability_version` is `train_fitted_platt` or diagnostic `raw`; `probability_role` identifies shipping versus diagnostic results.

Columns, in on-disk order:

```text
split
horizon_minutes
basis_bps
probability_version
breakdown
breakdown_value
threshold_role
probability_role
n_common
n_stale_excluded
n_best_side_ties
n_positive_gross_best_side
n_positive_net_best_side
n_threshold_clear
stale_excluded_share
positive_gross_best_side_share
positive_net_best_side_share
threshold_clear_share
median_best_side_net_edge_probability_units
p90_best_side_net_edge_probability_units
required_net_edge_min_probability_units
required_net_edge_max_probability_units
n_unique_markets_threshold_clear
n_dual_horizon_markets_threshold_clear
n_dual_horizon_same_side
n_dual_horizon_opposite_side
n_dual_horizon_yes_yes
n_dual_horizon_no_no
```

`basis_bps` is in basis points. Best-side edge and required-edge fields are raw probability units. Share fields are proportions in `[0, 1]`; `n_*` fields are counts and are null where a breakdown does not use that statistic. The artifact's purpose is to record the outcome-free candidate-signal waterfall, side and price-bucket breakdowns, clearing-edge summaries and descriptive cross-horizon market counts for both frozen basis settings and both probability versions.

### KXBTC15M Series Snapshot

Path: `data/execution/kxbtc15m_series_snapshot.json`

Format and grain: JSON, one response snapshot document rather than a tabular row set. Its top-level keys are:

```text
http_status
request_method
response
retrieved_at_utc
source_url
timestamp_basis
```

`response.series` contains `additional_prohibitions`, `category`, `contract_terms_url`, `contract_url`, `exchange_index`, `fee_multiplier`, `fee_type`, `frequency`, `last_updated_ts`, `product_metadata`, `settlement_sources`, `tags`, `ticker` and `title`. The stored response is HTTP 200 from `GET https://external-api.kalshi.com/trade-api/v2/series/KXBTC15M`, retrieved at `2026-09-14T20:00:49Z`; its timestamp basis is the HTTP Date header. The inspected series identifies ticker `KXBTC15M`, `fee_type = quadratic`, `fee_multiplier = 1`, fifteen-minute frequency and CF Benchmarks as the settlement source. Its purpose is to preserve the exact series metadata used in Day 10 fee research.

### KXBTC15M Fee-Changes Snapshot

Path: `data/execution/kxbtc15m_fee_changes_snapshot.json`

Format and grain: JSON, one primary response snapshot plus one endpoint sanity-check snapshot. Its top-level keys are:

```text
endpoint_sanity_check
http_status
request_method
response
retrieved_at_utc
source_url
timestamp_basis
```

The primary `response` contains `series_fee_change_arr`; it is empty for KXBTC15M. `endpoint_sanity_check` contains `http_status`, `request_method`, `response`, `retrieved_at_utc`, `source_url` and `timestamp_basis`; its response contains one KXINX fee-change record with `fee_multiplier`, `fee_type`, `id`, `scheduled_ts` and `series_ticker`.

The KXBTC15M response is HTTP 200 from `GET https://external-api.kalshi.com/trade-api/v2/series/fee_changes?series_ticker=KXBTC15M&show_historical=true`, retrieved at `2026-09-14T20:01:44Z` using the HTTP Date header. The KXINX sanity check was retrieved at `2026-09-14T20:01:48Z`. Its purpose is to preserve both the empty KXBTC15M series fee-change response and evidence that the endpoint returned a known record for another series; an empty array is not proof that no global, event-level or account-route historical change occurred.
