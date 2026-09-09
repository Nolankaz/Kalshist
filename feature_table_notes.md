# Model-Ready Market Feature Table

## Row Definition and Population

The model-ready artifact is `data/features/market_features.parquet`. Each row is
one decision point identified by the unique key `(ticker, horizon_minutes)`.
Every eligible KXBTC15M market contributes a T-10 row and a T-5 row, where the
decision timestamp is respectively 10 or 5 minutes before `close_time`.

The final table contains exactly 17,182 rows: 8,591 at T-10 and 8,591 at T-5.
The source and eligibility counts reconcile as follows:

- 8,597 settled markets exist in the source range.
- 8,586 markets have all 15 expected candles, 5 have partial positive candle
  coverage, and 6 have no candles. No missing candles were synthesized.
- All 8,591 markets with at least one candle have a leakage-safe quote at both
  T-10 and T-5. The five partial markets remain eligible because an observed
  candle exists at or before each decision timestamp.
- 8,585 was a stale planning count, not an observed final cohort. Subtracting
  six again from 8,591 would double-remove the six zero-candle markets. The
  implemented and independently validated eligible-market count is 8,591.

T-10 and T-5 are stored in long format so one row always represents one
prediction timestamp and shares the same feature schema. The two rows for a
ticker are correlated: they describe the same contract, strike, and outcome at
nearby times. All future train/validation/test splits must therefore be
chronological or date-based, never random, and no ticker may appear in more
than one split.

## Target Definition

The primary target is:

y = 1 if settlement_result == "yes" else 0

`settlement_result` is Kalshi's authoritative categorical resolution and is
available for every retained market. `expiration_value > strike` is not the
primary target because it is a secondary reconstruction from numeric source
fields, and either `expiration_value` or `strike` can be missing. It is retained
only as an audit target using a strict greater-than comparison.

There are 17,158 rows where both numeric fields are defined. All 17,158 agree
with `y`, giving 0 disagreements. The remaining 24 rows are undefined for this
comparison: 22 rows have a missing strike, 4 have a missing expiration value,
and 2 rows are in both sets.

## Column Groups and Point-in-Time Semantics

The four column groups are disjoint and classify all 50 columns. Only
`FEATURE_COLUMNS` should be supplied as model inputs. Target and future-only
columns are retained for labeling, validation, and later analysis.

### KEY_COLUMNS

| Column | Point-in-time semantics |
| --- | --- |
| `ticker` | Stable Kalshi contract identifier. |
| `horizon_minutes` | Decision horizon, exactly 10 or 5 minutes. |
| `open_time` | Scheduled UTC contract open timestamp. |
| `close_time` | Scheduled UTC contract close timestamp. |
| `decision_time` | `close_time - horizon_minutes`; all predictive data are cut off at this timestamp. |
| `close_date` | UTC date string derived from `close_time`. |

### FEATURE_COLUMNS

| Column | Point-in-time semantics |
| --- | --- |
| `spot` | BTC VWAP proxy price at `s(decision_time)`, the final whole-second label strictly before the decision boundary. |
| `spot_age_seconds` | Age of the observed BTC proxy price used at `s(decision_time)` after the bounded fill. |
| `strike` | Contract strike from the market source; fixed contract metadata, with upstream nulls preserved. |
| `log_moneyness` | `log(spot / strike)` using the point-in-time spot. |
| `T_years` | Remaining horizon expressed as `horizon_minutes / (365 * 24 * 60)`. |
| `hour_utc` | UTC hour of `decision_time`. |
| `day_of_week` | UTC weekday of `decision_time`, with Monday equal to 0. |
| `quote_yes_bid` | YES bid close from the selected one-minute candle. |
| `quote_yes_ask` | YES ask close from the selected one-minute candle. |
| `quote_mid` | `(quote_yes_bid + quote_yes_ask) / 2`. |
| `quote_spread` | `quote_yes_ask - quote_yes_bid`. |
| `quote_last_price` | Traded-price close from the selected candle; may be missing for a valid no-trade candle. |
| `quote_volume` | Volume reported on the selected candle, known by its period end. This is not final market-level volume. |
| `quote_open_interest` | Open interest reported on the selected candle, known by its period end. |
| `quote_period_end_ts` | Selected candle end timestamp as Unix seconds. |
| `quote_time` | UTC timestamp converted from `quote_period_end_ts`. |
| `quote_age_seconds` | `(decision_time - quote_time)` in seconds; no maximum-age filter is imposed. |
| `5min_vol` | Simple annualized realized volatility from valid one-second returns in `[decision_time - 5min, decision_time)`, subject to the 80% coverage gate. |
| `5min_ewma_vol` | EWMA annualized realized volatility over the same leakage-safe 5-minute return window and coverage gate. |
| `5min_n_obs` | Count of usable one-second returns in the 5-minute window. |
| `5min_coverage` | `5min_n_obs / 300`. |
| `15min_vol` | Simple annualized realized volatility from valid one-second returns in `[decision_time - 15min, decision_time)`, subject to the 80% coverage gate. |
| `15min_ewma_vol` | EWMA annualized realized volatility over the same leakage-safe 15-minute return window and coverage gate. |
| `15min_n_obs` | Count of usable one-second returns in the 15-minute window. |
| `15min_coverage` | `15min_n_obs / 900`. |
| `1hr_vol` | Simple annualized realized volatility from valid one-second returns in `[decision_time - 1hr, decision_time)`, subject to the 80% coverage gate. |
| `1hr_ewma_vol` | EWMA annualized realized volatility over the same leakage-safe 1-hour return window and coverage gate. |
| `1hr_n_obs` | Count of usable one-second returns in the 1-hour window. |
| `1hr_coverage` | `1hr_n_obs / 3,600`. |
| `4hr_vol` | Simple annualized realized volatility from valid one-second returns in `[decision_time - 4hr, decision_time)`, subject to the 80% coverage gate. |
| `4hr_ewma_vol` | EWMA annualized realized volatility over the same leakage-safe 4-hour return window and coverage gate. |
| `4hr_n_obs` | Count of usable one-second returns in the 4-hour window. |
| `4hr_coverage` | `4hr_n_obs / 14,400`. |
| `24hr_vol` | Simple annualized realized volatility from valid one-second returns in `[decision_time - 24hr, decision_time)`, subject to the 80% coverage gate. |
| `24hr_ewma_vol` | EWMA annualized realized volatility over the same leakage-safe 24-hour return window and coverage gate. |
| `24hr_n_obs` | Count of usable one-second returns in the 24-hour window. |
| `24hr_coverage` | `24hr_n_obs / 86,400`. |

Simple volatility is the sample standard deviation of valid log returns with
`ddof=1`, annualized by `sqrt(31,536,000)`. EWMA volatility uses the frozen
window-specific half-lives documented in `feature_notes.md`. Neither estimator
uses prices at or after `decision_time`.

### TARGET_COLUMNS

| Column | Point-in-time semantics |
| --- | --- |
| `y` | Primary binary label derived from the eventual categorical settlement; not available at decision time. |
| `settlement_result` | Eventual Kalshi `yes` or `no` result; retained as target provenance. |
| `settlement_value` | Eventual numeric payout, 1.0 for YES and 0.0 for NO; retained as a target cross-check. |

### FUTURE_ONLY_COLUMNS

| Column | Point-in-time semantics |
| --- | --- |
| `expiration_value` | Eventual Kalshi BTC reference value used at settlement; unavailable at decision time. |
| `y_from_expiration` | Audit-only strict comparison `expiration_value > strike`, where both fields are defined. |
| `target_agrees` | Whether `y_from_expiration` equals `y`, or missing when the numeric comparison is undefined. |
| `fwd_log_return` | `log(expiration_value / spot)` from the decision spot to the future settlement reference value. |

Raw market-level `volume` and `settlement_ts` are deliberately excluded. Both
describe information accumulated or recorded after the decision boundary and
would introduce future information. `quote_volume` is different: it belongs to
the selected candle ending at or before the decision timestamp.

## Quote and Spot Rules

For each row, the quote is the latest real source candle for the same ticker
whose `period_end_ts <= decision_time`. The stored values use that candle's
`yes_bid_close`, `yes_ask_close`, and `price_close`; candle highs and lows are
not decision-time prices. Equality is an accepted knife edge: a candle ending
exactly at `decision_time` is eligible. Independent as-of selection reproduced
all 17,182 stored quote timestamps.

There is no stale-quote cutoff and no quote filling. Partial markets can reuse
an older observed candle at one or both horizons. `quote_age_seconds` exposes
this staleness so later modeling can filter or control for it explicitly.

Spot uses the same filled one-second cross-exchange BTC grid as realized
volatility. For decision timestamp `t`, `s(t) = floor(t - 1ns, 1 second)`, so
spot is strictly pre-decision. Observed prices may be previous-tick filled for
at most 10 seconds; a longer gap remains missing. The final table has 876
missing spot rows, approximately 5.10% of 17,182.

## Final NaN Inventory

The following counts are from the final feature table. Zero-count rows are
included so the inventory covers every output column.

| Column | NaNs | Reason when nonzero |
| --- | ---: | --- |
| `ticker` | 0 | |
| `horizon_minutes` | 0 | |
| `open_time` | 0 | |
| `close_time` | 0 | |
| `decision_time` | 0 | |
| `close_date` | 0 | |
| `spot` | 876 | No usable BTC proxy price at `s(decision_time)` within the frozen 10-second fill limit. |
| `spot_age_seconds` | 876 | Missing whenever spot has no valid fill source. |
| `strike` | 22 | Eleven source markets have a missing strike, producing two horizon rows each. |
| `log_moneyness` | 897 | Missing when either spot or strike is missing; this is the union of those null masks. |
| `T_years` | 0 | |
| `hour_utc` | 0 | |
| `day_of_week` | 0 | |
| `quote_yes_bid` | 0 | |
| `quote_yes_ask` | 0 | |
| `quote_mid` | 0 | |
| `quote_spread` | 0 | |
| `quote_last_price` | 3 | The selected source candles are valid no-trade minutes with missing `price_close`; bid and ask remain present. |
| `quote_volume` | 0 | |
| `quote_open_interest` | 0 | |
| `quote_period_end_ts` | 0 | |
| `quote_time` | 0 | |
| `quote_age_seconds` | 0 | |
| `5min_vol` | 858 | Five-minute usable-return coverage is below 80%. |
| `5min_ewma_vol` | 858 | Same five-minute coverage gate. |
| `5min_n_obs` | 0 | |
| `5min_coverage` | 0 | |
| `15min_vol` | 766 | Fifteen-minute usable-return coverage is below 80%. |
| `15min_ewma_vol` | 766 | Same fifteen-minute coverage gate. |
| `15min_n_obs` | 0 | |
| `15min_coverage` | 0 | |
| `1hr_vol` | 686 | One-hour usable-return coverage is below 80%. |
| `1hr_ewma_vol` | 686 | Same one-hour coverage gate. |
| `1hr_n_obs` | 0 | |
| `1hr_coverage` | 0 | |
| `4hr_vol` | 622 | Four-hour usable-return coverage is below 80%. |
| `4hr_ewma_vol` | 622 | Same four-hour coverage gate. |
| `4hr_n_obs` | 0 | |
| `4hr_coverage` | 0 | |
| `24hr_vol` | 352 | Twenty-four-hour usable-return coverage is below 80%; a literal complete 24-hour history is not required. |
| `24hr_ewma_vol` | 352 | Same twenty-four-hour coverage gate. |
| `24hr_n_obs` | 0 | |
| `24hr_coverage` | 0 | |
| `y` | 0 | |
| `settlement_result` | 0 | |
| `settlement_value` | 0 | |
| `expiration_value` | 4 | Two source markets have no expiration value, producing two horizon rows each. |
| `y_from_expiration` | 24 | Undefined when either expiration value or strike is missing: 12 unique markets. |
| `target_agrees` | 24 | Missing for the same rows where `y_from_expiration` is undefined. |
| `fwd_log_return` | 880 | Missing when either spot or expiration value is missing. |

The stale Day 7 expectation of at least 384 missing `24hr_vol` rows assumed
that every row lacking a literal 24-hour history must be null. Frozen Day 6
semantics instead require 80% coverage, or 69,120 of 86,400 expected returns.
The last failing row is at `2026-05-27 19:40:00 UTC` with 68,726 observations
and 79.5440% coverage. The first passing row is at
`2026-05-27 19:50:00 UTC` with 69,322 observations and 80.2338% coverage.
All 352 `24hr_vol` nulls exactly match failures of the 80% gate.

## Final Validation Results

Structural and source validation passed with:

- 17,182 unique structural rows.
- 0 future-quote violations.
- 0 target disagreements across the 17,158 comparable rows.
- 0 independently re-derived quote-selection mismatches.

The future-corruption test used all 192 decision timestamps on
`2026-07-15` UTC. Multiplying every filled-grid price at or after each tested
decision boundary by 1.5 produced 0 volatility, spot, or log-moneyness
mismatches. A bounded perturbation strictly before the test boundary changed
all 10 volatility/EWMA values while leaving every `n_obs` and `coverage` value
unchanged.

## Market-Relative Diagnostics

The rank-based AUC smell test and market Brier scores are:

| Horizon | `quote_mid` AUC | `log_moneyness` AUC | Market Brier score |
| --- | ---: | ---: | ---: |
| T-10 | 0.791470 | 0.771496 | 0.186282 |
| T-5 | 0.919115 | 0.902318 | 0.113732 |

The 20 volatility-family AUCs range from 0.491553 to 0.557552 and remain near
0.5. No feature exceeded the same-horizon `quote_mid` AUC by the informational
0.10 investigation margin. This is a relative leakage smell test, not an
absolute performance threshold; no stale `AUC > 0.75` assertion is used.

A previously measured comparison found an approximately 1.89× scale gap
between one-second realized volatility and effective terminal volatility. That
gap is preserved as an open modeling issue rather than resolved by changing
the Day 7 table. Selecting and calibrating the appropriate sigma is a later-day
problem.

The following candidate features were explicitly not engineered on Day 7:

- Volatility ratios.
- Volatility-of-volatility.
- Regime flags.
- Z-score.
