# Evaluation Design

## Chronological Dataset Split

The split key is the stored string column `close_date`, which is the UTC date
derived from `close_time`. The frozen split is:

| Period | `close_date` range | Rows | Markets |
| --- | --- | ---: | ---: |
| Train | 2026-05-26 through 2026-07-19 | 10,392 | 5,196 |
| Validation | 2026-07-20 through 2026-08-09 | 3,966 | 1,983 |
| Test | 2026-08-10 through 2026-08-24 | 2,824 | 1,412 |
| **Total** | **2026-05-26 through 2026-08-24** | **17,182** | **8,591** |

These measured counts are assertions, not rough expectations. Any mismatch
means the split key, a boundary, or the dataset has changed.

`close_date` is intentional rather than the UTC date of `decision_time`. There
are 91 markets that close at 00:00 UTC, so their T-10 and T-5 decision
timestamps fall on the previous UTC date.

**Correction (2026-09-10):** With the frozen inclusive train range of
2026-05-26 through 2026-07-19, splitting by decision date produces 10,392 train
rows, 3,966 validation rows, and 2,822 test rows, with 2 rows on 2026-05-25
unassigned. The previously stated 10,394 train count occurs only when the train
split has no lower bound. `close_date` remains the frozen split key.

Ticker disjointness would still pass because both horizons remain on the same
decision date. The row-count assertions are therefore the main guard against
accidentally switching split keys.

## Why the Split Must Be Chronological and Never Random

The T-10 and T-5 rows from one ticker share the same target and strike. They
also use heavily overlapping price history, including 24-hour volatility
windows that overlap almost completely. Adjacent 15-minute markets occur in
the same volatility regime. A random split would place highly correlated or
near-duplicate examples on both sides of the boundary and overstate
generalization.

No ticker may appear in more than one split.

## Allowed Base Rates

| Period | Base rate |
| --- | ---: |
| Train | 0.490377 |
| Validation | 0.489662 |

The test base rate is deliberately not computed until Day 17.

## What Validation Is Allowed to Decide

Validation may later inform:

- which sigma / volatility window feeds Stage 0
- feature inclusion
- regularization strength
- calibration mapping
- edge threshold
- position sizing

## Test-Set One-Shot Rule

The test set is read for outcome-derived evaluation exactly once, on Day 17.

Before Day 17, it cannot influence any modeling, feature, calibration,
threshold, or sizing decision.

If the test result disappoints, that is the result.

Repeatedly checking test performance would turn the test set into another
validation set and remove the project's honest final out-of-sample estimate.

## Permitted Test-Set Operations Before Day 17

Only structural checks are allowed:

- row count
- market count
- date range
- contiguity
- ticker disjointness against train and validation
- per-column NaN counts

Before Day 17, computing the test base rate, Brier score, AUC, calibration, log
loss, or any other outcome-derived statistic is prohibited.

## Walk-Forward Evaluation Schedule

The walk-forward schedule is frozen over train and validation combined. The
initial fit window is 2026-05-26 through 2026-06-28 and contains 6,414 rows.
Each fold scores the next chronological week and then adds it to the expanding
fit window.

| Fold | Score dates | Scored rows | Expanding fit rows |
| --- | --- | ---: | ---: |
| 1 | 2026-06-29 through 2026-07-05 | 1,326 | 6,414 |
| 2 | 2026-07-06 through 2026-07-12 | 1,326 | 7,740 |
| 3 | 2026-07-13 through 2026-07-19 | 1,326 | 9,066 |
| 4 | 2026-07-20 through 2026-07-26 | 1,326 | 10,392 |
| 5 | 2026-07-27 through 2026-08-02 | 1,314 | 11,718 |
| 6 | 2026-08-03 through 2026-08-09 | 1,326 | 13,032 |
| **Total** | **2026-06-29 through 2026-08-09** | **7,944** | |

This schedule is expanding rather than rolling because the dataset covers
only 91 days. Data scarcity currently matters more than regime obsolescence,
although that choice may be revisited later.

## Volatility Regime Warning

Historical feature distributions differ substantially between train and
validation. Approximate simple realized-volatility medians are:

| Feature | Train median | Validation median |
| --- | ---: | ---: |
| `5min_vol` | 0.7519 | 0.4536 |
| `15min_vol` | 0.7460 | 0.4524 |
| `1hr_vol` | 0.7520 | 0.4629 |
| `4hr_vol` | 0.7513 | 0.4773 |
| `24hr_vol` | 0.7415 | 0.4865 |

The train period was roughly 1.6x more volatile than validation. This is only
a feature-side descriptive fact; nothing has been trained or validated yet.
Train-period metrics should therefore not automatically be expected to match
validation-period metrics.

## Day 8 Stage 0 Definition

Stage 0 uses a zero-drift lognormal price model:

```text
z = log_moneyness / (sigma * sqrt(T_years))
P(YES) = Phi(z)
```

The sign checks are:

- `spot == strike` implies `P(YES) = 0.5`.
- `spot > strike` implies `P(YES) > 0.5`.
- `spot < strike` implies `P(YES) < 0.5`.

The time conversion is consistent with 31,536,000 seconds per year and
525,600 minutes per year:

- T-10: `T_years = 1.9025875190258754e-05`.
- T-5: `T_years = 9.512937595129377e-06`.

The ten annualized sigma candidates used on Day 8 were:

1. `5min_vol`
2. `5min_ewma_vol`
3. `15min_vol`
4. `15min_ewma_vol`
5. `1hr_vol`
6. `1hr_ewma_vol`
7. `4hr_vol`
8. `4hr_ewma_vol`
9. `24hr_vol`
10. `24hr_ewma_vol`

No sigma candidate was selected on Day 8.

`Phi` was implemented with `math.erf` and NumPy, with no SciPy dependency.
Float64 saturation begins around `|z| ≈ 8.3`. On train, `24hr_vol` produced
one probability exactly equal to `0.0`, and `24hr_ewma_vol` produced one
probability exactly equal to `0.0`. No other candidate produced a saturated
train probability.

## Day 8 Train-Only Stage 0 Scoring

Each candidate was scored separately by horizon on train rows where its Stage
0 probability was non-null. The constant benchmark used the frozen train base
rate of `0.490377` on those exact rows. The market benchmark used `quote_mid`
on the same rows. Log loss used probabilities clipped to `[1e-15, 1 - 1e-15]`.
The log-moneyness AUC is also computed on the same candidate-specific rows.

| Candidate | Horizon | n | Brier | Log loss | AUC | Log-moneyness AUC | Constant Brier | Market Brier | Saturated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5min_vol` | T-10 | 4,910 | 0.203782 | 0.594685 | 0.777383 | 0.774910 | 0.249916 | 0.186683 | 0 |
| `5min_vol` | T-5 | 4,863 | 0.143036 | 0.451740 | 0.911452 | 0.908648 | 0.249893 | 0.111385 | 0 |
| `5min_ewma_vol` | T-10 | 4,910 | 0.203371 | 0.593816 | 0.777906 | 0.774910 | 0.249916 | 0.186683 | 0 |
| `5min_ewma_vol` | T-5 | 4,863 | 0.142541 | 0.450212 | 0.911306 | 0.908648 | 0.249893 | 0.111385 | 0 |
| `15min_vol` | T-10 | 4,917 | 0.203380 | 0.593614 | 0.777829 | 0.774925 | 0.249911 | 0.186724 | 0 |
| `15min_vol` | T-5 | 4,890 | 0.144111 | 0.454602 | 0.911502 | 0.908293 | 0.249884 | 0.111855 | 0 |
| `15min_ewma_vol` | T-10 | 4,917 | 0.203447 | 0.593876 | 0.777901 | 0.774925 | 0.249911 | 0.186724 | 0 |
| `15min_ewma_vol` | T-5 | 4,890 | 0.144007 | 0.454444 | 0.911490 | 0.908293 | 0.249884 | 0.111855 | 0 |
| `1hr_vol` | T-10 | 4,929 | 0.203722 | 0.594287 | 0.777499 | 0.774443 | 0.249903 | 0.186719 | 0 |
| `1hr_vol` | T-5 | 4,903 | 0.144864 | 0.456210 | 0.911879 | 0.908232 | 0.249890 | 0.111623 | 0 |
| `1hr_ewma_vol` | T-10 | 4,929 | 0.203712 | 0.594330 | 0.777596 | 0.774443 | 0.249903 | 0.186719 | 0 |
| `1hr_ewma_vol` | T-5 | 4,903 | 0.144760 | 0.456113 | 0.911977 | 0.908232 | 0.249890 | 0.111623 | 0 |
| `4hr_vol` | T-10 | 4,927 | 0.204097 | 0.595227 | 0.777414 | 0.774671 | 0.249923 | 0.186665 | 0 |
| `4hr_vol` | T-5 | 4,906 | 0.146026 | 0.458845 | 0.910908 | 0.908050 | 0.249900 | 0.111804 | 0 |
| `4hr_ewma_vol` | T-10 | 4,927 | 0.204046 | 0.595119 | 0.777559 | 0.774671 | 0.249923 | 0.186665 | 0 |
| `4hr_ewma_vol` | T-5 | 4,906 | 0.145827 | 0.458483 | 0.911225 | 0.908050 | 0.249900 | 0.111804 | 0 |
| `24hr_vol` | T-10 | 4,888 | 0.204406 | 0.595768 | 0.777078 | 0.774943 | 0.249935 | 0.186556 | 0 |
| `24hr_vol` | T-5 | 4,869 | 0.147465 | 0.461946 | 0.909606 | 0.908044 | 0.249913 | 0.111580 | 1 |
| `24hr_ewma_vol` | T-10 | 4,888 | 0.204388 | 0.595758 | 0.777183 | 0.774943 | 0.249935 | 0.186556 | 0 |
| `24hr_ewma_vol` | T-5 | 4,869 | 0.147336 | 0.461732 | 0.909729 | 0.908044 | 0.249913 | 0.111580 | 1 |

Stage 0 clearly beats the constant baseline but loses to the market at both
horizons. For `5min_vol`, T-10 Stage 0 Brier is `0.203782`, versus constant
`0.249916` and market `0.186683`. At T-5, Stage 0 Brier is `0.143036`, versus
constant `0.249893` and market `0.111385`.

## Day 8 Train-Only Reliability

The `5min_vol` train-only reliability endpoints were:

| Horizon and decile | Mean predicted probability | Observed YES frequency |
| --- | ---: | ---: |
| T-10 lowest decile | 0.231346 | 0.083503 |
| T-10 highest decile | 0.729891 | 0.894094 |
| T-5 lowest decile | 0.095360 | 0.018480 |
| T-5 highest decile | 0.879721 | 0.983573 |

Observed YES frequency rises monotonically across deciles at both horizons.
Stage 0 is systematically under-confident: its probabilities are compressed
toward `0.5`, with low predicted probabilities above observed frequencies and
high predicted probabilities below observed frequencies.

## Day 8 Volatility Scale Findings

The train-only volatility-scale gap was approximately `1.78x` at T-10 and
`1.87x` at T-5. Day 7's frozen `~1.89x` figure was effectively the
train-period T-5 measurement.

The train-versus-validation volatility regime gap remains material: train
medians were roughly `~0.75`, while validation medians were roughly `~0.45`.
The train period was therefore about `1.6x` more volatile.

**Test-set status (2026-09-10):** The test split was not scored and no outcome-derived test statistic was computed on Day 8.

## Day 9 sigma-selection rule — frozen before validation scoring

This rule was written before any validation Brier, log loss, AUC, reliability curve, or other outcome-derived validation metric was computed.

### Selection population

Sigma selection will use the common row population separately at T-10 and T-5: only rows for which all ten Stage 0 probability candidates are non-null.

Per-candidate-row metrics may be reported for coverage diagnostics, but they do not enter the sigma-selection decision.

### Primary metric

The primary selection metric is validation Brier score on the uncalibrated Stage 0 probabilities.

Log loss and AUC will be reported as secondary diagnostics but will not determine the selected sigma.

### Horizon policy

One sigma estimator will be selected jointly for T-10 and T-5.

The primary score is the row-count-weighted mean of the common-row T-10 and T-5 Brier scores.

Both horizon-specific scores will also be reported. Any disagreement between the horizons will be recorded as a finding but will not create separate sigma estimators.

### Meaningful-difference rule

Candidate A is meaningfully worse than candidate B only when the paired mean difference in per-row squared error exceeds two paired standard errors on the common rows.

Differences within two paired standard errors are treated as ties.

The pooled comparison will use the same row-count weighting as the primary metric. Per-horizon paired comparisons will also be reported. A difference that appears meaningful only after pooling the two correlated horizons will be treated cautiously and will not override a per-horizon tie.

### Tie-break 1 — volatility window length

Among candidates not meaningfully worse than the numerically best candidate, prefer the shortest volatility window.

### Tie-break 2 — estimator simplicity

Within the selected volatility window, prefer the simple volatility estimator over its EWMA counterpart unless the simple estimator is meaningfully worse under the paired rule.

### Calibration

Calibration does not participate in sigma selection.

Sigma is selected using uncalibrated Stage 0 probabilities only. Calibrated metrics may later be reported, but they cannot change today's sigma selection.

### Train versus validation

If the validation ranking disagrees with the Day 8 train ranking, validation determines the sigma selection.

The disagreement will be recorded as evidence about estimator stability across regimes.

### Coverage floor

A sigma candidate with less than 80% coverage at either horizon is excluded from selection regardless of score.

### Irreversibility

Once validation metrics are computed, this rule will not be changed in response to those results. Any future alternative selection method will be treated as a separate experiment rather than a revision of this rule.

## Day 9 Stage 0 Selection and Calibration Conclusions

### Selected Stage 0 sigma

The frozen validation/common Brier and paired 2-SE selection rule was applied without modification after validation results were observed. All ten candidates passed the 80% coverage floor, and `5min_ewma_vol` was the final winner.

### Legitimate Platt calibration

Only parameters with `fit_split == "train"` and `parameter_role == "legitimate_train_fit"` are legitimate production calibration:

| Horizon | `a` | `b` |
| --- | ---: | ---: |
| T-10 | 0.009439668 | 2.161921969 |
| T-5 | 0.052819163 | 2.500293302 |

Validation-fitted parameters have `parameter_role == "validation_in_sample_ceiling"`. They are diagnostic in-sample ceilings only and must not be used as production calibration.

### Evaluation conclusions

- Train-fitted Platt calibration improved validation Brier and log loss at both horizons; AUC was unchanged.
- Market `quote_mid` retained lower validation/common Brier than both raw and Platt-calibrated Stage 0.
- Model-market disagreement is not considered tradable edge.
- Basis risk is material enough that later trading thresholds must account for it.
- Test outcomes remain untouched under the Day 17 one-shot rule.

Detailed Day 9 results and interpretation are recorded in [`calibration_notes.md`](calibration_notes.md).

## Day 10 Edge Threshold Rule

This Section 3.1 rule is pre-registered before building any threshold value, observing any threshold-clearance count, or computing any trading P&L. It leaves Section 3.2 only the mechanical work of applying the definitions below.

### Threshold scale, probability, and order size

The threshold applies to **net executable edge**, not model-versus-mid disagreement. A candidate on side `s` clears only when:

`net_edge_s >= required_net_edge`

Using the frozen Section 2.1 definitions:

`net_edge_yes = p - ask - fee_yes`

`net_edge_no = bid - p - fee_no`

The executable bid/ask price has already crossed the spread. Do not subtract or add half-spread again when applying the threshold to these executable-edge quantities. A midpoint-scale equivalent may be reported later for interpretation, but it is not the threshold applied to a trade.

The sole trading probability `p` is the train-fitted Platt calibration applied to `p_5min_ewma_vol`, separately by horizon, using only rows from `stage0_platt_parameters.parquet` with `fit_split == "train"` and `parameter_role == "legitimate_train_fit"`. Raw Stage 0 probabilities, validation-fitted in-sample ceiling parameters, other sigma candidates, and refitted parameters are prohibited. Order size is frozen at `C = 1`, consistent with execution assumption A7; no sizing rule enters this threshold.

### Price buckets and sparse-cell rule

Rows are bucketed by `quote_mid` using the exact seven Section 2.2 buckets:

1. `[0.00, 0.10)`
2. `[0.10, 0.25)`
3. `[0.25, 0.40)`
4. `[0.40, 0.60)`
5. `[0.60, 0.75)`
6. `[0.75, 0.90)`
7. `[0.90, 1.00]`

For each `(horizon, price_bucket)` cell, if the cell contains fewer than **50 validation common rows**, merge it with its adjacent bucket toward `0.50` before calculating the bucket statistic. Low-side buckets merge upward and high-side buckets merge downward. This rule is frozen before seeing threshold values and cannot be changed in response to them. The current minimum validation common bucket count is expected to be 54, so no merge is expected to occur; a mismatch must be treated as a regression failure rather than permission to change the buckets or cutoff.

### Basis-risk term

The **primary basis-error size is `basis_bps == 1.2`**, the artifact representation of the approximately `1.191 bps` basis standard deviation measured on Day 3. This is the measured typical-size proxy error and is paired with a model-error term intended to represent typical-size model error.

For each horizon and price bucket, the primary `basis_term` is the **median `max_abs_shift`** from `data/models/stage0_basis_probability_sensitivity.parquet` after filtering to:

- `split == "validation"`;
- `probability_version == "train_fitted_platt"`;
- `basis_bps == 1.2`.

Join these sensitivity rows to `quote_mid` on the complete established row key `(ticker, horizon_minutes)` and apply the price buckets above. The population is the validation common population already represented by the Day 9 sensitivity artifact. Median is frozen instead of mean or p90 because the basis-sensitivity distribution is long-tailed and the primary term is intended to be a stable typical-size margin.

The `basis_bps == 5.0` version uses the identical validation population, row-key join, price buckets, `train_fitted_platt` probability version, and median `max_abs_shift` statistic. It is a **conservative sensitivity, not primary**: approximately a four-standard-deviation judgmental buffer rather than the measured typical basis error. Both versions must be built and labeled separately; the choice between them may not depend on signal counts or later profitability.

Validation is used for the basis term because this sensitivity calculation is outcome-free. It depends on model inputs and fitted calibration parameters, not settlement results. Validation represents the more recent regime, and Day 9 found basis sensitivity materially larger there than on train, making it the more conservative available regime for this component without consuming validation outcomes. Test rows are prohibited.

### Model-error term

The `model_error_term` is one number per horizon: the **10-decile expected calibration error (ECE)** of the train-fitted Platt probability on the validation common population. It is defined as:

`ECE = Σ_k (n_k / n) * |observed_yes_rate_k - mean_predicted_k|`

Section 3.2 must first apply the legitimate train-fitted Platt parameters to `p_5min_ewma_vol`, then pass that calibrated probability column to the existing `build_reliability_table(...)` helper in `scripts/analyze_calibration.py`. The deciles are therefore formed on the train-fitted calibrated probability, never the raw probability. The validation-fitted in-sample ceiling is prohibited.

ECE is calculated across the full validation common population separately at T-10 and T-5. It is **not price-bucket-specific**. Calibration error requires validation outcomes because train is in-sample for the Platt fit, while validation is out of sample for the fitted calibrator. Subdividing ten reliability bins across seven price buckets would create thin, noisy cells and is not allowed.

Ten-bin ECE on approximately 1,650 rows is biased upward by binomial sampling noise. Near `p = 0.5`, approximately 165 rows per decile imply noise on the order of `sqrt(p(1-p)/165)`, or roughly 3–4 percentage points per bin. This makes the chosen ECE margin conservative. Do not bias-correct it, and do not switch to another calibration statistic after observing its value.

### Combination, primary/sensitivity outputs, and no iteration

For every horizon and effective price bucket, combine the terms by simple addition:

`required_net_edge = basis_term + model_error_term`

Quadrature is explicitly rejected for the primary rule because it would implicitly assume independence and roughly Gaussian error structure between basis error and model calibration error. Neither assumption is established, so the conservative additive sum is frozen. The formula may not be changed because the resulting threshold appears too high or too low.

Section 3.2 must mechanically produce two versions:

- **Primary:** `basis_bps = 1.2`.
- **Conservative sensitivity, not primary:** `basis_bps = 5.0`.

Both versions use the same train-fitted calibrated probability, validation population, price buckets, horizon-level model-error ECE, and additive combination rule. Only the basis-error size differs.

The threshold is computed once from this rule. Section 3.3's clearance count is a **result, not an optimization target**. Whether zero rows, almost no rows, or many rows clear, no component, statistic, population, bucket rule, or combination method may be altered after clearance is observed.

This threshold is specific to Stage 0, `5min_ewma_vol`, and the current train-fitted Platt calibrator. Any Day 12 or later model must recompute its own basis-risk and model-error terms using the same general rule rather than inheriting Stage 0's numerical threshold.

### Validation-query ledger

| # | Day | Validation outcome use | Type |
| - | --- | --- | --- |
| 1 | 9 | σ selection across ten candidates, two horizons, common rows | Decision |
| 2 | 9 | Raw reliability deciles with Wilson bands | Diagnostic |
| 3 | 9 | Train-fitted Platt evaluation (Brier, log loss, AUC) against `quote_mid` and constant | Decision support — confirmed calibrator ships |
| 4 | 9 | Validation-fitted in-sample ceiling | Diagnostic, never shipped |
| 5 | 9 | Settlement proximity on validation using `expiration_value` | Diagnostic |
| 6 | 10 | Model-error term: ECE of train-fitted Platt on validation | **Decision — edge threshold** |

Every validation-based decision makes validation progressively more optimistic as an estimate of future test performance. This ledger exists so Day 17 can state exactly how many decisions validation informed instead of pretending validation remained untouched.

Not every validation use has the same implication. Validation basis sensitivity is outcome-free and therefore is not a validation-outcome query. Validation ECE uses outcomes and belongs in the decision ledger. Raw reliability is diagnostic. The validation-fitted in-sample ceiling is diagnostic only and must never ship.

No threshold component value, combined threshold, or clearance count is computed in this section. No test row or trading P&L is read. **This rule was written and saved before any Day 10 threshold table or clearance count was computed.**

## Day 10 Threshold Result

For Stage 0 using `5min_ewma_vol` and the train-fitted Platt calibration, validation 10-decile ECE was `0.035595749` at T-10 and `0.021091839` at T-5. With the primary `1.2 bps` basis size, required net edge ranged from `0.045622514–0.087665994` at T-10 and `0.030558963–0.107034755` at T-5. The `5.0 bps` version remains a conservative sensitivity and is explicitly non-primary. Full Day 10 execution-cost, threshold and outcome-free clearance findings are recorded in [`execution_notes.md`](execution_notes.md).

**Test-set status (2026-09-14): the test split was not read on Day 10.**

## Day 11 Tier-2 Backtest Protocol

This protocol is frozen before any Day 11 code loads an outcome. All results under it are **Tier 2 — quote-aware, top-of-book, size-unaware**. Historical top-of-book prices do not establish fillable size, depth, latency or realized slippage.

### Scope and inherited rules

Evaluate Stage 0 with the selected `5min_ewma_vol` sigma candidate and only the horizon-specific, train-fitted Platt probabilities (`fit_split == "train"`, `parameter_role == "legitimate_train_fit"`). Use the Day 9 common-row population: rows at T-10 or T-5 for which all ten Stage 0 candidate probabilities are non-null. Use the frozen `close_date` split: train is 2026-05-26 through 2026-07-19 and validation is 2026-07-20 through 2026-08-09, inclusive. Day 11 uses train and validation only. **The test split is not loaded or inspected on Day 11.**

Import the already-frozen Day 10 definitions in [`execution_notes.md`](execution_notes.md) and the Day 10 Edge Threshold Rule above without modifying them: executable YES/NO prices, the Direct Member one-contract `net_cash_fee` implementation, the `quote_age_seconds > 60` stale-quote exclusion, the seven `quote_mid` price buckets, the saved `data/execution/edge_threshold.parquet` thresholds, one-contract taker entry, holding to settlement, and Kalshi settlement as contract truth. The historical fee-applicability and top-of-book fillability qualifications in the Day 10 execution-assumption register remain in force. Do not refit Platt, rebuild thresholds, or substitute midpoint prices.

### Outcome-free trade selection

Reuse the Day 10 row-level signal definition from `scripts/count_threshold_clearance.py`: a row clears only when `signal_eligible` is true and `best_net_edge >= required_net_edge`. Use its best-side and net-edge calculations and the saved threshold for that row's horizon, price bucket and `basis_bps`; do not reimplement the clearance rule differently.

Apply the one-position-per-market rule separately for each `basis_bps` setting. For each ticker, select the clearing row with the earliest `decision_time`. If T-10 clears, enter at T-10 and suppress T-5 regardless of its side or edge. If T-10 does not clear, T-5 may become the trade if it clears. There is no adding, reversing, hedging or exit. Comparing the T-10 and T-5 edges to select the larger one would use information unavailable at T-10 and introduce lookahead bias. Freeze the selected trade lists before joining outcomes.

### Payoff and per-trade accounting

Kalshi `settlement_value` determines the binary payoff: `payoff = settlement_value` for YES and `payoff = 1 - settlement_value` for NO. Later, `y` may be loaded only to assert `y == settlement_value`; `expiration_value` must not determine payoff.

For each one-contract trade, `entry_price` is the frozen executable price for its side, and `fee` is `net_cash_fee` from the frozen Direct Member fee implementation at that price. Calculate:

```text
gross_pnl = payoff - entry_price
net_pnl = gross_pnl - fee
hit = (payoff == 1) for the traded side
capital = entry_price + fee
```

For reference lines, the model-expected net P&L of a trade is its selected-side train-fitted Platt payoff probability minus `entry_price + fee` (equivalently its `best_net_edge`). The market-fair expected net P&L uses `quote_mid` for a YES payoff probability or `1 - quote_mid` for NO, minus the same `entry_price + fee`. `surprise` is realized `net_pnl` minus model-expected net P&L. These are comparisons, not alternative trade-selection rules.

### Required reports and primary result

Each split × `basis_bps` setting must report trade count; distinct markets; days with trades; total close_date days represented in the frozen Day 9 common-row population; T-10 and T-5 counts; YES and NO counts; hit rate; mean entry price; break-even hit rate `mean(entry_price + fee)`; gross P&L; fees on a separate line; the fee rounding component; net P&L; net P&L per trade; mean net edge at entry; model-expected total P&L; market-fair expected total P&L; mean surprise; total capital; and net P&L / capital. Show monetary P&L and fees in dollars for one-contract trades. The fee rounding component is the frozen fee result's `rounding_adjustment - rebate`, summed across trades; it is part of `net_cash_fee`, not an additional charge. Totals and counts must reconcile to the selected trade list.

The **primary result** is validation, `basis_bps = 1.2`, net P&L per trade with a day-clustered ±2 standard-error interval, clustering trades by `close_date`. For `N` trades on `D` traded days, with mean net P&L `m`, use `SE = sqrt(D / (D - 1) * sum_d (sum_{i in d}(net_pnl_i - m))^2 / N^2)`; an interval requires at least two traded days. Train is in-sample for the Platt calibrator. `basis_bps = 5.0` is sensitivity only and can never become primary based on observed P&L.

### Pre-registered descriptive views

Break down each trade list by horizon; side; the existing seven `quote_mid` price buckets (`p00_10`, `p10_25`, `p25_40`, `p40_60`, `p60_75`, `p75_90`, `p90_100`); and `abs(z_5min_ewma_vol)` buckets `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 1.0)`, and `[1.0, infinity)`. Use UTC decision-hour blocks `00–05`, `06–11`, `12–17`, and `18–23`. A full 24-hour table may be stored, but it is not a primary interpretation surface. Group weeks by Monday-start week based on `close_date`. Flag every cell with fewer than 30 trades as **thin** and do not interpret it.

Report a descriptive-only train sensitivity excluding `close_date` 2026-05-27 through 2026-05-31, inclusive. This interval was identified before observing Day 11 outcomes. The sensitivity never replaces the full train result.

For concentration, report the largest single day's share of total net P&L, the largest single Monday-start week's share of total net P&L, and the number of validation weeks whose net P&L per trade has the same sign as the validation total. Do not interpret a share with a zero total net P&L as a finite ratio.

### Interpretation and no-iteration rule

These categories are fixed before outcomes are joined. Let `m` be validation 1.2 bps net P&L per trade and `SE` its day-clustered standard error:

- If `m + 2*SE < 0`: "Stage 0 at the frozen threshold loses money after costs on validation."
- If the interval contains zero: "No tier-2 evidence of edge either way; not established."
- If `m - 2*SE > 0`: "A positive tier-2 validation result, not an established edge: one three-week period, validation already used for the threshold's ECE, optimistic fills."

The final interpretation must compare realized P&L per trade with both the market-fair and model-expected reference lines. None of the inherited choices or Day 11 rules above may change after outcomes are joined. Only a genuine implementation bug violating a written invariant or accounting identity may be fixed. If a fix occurs, record its before/after behavior and justification in `backtest_notes.md` and rerun both splits.

### Validation-query ledger and frozen trade-list placeholders

Pre-declared ledger entry **7**: **Day 11**; validation outcome use: **tier-2 P&L of the frozen Stage 0 strategy on validation**; **1.2 bps primary**, **5.0 bps sensitivity**; purpose: **Evaluation**; **no parameter chosen**. This entry records the planned query, not a result.

| Frozen trade list | Trade count |
| --- | ---: |
| Train, 1.2 bps | 1,881 |
| Validation, 1.2 bps | 719 |
| Train, 5.0 bps | 277 |
| Validation, 5.0 bps | 100 |

Decision fingerprint: **f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96**. These placeholders must be filled from the actual outcome-free trade-list output in Section 1.3; no expected planning counts are inserted here.

## Day 11 Backtest Result

### Validation-query ledger — Day 11 addition

| # | Day | Validation outcome use | Purpose | Parameter chosen |
| --- | --- | --- | --- | --- |
| 7 | 11 | Tier-2 P&L of the frozen Stage 0 strategy on validation; 1.2 bps primary, 5.0 bps conservative sensitivity | Evaluation | None |

The outcome-free decision list had SHA-256 fingerprint `f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96`. No parameter, model, threshold, or trade rule was selected from the Day 11 outcomes.

**Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.** The primary validation 1.2-bps book had **719 trades on 21 population days and 21 traded days**. Its mean realized net P&L was **-$0.017710570236 per trade**; the pre-registered traded-day-clustered SE was **$0.012277194816**, giving the clustered **±2SE interval [-$0.042264959869, +$0.006843819396] per trade**. The interval crosses zero. The exact frozen interpretation is: **"No tier-2 evidence of edge either way; not established."** The 5.0-bps book remains sensitivity only. Full accounting, breakdowns, daily/weekly concentration, and statistical caveats are in [`backtest_notes.md`](backtest_notes.md).

Test-set status (Day 11): the test split was not read.

## Day 12 Model Protocol

This protocol is frozen before any Day 12 feature is created or any Day 12 model is fitted. It is documentation and pre-registration only.

### 1. Scope

Day 12 uses only the Day 9 common rows: rows at T-10 or T-5 for which all ten Stage 0 probability candidates are non-null. It uses train and validation only, with the stored string column `close_date` as the split key. **The test split is not loaded or inspected.** Models are fitted separately by horizon. Stage 0 re-scored on the identical rows is the sole baseline of record; no alternative baseline may replace it after results are observed.

### 2. Inherited frozen rules

Day 12 inherits the following rules unchanged:

- The six Day 8 walk-forward fold boundaries and expanding-fit schedule.
- The Day 9 common population.
- `PLATT_CLIP = 1e-6` in `platt_feature(...)`.
- The Day 10 execution machinery and additive threshold composition rule, `required_net_edge = basis_term + model_error_term`.
- The Day 11 accounting identities, one-position-per-market rule, and traded-`close_date` clustered-SE formula. For `N` trades on `D` traded days with mean net P&L `m`, that formula remains `SE = sqrt(D / (D - 1) * sum_d (sum_{i in d}(net_pnl_i - m))^2 / N^2)`.

No inherited rule is reopened by this protocol.

### 3. Frozen feature list and hypotheses

The model has exactly the following 11 features, plus an intercept. Each stated sign is a pre-fit hypothesis, not a result.

1. `stage0_logit = platt_feature(p_5min_ewma_vol)`
   - This is the baseline carrier: it preserves the selected Stage 0 probability signal on the log-odds scale, so the model nests horizon-specific Platt calibration when all other coefficients are zero.
   - Its expected sign is strongly positive, around the existing Platt slope scale, because higher Stage 0 YES probability should map to higher fitted YES probability and the current Platt fits already establish that monotone direction.

2. `log_sigma = log(5min_ewma_vol)`
   - This represents the absolute short-window volatility-level regime, allowing a common Stage 0 logit to receive a level-dependent intercept adjustment.
   - Its expected sign is not predicted and should be small because volatility already enters Stage 0's denominator; any remaining standalone level effect could reasonably operate in either direction.

3. `log_ratio_5m_1h = log(5min_vol / 1hr_vol)`
   - This measures short-horizon volatility expansion or contraction relative to the recent one-hour regime.
   - Its expected main-effect sign is approximately zero because the hypothesis is that this ratio changes the reliability or scale of Stage 0 confidence, not the unconditional YES direction.

4. `log_ratio_15m_4h`
   - This captures medium-horizon volatility expansion or contraction relative to the broader four-hour regime.
   - Its expected main-effect sign is approximately zero because a symmetric change in volatility regime should affect confidence calibration rather than systematically favor YES or NO.

5. `log_ratio_1h_24h`
   - This captures session-scale volatility relative to the trailing daily regime.
   - Its expected main-effect sign is approximately zero because the regime comparison is intended to describe calibration conditions, not provide directional price information.

6. `log_vol_of_vol`
   - This measures recent instability in short-window volatility, distinguishing a stable volatility estimate from one moving through a turbulent regime.
   - Its expected main-effect sign is approximately zero because instability alone should change confidence rather than create a persistent YES or NO tilt.

7. `sin_hour`
   - This is the sine component of UTC intraday seasonality, allowing a smooth cyclical time-of-day effect without a discontinuity at midnight.
   - Its expected effect is small and no sign is predicted because the phase and direction of any intraday calibration pattern are not known in advance.

8. `cos_hour`
   - This is the complementary cosine component needed to represent UTC intraday seasonality without forcing a fixed phase.
   - Its expected effect is small and no sign is predicted because any time-of-day calibration pattern is cyclical and has no pre-registered directional phase.

9. `stage0_logit × log_ratio_5m_1h`
   - This permits regime-dependent rescaling of Stage 0 confidence when short-window volatility expands or contracts relative to the one-hour window.
   - Its expected sign is positive under the Day 12 hypothesis: short-horizon volatility expansion should strengthen the mapping from the Stage 0 logit to the outcome rather than create a standalone directional shift.

10. `stage0_logit × log_vol_of_vol`
    - This permits the Stage 0 confidence scale to respond to instability in the volatility estimate.
    - Its expected sign is negative because unstable volatility should shrink confidence toward 0.5 rather than leave extreme Stage 0 logits fully trusted.

11. `stage0_logit × log_sigma`
    - This permits volatility-level-dependent rescaling of Stage 0 confidence beyond the standalone volatility-level term.
    - Its expected sign is not predicted because the existing train/validation regime shift does not establish whether a higher volatility level should strengthen or weaken the Stage 0 logit after the other frozen terms are included.

### 4. Excluded features

The model will not use:

- Any quote-derived feature, including `quote_mid`, `quote_yes_bid`, `quote_yes_ask`, `quote_spread`, or any other bid/ask/mid/price-derived column.
- Raw `z_5min_ewma_vol` as a model input.
- `day_of_week`.
- `fwd_log_return`.
- Any `FUTURE_ONLY` column.
- Any feature outside the frozen list above.

No feature may be added after any score is seen. Quote-derived columns remain available only where a frozen execution or descriptive rule explicitly requires them; they are not model inputs.

### 5. Preprocessing

Every model feature is standardized to zero mean and unit variance. The mean and standard deviation are fitted from fit-window rows only and then applied unchanged to the corresponding scored rows. Interaction features are constructed from their unstandardized components first; the resulting interaction column is then standardized using fit-window statistics. The intercept remains unpenalized.

### 6. Configurations

#### M1 — walk-forward

M1 uses the six frozen Day 8 folds and an expanding fit window:

| Fold | Fit dates | Score dates |
| --- | --- | --- |
| 1 | 2026-05-26 through 2026-06-28 | 2026-06-29 through 2026-07-05 |
| 2 | 2026-05-26 through 2026-07-05 | 2026-07-06 through 2026-07-12 |
| 3 | 2026-05-26 through 2026-07-12 | 2026-07-13 through 2026-07-19 |
| 4 | 2026-05-26 through 2026-07-19 | 2026-07-20 through 2026-07-26 |
| 5 | 2026-05-26 through 2026-07-26 | 2026-07-27 through 2026-08-02 |
| 6 | 2026-05-26 through 2026-08-02 | 2026-08-03 through 2026-08-09 |

Its expected out-of-fold population is 7,012 Day 9 common rows: 3,531 T-10 rows and 3,481 T-5 rows.

#### M2 — train-only

M2 fits once on train common rows and scores validation common rows. Its fit population is 4,792 T-10 rows and 4,757 T-5 rows. Its score population is 1,685 T-10 rows and 1,649 T-5 rows, or 3,334 validation rows in total. M2 is the primary head-to-head configuration against Stage 0.

#### B1 — walk-forward Stage 0

B1 uses the same walk-forward harness with one feature only, `stage0_logit`. It is effectively unregularized and is used to prove equivalence to the existing `fit_platt(...)` implementation on identical fit and score rows.

### 7. Regularization

The M1 and M2 logistic fits use exactly:

```text
penalty="l2"
solver="lbfgs"
max_iter=1000
tol=1e-6
C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10}
```

For each regularized configuration and horizon, the last seven `close_date` days of the current fit window form an inner holdout. Choose `C` by the lowest Brier score on that inner holdout. A tie selects the smaller `C`, meaning stronger regularization. Then refit on the entire fit window using the selected `C`. Outer scored rows never participate in `C` selection. B1 retains its separately frozen effectively unregularized role so it can test equivalence to `fit_platt(...)`.

A fixed `C = 1` sensitivity run is also pre-registered. It is a sensitivity analysis and cannot replace the selected-`C` primary result after scores are observed.

### 8. Comparison populations

- **Primary:** folds 4–6, which are the validation common rows: 3,334 rows.
- **Secondary:** folds 1–3, which are train dates: 3,678 rows. This population is explicitly labeled Stage 0 in-sample for its calibrator.
- **Descriptive:** all six folds: 7,012 rows.

These populations are frozen and cannot be changed after scoring.

### 9. Probability metrics

The primary probability metric is Brier score. Comparison with Stage 0 is performed per horizon and as a pooled row-count-weighted result. On each identical scored row, define the paired squared-error difference:

`d_i = (p_model - y)^2 - (p_stage0 - y)^2`

The paired standard error is:

`SE(d) = sd(d) / sqrt(n)`

A difference is meaningful only when `abs(mean(d)) > 2 * SE(d)`. A negative `mean(d)` favors the Day 12 model; a positive `mean(d)` favors Stage 0.

Secondary diagnostics are log loss, AUC, and 10-decile ECE. They are reported but do not decide the verdict.

### 10. Economic comparisons

Both economic comparisons below are pre-registered.

#### A. Fixed-bar comparison

Use Stage 0's frozen 1.2 bps `data/execution/edge_threshold.parquet`. This is Stage 0's fixed error-budget bar, not the Day 12 model's own threshold. The Stage 0 comparison uses the frozen Day 11 results restricted to the comparable calendar. The validation Stage 0 book contains 719 trades.

#### B. Model-own-threshold comparison

Reapply the Day 10 composition rule:

`required_net_edge = basis_term + model_error_term`

The `basis_term` comes from the Day 12 model's own ±1.2 bps spot-perturbation sensitivity. The `model_error_term` comes from the Day 12 model's 10-decile validation ECE. Use the same seven frozen `quote_mid` price buckets:

1. `[0.00, 0.10)`
2. `[0.10, 0.25)`
3. `[0.25, 0.40)`
4. `[0.40, 0.60)`
5. `[0.60, 0.75)`
6. `[0.75, 0.90)`
7. `[0.90, 1.00]`

Use the same sparse-cell merge rule: a `(horizon, price_bucket)` cell with fewer than 50 validation common rows merges with its adjacent bucket toward `0.50`, with low-side buckets merging upward and high-side buckets merging downward. Compute this model-own threshold once. It may not be revised after clearance or P&L is observed.

### 11. Paired economic statistic

The paired economic comparison uses all 21 validation `close_date` days. For day `i`, define the daily total net P&L difference:

`d_i = model_net_pnl(day i) - stage0_net_pnl(day i)`

Report:

`mean(d) ± 2 * sd(d) / sqrt(21)`

Zero-trade days are included with zero daily total net P&L for the corresponding book. Also report each book's trade count, population-day and traded-day counts, total fees, and its own Day 11 clustered interval. The two books may differ in size; this is not a capital-matched comparison.

### 12. Frozen descriptive breakdowns

The following breakdowns are pre-registered:

- Horizon.
- Frozen `abs(z_5min_ewma_vol)` buckets: `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 1.0)`, and `[1.0, infinity)`.
- The frozen seven `quote_mid` price buckets.

Every cell with fewer than 30 rows is labeled `thin`. These breakdowns are descriptive only and may not become a new filter or model-selection rule.

### 13. Verdict categories

Exactly four verdict categories are frozen for M2 on validation:

**A — improves on Stage 0**

- Brier is meaningfully better at both horizons, or pooled with no per-horizon reversal.
- And `mean(d) - 2*SE(d) > 0` for the paired daily net P&L difference.

**B — better probabilities, no economic improvement**

- Brier is meaningfully better.
- The P&L-difference interval contains zero or is below zero.

**C — no meaningful difference; did not beat Stage 0**

- The Brier difference is within two paired standard errors.
- The P&L-difference interval contains zero.
- Ties go to Stage 0.

**D — worse**

- Brier is meaningfully worse at either horizon.
- Or `mean(d) + 2*SE(d) < 0` for the paired daily net P&L difference.

M1's verdict will be recorded separately later, with the caveat that folds 5–6 fit on earlier validation outcomes under the frozen Day 8 schedule.

### 14. Determinism and no iteration

Fitting must be deterministic. A second run must reproduce every coefficient to `1e-10`. No feature, transform, `C` grid, fold boundary, population, metric, or verdict rule may change after scores are seen. Only a bug that violates a written invariant may be fixed. Any such fix must later be documented in `model_notes.md` with its justification and before/after behavior.

### 15. Validation-query ledger

Append the following two Day 12 entries to the validation-query ledger; earlier entries remain unchanged:

| # | Day | Validation outcome use | Purpose | Classification / caveat |
| --- | --- | --- | --- | --- |
| 8 | 12 | Day 12 model-error term: 10-decile ECE of the Day 12 model on validation | Derive the model's own edge threshold | Decision |
| 9 | 12 | Day 12 fitted-model probability scores and Tier-2 validation P&L | Decision support for whether logistic proceeds to Day 13/14 | M1 caveat: folds 5–6 fit on earlier validation outcomes under the frozen Day 8 schedule |

These are pre-registered validation uses, not results.

### 16. Frozen prediction-table placeholders

These fields are intentionally blank until Day 12 §2.2 produces the predictions:

| Prediction artifact field | Value |
| --- | --- |
| M1 row count | 7,012 |
| M2 row count | 3,334 |
| B1 row count | 7,012 |
| Total prediction rows | 17,358 |
| Prediction SHA-256 fingerprint | `78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512` |

## Day 12 Model Protocol Amendment — Vol-of-Vol Removed Pre-Fit

During Day 12 §1.3 feature construction, before any Day 12 model fit or score, the frozen `log_vol_of_vol` completeness invariant failed against the train/validation source data. Among scored Day 9 common T-10 rows, 103 had fewer than the required 20 non-null `5min_vol` observations in the previous 24 T-10 decisions: fold 2 had 16, fold 3 had 20, fold 4 had 22, fold 5 had 13, and fold 6 had 32. The provisional eight-hour maximum-span guard also produced fold-1 nulls, but widening that guard cannot repair these 103 insufficient-history rows. No derived-feature Parquet was successfully written during that attempt; the provisional builder was removed.

This affects more than a handful of scored rows, so the Day 12 pre-fit failure rule is invoked. No imputation rule is introduced, the 20-observation minimum is not weakened, and the frozen scored population is not changed. Instead, `log_vol_of_vol` and its dependent `stage0_logit × log_vol_of_vol` interaction are removed from the Day 12 model specification. The model now has exactly nine features plus an intercept:

1. `stage0_logit`
2. `log_sigma`
3. `log_ratio_5m_1h`
4. `log_ratio_15m_4h`
5. `log_ratio_1h_24h`
6. `sin_hour`
7. `cos_hour`
8. `stage0_logit × log_ratio_5m_1h`
9. `stage0_logit × log_sigma`

The two remaining interaction terms are exactly `stage0_logit × log_ratio_5m_1h` and `stage0_logit × log_sigma`. They will be constructed inside the later model implementation, not stored by the §1.3 feature builder. Any later Day 12 instruction referring to “11 features,” “three interactions,” or `log_vol_of_vol` is superseded by this amendment.

This amendment precedes every Day 12 model coefficient and model score; it was not chosen in response to model performance. Every other Day 12 protocol choice remains frozen and unchanged: M1/M2/B1, Day 8 fold boundaries, Day 9 common and scored populations, standardization rules, the `C` grid and inner-holdout selection, probability metrics, economic comparisons, verdict categories, validation-query ledger entries, and the prohibition on reading the test split. No earlier protocol or ledger entry is edited.

### Validation-query ledger — Day 12 §3.2 additions

The pre-registered Day 12 validation uses numbered 8 and 9 have now occurred. Entry 8 used the exact §3.1 M2 validation ten-decile ECE as the model-error term for the separately labelled model-own threshold. Entry 9 scored frozen Day 12 probabilities and evaluated Tier-2 validation P&L under the fixed Stage 0 bar and the secondary model-own bar. No parameter was changed after these outcomes. The numerical results are recorded in `model_notes.md`; the final Day 12 verdict is reserved for §3.3.

| # | Day | Validation outcome use | Purpose | Classification / caveat |
| --- | --- | --- | --- | --- |
| 8 | 12 | Day 12 model-error term: 10-decile ECE of the Day 12 M2 model on validation | Derive the model's own edge threshold | Decision |
| 9 | 12 | Day 12 fitted-model probability scores and Tier-2 validation P&L | Decision support for whether logistic proceeds to Day 13/14 | M1 caveat: folds 5–6 fit on earlier validation outcomes under the frozen Day 8 schedule |

## Day 12 Model Result

The pre-registered M2 validation verdict is **C — no meaningful difference; did not beat Stage 0**. The frozen prediction fingerprint is `78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512`.

On identical validation common rows, T-10 M2 versus Stage 0 Brier was 0.189066840 versus 0.189792806, paired difference −0.000725966, paired SE 0.000889613, 2SE 0.001779227, not meaningful. T-5 was 0.116485755 versus 0.116474357, difference +0.000011398, paired SE 0.000554122, 2SE 0.001108244, not meaningful. The pooled difference −0.000361265 with paired SE 0.000526518 was not meaningful. Thus the meaningful-probability condition for A or B was absent; neither horizon was meaningfully worse for D.

The primary Tier-2 fixed-bar comparison used the frozen Stage 0 1.2 bps threshold. M2 made 728 validation trades, net P&L −$18.226600, net/trade −$0.025036538, with traded-day-clustered per-trade interval [−$0.046743937, −$0.003329140]. Frozen Stage 0 made 719 trades, net P&L −$12.733900, net/trade −$0.017710570. The pre-registered 21-day paired M2-minus-Stage-0 daily net P&L mean was −$0.261557143, SE $0.321544483, 2SE $0.643088966, interval [−$0.904646109, +$0.381531823]. Zero lies inside it, so the economic requirement for A and the wholly negative interval condition for D were absent. The numerical P&L gap does not override the frozen category.

The separately labelled **secondary** `model_own_bar` used the additive model-own basis-sensitivity plus M2 ECE threshold. It made 762 validation trades, net P&L −$19.556200, net/trade −$0.025664304; it does not determine the primary verdict. M1 validation folds 4–6 are context only: folds 5–6 fitted on earlier validation outcomes and do not replace the M2 train-only protocol-identical comparison. The deterministic read-only rerun reproduced all 14 selected C values, coefficients, scalers, and predictions exactly. No parameter changed after validation outcomes. Full results and the four fingerprints are in `model_notes.md` §3.3.

**Test-set status (2026-09-25): the test split was not read on Day 12.** Day 12 is frozen.

## Day 13 HAR-RV Protocol

This protocol is frozen at Day 13 Step 1.2, before any forward-volatility target value is built or inspected, any HAR/N1 fit, or any Day 13 score. Day 13 uses only the Day 9 common population, separately by horizon, on the frozen `close_date` train (2026-05-26 through 2026-07-19) and validation (2026-07-20 through 2026-08-09) splits. The test split must not be loaded; no forward target may be computed for a test row. The frozen train-fitted Stage 0 Platt probability using `5min_ewma_vol` is the baseline of record.

### 1. Inherited rules and sequence

Reuse unchanged `FROZEN_FOLDS` and the six expanding Day 8 folds; the Day 9 common population (9,549 train rows: 4,792 T-10 / 4,757 T-5; 3,334 validation rows: 1,685 / 1,649); `PLATT_CLIP = 1e-6`; the Day 9/12 paired per-row 2-SE Brier rule; and Day 12's B1 rows after verification of prediction fingerprint `78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512`. The frozen Day 10 threshold composition, Day 11 one-contract execution/accounting and one-position-per-market rules, and Day 12 paired economic comparison remain in force for the deferred economic comparison. No Day 7–12 script or artifact is changed.

Save this protocol before Step 1.3 builds a target. Quarantine and prove the target before model code runs. Write and fingerprint forecasts and HAR-fed probabilities before scoring either. Read coefficients before volatility scoring, and record the volatility verdict before probability scoring. Validation outcomes cannot change these choices.

### 2. Forward target and quarantine

For each row, `fwd_rv` is annualized **simple** realized volatility of one-second cross-exchange VWAP-proxy log returns labelled in `[decision_time, close_time)`: `std(ddof=1) * sqrt(31,536,000)`. The first return uses the decision-time spot from the prior second; the close-second price is excluded. Apply the existing frozen forward fill of at most 10 seconds. Coverage must be at least 80% of the expected 300 returns at T-5 or 600 at T-10; otherwise the target is null. Define `fwd_log_rv = ln(fwd_rv)` where `fwd_rv` is non-null. There is no imputation, no coverage relaxation, and no use of `fwd_log_return` or `expiration_value` to construct this target.

Every forward target value or diagnostic column is `FUTURE_ONLY`, prefixed `fwd_`, and lives only in `data/targets/forward_vol_target.parquet`. It is loaded as a separate target lookup and never enters a design matrix. A T-5 row's trailing features must never be joined onto the same market's T-10 design: the T-5 trailing five-minute window overlaps the T-10 forward target.

### 3. Volatility models and benchmarks

Fit separately per horizon with an intercept and `numpy.linalg.lstsq`. Do not standardize, regularize, weight, or apply a Jensen/lognormal correction. All logs below are natural logs of positive, annualized volatility features from the row's own horizon.

| Name | Frozen log-volatility forecast | Role |
| --- | --- | --- |
| HAR | `fwd_log_rv ~ 1 + ln 5min_vol + ln 1hr_vol + ln 24hr_vol` | Multi-horizon model |
| N1 | `fwd_log_rv ~ 1 + ln 5min_ewma_vol` | Fitted level/slope control |
| Naive-S0 | `ln 5min_ewma_vol` | Unfitted **primary benchmark**; Stage 0's actual sigma assumption |
| Naive-15 | `ln 15min_vol` | Unfitted roadmap benchmark |

OLS fit rows are fit-window Day 9 common rows with non-null `fwd_log_rv`. Forecast rows remain **every** common row in each fit and score window, including rows with a null forward target. A null target only removes its row from OLS fitting and forward-volatility scoring; it never removes the row from the probability population. No input is joined from another horizon.

### 4. Probability feed-through and configurations

For HAR and N1, set `sigma_hat = exp(log-vol forecast)` and call the existing `stage0_probability(log_moneyness, sigma_hat, T_years)` for the raw probability. Fit a separate horizon-specific Platt map for each forecast using existing `fit_platt` and `platt_feature`, then score with `stable_sigmoid(a + b * platt_feature(p_raw))`. The Platt fit uses **all** fit-window common rows and their outcomes, including rows whose own forward target is null; their log-vol forecasts still exist. The OLS forecast fed into this Platt fit is in-sample on OLS fit rows, a known asymmetry analogous to Stage 0's train-in-sample calibrator.

- **H2 train-only is primary for both verdicts:** fit HAR/N1 on train target-bearing common rows, fit their Platt maps on all train common rows, and score all 3,334 validation common rows. Compare with the frozen train-fitted Stage 0 Platt probability on exactly those rows.
- **H1 walk-forward is secondary:** use the unchanged six `FROZEN_FOLDS` and score all 7,012 out-of-fold common rows (3,531 T-10 / 3,481 T-5). Compare with the frozen Day 12 B1 walk-forward Stage 0 rows on identical keys. Folds 5–6 consume earlier validation **forward volatility and outcomes** in their expanding fits; H1 cannot replace H2's primary verdicts.

### 5. Plumbing proofs before trusting HAR-fed probabilities

Run the sigma-replacement plumbing path before writing a HAR-fed probability. Achieved maximum absolute errors remain blank until Step 2.1. The numerical tolerances below are frozen now from the Day 13 plan and Day 12 precedent; failure stops the later fit/score sequence for investigation.

| Proof | Maximum allowed error | Achieved error |
| --- | ---: | --- |
| `sigma := 5min_ewma_vol` raw feed-through versus saved `p_5min_ewma_vol` | `1e-12` | `0` |
| H2 plumbing versus frozen train-Platt Stage 0 probability | `1e-10` | `2.2204460492503131e-16` |
| H1 plumbing versus fingerprint-verified Day 12 B1 probability | `1e-8` | `1.5254186802593495e-12` |
| `numpy.linalg.lstsq` coefficients versus an independent OLS solution, on every HAR and N1 fit | `1e-10` | `0` maximum |

### 6. Forward-volatility scoring and V verdict

The **primary** scoring population is H2 validation common rows with non-null `fwd_log_rv`; the **secondary** population is H1 out-of-fold common rows with non-null target, separately identifying train-date folds 1–3 and validation folds 4–6. Primary loss is squared error in log-volatility. For each horizon and each benchmark, define the paired row loss differential `d_i = (HAR forecast_i - fwd_log_rv_i)^2 - (benchmark forecast_i - fwd_log_rv_i)^2`. Report `mean(d)` with `clustered_mean_se(d, close_date)`. A difference is meaningful iff `abs(mean(d)) > 2 * clustered SE`; a negative difference favors HAR. Report the naive per-row SE and design effect, but neither decides the verdict. Make HAR versus **Naive-S0** the primary comparison; also compare HAR versus N1 and Naive-15 on identical target-bearing keys.

Report MSE, MAE, mean residual (`fwd_log_rv - forecast`), and `R²_vs_naive = 1 - MSE_model / MSE_Naive-S0`. Also report R² against the fit-window target mean, explicitly labelled **inflated by regime shift; not a skill measure**. Do not substitute the scored-window mean for the fit-window reference.

Assign a V category **per horizon on H2 validation**, checking V4 first:

- **V4 — worse:** HAR is meaningfully worse than Naive-S0.
- **V1 — structure adds skill:** HAR is meaningfully better than Naive-S0 **and** meaningfully better than N1.
- **V2 — level/scale correction only:** HAR is meaningfully better than Naive-S0 but not meaningfully better than N1.
- **V3 — no meaningful difference:** “Stage 0's trailing-σ assumption was already adequate as a volatility forecast.”

### 7. Probability comparison and P verdict

The **primary** comparison is H2 HAR-fed Stage 0 against the frozen train-fitted Stage 0 on the exact 3,334 validation common rows. Per horizon and pooled by row count, use the inherited Day 9/12 paired per-row rule on `d_i = (p_HAR,i - y_i)^2 - (p_S0,i - y_i)^2`, with `SE = sd(d) / sqrt(n)` and meaningful iff `abs(mean(d)) > 2 * SE`. A negative difference favors HAR. Log loss, AUC, and 10-decile ECE are secondary. Report market-mid Brier and market-gap closure `(Brier_S0 - Brier_H2) / (Brier_S0 - Brier_market)` on the same rows.

Break down the paired probability scores using **Stage 0's** `abs(z_5min_ewma_vol)` buckets `[0, 0.25)`, `[0.25, 0.5)`, `[0.5, 1.0)`, `[1.0, infinity)` and the frozen seven `quote_mid` price buckets `[0, 0.10)`, `[0.10, 0.25)`, `[0.25, 0.40)`, `[0.40, 0.60)`, `[0.60, 0.75)`, `[0.75, 0.90)`, `[0.90, 1.00]`. These definitions keep row sets identical between models. Label cells with fewer than 30 rows **thin**. Score H1 versus B1 secondarily on all 7,012 rows and separately on validation folds 4–6. N1-fed H2 versus Stage 0 is diagnostic only.

Assign the **H2 probability-only** category, checking P4 first:

- **P4 — worse:** Brier is meaningfully worse at either horizon.
- **P1 — improves:** Brier is meaningfully better at both horizons.
- **P2 — mixed:** Brier is meaningfully better at exactly one horizon and not meaningfully worse at the other; a pooled result cannot upgrade P2.
- **P3 — no meaningful difference:** “Forecast σ did not improve Stage 0's probabilities; Stage 0 remains the baseline of record.”

Ties go to Stage 0. Report an H1 category separately with the folds 5–6 caveat. No economic result enters a P category.

### 8. Oracle headroom diagnostic

Feed through with `sigma := fwd_rv`; fit its Platt calibrator on train common rows with a non-null forward target, then score validation common rows with a non-null target. Re-score Stage 0 and H2 on **that exact same validation subset**. Per horizon, report both Brier gaps and `headroom captured = (Brier_S0 - Brier_H2) / (Brier_S0 - Brier_oracle)`, flagging an unstable ratio if the denominator is near zero. The oracle is diagnostic only: it is never written as a per-row probability artifact, never thresholded, never traded, and never treated as a model.

### 9. Residual analysis and frozen terciles

Use H1 out-of-fold rows with non-null target; define residual `fwd_log_rv - log_rv_har` and report the validation subset separately. Report per horizon by UTC decision-hour blocks `00–05 / 06–11 / 12–17 / 18–23`; Saturday/Sunday UTC versus weekday; trailing `5min_ewma_vol` terciles; HAR forecast deciles; and fold. Compute lag-1 residual autocorrelation within each horizon and `close_date`, in `decision_time` order, never across a date boundary. Mean-residual cells use day-clustered SEs. Forecast deciles are equal-frequency bins of the H1 HAR log-vol forecast within each horizon's target-bearing out-of-fold population, then reused unchanged for its validation subset.

The trailing-volatility tercile edges below were computed **now, before any forward target or residual exists**, using only `5min_ewma_vol` from `market_features.parquet` joined by `(ticker, horizon_minutes, close_date)` to `is_common` keys from `derived_features.parquet`. Both reads were column-projected and Parquet-filtered to **train** / `close_date <= 2026-07-19`; the resulting population is exactly 9,549 Day 9 train common rows. Edges are pandas linear-interpolation quantiles at `1/3` and `2/3`, computed separately by horizon. Freeze bins as `[0, lower)`, `[lower, upper)`, and `[upper, infinity)`; an observation exactly on an edge enters the higher bin.

| Horizon | Train common feature rows | Lower edge (`1/3`) | Upper edge (`2/3`) |
| --- | ---: | ---: | ---: |
| T-10 | 4,792 | `0.6337964396166403` | `0.8837207863551418` |
| T-5 | 4,757 | `0.623421011561805` | `0.8606540123954035` |

Call a residual cut **structure** only for a non-thin cell (`n >= 30`) with `abs(mean residual) > 2 * day-clustered SE` **and the same sign** in comparable train-date folds 1–3 and validation folds 4–6. Fold cells and lag-1 autocorrelation are descriptive; a structure flag needs a comparable cell on both sides of the boundary. No residual cut may become a new feature, filter, or model change during Day 13.

### 10. OFI decision, pre-registered

Choose one of these three categories after the probability and residual diagnostics, and record the evidence and data-availability facts:

1. **Remain deferred.** No evidence that remaining probability error is directional rather than sigma or basis, or the evidence is not robust across the train/validation boundary.
2. **Reopen as a scoped post-Day-17 experiment.** Evidence points to a directional gap, such as small oracle headroom, a persistent market gap, and low-`abs(z)` concentration. This requires a **new Kraken + Crypto.com backfill with trade side retained**; Bullish is excluded for the roadmap's coverage-discontinuity reason. The repository has no retained BTC trade-side data, so OFI does not reopen for free.
3. **Reopen within Days 14–16.** This is not realistically available: it needs a multi-day backfill and a new feature with changing coverage shortly before the one-shot test. Such a schedule change would require written justification.

HAR residuals concern volatility magnitude; they are not by themselves evidence of directional OFI. The OFI decision must also consider the oracle, persistent market gap, low-`abs(z)` probability errors, and the Day 11 market-fair P&L reference.

### 11. Economic comparison frozen for later execution

Freeze the Day 12 §10–11 comparison for H2 HAR-fed probabilities now; **do not run it in Step 1.2**. The primary bar is Stage 0's fixed **1.2 bps** `data/execution/edge_threshold.parquet` through unchanged `generic_select`, against the frozen 719-trade Stage 0 validation book. The secondary model-own bar uses `required_net_edge = basis_term + model_error_term` under the frozen Day 10 composition and seven price buckets: the basis term is the model's own ±1.2 bps spot-perturbation sensitivity, and the model-error term is H2 validation 10-decile ECE. Perturbing basis changes `log_moneyness` only; forecast sigma is spot-independent and must show exactly zero spot sensitivity. Apply the inherited executable prices, fee, stale-quote, one-contract, earliest-clearing-row, settlement, and accounting rules without rewriting them.

The statistic is HAR-fed minus frozen Stage 0 **daily total net P&L** over all 21 validation `close_date` days, including zero-trade days, reported as the mean paired daily difference ± `2 * sd(daily differences) / sqrt(21)`. Report the inherited book counts, fees, and Day 11 clustered intervals as context. Execution is Day 14's four-way comparison unless optional O1 is later run. **Do not add validation ledger entry 13 unless this comparison actually runs.**

### 12. Determinism, no iteration, and validation-query ledger

A second run must reproduce every coefficient to `1e-12` and both forecast/probability fingerprints exactly. After any score is seen, no input window, transform, target definition, coverage rule, benchmark, population, metric, or verdict category/rule may change. Only a bug violating an already-written invariant may be fixed; document its justification and before/after behavior later in `model_notes.md`. Do not change a negative HAR coefficient by dropping a horizon.

Declare the following Day 13 validation uses now; entries 1–9 above remain untouched. H1 entries 10 and 11 carry the caveat that folds 5–6 consume validation forward volatility **and** outcomes.

| # | Day | Validation outcome use | Purpose | Classification / caveat |
| --- | --- | --- | --- | --- |
| 10 | 13 | HAR, N1, and naive forward-volatility losses against validation forward realized volatility | Decision support — Day 14 gate input | H2 primary; H1 folds 5–6 consume validation forward volatility and outcomes |
| 11 | 13 | HAR-fed and N1-fed Stage 0 probability scores on validation | Decision support — whether HAR-fed Stage 0 is a candidate in Day 14's comparison | H2 primary; H1 folds 5–6 consume validation forward volatility and outcomes |
| 12 | 13 | Oracle headroom and residual analysis on validation | Diagnostic — no parameter chosen | Oracle never becomes a model or per-row artifact |

### 13. Unresolved values reserved for Steps 1.3 and 2.1

These are placeholders for later **measurements**, not open protocol choices. Step 1.3 fills target counts from the builder's authoritative output; Step 2.1 fills fit counts, proof errors above, and fingerprints before any score. Planning-time forward-window observation counts do not replace builder-verified counts.

| Measurement | T-10 | T-5 | Fill at |
| --- | --- | --- | --- |
| Train common null `fwd_rv` / `fwd_log_rv` rows | **27** | **62** | Step 1.3 |
| Validation common null `fwd_rv` / `fwd_log_rv` rows | **41** | **61** | Step 1.3 |
| H2 HAR OLS fit rows after null-target exclusion | **4,765** | **4,695** | Step 2.1 |
| H1 HAR OLS fit rows after null-target exclusion, each fold | F1 **2,936**; F2 **3,573**; F3 **4,184**; F4 **4,765**; F5 **5,310**; F6 **5,891** | F1 **2,903**; F2 **3,529**; F3 **4,125**; F4 **4,695**; F5 **5,226**; F6 **5,797** | Step 2.1 |

| Fingerprint | Value | Fill at |
| --- | --- | --- |
| Forecast table SHA-256 | `d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee07ee0b31aa472eeaa50acc9` | Step 2.1 |
| HAR-fed probability table SHA-256 | `e864a6377c8d520b04eae3dfb6664feaacdb5c35cb9708a6c0755b741d17fa0b` | Step 2.1 |

**Step 1.2 status:** no forward target has been built or inspected, no HAR/N1 model has been fitted, no validation metric or probability score has been computed, and no test-period row has been read. Step 1.3 has not started.

## Day 13 HAR-RV Result

This result section records the completed, pre-registered validation uses. The Day 13 protocol and prior ledger entries remain unchanged.

| # | Day | Validation outcome use | Purpose | Classification / caveat |
| --- | --- | --- | --- | --- |
| 10 | 13 | HAR, N1, and naive forward-volatility losses against validation forward realized volatility | Decision support — Day 14 gate input | H2 primary; H1 folds 5–6 consume validation forward volatility and outcomes |
| 11 | 13 | HAR-fed and N1-fed Stage 0 probability scores on validation | Decision support — whether HAR-fed Stage 0 is a candidate in Day 14's comparison | H2 primary; H1 folds 5–6 consume validation forward volatility and outcomes |
| 12 | 13 | Oracle headroom and residual analysis on validation | Diagnostic — no parameter chosen | Oracle never becomes a model or per-row artifact |

Frozen identities: forecast fingerprint `d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee07ee0b31aa472eeaa50acc9`; HAR-fed probability fingerprint `e864a6377c8d520b04eae3dfb6664feaacdb5c35cb9708a6c0755b741d17fa0b`.

| Primary H2 validation target-bearing population | Horizon | n | HAR MSE | Naive-S0 MSE | N1 MSE | HAR−Naive-S0 paired loss; day-clustered SE | HAR−N1 paired loss; day-clustered SE | Official V verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| H2 train-only | T-10 | 1,644 | 0.058826112 | 0.089998530 | 0.091657356 | −0.031172418; 0.004112423 | −0.032831244; 0.005616921 | **V1 — structure adds skill** |
| H2 train-only | T-5 | 1,588 | 0.075463723 | 0.099324391 | 0.096687468 | −0.023860668; 0.004099138 | −0.021223745; 0.004020634 | **V1 — structure adds skill** |

Both V1 verdicts use the frozen close-date-clustered two-SE rule: HAR meaningfully beat Naive-S0 and N1 at each horizon. They concern forward-volatility skill only.

| Primary H2 validation common probability population | Horizon | n | HAR Brier | Frozen Stage 0 Brier | HAR−Stage 0 paired difference; per-row SE | Meaningful under inherited two-SE rule |
| --- | --- | ---: | ---: | ---: | --- | --- |
| H2 train-only | T-10 | 1,685 | 0.189033084 | 0.189792806 | −0.000759722; 0.000603548 | No |
| H2 train-only | T-5 | 1,649 | 0.117061648 | 0.116474357 | +0.000587291; 0.000476165 | No |

**H2 P3 — no meaningful difference:** “Forecast σ did not improve Stage 0's probabilities; Stage 0 remains the baseline of record.” The pooled H2 validation common result (`n=3,334`) was likewise not meaningful; the pooled result and secondary metrics cannot upgrade P3. The volatility and probability verdict rules remain separate.

**Oracle diagnostic only — not a model:** on the target-bearing H2 validation subsets, T-10 (`n=1,644`) had raw Stage 0-minus-oracle Brier gap `0.002185453` and HAR headroom captured `+0.344385`; T-5 (`n=1,588`) had raw gap `0.000443444` and captured fraction `−1.098620`. The small T-5 denominator makes that ratio sensitive; the raw gap is the primary context. No per-row oracle probability artifact exists.

**OFI verdict: Reopen as a scoped post-Day-17 experiment.** It requires a new Kraken + Crypto.com trade-side backfill, with Bullish excluded, and does not modify Days 14–17 or validate OFI.

The HAR-fed economic-comparison specification was frozen on Day 13; execution was deferred to Day 14. No optional O1 economic comparison ran and **no validation ledger entry 13** is added. Detailed target, proof, coefficient, forecast, probability, residual, and OFI readings are in `# Day 13 — HAR-RV Forward-Volatility Notes` in `model_notes.md`.

Test-set status (Day 13): the test split was not read on Day 13.
