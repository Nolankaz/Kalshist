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
