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
