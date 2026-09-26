# Day 12 Model Notes

## Day 12 §2.3 — Coefficients Against Pre-Registered Hypotheses

### Integrity and interpretation basis

This reading uses the frozen `data/models/logistic_coefficients.parquet` and selected-C fields from `data/models/logistic_fold_selection.parquet`. The §2.2 prediction fingerprint was already frozen as `78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512`. No prediction or model-performance result was read; this interpretation precedes §3.1. No model was fitted or changed here.

The coefficient artifact has 138 rows: M1 has six folds × two horizons × nine features (108); M2 has one fold × two horizons × nine features (18); B1 has six folds × two horizons × one Stage 0 feature (12). Keys are unique, horizons are T-10 and T-5 only, the amended nine-feature order is exact in every M1/M2 fit, coefficients and fit means/SDs are finite, SDs are positive, and selected C is constant within each fit. The C-selection artifact has 98 rows, one selected C in each of 14 M1/M2 fit groups, matching the coefficient records. No test information was loaded. B1 is a one-feature reference, not part of the nine-feature analysis.

“Sign agreement” below counts positive, negative, and exact-zero M1 coefficients across the six folds, separately by horizon. The observed sign is descriptive; a coefficient is conditional on the other terms. Standardized coefficients support cross-feature magnitude comparisons. Original-unit coefficients support comparison with the historical Platt slope scale. Neither coefficient scale measures predictive performance or causal importance.

### Coefficients against the frozen hypotheses

| Feature | Pre-registered hypothesis | Predicted sign | T-10 M1 sign agreement | T-5 M1 sign agreement | M2 sign T-10 / T-5 | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| `stage0_logit` | Baseline carrier, strongly positive; original slope near historical Platt scale. | Positive | 6/6 positive | 6/6 positive | + / + | Baseline signal remains positive and on the same broad original-unit scale at both horizons. |
| `log_sigma` | Absolute volatility-level shift, expected relatively small. | None | 6/6 negative | 4/6 negative, 2/6 positive | − / − | Small beside the carrier; T-5 sign is unstable. No sign was predicted. |
| `log_ratio_5m_1h` | Short-window versus one-hour ratio, main effect expected near zero. | Near zero | 6/6 positive | 6/6 positive | + / + | Consistently positive and larger than the other ratio main effects at T-10, though below the carrier; this differs from the near-zero expectation. |
| `log_ratio_15m_4h` | Medium-window versus four-hour ratio, main effect expected near zero. | Near zero | 6/6 negative | 6/6 positive | − / + | Small within each horizon but reverses sign between horizons; no directional story follows. |
| `log_ratio_1h_24h` | Session versus day ratio, main effect expected near zero. | Near zero | 5/6 positive, 1/6 negative | 4/6 positive, 2/6 negative | + / + | Small and sign-unstable across folds. |
| `sin_hour` | Small cyclical intraday term. | None | 6/6 positive | 6/6 positive | + / + | Consistently positive but small; no sign was predicted. |
| `cos_hour` | Complementary small cyclical intraday term. | None | 6/6 negative | 3/6 positive, 3/6 negative | − / + | Very small, with a mixed T-5 sign; no sign was predicted. |
| `stage0_logit_x_log_ratio_5m_1h` | Ratio-dependent Stage 0 sharpness. | Positive | 6/6 positive | 6/6 positive | + / + | Agrees with the positive interaction hypothesis at both horizons. |
| `stage0_logit_x_log_sigma` | Volatility-level-dependent Stage 0 sharpness. | None | 5/6 negative, 1/6 positive | 5/6 negative, 1/6 positive | − / − | Mostly negative, but fold 6 reverses at both horizons; no sign was predicted. |

There are no exactly zero M1 coefficients. “Near zero” in the hypotheses is a magnitude expectation, not a predicted positive or negative sign.

### M1 descriptive summaries

Each row summarizes six fold coefficients. Columns marked `std` use standardized feature units; columns marked `orig` use the original feature units. These are descriptions of frozen fits, not performance measures.

| Horizon | Feature | Positive / negative / zero | Dominant sign | Mean std | Median std | Min std | Max std | Mean orig | Median orig |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T-10 | `stage0_logit` | 6 / 0 / 0 | Positive | +1.2785 | +1.3239 | +0.9926 | +1.3705 | +2.0024 | +2.0729 |
| T-10 | `log_sigma` | 0 / 6 / 0 | Negative | −0.0690 | −0.0774 | −0.1112 | −0.0107 | −0.1767 | −0.2008 |
| T-10 | `log_ratio_5m_1h` | 6 / 0 / 0 | Positive | +0.1924 | +0.1997 | +0.1384 | +0.2112 | +0.9333 | +0.9789 |
| T-10 | `log_ratio_15m_4h` | 0 / 6 / 0 | Negative | −0.0500 | −0.0440 | −0.0933 | −0.0232 | −0.2649 | −0.2382 |
| T-10 | `log_ratio_1h_24h` | 5 / 1 / 0 | Positive | +0.0443 | +0.0551 | −0.0358 | +0.0846 | +0.2259 | +0.2972 |
| T-10 | `sin_hour` | 6 / 0 / 0 | Positive | +0.0214 | +0.0242 | +0.0050 | +0.0322 | +0.0305 | +0.0346 |
| T-10 | `cos_hour` | 0 / 6 / 0 | Negative | −0.0170 | −0.0176 | −0.0361 | −0.0005 | −0.0238 | −0.0247 |
| T-10 | `stage0_logit_x_log_ratio_5m_1h` | 6 / 0 / 0 | Positive | +0.1939 | +0.1789 | +0.1678 | +0.2343 | +1.2318 | +1.1833 |
| T-10 | `stage0_logit_x_log_sigma` | 1 / 5 / 0 | Negative | −0.0533 | −0.0559 | −0.1453 | +0.0529 | −0.2021 | −0.1962 |
| T-5 | `stage0_logit` | 6 / 0 / 0 | Positive | +3.0294 | +3.1620 | +2.3753 | +3.5247 | +2.2439 | +2.3556 |
| T-5 | `log_sigma` | 2 / 4 / 0 | Negative | −0.0447 | −0.0575 | −0.1264 | +0.0601 | −0.1004 | −0.1529 |
| T-5 | `log_ratio_5m_1h` | 6 / 0 / 0 | Positive | +0.1245 | +0.1342 | +0.0714 | +0.1675 | +0.5989 | +0.6386 |
| T-5 | `log_ratio_15m_4h` | 6 / 0 / 0 | Positive | +0.0330 | +0.0309 | +0.0141 | +0.0507 | +0.1792 | +0.1672 |
| T-5 | `log_ratio_1h_24h` | 4 / 2 / 0 | Positive | +0.0294 | +0.0505 | −0.0665 | +0.0800 | +0.1426 | +0.2650 |
| T-5 | `sin_hour` | 6 / 0 / 0 | Positive | +0.0218 | +0.0225 | +0.0112 | +0.0283 | +0.0313 | +0.0322 |
| T-5 | `cos_hour` | 3 / 3 / 0 | Mixed | −0.0053 | −0.0013 | −0.0294 | +0.0136 | −0.0075 | −0.0018 |
| T-5 | `stage0_logit_x_log_ratio_5m_1h` | 6 / 0 / 0 | Positive | +0.4289 | +0.4339 | +0.2968 | +0.6492 | +1.3032 | +1.2193 |
| T-5 | `stage0_logit_x_log_sigma` | 1 / 5 / 0 | Negative | −0.3049 | −0.3881 | −0.5104 | +0.0709 | −0.4685 | −0.5165 |

For each M1 fold, the following ratio is `abs(feature standardized coefficient) / abs(stage0_logit standardized coefficient)`; the table reports its six-fold mean. This compares coefficient scales within a fit and does not assign feature importance.

| Feature | T-10 mean relative magnitude | T-5 mean relative magnitude |
| --- | ---: | ---: |
| `log_sigma` | 5.3% | 2.5% |
| `log_ratio_5m_1h` | 15.2% | 4.3% |
| `log_ratio_15m_4h` | 3.9% | 1.1% |
| `log_ratio_1h_24h` | 4.3% | 1.9% |
| `sin_hour` | 1.6% | 0.7% |
| `cos_hour` | 1.3% | 0.5% |
| `stage0_logit_x_log_ratio_5m_1h` | 15.3% | 14.0% |
| `stage0_logit_x_log_sigma` | 5.9% | 11.7% |

### Stage 0 carrier and interaction detail

The M1 `stage0_logit` original-unit slopes are all positive. T-10 is near the historical Platt anchor of about 2.16 except fold 3 at 1.57; T-5 is near the anchor of about 2.50 except folds 3 and 5 at about 1.78–1.79. M2 is 2.0610 at T-10 and 2.1791 at T-5. No slope is far below 1 or far above 3. The interaction terms can absorb part of the carrier's effective slope, so the carrier coefficient alone is not a full effective-slope calculation.

| Horizon | Feature, original units | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | M2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-10 | `stage0_logit` | +2.0610 | +2.1131 | +1.5660 | +2.0610 | +2.0848 | +2.1286 | +2.0610 |
| T-5 | `stage0_logit` | +2.6069 | +2.5724 | +1.7812 | +2.1791 | +1.7917 | +2.5321 | +2.1791 |
| T-10 | `stage0_logit_x_log_ratio_5m_1h` | +1.2065 | +1.1601 | +1.1187 | +1.4100 | +1.3917 | +1.1039 | +1.4100 |
| T-5 | `stage0_logit_x_log_ratio_5m_1h` | +2.1476 | +1.5062 | +0.9286 | +1.3453 | +0.7980 | +1.0934 | +1.3453 |
| T-10 | `stage0_logit_x_log_sigma` | −0.3345 | −0.2100 | −0.5214 | −0.1825 | −0.1027 | +0.1385 | −0.1825 |
| T-5 | `stage0_logit_x_log_sigma` | −0.8365 | −0.2770 | −0.7428 | −0.4557 | −0.5772 | +0.0784 | −0.4557 |

The ratio interaction is positive in all six folds at both horizons and positive in M2. This agrees with the preregistered sign: as the short-window to one-hour volatility ratio rises, it increases the fitted sharpness of the Stage 0 logit, pushing positive logits more positive and negative logits more negative. It is not a standalone YES/NO effect. Its M1 mean standardized coefficient is +0.1939 at T-10 and +0.4289 at T-5, versus carrier means of +1.2785 and +3.0294.

The volatility-level interaction is negative in folds 1–5 and positive in fold 6 at both horizons; M2 is negative. With the other terms fixed, a negative coefficient means a higher volatility level tends to weaken the Stage 0 logit toward zero, while a positive coefficient tends to strengthen it. Because fold 6 reverses, the direction is not fully stable. No sign was predicted. Its M1 mean standardized coefficient is −0.0533 at T-10 and −0.3049 at T-5; the larger T-5 magnitude still remains below the carrier in every fold.

### Cross-feature magnitude and stability

`stage0_logit` has the largest absolute standardized coefficient in **every** M1 fold/horizon and both M2 horizons. No other feature exceeds it in any fit. The table gives that largest coefficient for each fit; all entries are the carrier.

| Fit | T-10 largest standardized coefficient | T-5 largest standardized coefficient |
| --- | ---: | ---: |
| M1 fold 1 | +1.3165 | +3.4494 |
| M1 fold 2 | +1.3314 | +3.3839 |
| M1 fold 3 | +0.9926 | +2.3753 |
| M1 fold 4 | +1.3115 | +2.9402 |
| M1 fold 5 | +1.3485 | +2.5028 |
| M1 fold 6 | +1.3705 | +3.5247 |
| M2 | +1.3115 | +2.9402 |

Across M1 folds, `stage0_logit`, the ratio interaction, `log_ratio_5m_1h`, and `sin_hour` are positive at both horizons. `log_ratio_15m_4h` is stable within each horizon but changes sign between T-10 and T-5. `log_sigma` at T-5, `log_ratio_1h_24h` at both horizons, `cos_hour` at T-5, and the volatility-level interaction at both horizons have sign flips. The carrier dominates standardized magnitude; the ratio main effect and ratio interaction are the most notable departures from the near-zero/small hypotheses at T-10, while the volatility-level interaction has a wider T-5 range (−0.5104 to +0.0709). The ratio main effect and ratio interaction are both positive, but their joint interpretation can be affected by collinearity. The volatility-level main effect and interaction also share an input, with T-5 sign changes. These patterns warrant caution, not a causal or performance claim.

### M1 versus M2 and frozen C context

The table compares standardized coefficients. M1 fold 4 and M2 have the same train-only fit window and selected C at each horizon, so their coefficients agree to the displayed precision. Folds 5–6 use expanding fit windows and later information; changes relative to M2 are a stability diagnostic.

| Horizon | Feature | M1 fold 4 | M1 fold 5 | M1 fold 6 | M2 | Descriptive comparison |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T-10 | `stage0_logit` | +1.3115 | +1.3485 | +1.3705 | +1.3115 | Broadly consistent. |
| T-10 | `log_sigma` | −0.1023 | −0.1112 | −0.0813 | −0.1023 | Broadly consistent. |
| T-10 | `log_ratio_5m_1h` | +0.2102 | +0.2112 | +0.2033 | +0.2102 | Broadly consistent. |
| T-10 | `log_ratio_15m_4h` | −0.0472 | −0.0630 | −0.0933 | −0.0472 | Same sign, changing magnitude. |
| T-10 | `log_ratio_1h_24h` | +0.0710 | +0.0831 | +0.0846 | +0.0710 | Broadly consistent. |
| T-10 | `sin_hour` | +0.0132 | +0.0197 | +0.0286 | +0.0132 | Same sign, changing magnitude. |
| T-10 | `cos_hour` | −0.0192 | −0.0271 | −0.0361 | −0.0192 | Same sign, changing magnitude. |
| T-10 | `stage0_logit_x_log_ratio_5m_1h` | +0.2313 | +0.2343 | +0.1841 | +0.2313 | Broadly consistent sign; fold 6 is smaller. |
| T-10 | `stage0_logit_x_log_sigma` | −0.0581 | −0.0380 | +0.0529 | −0.0581 | Sign disagreement in fold 6. |
| T-5 | `stage0_logit` | +2.9402 | +2.5028 | +3.5247 | +2.9402 | Same sign, changing magnitude. |
| T-5 | `log_sigma` | −0.0803 | −0.1264 | −0.1124 | −0.0803 | Same sign, changing magnitude. |
| T-5 | `log_ratio_5m_1h` | +0.1675 | +0.1559 | +0.1310 | +0.1675 | Broadly consistent. |
| T-5 | `log_ratio_15m_4h` | +0.0141 | +0.0480 | +0.0340 | +0.0141 | Same sign, changing magnitude. |
| T-5 | `log_ratio_1h_24h` | +0.0800 | +0.0712 | +0.0745 | +0.0800 | Broadly consistent. |
| T-5 | `sin_hour` | +0.0221 | +0.0275 | +0.0283 | +0.0221 | Broadly consistent. |
| T-5 | `cos_hour` | +0.0095 | +0.0136 | −0.0116 | +0.0095 | Sign disagreement in fold 6. |
| T-5 | `stage0_logit_x_log_ratio_5m_1h` | +0.4625 | +0.2969 | +0.4090 | +0.4625 | Same sign, changing magnitude. |
| T-5 | `stage0_logit_x_log_sigma` | −0.3239 | −0.5104 | +0.0709 | −0.3239 | Sign disagreement in fold 6. |

| Fold | M1 selected C, T-10 | M1 selected C, T-5 |
| --- | ---: | ---: |
| 1 | 10 | 10 |
| 2 | 10 | 10 |
| 3 | 0.01 | 0.03 |
| 4 | 10 | 0.1 |
| 5 | 10 | 0.03 |
| 6 | 10 | 10 |

M2 selected C is 10 at T-10 and 0.1 at T-5. Smaller C means stronger shrinkage. Fold 3 and some T-5 fits therefore have a different shrinkage context, which limits direct coefficient comparisons across folds. These are the frozen §2.2 selections; none were changed here.

## Day 12 §3.1 — Probability Comparison Against Stage 0

### Frozen inputs and comparison populations

The §2.2 prediction fingerprint `78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512` matched before scoring and again afterward. The official baseline is the frozen legitimate train-fitted Stage 0 Platt probability carried on each prediction row. Every model and Stage 0 calculation uses identical `(ticker, horizon_minutes)` keys and the same targets. M2 scores all 3,334 validation common rows (T-10: 1,685; T-5: 1,649). M1 validation uses folds 4–6 on those same keys; M1 train secondary uses folds 1–3 on 3,678 rows; all-OOF descriptive uses 7,012 rows. B1 is a walk-forward Stage 0 reference, not the official baseline.

The frozen Stage 0 validation Brier values in `stage0_platt_validation_scores.parquet` were reproduced on the exact common rows within `1e-12`. Differences below are model minus Stage 0: negative favors the model on that row set. Paired SE is the sample SD of rowwise squared-error differences divided by `sqrt(n)`. The pre-registered meaningful flag requires `abs(difference) > 2 × paired SE`. These probability results do not assign the final Day 12 verdict.

### Primary M2 validation comparison

| Horizon | n | M2 Brier | Stage 0 Brier | Difference | Paired SE | 2 SE | Meaningful | M2 ECE | Stage 0 ECE | Market Brier | M2 log loss | Stage 0 log loss | M2 AUC | Stage 0 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-10 | 1,685 | 0.189066840 | 0.189792806 | −0.000725966 | 0.000889613 | 0.001779227 | No | 0.037055016 | 0.035595749 | 0.182157128 | 0.563109370 | 0.564414415 | 0.786337115 | 0.784119142 |
| T-5 | 1,649 | 0.116485755 | 0.116474357 | +0.000011398 | 0.000554122 | 0.001108244 | No | 0.018318881 | 0.021091839 | 0.108801868 | 0.378247281 | 0.378477904 | 0.914274783 | 0.914429288 |

The pooled row-count-weighted validation Brier is 0.153168157 for M2 and 0.153529422 for Stage 0, a paired difference of −0.000361265 with SE 0.000526518; the two-SE rule is not met. T-10 M2 Brier is lower while its ECE is higher, so those facts must be kept separate. T-5 Brier is slightly higher while ECE is lower. AUC is a ranking diagnostic, listed after the probability-pricing and market context above. Market mid has lower Brier than both probability columns on these validation rows, but it is only a benchmark and was never a model feature. No paired significance comparison with market was made.

ECE uses the repository's ten probability-quantile deciles and the row-count-weighted absolute gap between observed and mean predicted probability. Log loss uses the frozen `1e-15` clip. The [reliability plot](data/models/plots/logistic_vs_stage0_reliability.png) shows M2, Stage 0, and market mid separately at T-10 and T-5, with Wilson intervals and the perfect-calibration diagonal. It is descriptive, not a visual significance test.

### Frozen `abs(z_5min_ewma_vol)` breakdown — M2 validation

| Horizon | Frozen bucket | n | M2 Brier | Stage 0 Brier | Paired difference | Thin |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T-10 | `[0, 0.25)` | 879 | 0.231249703 | 0.231821412 | −0.000571709 | No |
| T-10 | `[0.25, 0.5)` | 469 | 0.172102290 | 0.172934982 | −0.000832692 | No |
| T-10 | `[0.5, 1.0)` | 310 | 0.102171488 | 0.103222139 | −0.001050651 | No |
| T-10 | `[1.0, infinity)` | 27 | 0.108148167 | 0.108314353 | −0.000166186 | **Yes** |
| T-5 | `[0, 0.25)` | 432 | 0.227357481 | 0.228477742 | −0.001120261 | No |
| T-5 | `[0.25, 0.5)` | 408 | 0.131690422 | 0.131369132 | +0.000321290 | No |
| T-5 | `[0.5, 1.0)` | 494 | 0.063421226 | 0.062512423 | +0.000908804 | No |
| T-5 | `[1.0, infinity)` | 315 | 0.027958095 | 0.028203451 | −0.000245356 | No |

The pre-registered low-|z| bucket `[0, 0.25)` has a negative M2-minus-Stage-0 Brier difference at both horizons. Its cells contain 879 and 432 rows. The T-10 highest-|z| cell has only 27 rows and is marked thin; it is not interpreted. Bucket differences are descriptive and do not create a new filter.

### Frozen quote-mid price buckets — M2 validation

| Horizon | Frozen bucket | n | M2 Brier | Stage 0 Brier | Paired difference |
| --- | --- | ---: | ---: | ---: | ---: |
| T-10 | `p00_10` | 54 | 0.107283594 | 0.108387601 | −0.001104006 |
| T-10 | `p10_25` | 279 | 0.123905055 | 0.123017217 | +0.000887838 |
| T-10 | `p25_40` | 317 | 0.223136036 | 0.223512402 | −0.000376367 |
| T-10 | `p40_60` | 442 | 0.253397874 | 0.251682226 | +0.001715648 |
| T-10 | `p60_75` | 275 | 0.208661875 | 0.211800934 | −0.003139059 |
| T-10 | `p75_90` | 240 | 0.131987305 | 0.135703643 | −0.003716339 |
| T-10 | `p90_100` | 78 | 0.082305747 | 0.086090223 | −0.003784477 |
| T-5 | `p00_10` | 404 | 0.033948306 | 0.033423316 | +0.000524990 |
| T-5 | `p10_25` | 217 | 0.147841301 | 0.146693318 | +0.001147983 |
| T-5 | `p25_40` | 147 | 0.206920503 | 0.204911280 | +0.002009223 |
| T-5 | `p40_60` | 165 | 0.249506481 | 0.249997056 | −0.000490574 |
| T-5 | `p60_75` | 147 | 0.219977026 | 0.217785954 | +0.002191072 |
| T-5 | `p75_90` | 183 | 0.144376900 | 0.145548289 | −0.001171389 |
| T-5 | `p90_100` | 386 | 0.041307908 | 0.043288601 | −0.001980693 |

All M2 validation price cells have at least 30 rows. T-10 differences are negative in the three upper price buckets and positive in `p40_60`; T-5 signs vary across buckets. The per-bucket differences are descriptive, not new model-selection rules. For every comparison window and horizon, the frozen |z| and price-bucket counts sum to their parent population; every cell with fewer than 30 rows is marked thin in the artifact.

### M1 windows and B1 reference

| Configuration and window | Horizon | n | Model/reference Brier | Frozen Stage 0 Brier | Paired difference | Paired SE | Meaningful |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M1 validation, folds 4–6 | T-10 | 1,685 | 0.189043471 | 0.189792806 | −0.000749335 | 0.000860063 | No |
| M1 validation, folds 4–6 | T-5 | 1,649 | 0.117243515 | 0.116474357 | +0.000769159 | 0.000611496 | No |
| M1 train secondary, folds 1–3 | T-10 | 1,846 | 0.191773513 | 0.191693384 | +0.000080129 | 0.000743617 | No |
| M1 train secondary, folds 1–3 | T-5 | 1,832 | 0.127315090 | 0.127046751 | +0.000268339 | 0.000665600 | No |
| M1 all-OOF descriptive | T-10 | 3,531 | 0.190470732 | 0.190786424 | −0.000315693 | 0.000565279 | No |
| M1 all-OOF descriptive | T-5 | 3,481 | 0.122544039 | 0.122038455 | +0.000505584 | 0.000454508 | No |
| B1 validation reference | T-10 | 1,685 | 0.189889425 | 0.189792806 | +0.000096619 | 0.000064117 | No |
| B1 validation reference | T-5 | 1,649 | 0.116506419 | 0.116474357 | +0.000032062 | 0.000062917 | No |
| B1 train reference | T-10 | 1,846 | 0.192128196 | 0.191693384 | +0.000434812 | 0.000197425 | Yes |
| B1 train reference | T-5 | 1,832 | 0.127501207 | 0.127046751 | +0.000454456 | 0.000242852 | No |
| B1 all-OOF reference | T-10 | 3,531 | 0.191059850 | 0.190786424 | +0.000273426 | 0.000107676 | Yes |
| B1 all-OOF reference | T-5 | 3,481 | 0.122292817 | 0.122038455 | +0.000254362 | 0.000131270 | No |

M1 folds 1–3 are secondary because the frozen Stage 0 Platt calibrator is **in-sample** on those train dates. M1 all-OOF combines train and validation dates and is descriptive only; it must not be compared with a validation-only Stage 0 aggregate. M1 folds 5–6 also fit on earlier validation outcomes under the frozen walk-forward schedule. B1 demonstrates the difference between walk-forward Stage 0 fitting and the official train-fitted Stage 0 baseline; it does not replace that baseline. The B1 train T-10 and all-OOF T-10 flags belong only to those reference populations.

The [comparison table](data/models/model_comparison.parquet) stores overall, pooled Brier, frozen |z|, and frozen price-bucket results with paired SE and thin-cell flags where defined. It passed an exact Parquet round trip. No model was refitted or recalibrated, no C or frozen prediction changed, and no test row was read. The next economic comparison has not begun.

## Day 12 §3.2 — Economic Comparison Against Stage 0

### Integrity and fixed trade rules

The frozen Day 12 prediction fingerprint `78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512` matched before trade selection and after settlement. The generic selector reproduced Day 11's Stage 0 fingerprint `f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96` and its four counts exactly: train/validation at 1.2 bps, 1,881/719; train/validation at 5.0 bps, 277/100. The one-position rule retained the earliest clearing decision per market. The accounting self-check and one-to-one validation settlement joins passed. The frozen Stage 0 validation book reproduced 719 trades, net P&L −$12.7339, and mean net P&L −$0.017710570 per trade against the saved Day 11 summary.

The primary comparison uses the **same frozen Stage 0 required-net-edge threshold at 1.2 bps** for both probability sources. M2 uses 3,334 validation common predictions before selection (T-10: 1,685; T-5: 1,649). M1 uses only its validation folds 4–6 and is context, since folds 5–6 fit on earlier validation outcomes. The model trade selection read no settlement or outcome field; only validation settlements were loaded later. These are Tier 2, quote-aware, top-of-book, size-unaware, one-contract taker books held to settlement.

### Primary fixed-bar validation books

| Book | Trades | Traded/population days | Hit rate | Break-even hit rate | Gross P&L | Fees | Net P&L | Net P&L/trade | Mean net edge | Market-fair expected P&L | Mean surprise | Clustered SE/trade | Clustered ±2 SE interval/trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M2, Stage 0 fixed bar | 728 | 21/21 | 0.445054945 | 0.470091484 | −$10.151000 | $8.075600 | −$18.226600 | −$0.025036538 | +0.112568396 | −$11.008100 | −0.137604935 | 0.010853699 | [−0.046743937, −0.003329140] |
| Frozen Stage 0, Stage 0 fixed bar | 719 | 21/21 | 0.432545202 | 0.450255772 | −$5.280000 | $7.453900 | −$12.733900 | −$0.017710570 | +0.114468673 | −$10.219900 | −0.132179243 | 0.012277195 | [−0.042264960, +0.006843819] |

The M2 book has nine more trades and a more negative mean surprise (−0.137604935 versus −0.132179243). Its net P&L is $5.492700 lower on these different-sized books. Trade count by itself is not a quality measure. Both market-fair expected totals are negative. Fees are displayed separately from gross and net P&L.

Across **all 21 validation calendar days**, including zero-trade days, the mean daily M2-minus-Stage-0 net P&L difference is −$0.261557143; sample SD is 1.473501933, SE is 0.321544483, and 2 SE is 0.643088966. The paired interval is [−$0.904646109, +$0.381531823]. It crosses zero. This daily strategy-difference SE is separate from either book's traded-day clustered per-trade SE.

M1 validation folds 4–6 produced 756 fixed-bar trades across 21 days: gross −$6.638000, fees $7.996200, net −$14.634200, net/trade −$0.019357407, mean net edge +0.111362734, market-fair expected −$10.939700, mean surprise −0.130720142, clustered SE 0.011121894, interval [−0.041601195, +0.002886380]. This is context only and does not replace the primary M2 comparison.

### Model-own threshold — separate secondary book

The model-error terms are the exact stored §3.1 M2 validation ten-decile ECE values, not newly binned estimates: T-10 `0.037055016` and T-5 `0.018318881` when displayed to nine places. The basis calculation used the frozen ±1.2 bps `log1p(±1.2/10,000)` spot convention. It changed only `stage0_logit` and the two surviving Stage 0 interactions; all six spot-independent raw features had **exactly zero** shift. Applying the stored M2 train-fit scaler and coefficients reconstructed the unperturbed frozen probabilities with maximum error `1.110e-16`. No scaler or model was refitted. Each basis term below is the median of the rowwise maximum absolute plus/minus probability shift. All frozen price buckets had at least 50 rows, so the Day 10 sparse-cell merge rule did not fire. The required edge is the additive `basis term + M2 validation ECE`.

| Horizon | Frozen price bucket | n | M2 ECE | Basis term | Required net edge | Sparse merge |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| T-10 | `p00_10` | 54 | 0.037055016 | 0.010662110 | 0.047717126 | No |
| T-10 | `p10_25` | 279 | 0.037055016 | 0.032039445 | 0.069094461 | No |
| T-10 | `p25_40` | 317 | 0.037055016 | 0.048737106 | 0.085792122 | No |
| T-10 | `p40_60` | 442 | 0.037055016 | 0.051957267 | 0.089012283 | No |
| T-10 | `p60_75` | 275 | 0.037055016 | 0.044303514 | 0.081358530 | No |
| T-10 | `p75_90` | 240 | 0.037055016 | 0.026211394 | 0.063266410 | No |
| T-10 | `p90_100` | 78 | 0.037055016 | 0.009259338 | 0.046314354 | No |
| T-5 | `p00_10` | 404 | 0.018318881 | 0.010769621 | 0.029088502 | No |
| T-5 | `p10_25` | 217 | 0.018318881 | 0.055977510 | 0.074296391 | No |
| T-5 | `p25_40` | 147 | 0.018318881 | 0.076598893 | 0.094917774 | No |
| T-5 | `p40_60` | 165 | 0.018318881 | 0.083926869 | 0.102245750 | No |
| T-5 | `p60_75` | 147 | 0.018318881 | 0.078986729 | 0.097305609 | No |
| T-5 | `p75_90` | 183 | 0.018318881 | 0.050811298 | 0.069130179 | No |
| T-5 | `p90_100` | 386 | 0.018318881 | 0.010177721 | 0.028496602 | No |

The **secondary** `model_own_bar` book has 762 trades on 21 days: hit rate 0.450131234, break-even hit rate 0.475795538, gross −$11.173000, fees $8.383200, net −$19.556200, net/trade −$0.025664304, mean net edge +0.110879787, market-fair expected −$11.424700, mean surprise −0.136544091, clustered SE 0.009996107, and clustered interval [−0.045656519, −0.005672090]. This threshold book is not an apples-to-apples replacement for the primary fixed-bar comparison.

The final model trade-decision fingerprint is `1b8df668ae4d6e9ce77dc08abb11f1ec710361c8f2258bbf2505ba0c7306dc18`; the fixed-bar-only fingerprint before adding the secondary book was `72a50a6494e75f2de221756e22aba66053f9e7a597b2f809ec606a0274c828a2`. The [trade decisions](data/backtest/model_trade_decisions.parquet), [validation results](data/backtest/model_results_validation.parquet), and [model-own threshold](data/models/model_edge_threshold.parquet) passed exact Parquet round trips. The [daily P&L plot](data/backtest/plots/model_vs_stage0_daily_net_pnl.png) displays the primary fixed-bar 21-day books and their paired difference. No test row was read, no frozen Stage 0 artifact changed, and no final Day 12 verdict was assigned. §3.3 has not begun.

## Day 12 §3.3 — Final Result and Freeze

### Official verdict

**Category C — no meaningful difference; did not beat Stage 0.** This is the exact pre-registered M2 validation label. Neither T-10 nor T-5 Brier difference exceeded two paired standard errors; the pooled difference also failed that rule. The paired 21-day M2-minus-Stage-0 net P&L interval crossed zero. A and B require meaningful probability improvement, which was absent. D requires a meaningfully worse Brier at either horizon or a wholly negative paired P&L interval; neither occurred. The raw P&L ordering does not change the category. Under the frozen tie rule, Stage 0 remains the baseline of record.

### Probability result

On the identical validation common rows, M2 T-10 Brier was 0.189066840 versus Stage 0's 0.189792806 (difference −0.000725966, paired SE 0.000889613, 2SE 0.001779227): numerically lower, not meaningful. T-5 was 0.116485755 versus 0.116474357 (difference +0.000011398, paired SE 0.000554122, 2SE 0.001108244): essentially tied and numerically higher, not meaningful. Pooled Brier was 0.153168157 versus 0.153529422 (difference −0.000361265, paired SE 0.000526518): not meaningful. Market quote-mid Brier was lower than both on the same rows at T-10 (0.182157128) and T-5 (0.108801868); market mid was a benchmark, never a model input. No meaningful probability improvement was established.

### Primary economic result

**Tier 2 — quote-aware, top-of-book, size-unaware; one contract, taker entry, held to settlement.** The primary comparison applies the same frozen Stage 0 1.2 bps fixed bar to both books. M2 made 728 trades: gross P&L −$10.151000, fees $8.075600, net P&L −$18.226600, net/trade −$0.025036538, mean net edge +$0.112568396, market-fair expected total P&L −$11.008100, and mean surprise −$0.137604935. Its traded-day-clustered per-trade ±2SE interval was [−$0.046743937, −$0.003329140]. Frozen Stage 0 made 719 trades: gross −$5.280000, fees $7.453900, net −$12.733900, and net/trade −$0.017710570. M2 was numerically worse on these different-sized books, but the pre-registered direct comparison did not establish a meaningful difference: mean daily M2-minus-Stage-0 P&L over all 21 validation days was −$0.261557143, SE $0.321544483, 2SE $0.643088966, interval [−$0.904646109, +$0.381531823].

### Secondary model-own threshold

**Secondary Tier 2 book; not the primary verdict basis.** The model-own required net edge was the sum of the M2 ±1.2 bps spot-perturbation basis-sensitivity term and its stored ten-decile validation ECE term, by horizon and frozen quote-mid price bucket. Reapplying saved train-fit coefficients and scalers reconstructed unperturbed probabilities with maximum absolute error `1.1102230246251565e-16`; the six spot-independent raw features shifted by exactly `0.0`. The `model_own_bar` book made 762 trades, net P&L −$19.556200, net/trade −$0.025664304, mean surprise −$0.136544091, and traded-day-clustered per-trade interval [−$0.045656519, −$0.005672090].

### M1 caveat

M1 validation folds 4–6 made 756 fixed-bar trades and net P&L −$14.634200. Folds 5–6 fit on earlier validation outcomes under the frozen walk-forward schedule. M1 remains context and does not replace the primary protocol-identical M2 versus train-fitted Stage 0 comparison.

### Final research interpretation

The logistic fits learned some stable coefficient structure, including a positive Stage 0 carrier and short-window-ratio interaction across M1 folds. The richer linear corrections did not establish meaningful probability improvement, did not fix the negative-surprise problem, and did not improve economic performance numerically. Day 12 therefore does not establish the fitted linear correction as an improvement over Stage 0. This describes the completed validation experiment; it is not a new selection rule or a test-period result.

### Integrity

The read-only deterministic rerun used the amended nine-feature design, frozen six M1 folds, M2 train-only fit, B1 reference, fit-only standardization, seven-value C grid, last-seven-day inner holdout, Brier selection with smaller-C tie break, and frozen solver settings. All 14 selected C values reproduced exactly. Maximum absolute difference was `0.0` for standardized and original coefficients, intercepts, scaler means and SDs, and predictions; the required coefficient tolerance was `1e-10`. The prediction fingerprint remained `78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512`. The Day 11 Stage 0 selector fingerprint remained `f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96`. The fixed-bar-only model decision fingerprint was `72a50a6494e75f2de221756e22aba66053f9e7a597b2f809ec606a0274c828a2`; the final two-bar decision fingerprint was `1b8df668ae4d6e9ce77dc08abb11f1ec710361c8f2258bbf2505ba0c7306dc18`. The test split was not read or scored. No feature, C, fold, model, threshold, execution, or accounting parameter changed after validation outcomes; no official artifact was overwritten during §3.3. Day 12 is frozen.
