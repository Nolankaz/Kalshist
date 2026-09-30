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

# Day 13 — HAR-RV Scratch Notes

## Step 2.2 — Coefficients Before Any Forecast Score

This is a read-only interpretation of the frozen `data/models/har_coefficients.parquet`, recorded **before Step 2.3 forecast scoring**. Before reading coefficients, the saved forecast fingerprint `d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee07ee0b31aa472eeaa50acc9` and HAR-fed probability fingerprint `e864a6377c8d520b04eae3dfb6664feaacdb5c35cb9708a6c0755b741d17fa0b` were recomputed from their artifacts and matched `evaluation_notes.md`. No prediction, target, or outcome was compared with a forecast or probability in this step.

The coefficient artifact has 140 rows with unique `(configuration, fold, horizon_minutes, model, term)` keys: six H1 folds and one H2 train-only fit per horizon, each carrying HAR (`intercept`, `ln_5min_vol`, `ln_1hr_vol`, `ln_24hr_vol`), N1 (`intercept`, `ln_5min_ewma_vol`), and separate HAR/N1 Platt (`intercept`, `stage0_raw_logit`) coefficients. H1 folds 1–3 score train dates; folds 4–6 score validation dates. H1 folds 5–6 add earlier validation forward-volatility targets and outcomes to their expanding fits. The H2 fit and H1 fold 4 use the same train common rows. `n_OLS` excludes null forward targets; the Platt fit uses all fit-window common rows. All coefficients and fit counts below are from this saved artifact. Classical OLS standard errors are **classical; understated under residual autocorrelation** and are not used for t-statistics, significance, or a decision.

### HAR coefficients and slope sums

`Σβ` is the sum of the three HAR log-volatility slopes. Each row identifies its configuration, fold, horizon, and target-bearing OLS fit population.

| Configuration | Fold | Horizon | n_OLS | Intercept | ln 5min | ln 1hr | ln 24hr | Σβ |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 walk-forward | 1 | T-10 | 2,936 | -0.034832 | +0.315580 | +0.489132 | +0.200535 | 1.005247 |
| H1 walk-forward | 2 | T-10 | 3,573 | -0.035336 | +0.309660 | +0.532851 | +0.169163 | 1.011674 |
| H1 walk-forward | 3 | T-10 | 4,184 | -0.035381 | +0.308614 | +0.531203 | +0.175997 | 1.015814 |
| H1 walk-forward | 4 | T-10 | 4,765 | -0.036023 | +0.307413 | +0.530693 | +0.174705 | 1.012811 |
| H1 walk-forward | 5 | T-10 | 5,310 | -0.034050 | +0.302046 | +0.539959 | +0.183651 | 1.025655 |
| H1 walk-forward | 6 | T-10 | 5,891 | -0.035465 | +0.314216 | +0.533837 | +0.171039 | 1.019092 |
| H2 train-only | 1 | T-10 | 4,765 | -0.036023 | +0.307413 | +0.530693 | +0.174705 | 1.012811 |
| H1 walk-forward | 1 | T-5 | 2,903 | -0.041970 | +0.386291 | +0.522751 | +0.139238 | 1.048281 |
| H1 walk-forward | 2 | T-5 | 3,529 | -0.042969 | +0.375523 | +0.547294 | +0.121223 | 1.044040 |
| H1 walk-forward | 3 | T-5 | 4,125 | -0.043856 | +0.364498 | +0.546377 | +0.138245 | 1.049120 |
| H1 walk-forward | 4 | T-5 | 4,695 | -0.043270 | +0.360942 | +0.546573 | +0.146947 | 1.054462 |
| H1 walk-forward | 5 | T-5 | 5,226 | -0.043270 | +0.352581 | +0.547977 | +0.157755 | 1.058313 |
| H1 walk-forward | 6 | T-5 | 5,797 | -0.044280 | +0.365724 | +0.545591 | +0.140259 | 1.051574 |
| H2 train-only | 1 | T-5 | 4,695 | -0.043270 | +0.360942 | +0.546573 | +0.146947 | 1.054462 |

Every HAR slope is positive; none changes sign. The H1 sign counts are:

| Horizon | HAR term | Positive | Negative | Exact zero |
| --- | --- | ---: | ---: | ---: |
| T-10 | `ln_5min_vol` | 6/6 | 0/6 | 0/6 |
| T-10 | `ln_1hr_vol` | 6/6 | 0/6 | 0/6 |
| T-10 | `ln_24hr_vol` | 6/6 | 0/6 | 0/6 |
| T-5 | `ln_5min_vol` | 6/6 | 0/6 | 0/6 |
| T-5 | `ln_1hr_vol` | 6/6 | 0/6 | 0/6 |
| T-5 | `ln_24hr_vol` | 6/6 | 0/6 | 0/6 |

The T-10 slope sum is just above 1 in every fold (`1.005247–1.025655`); T-5 is further above 1 (`1.044040–1.058313`). Thus the combined HAR log-volatility slope suggests near-unit to mildly extrapolative persistence rather than the subunit mean-reversion example in the plan. This is a conditional regression description, not a causal or performance claim. The nested windows could make individual coefficients unstable even with stable fitted values; no negative coefficient or sign reversal appeared here, and the frozen three-window specification remains unchanged.

### N1, intercepts, and Platt calibration

`n_OLS` is the same target-bearing fit population as HAR; `n_Platt` is every common fit row, including rows without a forward target. The two Platt columns for each model are its intercept `a` and `stage0_raw_logit` slope `b`.

| Configuration | Fold | Horizon | n_OLS / n_Platt | N1 intercept | N1 ln EWMA-5 slope | HAR Platt a | HAR Platt b | N1 Platt a | N1 Platt b |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 walk-forward | 1 | T-10 | 2,936 / 2,946 | -0.072848 | 0.704743 | -0.088936 | 2.100611 | -0.067219 | 2.105342 |
| H1 walk-forward | 2 | T-10 | 3,573 / 3,590 | -0.076453 | 0.720053 | -0.064015 | 2.158075 | -0.040696 | 2.162869 |
| H1 walk-forward | 3 | T-10 | 4,184 / 4,206 | -0.087905 | 0.724761 | -0.047128 | 2.085345 | -0.023587 | 2.091933 |
| H1 walk-forward | 4 | T-10 | 4,765 / 4,792 | -0.095582 | 0.738083 | -0.033437 | 2.109892 | -0.008302 | 2.111973 |
| H1 walk-forward | 5 | T-10 | 5,310 / 5,348 | -0.097207 | 0.782905 | -0.021255 | 2.103306 | +0.007409 | 2.114991 |
| H1 walk-forward | 6 | T-10 | 5,891 / 5,942 | -0.102769 | 0.786006 | -0.021593 | 2.054371 | +0.005807 | 2.072587 |
| H2 train-only | 1 | T-10 | 4,765 / 4,792 | -0.095582 | 0.738083 | -0.033437 | 2.109892 | -0.008302 | 2.111973 |
| H1 walk-forward | 1 | T-5 | 2,903 / 2,925 | -0.065946 | 0.782379 | -0.054652 | 2.683678 | -0.039935 | 2.657991 |
| H1 walk-forward | 2 | T-5 | 3,529 / 3,565 | -0.069268 | 0.782386 | -0.022494 | 2.572215 | -0.004806 | 2.553584 |
| H1 walk-forward | 3 | T-5 | 4,125 / 4,178 | -0.081351 | 0.779135 | +0.002010 | 2.493566 | +0.021813 | 2.474552 |
| H1 walk-forward | 4 | T-5 | 4,695 / 4,757 | -0.088686 | 0.794081 | +0.023052 | 2.473859 | +0.044030 | 2.456460 |
| H1 walk-forward | 5 | T-5 | 5,226 / 5,306 | -0.091132 | 0.824366 | +0.040075 | 2.437464 | +0.063063 | 2.426371 |
| H1 walk-forward | 6 | T-5 | 5,797 / 5,891 | -0.095574 | 0.829803 | +0.041400 | 2.411435 | +0.063910 | 2.401217 |
| H2 train-only | 1 | T-5 | 4,695 / 4,757 | -0.088686 | 0.794081 | +0.023052 | 2.473859 | +0.044030 | 2.456460 |

N1 slopes are consistently below 1: `0.704743–0.786006` at T-10 and `0.779135–0.829803` at T-5. Under the frozen interpretation, trailing EWMA-5 volatility is over-dispersed relative to forward volatility, and N1 compresses its high and low readings toward each other. Its negative intercepts (`-0.102769–-0.072848` at T-10; `-0.095574–-0.065946` at T-5) also shift the volatility level after exponentiation, but these fits are more than a constant multiplicative level correction because their slopes are clearly below 1. Platt can absorb a constant sigma scaling; neither the N1 level adjustment nor its compression establishes a probability improvement without scoring.

**Day 12 connection:** N1's consistently subunit slopes conceptually agree with Day 12's positive `stage0_logit × log_ratio_5m_1h` interaction: a short-window volatility spike relative to the hour calls for less forward-sigma extrapolation and sharper Stage 0 confidence, without establishing that either correction performs better.

HAR intercepts are small and negative (`-0.036023–-0.034050` at T-10; `-0.044280–-0.041970` at T-5), while the HAR slope sums remain slightly above 1. The HAR-fed Platt slopes stay near the frozen Stage 0 reference scale of about `2.162` at T-10 and `2.500` at T-5: H2 is `2.109892` / `2.473859`; H1 spans `2.054371–2.158075` / `2.411435–2.683678`. None is near 1 or dramatically separated from that scale. This is a calibration sanity observation, not a performance claim.

### Fit-window equality and later-fold drift

For each horizon, every H1 fold-4 coefficient was compared by `(model, term)` with H2, including all HAR/N1 and both Platt intercepts and slopes. The **maximum absolute coefficient difference was `0`** at T-10 and `0` at T-5, below the required `1e-10`. HAR and N1 OLS fit counts matched exactly (`4,765` T-10; `4,695` T-5), as did the all-common Platt fit counts (`4,792` T-10; `4,757` T-5) and null-target exclusions. This confirms the shared train fit population and path.

As earlier validation dates enter H1's expanding fits, folds 5–6 show **modest HAR drift** rather than a sign change: T-10 `Σβ` moves from fold 4's `1.012811` to `1.025655` / `1.019092`; T-5 moves from `1.054462` to `1.058313` / `1.051574`. T-10 short/hour/day slopes move from `0.307413 / 0.530693 / 0.174705` to `0.302046 / 0.539959 / 0.183651` and `0.314216 / 0.533837 / 0.171039`; T-5 moves from `0.360942 / 0.546573 / 0.146947` to `0.352581 / 0.547977 / 0.157755` and `0.365724 / 0.545591 / 0.140259`. N1 slope rises more visibly, from `0.738083` to `0.782905 / 0.786006` at T-10 and from `0.794081` to `0.824366 / 0.829803` at T-5. HAR Platt slope moves from `2.109892` to `2.103306 / 2.054371` at T-10 and from `2.473859` to `2.437464 / 2.411435` at T-5; N1 Platt slope moves from `2.111973` to `2.114991 / 2.072587`, and from `2.456460` to `2.426371 / 2.401217`. This is a regime-stability description only. No model, window, feature, or calibration rule was changed, and no forecast or probability score was computed.

## Step 2.3 — Forward-Volatility Scores Before Any Probability Score

The frozen `har_forecasts.parquet` fingerprint matched the protocol's `d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee07ee0b31aa472eeaa50acc9` **before** loading the separate forward target. The forecast file had 10,346 exact common score rows (H1 7,012; H2 3,334), the registered horizon and fold keys, and no `fwd_` column or test date. Target joins were one-to-one within each configuration on `(ticker, horizon_minutes)`; split and close date agreed. Only non-null `fwd_log_rv` rows entered the scores. The 184-row `har_vol_comparison.parquet` passed its schema, key, numerical-identity, and exact Parquet round-trip checks; a second run reproduced it exactly.

Loss is squared error in log-volatility. In every table below, a negative HAR-minus-benchmark paired difference favors HAR. The naive SE is descriptive; the verdict uses `abs(mean difference) > 2 × close_date-clustered SE`. Mean residual is `fwd_log_rv - forecast`: positive means underforecasting, negative means overforecasting. `R²_vs_naive` compares MSE to Naive-S0 on the same rows. Fit-mean R² uses the **train target-bearing common-row mean** for H2, and each scored row's own fold/horizon fit-window target-bearing common-row mean for H1 before aggregation. The fit-mean R² is **inflated by regime shift; not a skill measure**. No scored-window target mean is used as its reference.

### H2 validation T-10 — 1,644 target-bearing rows, 21 close-date clusters

1. **HAR vs Naive-S0:** HAR MSE `0.058826112` versus Naive-S0 `0.089998530`; paired difference `-0.031172418065`, naive SE `0.003138915715`, clustered SE `0.004112422816`, design effect `1.716470`. This is meaningfully better under the clustered rule.
2. **HAR vs N1:** N1 MSE `0.091657356`; paired difference `-0.032831244267`, naive SE `0.002626039522`, clustered SE `0.005616920930`, design effect `4.575032`. This is meaningfully better; the registered multi-horizon HAR structure adds forecast skill beyond N1's fitted EWMA-5 level/slope control on these rows.
3. **Pre-registered V category: V1 — structure adds skill.** HAR is meaningfully better than both Naive-S0 and N1 on H2 validation. V4 was checked first and did not apply. This V verdict was recorded **before Step 3.1 probability scoring**.
4. HAR `R²_vs_naive` is `0.346366`.

| Model | MSE | MAE | Mean residual (`target − forecast`) | R²_vs_naive | Fit-mean R²: inflated by regime shift; not a skill measure |
| --- | ---: | ---: | ---: | ---: | ---: |
| HAR | 0.058826112 | 0.184528 | -0.012965 | 0.346366 | 0.844575 |
| N1 | 0.091657356 | 0.231943 | -0.136942 | -0.018432 | 0.757831 |
| Naive-S0 | 0.089998530 | 0.228475 | -0.027597 | 0.000000 | 0.762213 |
| Naive-15 | 0.071419926 | 0.204139 | -0.034767 | 0.206432 | 0.811300 |

HAR's MAE is smallest, and its negative mean residual indicates slight average overforecasting in log space. Naive-15's MSE is **below** Naive-S0's on these same rows. HAR minus Naive-15 is `-0.012593813872` (naive SE `0.002033301238`, clustered SE `0.002440290674`, design effect `1.440389`), meaningful under the frozen rule; Naive-15 does not enter V. For the primary HAR–Naive-S0 comparison, the clustered SE is `1.31×` the naive SE (design effect `1.716470`), so row-independent uncertainty would understate the interval. Against N1 the ratio is `2.14×` (design effect `4.575032`).

### H2 validation T-5 — 1,588 target-bearing rows, 21 close-date clusters

1. **HAR vs Naive-S0:** HAR MSE `0.075463723` versus Naive-S0 `0.099324391`; paired difference `-0.023860667932`, naive SE `0.003095726384`, clustered SE `0.004099138244`, design effect `1.753315`. This is meaningfully better under the clustered rule.
2. **HAR vs N1:** N1 MSE `0.096687468`; paired difference `-0.021223745465`, naive SE `0.002660049550`, clustered SE `0.004020633895`, design effect `2.284597`. This is meaningfully better; the registered multi-horizon HAR structure adds forecast skill beyond N1's fitted EWMA-5 level/slope control on these rows.
3. **Pre-registered V category: V1 — structure adds skill.** HAR is meaningfully better than both Naive-S0 and N1 on H2 validation. V4 was checked first and did not apply. This V verdict was recorded **before Step 3.1 probability scoring**.
4. HAR `R²_vs_naive` is `0.240230`.

| Model | MSE | MAE | Mean residual (`target − forecast`) | R²_vs_naive | Fit-mean R²: inflated by regime shift; not a skill measure |
| --- | ---: | ---: | ---: | ---: | ---: |
| HAR | 0.075463723 | 0.210878 | +0.003923 | 0.240230 | 0.812714 |
| N1 | 0.096687468 | 0.238582 | -0.104507 | 0.026549 | 0.760041 |
| Naive-S0 | 0.099324391 | 0.240204 | -0.026355 | 0.000000 | 0.753497 |
| Naive-15 | 0.091209495 | 0.231027 | -0.075124 | 0.081701 | 0.773636 |

HAR's MAE is smallest, and its positive mean residual indicates slight average underforecasting in log space. Naive-15's MSE is **below** Naive-S0's on these same rows. HAR minus Naive-15 is `-0.015745772677` (naive SE `0.002337038458`, clustered SE `0.002913104236`, design effect `1.553747`), meaningful under the frozen rule; Naive-15 does not enter V. For the primary HAR–Naive-S0 comparison, the clustered SE is `1.32×` the naive SE (design effect `1.753315`); against N1 it is `1.51×` (design effect `2.284597`). Both show why the day-clustered rule, rather than the smaller row-independent SE, controls the verdict.

### Secondary H1 out-of-fold context

These are forward-volatility scores only. H1 folds 1–3 are **train-date OOF**; folds 4–6 are **validation-date OOF**. Folds 5–6 consume earlier validation forward-volatility targets and outcomes in their expanding fits, so H1 remains secondary to H2. Every paired comparison below is meaningful under its own day-clustered 2-SE check, with HAR favored; these rows do not assign or revise an H2 V category.

| H1 population | Horizon | Target-bearing n | Close-date clusters | HAR MSE | Naive-S0 MSE | HAR `R²_vs_naive` | HAR − Naive-S0 | HAR − N1 | HAR − Naive-15 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All OOF | T-10 | 3,473 | 42 | 0.049356363 | 0.079233020 | 0.377073 | -0.029877 | -0.023707 | -0.009916 |
| All OOF | T-5 | 3,380 | 42 | 0.068669831 | 0.093002379 | 0.261634 | -0.024333 | -0.018193 | -0.012492 |
| Folds 1–3, train dates | T-10 | 1,829 | 21 | 0.040823668 | 0.069556422 | 0.413086 | -0.028733 | -0.019197 | -0.007531 |
| Folds 1–3, train dates | T-5 | 1,792 | 21 | 0.062613120 | 0.087400060 | 0.283603 | -0.024787 | -0.017311 | -0.009646 |
| Folds 4–6, validation dates | T-10 | 1,644 | 21 | 0.058849247 | 0.089998530 | 0.346109 | -0.031149 | -0.028724 | -0.012571 |
| Folds 4–6, validation dates | T-5 | 1,588 | 21 | 0.075504607 | 0.099324391 | 0.239818 | -0.023820 | -0.019188 | -0.015705 |

All paired mean differences matched the corresponding HAR-minus-benchmark MSE differences to `1e-12` or better. No probability score, oracle, residual cut, economic result, or test row was accessed; the HAR/N1 specification, target, folds, and frozen artifacts were unchanged.

## Step 3.1 — Controlled Sigma Swap, Probability Quality Only

The saved HAR-fed probability fingerprint matched the frozen protocol value `e864a6377c8d520b04eae3dfb6664feaacdb5c35cb9708a6c0755b741d17fa0b` **before any probability score**. The artifact retained all 10,346 unique H1/H2 keys, including all 3,334 H2 validation common rows regardless of forward-target coverage. Its carried Stage 0 probabilities matched the frozen Stage 0 artifact exactly on identical keys. Recomputed Stage 0 validation Brier matched `stage0_platt_validation_scores.parquet` to a maximum absolute error of `1.3877787807814457e-17` across T-10 and T-5, below `1e-12`. The frozen Day 12 B1 fingerprint and its carried H1 probabilities also matched exactly.

The primary rule remains Day 9/12's **paired per-row Brier difference ± two paired SEs**; the Day 13 day-clustered volatility rule did not enter any P decision. A negative difference favors HAR. Secondary log loss, AUC, ECE, market gap, buckets, and the oracle are descriptive only.

### H2 T-10 — 1,685 validation common rows

HAR-fed Brier was `0.189033084` versus frozen Stage 0 `0.189792806`: paired difference `-0.000759722`, paired per-row SE `0.000603548`, and `abs(difference) < 2 × SE`, so **not meaningful**. Log loss was HAR `0.562951591` versus Stage 0 `0.564414415`; AUC `0.785220373` versus `0.784119142`; ten-decile ECE `0.030709132` versus `0.035595749`. The market-mid benchmark had Brier `0.182157128`, log loss `0.544429183`, AUC `0.802010981`, and ECE `0.024602374`.

Market-gap closure was `0.099496315`: Stage 0-to-HAR improvement numerator `0.000759722` divided by the **visible** Stage 0-to-market denominator `0.007635678`. The denominator is not numerically near zero under the script's predeclared `1e-12` stability check.

### H2 T-5 — 1,649 validation common rows

HAR-fed Brier was `0.117061648` versus frozen Stage 0 `0.116474357`: paired difference `+0.000587291`, paired per-row SE `0.000476165`, and `abs(difference) < 2 × SE`, so **not meaningful**. Log loss was HAR `0.379809030` versus Stage 0 `0.378477904`; AUC `0.913896614` versus `0.914429288`; ten-decile ECE `0.024833848` versus `0.021091839`. The market-mid benchmark had Brier `0.108801868`, log loss `0.352078390`, AUC `0.924170456`, and ECE `0.022625531`.

Market-gap closure was `-0.076545037`: Stage 0-to-HAR numerator `-0.000587291` divided by the **visible** Stage 0-to-market denominator `0.007672489`. That denominator is not numerically near zero under the same stability check.

### Pooled H2 verdict, N1 control, and ranking

Across all 3,334 validation common rows, HAR-fed Brier was `0.153435934` versus Stage 0 `0.153529422`: paired difference `-0.000093488`, paired SE `0.000385490`, not meaningful. The pre-registered overall verdict is **P3 — no meaningful difference:** “Forecast σ did not improve Stage 0's probabilities; Stage 0 remains the baseline of record.” Ties go to Stage 0. The pooled result cannot upgrade either horizon, and no secondary or economic metric enters this category.

N1-fed Brier was `0.190027892` at T-10, `0.116976624` at T-5, and `0.153896656` pooled. Relative to the same Stage 0 rows, its paired differences were `+0.000235086` (SE `0.000536517`), `+0.000502267` (SE `0.000365636`), and `+0.000367234` (SE `0.000325889`): none meaningful. HAR Brier was descriptively lower than N1 by `0.000994808` at T-10 and higher by `0.000085024` at T-5. N1 log loss / AUC / ECE were `0.562892604 / 0.784583041 / 0.040632161` at T-10 and `0.377152855 / 0.914479318 / 0.028020659` at T-5. N1 is a diagnostic control, with no separate probability verdict.

HAR's AUC changed slightly from Stage 0 at both horizons (up `0.001101232` at T-10, down `0.000532674` at T-5). The frozen Platt slopes are positive, so this ranking change comes from HAR's row-varying sigma rather than a monotonic calibration rescale. ECE and reliability moved in opposite directions by horizon: HAR's ECE improved at T-10 and worsened at T-5. The validated ten-decile reliability plot shows HAR and Stage 0 close together, with Wilson intervals; neither its appearance nor ECE changes the Brier verdict.

### Oracle headroom — diagnostic, never a candidate model

The oracle used `sigma := fwd_rv`, trained its horizon-specific Platt map only on target-bearing **train** common rows, and was scored only on the exact target-bearing H2 validation subsets. All figures below re-score HAR and Stage 0 on those same keys; they are distinct from the full H2 probability populations.

| Horizon | Subset n | Stage 0 Brier | HAR Brier | Oracle Brier | Oracle gap (`S0 − oracle`) | HAR improvement (`S0 − HAR`) | Headroom captured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-10 | 1,644 | 0.190329179 | 0.189576542 | 0.188143726 | 0.002185453 | +0.000752637 | +0.344385 |
| T-5 | 1,588 | 0.117578573 | 0.118065749 | 0.117135129 | 0.000443444 | -0.000487176 | -1.098620 |

The T-5 oracle gap is small in Brier units, so its headroom fraction is sensitive to the denominator; the raw gap and improvement are the clearer figures. It is above the script's predeclared `1e-12` numerical near-zero threshold, so the artifact's numerical `unstable` flag is false. The oracle is diagnostic only; no per-row oracle probability was written, thresholded, or traded.

### Frozen Stage 0 |z| and quote-mid buckets

The |z| cuts use only Stage 0's frozen `z_5min_ewma_vol`. In `[0, 0.25)`, T-10 had `n=879`, HAR minus Stage 0 Brier `-0.000244` (paired SE `0.000646`), and T-5 had `n=432`, difference `+0.001146` (SE `0.000940`); neither was meaningful. Thus the known low-|z| concentration did not show an established probability improvement. The other three |z| cells at each horizon also failed the 2-SE rule. T-10's `[1.0, infinity)` cell had `n=27` and is explicitly **thin**; all other |z| cells had at least 30 rows.

The seven frozen `quote_mid` price buckets partitioned every H2 horizon. At T-10, `[0.25, 0.40)` (`p25_40`, `n=317`) had HAR minus Stage 0 Brier `-0.004315` with paired SE `0.001954`, the sole price cell clearing two SEs. It is a bucket diagnostic, not a new verdict or feature. No T-5 price bucket cleared two SEs; its largest positive difference was `[0.75, 0.90)` (`p75_90`, `n=183`), `+0.002658` with SE `0.001620`. No price cell was thin. Market midpoint was used only as benchmark and bucketing context, never as a HAR input.

### Secondary H1 versus frozen B1

| H1 window | Horizon | n | HAR Brier | B1 Brier | Paired difference | Paired SE | Meaningful |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| All OOF | T-10 | 3,531 | 0.190535935 | 0.191059850 | -0.000523916 | 0.000389634 | No |
| All OOF | T-5 | 3,481 | 0.122177048 | 0.122292817 | -0.000115769 | 0.000312065 | No |
| All OOF | Pooled | 7,012 | 0.156600213 | 0.156921510 | -0.000321297 | 0.000249988 | No |
| Folds 4–6, validation dates | T-10 | 1,685 | 0.189122069 | 0.189889425 | -0.000767356 | 0.000600636 | No |
| Folds 4–6, validation dates | T-5 | 1,649 | 0.117113920 | 0.116506419 | +0.000607502 | 0.000478216 | No |
| Folds 4–6, validation dates | Pooled | 3,334 | 0.153506761 | 0.153594111 | -0.000087350 | 0.000384956 | No |

Both H1 windows receive separate **secondary P3** categories under the same per-row rule. H1 folds 1–3 score train dates; folds 4–6 score validation dates, and folds 5–6 consume earlier validation forward-volatility targets and outcomes in their expanding fits. H1 does not replace the H2 verdict.

The joint frozen reading is **V1/V1 for volatility forecasting and P3 for probability quality**: better forward-sigma forecasts did not materially improve Stage 0 probabilities. The small T-5 oracle gap and the remaining Stage 0-to-market Brier gap are diagnostics for later consideration, not grounds to alter HAR, the P rule, or the baseline of record. This step computed no economic result, residual analysis, or OFI decision and read no test row. The 58-row comparison artifact and reliability plot passed their structural, key, bucket, Wilson, exact Parquet round-trip, and deterministic second-run checks.

## Step 3.2 — Frozen H1 HAR Residual Diagnostics

The full 10,346-row HAR forecast artifact reproduced fingerprint `d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee0b31aa472eeaa50acc9` **before loading a forward target**. The residual analysis retained H1's 3,473 T-10 and 3,380 T-5 out-of-fold rows with non-null `fwd_log_rv`, preserving fold IDs and excluding no other rows. Define `residual = fwd_log_rv − log_rv_har`: positive means HAR underforecasted forward log volatility; negative means overforecasted. All uncertainty below is from the frozen `close_date` clustered helper, with the naive SE retained in the aggregate artifact. The formal structure rule requires `n >= 30`, `abs(mean) > 2 × clustered SE` **on both train-date folds 1–3 and validation folds 4–6**, with the same nonzero sign. All-OOF, fold, and autocorrelation results alone cannot establish structure.

### UTC decision-hour blocks

Cells use the frozen `00-05 / 06-11 / 12-17 / 18-23` UTC blocks. Each pair below is `n; mean residual; clustered SE`, first for train folds 1–3, then validation folds 4–6. Each horizon's four cells partition its parent on both sides.

| Horizon | UTC hour | Train n; mean; SE | Validation n; mean; SE | Formal structure |
| --- | --- | --- | --- | --- |
| T-10 | 00-05 | 473; -0.020743; 0.008108 | 415; -0.014452; 0.013224 | No |
| T-10 | 06-11 | 379; -0.032889; 0.010270 | 294; -0.013756; 0.013777 | No |
| T-10 | 12-17 | 491; +0.041042; 0.008818 | 465; +0.021643; 0.015044 | No |
| T-10 | 18-23 | 486; -0.021249; 0.007981 | 470; -0.032662; 0.012150 | **Yes** |
| T-5 | 00-05 | 455; -0.028320; 0.010327 | 383; -0.009007; 0.014796 | No |
| T-5 | 06-11 | 364; -0.001723; 0.013860 | 286; +0.001705; 0.014162 | No |
| T-5 | 12-17 | 490; +0.026469; 0.012406 | 463; +0.028351; 0.012015 | **Yes** |
| T-5 | 18-23 | 483; -0.017463; 0.010322 | 456; -0.006332; 0.014359 | No |

The T-10 12–17 all-OOF mean is positive and clears its clustered rule, but its validation-side cell does not; it is therefore **not** formal structure. The same distinction applies to other all-OOF-only flags.

### UTC weekend versus weekday

| Horizon | Cell | Train n; mean; SE | Validation n; mean; SE | Formal structure |
| --- | --- | --- | --- | --- |
| T-10 | Weekday | 1,392; -0.002429; 0.003448 | 1,322; -0.002461; 0.005732 | No |
| T-10 | Weekend | 437; -0.020757; 0.005670 | 322; -0.037503; 0.019150 | No |
| T-5 | Weekday | 1,385; -0.000992; 0.005597 | 1,294; +0.003214; 0.007185 | No |
| T-5 | Weekend | 407; -0.018683; 0.009589 | 294; +0.010608; 0.021184 | No |

T-10's weekend mean is negative on both sides and clears two clustered SEs on the train side, but not on validation; **no weekend cell is formal structure**. Weekend BTC proxy quality is a separate caveat: Bullish/Crypto.com low-volume proxy days are weekends, so an apparent weekend effect could reflect liquidity/measurement quality rather than a clean volatility relationship. This caveat does not alter the mechanical flag.

### Frozen trailing EWMA-5 volatility terciles

The protocol's train-common feature edges were reused exactly: T-10 `0.6337964396166403 / 0.8837207863551418`; T-5 `0.623421011561805 / 0.8606540123954035`. Bins are `[0, lower)`, `[lower, upper)`, `[upper, infinity)`; equality enters the higher bin. No validation or OOF quantile was recomputed.

| Horizon | Tercile | Train n; mean; SE | Validation n; mean; SE | Formal structure |
| --- | --- | --- | --- | --- |
| T-10 | Low | 968; -0.020067; 0.006321 | 1,312; -0.019971; 0.006510 | **Yes** |
| T-10 | Middle | 558; -0.001270; 0.007550 | 270; +0.022035; 0.012984 | No |
| T-10 | High | 303; +0.025352; 0.010282 | 62; +0.079404; 0.020546 | **Yes** |
| T-5 | Low | 924; -0.008845; 0.007743 | 1,273; -0.002137; 0.007740 | No |
| T-5 | Middle | 552; -0.012165; 0.009698 | 250; +0.024509; 0.016064 | No |
| T-5 | High | 316; +0.018703; 0.008164 | 65; +0.059553; 0.015046 | **Yes** |

The high-volatility validation cells are non-thin (`n=62` and `65`) but much smaller than their low-tercile counterparts, so the exact counts and clustered SEs matter. They satisfy the frozen rule without creating a new feature or filter.

### HAR forecast deciles and residual shape

Ten equal-frequency forecast deciles were formed once per horizon on the **entire target-bearing H1 OOF forecast** before examining residual cells. These exact log-volatility boundaries are stored in every matching decile row of `har_residual_summary.parquet` and reused unchanged on train and validation:

- **T-10 D1–D10 edges:** `[-1.8578270847428646, -1.0439264294361512, -0.9061781669964727, -0.8065486300308141, -0.7155595531755348, -0.6373288045104348, -0.5653565237662606, -0.4866595421217885, -0.37132573676715014, -0.21814736473304155, 0.27331555278390446]`.
- **T-5 D1–D10 edges:** `[-2.2980893538443397, -1.11111924701299, -0.9538896014095417, -0.8485785611564186, -0.7598899535468682, -0.6798190496133201, -0.5940775432508251, -0.5033463581567407, -0.3912929841897199, -0.23270343422168513, 0.3042119219722696]`.

Each sequence below is D1 through D10, as `mean residual` (the artifact carries each cell's `n` and clustered SE):

| Horizon and side | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T-10 train | +0.1076 | +0.0594 | -0.0335 | -0.0069 | -0.0449 | -0.0409 | -0.0142 | -0.0075 | +0.0003 | +0.0113 |
| T-10 validation | -0.0116 | -0.0216 | -0.0029 | -0.0221 | -0.0104 | -0.0242 | -0.0052 | +0.0137 | +0.0359 | +0.0168 |
| T-5 train | +0.0331 | +0.0295 | +0.0479 | -0.0155 | -0.0145 | -0.0155 | -0.0598 | -0.0193 | +0.0151 | +0.0051 |
| T-5 validation | +0.0181 | +0.0151 | +0.0247 | -0.0402 | -0.0163 | -0.0317 | +0.0350 | +0.0033 | +0.0308 | +0.0251 |

The visual curve suggests varying residual shape, especially positive low-forecast means on train and positive high-forecast means on validation. **No forecast-decile cell satisfies the formal cross-boundary structure rule.** In particular, T-10's low-forecast train-side positive residuals do not persist with the same sign on validation; T-5 D7 changes sign. Validation D10 has only `n=26` at each horizon and is thin. The plot uses no smoothing fit.

### Fold drift and within-day lag-1 correlation

Fold-level means are descriptive only; even a single fold clearing two clustered SEs receives no formal structure flag.

| Horizon | Fold 1 n; mean; SE | Fold 2 n; mean; SE | Fold 3 n; mean; SE | Fold 4 n; mean; SE | Fold 5 n; mean; SE | Fold 6 n; mean; SE |
| --- | --- | --- | --- | --- | --- | --- |
| T-10 | 637; -0.015616; 0.005986 | 611; -0.005659; 0.004885 | 581; +0.001640; 0.005691 | 545; -0.021309; 0.006835 | 581; +0.006131; 0.012420 | 518; -0.014049; 0.010292 |
| T-5 | 626; -0.002073; 0.006623 | 596; -0.008499; 0.007133 | 570; -0.004587; 0.012515 | 531; -0.006718; 0.013366 | 571; +0.005621; 0.012514 | 486; +0.015709; 0.010485 |

T-10 fold means alternate around zero after fold 3; T-5 moves from small negative train-fold means to positive folds 5–6. This is descriptive drift, not a model adjustment. Within each horizon and `close_date`, residuals were sorted by UTC `decision_time` (ticker breaks ties) and direct lag pairs were formed only inside that date. The correlation is calculated across all such valid pairs, never across a midnight boundary:

| Horizon | All OOF correlation; pairs; days | Train folds 1–3 correlation; pairs; days | Validation folds 4–6 correlation; pairs; days |
| --- | --- | --- | --- |
| T-10 | +0.098551; 3,431; 42 | +0.092852; 1,808; 21 | +0.102924; 1,623; 21 |
| T-5 | +0.022561; 3,338; 42 | +0.048890; 1,771; 21 | -0.002701; 1,567; 21 |

Autocorrelation is descriptive and never supplies a formal structure flag.

### Frozen structure inventory and Day 14 diagnostic implication

**Every formal structure cell:** T-10 UTC `18-23` (negative); T-10 trailing-volatility `low` (negative); T-10 trailing-volatility `high` (positive); T-5 UTC `12-17` (positive); T-5 trailing-volatility `high` (positive). All have at least 30 observations, clear two **day-clustered** SEs on both sides, and retain the same sign across the boundary. No weekend or forecast-decile cell qualifies. The complete train/validation/all-OOF rows, counts, uncertainties, boundary checks, and flags are in the 138-row aggregate artifact.

For the Day 14 gate, there is stable residual structure by hour and trailing-volatility regime, but no registered forecast-decile pattern that persists across the train/validation boundary. This is diagnostic evidence, **not** a decision to fit or reject a boosted model. Day 13's frozen verdicts remain **V1/V1** for volatility forecasting and **P3** for probability quality; any possible extra volatility skill must be considered alongside the limited oracle probability headroom. Folds 5–6 consumed earlier validation forward-volatility targets and outcomes in their expanding fits. No cut became a feature, filter, or model change; no gradient boosting, OFI verdict, economic comparison, or test read occurred. The summary's exact Parquet round trip, both plot hashes, and a deterministic second run passed.

## Step 3.3 — OFI Decision

**Pre-registered verdict: Reopen as a scoped post-Day-17 experiment.** This schedules a **new, separately pre-registered research experiment after Day 17**, preserving the Day 14–17 workflow and one-shot test. It does not validate OFI or authorize a current feature, model, or trade. Order-flow imbalance (OFI) would test whether net aggressive buying versus selling predicts the **direction** of the next move. HAR residuals measure error in **future volatility magnitude**; their structure cannot establish a directional signal.

The saved Day 13 artifacts retain **V1 at T-10 and V1 at T-5**: HAR's multi-horizon forecast improved on both Naive-S0 and N1 under the frozen volatility rule. The H2 probability verdict remains **P3**: “Forecast σ did not improve Stage 0's probabilities; Stage 0 remains the baseline of record.” On all validation common rows, T-10 HAR/Stage 0 Brier was `0.189033084 / 0.189792806` (paired difference `-0.000759722`, SE `0.000603548`, not meaningful); T-5 was `0.117061648 / 0.116474357` (`+0.000587291`, SE `0.000476165`, not meaningful). Better sigma forecasts therefore did not establish better final probabilities.

| Horizon | Oracle-subset n | Stage 0 / HAR / oracle Brier on identical target-bearing rows | Raw oracle gap (Stage 0 − oracle) | HAR improvement (Stage 0 − HAR) | Headroom captured | Full-H2 HAR / market-mid Brier | Remaining HAR − market gap | Stage 0-to-market gap closure |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| T-10 | 1,644 | `0.190329179 / 0.189576542 / 0.188143726` | `0.002185453` | `+0.000752637` | `+0.344385` | `0.189033084 / 0.182157128` | `0.006875957` | `+0.099496` |
| T-5 | 1,588 | `0.117578573 / 0.118065749 / 0.117135129` | `0.000443444` | `-0.000487176` | `-1.098620` | `0.117061648 / 0.108801868` | `0.008259780` | `-0.076545` |

The T-5 oracle gap is especially small in raw Brier units; its negative headroom-captured ratio is sensitive to that small denominator and is not a standalone headline. The oracle is diagnostic only. Within the frozen Stage 0 architecture, even realized future sigma would repair little T-5 probability error. That bounds a sigma-only remedy; it does **not** identify the remaining error as OFI. A market-mid probability advantage persists after the HAR correction at both horizons; market mid remains a benchmark, never a model input.

In the frozen Stage 0 `|z|` bucket `[0, 0.25)`, T-10 had `n=879` and HAR minus Stage 0 Brier `-0.000244` (paired SE `0.000646`); T-5 had `n=432`, difference `+0.001146` (SE `0.000940`). Neither clears the inherited two-SE rule. Thus better sigma has not been shown to solve the near-boundary weakness, but the bucket does not reveal its cause. The Day 11 primary validation 1.2 bps book had 719 trades: realized net P&L `-$0.017710570/trade`, near market-fair expected `-$0.014214047/trade` and far from model-expected `+$0.114468673/trade`. Its frozen low-`|z|` `[0, 0.25)` cell had 258 trades and `-$20.3721` net P&L. This supports the earlier market-fair interpretation, not an OFI performance claim; no new economic query was run.

Step 3.2 found formal volatility-residual structure in exactly T-10 UTC hour `18–23`, T-10 low and high trailing-volatility terciles, T-5 UTC hour `12–17`, and T-5 high trailing-volatility tercile. No weekend or forecast-decile cell qualified. These five cells show that some **volatility magnitude** structure remains and are a competing explanation, not direct evidence for OFI. Basis, settlement mechanics, and other market information also remain plausible. The V1/V1 plus P3 pattern and oracle results make a sigma-dominated explanation less compelling, while none of these diagnostics demonstrates a specifically directional or OFI-exploitable error.

Repository data make the scope consequential. Each of the three BTC one-second backfills (`kraken`, `crypto_com`, `bullish`) has 91 daily files from `2026-05-27` through `2026-08-25`; inspected pre-test Parquet schemas and the backfill output selections contain exactly `timestamp, exchange, price, volume, trade_count`, with **no retained trade side or aggressor side**. The Kalshi live recorder writes `ticker, observed_ts, yes_bid, yes_ask, yes_bid_size, yes_ask_size, last_price, liquidity, volume, volume_24h, open_interest` rather than underlying BTC aggressive trade side; its 11 saved files end `2026-09-15`. The roadmap's “reopens for free” premise did not hold. A future OFI experiment therefore requires a **new Kraken + Crypto.com backfill with trade side retained**. Bullish remains excluded because its early retention gap would create a chronological coverage discontinuity. The multi-day backfill, new side-aware data, and proximity to the one-shot Day 17 test rule out reopening within Days 14–16 on this evidence.

This is a documentation decision only: no OFI definition, lookback, feature, model, backfill, validation query, economic comparison, or test read was performed. The frozen V and P verdicts and all existing artifacts remain unchanged.

# Day 13 — HAR-RV Forward-Volatility Notes

This section consolidates the saved Day 13 artifacts and the earlier scratch readings without changing their pre-score interpretations or verdicts. It adds no fit, score, or economic query.

## Environment and forward target

The frozen contract used Python `3.13.7`, NumPy `2.5.2`, pandas `3.0.5`, pyarrow `25.0.1`, and scikit-learn `1.9.1` (transitive scipy `1.18.1`, joblib `1.6.0`, threadpoolctl `3.7.0`). The existing virtual environment still reports these versions. Day 12 did not preserve an environment record in `model_notes.md`; nothing new was installed on Day 13.

The quarantined `data/targets/forward_vol_target.parquet` contains 14,358 train and validation structural rows. `fwd_rv` is the annualized simple standard deviation of one-second cross-exchange VWAP-proxy log returns labelled in `[decision_time, close_time)`, with sample `ddof=1` and factor `sqrt(31,536,000)`. It uses the Day 6 engine's at-most-10-second forward fill. T-5 expects 300 returns and needs at least 240; T-10 expects 600 and needs at least 480. Below 80% coverage the target is null, with no imputation; otherwise `fwd_log_rv = ln(fwd_rv)`. No test row has a forward target. All forward values remain under `data/targets/`, outside model feature/design tables.

| Target population | Horizon | Common rows n | Null `fwd_rv`/`fwd_log_rv` | Target-bearing n |
| --- | --- | ---: | ---: | ---: |
| Train common | T-10 | 4,792 | 27 | 4,765 |
| Train common | T-5 | 4,757 | 62 | 4,695 |
| Validation common | T-10 | 1,685 | 41 | 1,644 |
| Validation common | T-5 | 1,649 | 61 | 1,588 |

The saved target and builder document four mandatory construction proofs. **A:** all 7,179 T-5 train/validation structural rows were compared with the trusted five-minute batch engine at the same close times; observation counts and null masks must agree exactly and value error must be at most `1e-12`. **B:** a seeded 120-row T-10 sample spans the train/validation period and uses independent Day 6 primitives for the complete 600-second return window; counts and masks must agree exactly and value error must be at most `1e-12`. **C:** a seeded 24-day boundary sample requires post-close price changes to leave targets unchanged and at least 95% of eligible inside-window perturbations to change them. **D:** all value columns have the `fwd_` prefix, the target has a separate path, and the existing safe-column guard must reject its value columns. The builder gates its write on all four passing and verifies an exact Parquet round trip; the saved artifact exists and its structural/null invariants agree with the protocol. **Record limitation:** the builder's achieved numeric A/B errors and C sample counts/change rate were printed at execution but were not persisted in the saved notes or artifact. They are not reconstructed or invented in this documentation step.

## Frozen fits, feed-through, and pre-score checks

The per-horizon OLS specifications were `ln fwd_rv = beta0 + beta_s ln 5min_vol + beta_m ln 1hr_vol + beta_l ln 24hr_vol + error` for HAR and `ln fwd_rv = alpha + beta ln 5min_ewma_vol + error` for N1. Each used `numpy.linalg.lstsq` with an intercept, without standardization, regularization, weighting, or Jensen/lognormal correction. H1 uses the six frozen walk-forward folds; H2 fits train only and scores validation. Null-target common rows leave only the OLS response fit; all common fit rows remain in the horizon-specific Platt calibration and all common score rows receive forecasts. The controlled path is `log_rv_hat → sigma_hat = exp(log_rv_hat) → existing Stage 0 closed form → fit-window Platt`; it changes sigma, not the rest of Stage 0.

| Fit configuration/population | Horizon | HAR/N1 target-bearing OLS fit n | All-common Platt fit n |
| --- | --- | ---: | ---: |
| H2 train-only, train common | T-10 | 4,765 | 4,792 |
| H2 train-only, train common | T-5 | 4,695 | 4,757 |
| H1 walk-forward folds 1–6, expanding common fits | T-10 | 2,936 / 3,573 / 4,184 / 4,765 / 5,310 / 5,891 | See the unchanged Step 2.2 fold table |
| H1 walk-forward folds 1–6, expanding common fits | T-5 | 2,903 / 3,529 / 4,125 / 4,695 / 5,226 / 5,797 | See the unchanged Step 2.2 fold table |

Before HAR mode, raw Stage 0 feed-through reproduced `p_5min_ewma_vol` with max error `0` (limit `1e-12`); H2 plumbing reproduced frozen train-Platt Stage 0 with max error `2.2204460492503131e-16` (limit `1e-10`); H1 plumbing reproduced fingerprint-verified Day 12 B1 with max error `1.5254186802593495e-12` (limit `1e-8`). Independent OLS cross-check maximum coefficient error was `0` (limit `1e-10`). The saved coefficient artifact's maximum HAR in-sample R² is `0.7834873138556784`, below the `0.9` leakage canary; every fitted Platt slope is positive (saved minimum `2.0543705071218668`). The second Step 2.1 run reproduced coefficients to the frozen `1e-12` tolerance and fingerprints exactly.

- Forecast fingerprint: `d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee07ee0b31aa472eeaa50acc9`.
- HAR-fed probability fingerprint: `e864a6377c8d520b04eae3dfb6664feaacdb5c35cb9708a6c0755b741d17fa0b`.

The **Step 2.2 coefficient reading above remains the pre-score record**, without performance-informed reinterpretation. It reports positive HAR slopes in all 6/6 H1 folds per term/horizon and slope sums just above one; subunit N1 EWMA-5 slopes; H1 fold 4–H2 coefficient equality with maximum difference `0`; modest folds 5–6 drift; HAR-fed Platt slopes near the frozen Stage 0 scale; and conceptual agreement between N1 compression and Day 12's stable positive `stage0_logit × log_ratio_5m_1h` interaction. Classical OLS SEs remain descriptive and understated under residual autocorrelation.

## Forward-volatility skill — H2 primary, H1 secondary

Primary loss is squared error in log annualized volatility. HAR-minus-benchmark paired differences use close-date-clustered SEs for the V verdict; the naive row SE and design effect are context. Mean residual is `fwd_log_rv − forecast` (positive means underforecasting). The table uses target-bearing **H2 validation common** rows only, with 21 close-date clusters per horizon.

| Population/configuration | Horizon | n | HAR MSE | N1 MSE | Naive-S0 MSE | Naive-15 MSE | HAR MAE | HAR mean residual | HAR R²_vs_naive | HAR−S0 mean; naive SE; clustered SE; design effect | HAR−N1 mean; clustered SE | Meaningful vs S0/N1 | V |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| H2 train-only, validation target-bearing | T-10 | 1,644 | 0.058826112 | 0.091657356 | 0.089998530 | 0.071419926 | 0.184528 | −0.012965 | 0.346366 | −0.031172418; 0.003138916; 0.004112423; 1.716470 | −0.032831244; 0.005616921 | Yes / yes | V1 |
| H2 train-only, validation target-bearing | T-5 | 1,588 | 0.075463723 | 0.096687468 | 0.099324391 | 0.091209495 | 0.210878 | +0.003923 | 0.240230 | −0.023860668; 0.003095726; 0.004099138; 1.753315 | −0.021223745; 0.004020634 | Yes / yes | V1 |

**T-10 = V1 — structure adds skill; T-5 = V1 — structure adds skill.** HAR meaningfully beat both Naive-S0 and N1 at each H2 horizon, so the multi-horizon structure added forward-volatility skill under the frozen day-clustered rule. The conventional H2 HAR fit-mean R² values were `0.844575` / `0.812714` at T-10/T-5: **inflated by regime shift; not a skill measure**. H1 all-OOF target-bearing rows (T-10 `n=3,473`, T-5 `n=3,380`) and the train-date folds 1–3 / validation-date folds 4–6 are secondary; their saved scores are above in Step 2.3. H1 folds 5–6 consume earlier validation forward volatility and outcomes in their expanding fits.

## Controlled sigma swap — H2 probability quality

H2 uses **all** 3,334 validation common rows, including forward-target nulls. The inherited paired per-row two-SE Brier rule controls P; log loss, AUC, ECE, market mid, and bucket results are secondary. Each row below is H2 train-only versus frozen train-Platt Stage 0 on identical validation keys.

| Population/configuration | Horizon | n | HAR / Stage 0 Brier | HAR−S0 paired diff; SE | Meaningful | HAR / S0 log loss | HAR / S0 AUC | HAR / S0 ECE | Market-mid Brier | S0-to-market gap closure |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |
| H2 train-only, validation common | T-10 | 1,685 | 0.189033084 / 0.189792806 | −0.000759722; 0.000603548 | No | 0.562951591 / 0.564414415 | 0.785220373 / 0.784119142 | 0.030709132 / 0.035595749 | 0.182157128 | +0.099496315 |
| H2 train-only, validation common | T-5 | 1,649 | 0.117061648 / 0.116474357 | +0.000587291; 0.000476165 | No | 0.379809030 / 0.378477904 | 0.913896614 / 0.914429288 | 0.024833848 / 0.021091839 | 0.108801868 | −0.076545037 |

Pooled H2 validation common (`n=3,334`) HAR / Stage 0 Brier was `0.153435934 / 0.153529422`, paired difference `−0.000093488` with SE `0.000385490`, also not meaningful. **P3 — no meaningful difference:** “Forecast σ did not improve Stage 0's probabilities; Stage 0 remains the baseline of record.” Ties go to Stage 0; secondary metrics do not change P3.

The H2 N1-fed diagnostic control, on the same validation common keys, had Brier `0.190027892` at T-10 (`n=1,685`), `0.116976624` at T-5 (`n=1,649`), and `0.153896656` pooled (`n=3,334`); none improved meaningfully on Stage 0 under the inherited rule. Frozen Stage 0 `|z| < 0.25` cells likewise had no established HAR gain: T-10 `n=879`, HAR−S0 Brier `−0.000244` (SE `0.000646`); T-5 `n=432`, `+0.001146` (SE `0.000940`). A T-10 quote-mid price bucket `[0.25,0.40)` (`n=317`) had diagnostic HAR−S0 Brier `−0.004315` (SE `0.001954`); this is a bucket observation, not a verdict or new filter. H1 versus B1 was separately secondary P3 on all OOF rows (`n=7,012`) and folds 4–6 validation-date rows (`n=3,334`); folds 5–6 fit using earlier validation forward targets and outcomes.

## Oracle headroom — diagnostic only; oracle is not a model

Oracle Platt was fit on train target-bearing common rows and scored only on the same target-bearing H2 validation subset used for the Stage 0/HAR numbers below. It is an unreachable sigma diagnostic, not a deployable or row-level probability artifact.

| Population/configuration | Horizon | n | Stage 0 subset Brier | HAR subset Brier | Oracle Brier | Raw oracle gap S0−oracle | HAR improvement S0−HAR | Headroom captured |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 validation target-bearing, oracle diagnostic | T-10 | 1,644 | 0.190329179 | 0.189576542 | 0.188143726 | 0.002185453 | +0.000752637 | +0.344385 |
| H2 validation target-bearing, oracle diagnostic | T-5 | 1,588 | 0.117578573 | 0.118065749 | 0.117135129 | 0.000443444 | −0.000487176 | −1.098620 |

T-10 retains some sigma-related probability headroom, and HAR captured part of it. T-5's raw oracle gap is very small; its negative captured fraction is denominator-sensitive and must not be read alone. The oracle was never thresholded or traded.

## Residual structure, OFI, and limits of the result

The H1 target-bearing residual is `fwd_log_rv − log_rv_har`; formal structure needs non-thin same-sign train-fold 1–3 and validation-fold 4–6 cells, each beyond two day-clustered SEs. The complete Step 3.2 tables and frozen decile boundaries remain above.

| H1 target-bearing cut | Horizon | Train folds 1–3 n; sign | Validation folds 4–6 n; sign | Formal structure |
| --- | --- | --- | --- | --- |
| UTC hour 18–23 | T-10 | 486; negative | 470; negative | Yes |
| Trailing-volatility low tercile | T-10 | 968; negative | 1,312; negative | Yes |
| Trailing-volatility high tercile | T-10 | 303; positive | 62; positive | Yes |
| UTC hour 12–17 | T-5 | 490; positive | 463; positive | Yes |
| Trailing-volatility high tercile | T-5 | 316; positive | 65; positive | Yes |

No weekend or HAR forecast-decile cell qualified; the forecast-decile train patterns did not persist across the boundary. Within-day lag-1 residual autocorrelation at T-10 was modest positive and similar on train/validation (`+0.092852 / +0.102924`); at T-5 it largely vanished on validation (`+0.048890 / −0.002701`). No residual cut became a feature or filter.

**OFI verdict: Reopen as a scoped post-Day-17 experiment.** The V1/V1 plus P3 combination, small T-5 oracle headroom, persistent HAR-to-market Brier gaps (`0.006876` T-10 and `0.008260` T-5), unresolved low-`|z|` weakness, and Day 11 market-fair P&L motivate a separately pre-registered directional investigation. The five HAR residual cells concern **volatility magnitude**, not OFI direction; basis, settlement mechanics, and remaining volatility structure remain competing explanations. Stored BTC one-second data have no trade/aggressor side, so OFI did not reopen “for free.” A later experiment needs a **new Kraken + Crypto.com backfill with side retained**; Bullish stays excluded for chronological coverage discontinuity. OFI is not validated and does not enter Days 14–16.

Day 13 establishes whether frozen linear HAR beat Stage 0's trailing-sigma assumption for forward proxy realized volatility, whether that sigma swap materially improved calibrated probabilities on the frozen three-week validation window, and where pre-registered H1 residual structure persisted across the boundary. It establishes **no** net economic or post-fee superiority (Day 14), test-period performance (Day 17), gradient-boosting justification (Day 14's gate), OFI efficacy, or identification of basis or settlement mechanics as the remaining cause. The HAR-fed economic-comparison specification was frozen, with execution deferred to Day 14.

For the Day 14 gate, five cross-boundary residual cells and high trailing-volatility underforecasting at both horizons suggest a nonlinear volatility model may find structure; Day 12's stable positive ratio interaction is prior related evidence. P3 despite V1/V1, the especially small T-5 oracle probability gap, and absent stable forecast-decile shape limit the likely probability payoff. This records balanced gate evidence without deciding or fitting a boosted model.

**Deliberate anti-iteration exclusions:** reduced-horizon HAR, EWMA-input HAR, HAR-fed logistic, Jensen/lognormal bias correction, and a market-price model were possible *separately pre-registered* experiments but were **not attempted on Day 13**. No new residual-cut feature, OFI model, economic threshold, test-period adjustment, or post-score HAR specification change was attempted. These are recorded exclusions, not forgotten work.
