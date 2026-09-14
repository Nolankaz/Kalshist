# Day 9 Stage 0 Selection, Calibration, and Basis-Risk Notes

These notes record the completed Day 9 analysis through Part 3.2. All reported values come from stored Day 9 artifacts and use the established train and validation common populations. No test outcomes are used.

## 1. Selected Stage 0 sigma

The selected Stage 0 candidate is `5min_ewma_vol`. It was also the numerically best joint-Brier candidate. All ten candidates passed the frozen 80% coverage floor, and `5min_ewma_vol` was the only candidate in the final tie set.

The frozen selection rule—joint Brier, paired 2-SE, shortest window, then simple unless meaningfully worse—was applied without modification. The simple `5min_vol` estimator was meaningfully worse than the winner, so the simplicity tie-break could not override the EWMA result.

## 2. Raw calibration diagnosis

Raw `5min_ewma_vol` Stage 0 probabilities were under-confident on validation at both horizons: probabilities were too compressed toward 0.5, with insufficient separation between low- and high-probability markets.

| Horizon | Reliability bins | Observed YES frequencies | Inversions | Diagonal outside Wilson interval |
|---|---:|---|---:|---:|
| T-10 | 10 | Monotone increasing | 0 | 8/10 |
| T-5 | 10 | Monotone increasing | 0 | 9/10 |

## 3. Train-fitted Platt calibration

The legitimate train-fitted Platt parameters are:

| Horizon | Fit rows | `a` | `b` | Iterations | Final gradient norm |
|---|---:|---:|---:|---:|---:|
| T-10 | 4,792 | 0.009440 | 2.161922 | 5 | 2.453e-13 |
| T-5 | 4,757 | 0.052819 | 2.500293 | 6 | 6.940e-14 |

Both slopes are greater than 1, consistent with stretching the under-confident probabilities away from 0.5 while preserving their ordering.

The train-fitted parameters have `fit_split == "train"` and `parameter_role == "legitimate_train_fit"`; these are the legitimate, shippable calibration parameters. The validation-fitted parameters have `parameter_role == "validation_in_sample_ceiling"` and are diagnostic in-sample ceilings only. They are not shippable and were not used for the reported calibrated validation performance.

## 4. Validation performance after calibration

All metrics below use the established validation/common population.

| Horizon | n | Raw Brier | Platt Brier | Market Brier | Raw log loss | Platt log loss | Raw AUC | Platt AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T-10 | 1,685 | 0.201599 | 0.189793 | 0.182157 | 0.591030 | 0.564414 | 0.784119 | 0.784119 |
| T-5 | 1,649 | 0.137512 | 0.116474 | 0.108802 | 0.438009 | 0.378478 | 0.914429 | 0.914429 |

Train-fitted Platt calibration improved Brier and log loss at both horizons. AUC was unchanged because the positive-slope transformation preserved model ordering. Market `quote_mid` still had lower validation/common Brier than both raw and Platt-calibrated Stage 0 at T-10 and T-5.

## 5. Basis-risk propagation

Channel A measures the median maximum absolute probability shift under symmetric basis perturbations. Values are percentage points of probability.

| Horizon | Basis magnitude | Raw median max shift | Train-fitted Platt median max shift |
|---|---:|---:|---:|
| T-10 | 1.2 bps | 2.2087 pp | 4.1154 pp |
| T-10 | 5.0 bps | 9.3797 pp | 18.0886 pp |
| T-5 | 1.2 bps | 2.8616 pp | 4.1430 pp |
| T-5 | 5.0 bps | 12.4831 pp | 22.7983 pp |

T-5 is more basis-sensitive than T-10 at both perturbation sizes. Platt calibration increases probability sensitivity because it stretches probabilities away from 0.5. A 1.2 bp basis is non-negligible, and a 5 bp basis is material.

## 6. Settlement proximity

Channel B measures the realized settlement distance from strike independently of Channel A.

| Population | Markets | Median settlement distance | Within 1.2 bps | Within 5 bps |
|---|---:|---:|---:|---:|
| Train | 5,189 | 10.1358 bps | 6.65% | 27.02% |
| Validation | 1,980 | 6.5350 bps | 11.01% | 41.62% |
| Train + validation | 7,169 | 9.0277 bps | 7.85% | 31.05% |

On validation, 11.01% of markets settled within 1.2 bps of strike and 41.62% settled within 5 bps. Settlement proximity and probability sensitivity are separate diagnostics; no synthetic label-flip probability was constructed.

## 7. Model-vs-market disagreement

Positive signed gap means the model assigns a higher YES probability than market `quote_mid`. Gap values are percentage points of probability.

| Horizon | Version | Median absolute gap | Mean signed gap | Spearman correlation |
|---|---|---:|---:|---:|
| T-10 | Raw | 9.0358 pp | -0.9388 pp | 0.951857 |
| T-10 | Train-fitted Platt | 5.2346 pp | -1.6862 pp | 0.951857 |
| T-5 | Raw | 13.0346 pp | -1.1102 pp | 0.967016 |
| T-5 | Train-fitted Platt | 3.8361 pp | -0.9254 pp | 0.967016 |

The model and market rank markets very similarly. Platt calibration substantially reduced median probability disagreement and did not change ranking. Mean signed gaps are much smaller than median absolute gaps, so the disagreement is not simply a one-sided bias.

## 8. Basis risk relative to model-market gap

The ratios below compare validation median basis-sensitivity shifts with the corresponding median absolute model-market gaps.

| Horizon | Version | 1.2 bp sensitivity / gap | 5 bp sensitivity / gap |
|---|---|---:|---:|
| T-10 | Raw | 24.4% | 103.8% |
| T-10 | Train-fitted Platt | 78.6% | 345.6% |
| T-5 | Raw | 22.0% | 95.8% |
| T-5 | Train-fitted Platt | 108.0% | 594.3% |

A 1.2 bp basis sensitivity is smaller than the raw gap but approaches or exceeds the Platt gap. A 5 bp sensitivity is approximately equal to the raw gap and several times the Platt gap.

## 9. What Day 9 establishes

1. **Beats constant benchmark: supported.** Raw validation/common Brier was below the constant benchmark at both horizons.
2. **Predicts the target: supported.** Validation discrimination and reliability results show meaningful predictive structure, and Platt calibration improves probability quality.
3. **Beats the market: not supported.** Market `quote_mid` still has lower validation/common Brier than both raw and train-fitted Platt Stage 0.
4. **Tradable edge: not established.** Day 9 evaluates probability quality, calibration, basis sensitivity, settlement proximity, and model-market disagreement—not executable trading returns.

**Model-market disagreement is not the same as tradable edge.** Transaction costs, spreads, execution, and trading thresholds remain for later days.
