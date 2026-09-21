# Stage 0 Tier-2 Backtest Notes

## Day 11 — Section 2.2: Ordered train and validation evaluation

### Frozen evaluation state

The Stage 0 trade decisions were frozen before settlement outcomes were loaded.

Frozen decision fingerprint:

`f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96`

Frozen trade counts:

| Split | Basis | Trades |
| --- | ---: | ---: |
| Train | 1.2 bps | 1,881 |
| Validation | 1.2 bps | 719 |
| Train | 5.0 bps | 277 |
| Validation | 5.0 bps | 100 |

The 1.2 bps threshold is the primary Stage 0 result. The 5.0 bps threshold is a conservative sensitivity result only.

The test split remains untouched.

### Accounting assertion correction

During the first train-only Section 2.1 execution, the aggregate headline fee-reconciliation assertion used the single-row floating-point tolerance on a sum across many trades.

The original aggregate check effectively allowed only:

`EDGE_TOLERANCE`

for the summed reconciliation.

It was changed to allow:

`EDGE_TOLERANCE * n_trades`

for each aggregate row.

This was an assertion-tolerance correction only. It did not change:

- the frozen trade list
- the decision fingerprint
- trade selection
- entry prices
- fees
- model probabilities
- thresholds
- payoff formulas
- gross P&L
- net P&L
- any individual trade result

The corrected train run passed all row-level accounting, fee, posted-cash, settlement-join, and Parquet round-trip checks.

## Train headline

The train split is in-sample for the Platt calibrator and is not the primary Stage 0 evaluation result.

```text
PASS: self_check_accounting — four YES/NO payoff and Direct Member fee examples
PASS: frozen decision fingerprint f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96 and all four trade counts verified
PASS: train-only settlement projection and Parquet close_date filter (2026-05-26 through 2026-07-19)
PASS: one-to-one settlement join for 2,158 frozen trades
PASS: accounting identities and fresh posted-cash checks for 2,158 train trades
PASS: accounting identities and fresh posted-cash checks for 2,158 train trades
PASS: train results Parquet round trip (data/backtest/stage0_results_train.parquet)

Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.
split  basis_bps           threshold_role                                                                                      tier_label  n_trades  n_distinct_markets  n_days_with_trade  n_population_days  n_t10  n_t5  n_yes  n_no  hit_rate  mean_entry_price  breakeven_hit_rate  gross_pnl    fees  fee_rounding_component  net_pnl  net_pnl_per_trade  mean_net_edge  model_expected_total_pnl  market_fair_expected_total_pnl  mean_surprise  total_capital  net_pnl_over_capital
train        1.2                  primary Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.      1881                1881                 54                 54    890   991    840  1041  0.368421          0.364951             0.37460      6.528 18.1511                0.091890 -11.6231          -0.006179       0.109527                206.019591                        -25.7611      -0.115706       704.6231             -0.016495
train        5.0 conservative_sensitivity Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.       277                 277                 46                 54     81   196    106   171  0.079422          0.083065             0.08826     -1.009  1.4391                0.012865  -2.4481          -0.008838       0.178655                 49.487554                         -2.0276      -0.187493        24.4481             -0.100135
```

## Validation headline

This is the pre-registered Stage 0 validation evaluation under the frozen Tier-2 assumptions.

The validation split had already contributed calibration-error information to the Day 10 execution threshold, so this is not a completely untouched holdout. The test split remains untouched for the later final evaluation.

```text
PASS: self_check_accounting — four YES/NO payoff and Direct Member fee examples
PASS: frozen decision fingerprint f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96 and all four trade counts verified
PASS: validation-only settlement projection and Parquet close_date filter (2026-07-20 through 2026-08-09)
PASS: one-to-one settlement join for 819 frozen trades
PASS: accounting identities and fresh posted-cash checks for 819 validation trades
PASS: accounting identities and fresh posted-cash checks for 819 validation trades
PASS: validation results Parquet round trip (data/backtest/stage0_results_validation.parquet)

Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.
     split  basis_bps           threshold_role                                                                                      tier_label  n_trades  n_distinct_markets  n_days_with_trade  n_population_days  n_t10  n_t5  n_yes  n_no  hit_rate  mean_entry_price  breakeven_hit_rate  gross_pnl   fees  fee_rounding_component  net_pnl  net_pnl_per_trade  mean_net_edge  model_expected_total_pnl  market_fair_expected_total_pnl  mean_surprise  total_capital  net_pnl_over_capital
validation        1.2                  primary Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.       719                 719                 21                 21    368   351    243   476  0.432545          0.439889            0.450256      -5.28 7.4539                0.035406 -12.7339          -0.017711       0.114469                 82.302976                        -10.2199      -0.132179       323.7339             -0.039334
validation        5.0 conservative_sensitivity Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.       100                 100                 14                 21     31    69     26    74  0.080000          0.098300            0.104081      -1.83 0.5781                0.004714  -2.4081          -0.024081       0.185164                 18.516391                         -0.7741      -0.209245        10.4081             -0.231368
```

## Day 11 — Section 3.2: Final findings

### 1. What was evaluated

Stage 0 used `p_5min_ewma_vol` (the `5min_ewma_vol` sigma candidate) with horizon-specific, train-fitted Platt calibration. The frozen `1.2 bps` threshold is primary; `5.0 bps` is a conservative sensitivity, not an alternative selected from outcomes. Each selected market received one contract at the historical top-of-book taker price and was held to Kalshi settlement. The one-position-per-market rule selected the earliest clearing decision: T-10 if it cleared, otherwise a clearing T-5 row.

**Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement.** Historical bid/ask quotes, the frozen Direct Member one-fill fee model, and actual Kalshi settlement enter the accounting. Displayed depth, fillable size, latency, and live slippage were not observed or modeled. These are optimistic historical top-of-book fills, not demonstrated executable capacity.

### 2. Evaluation sequence and integrity

The Day 11 protocol was frozen first. The outcome-free builder reproduced the Day 10 candidate-population and clearance counts, selected trades chronologically, and fingerprinted the decision list before settlement outcomes were joined. The backtest then ran train before validation. The Section 2.3 breakdowns and Section 3.1 traded-day-clustered analysis consumed the resulting frozen train/validation files. No Day 11 analysis read the test split.

The decision fingerprint recomputed from `stage0_trade_decisions.parquet` is `f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96`.

The train-only aggregate assertion correction documented above changed the gross-to-net reconciliation tolerance from effectively `EDGE_TOLERANCE` for an entire sum to `EDGE_TOLERANCE * n_trades`. It did **not** change the trades, fingerprint, entry prices, fees, probabilities, thresholds, payoff formulas, gross P&L, net P&L, or any individual result row.

### 3. Outcome-free frozen trade list

The following counts come from the decision Parquet, before outcome fields were joined. “Later horizon suppressed” means another clearing horizon was present but the market already had its first selected entry; “opposite side” counts suppressed clearing rows whose side differed from the selected row.

| Split | Basis (bps) | Trades | T-10 | T-5 | YES | NO | Later horizon suppressed | Of those, opposite side |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1.2 primary | 1,881 | 890 | 991 | 840 | 1,041 | 366 | 54 |
| Train | 5.0 sensitivity | 277 | 81 | 196 | 106 | 171 | 21 | 1 |
| Validation | 1.2 primary | 719 | 368 | 351 | 243 | 476 | 109 | 23 |
| Validation | 5.0 sensitivity | 100 | 31 | 69 | 26 | 74 | 4 | 0 |

The saved `price_bucket` counts in frozen bucket order are train 1.2: `319 / 363 / 189 / 128 / 204 / 380 / 298`; train 5.0: `98 / 7 / 1 / 0 / 0 / 43 / 128`; validation 1.2: `101 / 129 / 99 / 64 / 86 / 126 / 114`; validation 5.0: `24 / 2 / 0 / 1 / 3 / 17 / 53`. These are descriptions of the selected books, not bucket-level strategy choices. The carried `net_edge` is the model-expected reference; `market_fair_expected` is the midpoint-based reference, both net of the same executable entry price and fee.

### 4. Headline train and validation results

All monetary columns below are dollars for one-contract trades (numerically probability units). **Tier 2 — quote-aware, top-of-book, size-unaware.** Train is in-sample for the Platt calibrator; 5.0 bps remains sensitivity only. “Trading days” means `close_date` days with at least one trade; “population days” includes zero-trade dates in the frozen common-population calendar.

| Split | Basis (bps) | Trades | Population / trading days | Hit rate | Break-even hit rate | Gross P&L | Trade fee | Rounding less rebate | Total fees | Net P&L | Net/trade |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1.2 primary | 1,881 | 54 / 54 | 0.368421 | 0.374600 | 6.5280 | 18.059210 | 0.091890 | 18.1511 | -11.6231 | -0.00617921 |
| Train | 5.0 sensitivity | 277 | 54 / 46 | 0.079422 | 0.088260 | -1.0090 | 1.426235 | 0.012865 | 1.4391 | -2.4481 | -0.00883791 |
| Validation | 1.2 primary | 719 | 21 / 21 | 0.432545 | 0.450256 | -5.2800 | 7.418494 | 0.035406 | 7.4539 | -12.7339 | -0.01771057 |
| Validation | 5.0 sensitivity | 100 | 21 / 14 | 0.080000 | 0.104081 | -1.8300 | 0.573386 | 0.004714 | 0.5781 | -2.4081 | -0.02408100 |

The `trade_fee` plus `rounding_adjustment - rebate` equals the separately reported total `fee`; the rounding component is not an extra charge. **Tier 2 — quote-aware, top-of-book, size-unaware.**

| Split | Basis (bps) | Mean net edge / model expected per trade | Model-expected total P&L | Market-fair expected total P&L | Mean surprise | Total capital | Net P&L / capital |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1.2 primary | 0.109526630 | 206.019591 | -25.7611 | -0.115705843 | 704.6231 | -0.016495 |
| Train | 5.0 sensitivity | 0.178655430 | 49.487554 | -2.0276 | -0.187493336 | 24.4481 | -0.100135 |
| Validation | 1.2 primary | 0.114468673 | 82.302976 | -10.2199 | -0.132179243 | 323.7339 | -0.039334 |
| Validation | 5.0 sensitivity | 0.185163912 | 18.516391 | -0.7741 | -0.209244912 | 10.4081 | -0.231368 |

### 5. Primary validation reference comparison

For validation 1.2 bps, realized net P&L was **-$0.017710570 per trade**. Model-expected net edge was **+$0.114468673 per trade**, while market-fair expected net P&L was **-$0.014214047 per trade**. Realized performance was much closer to the market-fair reference than to the model-expected reference. The per-trade identity is `realized net P&L = model-expected net edge + mean surprise`: `-0.017710570 = 0.114468673 + (-0.132179243)`, subject only to displayed rounding. The negative surprise describes the model's error on the rows it selected, not a new trade-selection rule. **Tier 2 — quote-aware, top-of-book, size-unaware.**

### 6. Pre-registered breakdown diagnostics

These are descriptive cells from `stage0_summary.parquet` for primary validation 1.2 bps; none was promoted into a strategy filter. **Tier 2 — quote-aware, top-of-book, size-unaware.**

- Horizon: T-10 had 368 trades and -$13.0871 net P&L; the 351 conditional later T-5 entries had +$0.3532. T-5 exists only when that market was not already traded at T-10, so this is not a clean T-10-versus-T-5 experiment.
- Side: YES had 243 trades and -$5.7245; NO had 476 trades and -$7.0094. Both sides contributed to the negative total.
- Frozen price buckets (`p00_10` through `p90_100`): net P&L in order was `+$0.5604 / -$4.8299 / -$6.6732 / +$2.5310 / -$8.5344 / +$6.4491 / -$2.2369`. Signs vary across adjacent cells; the largest negative dollar contribution here was `p60_75`, but this is not a bucket-selection result.
- Frozen absolute-z buckets: `[0, 0.25)` had 258 trades and -$20.3721; `[0.25, 0.5)` had 216 and +$0.6026; `[0.5, 1.0)` had 204 and +$6.1694; `[1.0, infinity)` had 41 and +$0.8662. The low-|z| cell accounted for the largest negative dollar contribution, with offsets elsewhere; no |z| threshold was changed.
- UTC decision-hour blocks `00-05 / 06-11 / 12-17 / 18-23`: net P&L was `-$5.6154 / -$4.2544 / +$2.9451 / -$5.8092`. The full 24-hour table is stored but is a sparse, non-primary interpretation surface.
- Monday-start validation weeks `2026-07-20 / 2026-07-27 / 2026-08-03`: net P&L was `+$6.3894 / -$7.9974 / -$11.1259`. Two of three weeks matched the total's negative sign.

Each summary cell carries `thin = (n_trades < 30)`. No primary-validation cell in the six displayed breakdowns is thin, but thin cells elsewhere—especially individual hours and the 5.0 bps sensitivity—are descriptive only and should not be interpreted. Excluding the pre-identified first partial train week (`2026-05-27` through `2026-05-31`) leaves 1,585 train primary trades, 49 days, and -$9.2099 net P&L (-$0.005811 per trade); the train 5.0-bps sensitivity leaves 151 trades, 41 days, and +$0.0865. These exclusions do not replace the full train books and do not apply to validation. **Tier 2 — quote-aware, top-of-book, size-unaware.**

### 7. Correlation-aware uncertainty

The naive SE uses trade-level sample SD (`ddof=1`); the clustered SE uses traded `close_date` days in the frozen protocol formula. Intervals are exactly mean ±2 SE, not confidence intervals with a fitted critical value. **Tier 2 — quote-aware, top-of-book, size-unaware.**

| Split | Basis (bps) | Naive SE | Clustered SE | Naive ±2SE | Clustered ±2SE | Design effect | SE inflation | N_eff |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Train | 1.2 primary | 0.008149837 | 0.008963755 | [-0.022478887, 0.010120461] | [-0.024106723, 0.011748297] | 1.209712 | 1.099869 | 1,554.915 |
| Train | 5.0 sensitivity | 0.015971069 | 0.017403649 | [-0.040780045, 0.023104233] | [-0.043645204, 0.025969392] | 1.187443 | 1.089698 | 233.274 |
| Validation | 1.2 primary | 0.013393751 | 0.012277195 | [-0.044498073, 0.009076933] | [-0.042264960, 0.006843819] | 0.840222 | 0.916636 | 855.727 |
| Validation | 5.0 sensitivity | 0.026534840 | 0.023905295 | [-0.077150681, 0.028988681] | [-0.071891590, 0.023729590] | 0.811625 | 0.900902 | 123.210 |

The primary validation book has 719 trades on 21 population days and 21 traded-day clusters. Its clustered interval crosses zero. A design effect below one is permitted: here the day-clustered variance estimate is slightly lower than the naive trade-level estimate. `N_eff` is an algebraic design-effect equivalent, **not** a claim that more than 719 trades occurred or that there were more than 21 day clusters. With only 21 clusters, the clustered SE may itself be noisy. The 5.0-bps intervals are sensitivity diagnostics, not the basis for the primary category.

### 8. Daily and weekly concentration

For primary validation 1.2 bps, mean daily net P&L across all 21 population days was **-$0.606376**; sample SD was **$1.995355**. Positive trading-day share was **9/21 = 0.428571** (denominator: days with trades). The highest day was `2026-07-25` at **+$2.9082**; the lowest was `2026-08-05` at **-$4.6410**. The largest absolute daily contribution divided by absolute total net P&L was **0.364460**. These ratios are not capped at one, because offsetting periods can make a single contribution larger than the total. **Tier 2 — quote-aware, top-of-book, size-unaware.**

| Monday-start week | Trades | Net P&L | Net P&L/trade |
| --- | ---: | ---: | ---: |
| 2026-07-20 | 223 | +$6.3894 | +$0.028652 |
| 2026-07-27 | 250 | -$7.9974 | -$0.031990 |
| 2026-08-03 | 246 | -$11.1259 | -$0.045227 |

Two of the three validation weeks had the same sign as the negative total. The largest absolute week's contribution divided by absolute total net P&L was **0.873723**. These are concentration diagnostics, not a week-clustered inferential calculation. **Tier 2 — quote-aware, top-of-book, size-unaware.**

### 9. Frozen conclusion

**No tier-2 evidence of edge either way; not established.**

The primary point estimate was negative, but its pre-registered traded-day-clustered ±2SE interval included zero. Validation is not a completely untouched holdout: its calibration-error information contributed to the Day 10 threshold. The test split remains untouched. This is the frozen category; no outcome-derived parameter or filter was chosen.

### 10. What Day 11 establishes—and does not

Day 11 establishes a reproducible Tier-2 historical evaluation using recorded bid/ask inputs, frozen one-contract fee accounting, actual Kalshi settlement, decisions fingerprinted before outcomes, reconciled realized results, descriptive breakdowns, complete daily accounting, and a pre-registered day-clustered interval.

It does **not** establish depth-aware fills, size capacity, latency, slippage, live or maker execution, exits, dynamic sizing, Sharpe, drawdown, a regime-aware production edge, or untouched test performance. The settlement-index basis relative to the BTC exchange proxy remains a modeling limitation. None of these omissions is repaired by a favorable individual bucket or week.

### 11. Observations for later days, without changing Stage 0

Day 12 fitted models must improve behavior on selected disagreement rows, not merely improve global calibration, if they are to close the observed model-expected versus realized gap. The low-|z| and other pre-registered diagnostic patterns can be revisited in later, separately planned sensitivity or regime work. These observations do not modify today's probability, threshold, fee, trade list, or execution assumptions and are not recommendations to trade a subset.
