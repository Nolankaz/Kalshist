Realized volatility definition:

For a timestamp t and lookback window W:

1. Pull BTC VWAP proxy prices from the window ending at t.
2. Compute consecutive log returns:
   r_t = ln(price_t / price_(t-1))
3. Compute the standard deviation of those log returns.
4. Annualize:
   annualized_vol = std(log_returns) * sqrt(31,536,000)
   because the underlying bars are 1-second bars.
5. Repeat for:
   5min
   15min
   1hr
   4hr
   24hr


## Realized volatility data-quality rule

Realized volatility features are computed on a 1-second wall-clock BTC VWAP proxy.

The observed cross-exchange VWAP proxy is reindexed to a complete 1-second UTC grid. Missing prices are previous-tick forward-filled for at most 10 seconds. Gaps longer than 10 seconds remain missing so stale prices are not treated as current.

Features at prediction timestamp t use returns in [t - window, t), so no data at or after t is used.

For each window, n_obs records the number of usable 1-second returns and coverage is:

coverage = n_obs / expected_returns

Expected returns:
5min: 300
15min: 900
1hr: 3600
4hr: 14400
24hr: 86400

A minimum coverage threshold of 80% is required. Windows below 80% coverage return NaN for both simple and EWMA realized volatility rather than silently producing a weak estimate.

Simple realized volatility uses sample standard deviation (ddof=1) of log returns and annualizes by sqrt(31,536,000).

EWMA volatility uses exponentially weighted squared log returns with the configured per-window half-life, takes the square root, and annualizes by the same factor.


## Day 4 — Realized Volatility Features

### Implemented features

Realized volatility is computed for five lookback windows:

- 5 minutes
- 15 minutes
- 1 hour
- 4 hours
- 24 hours

For each window, two estimators are stored:

- Simple realized volatility
- EWMA realized volatility

This produces 10 volatility features per timestamp.

### Price series used

Volatility is computed from the validated cross-exchange BTC VWAP proxy built from:

- Bullish
- Kraken
- Crypto.com

The proxy is constructed from the stored 1-second exchange price/volume buckets.

Observed proxy prices are reindexed onto a complete 1-second UTC wall-clock grid.

Missing proxy prices are forward-filled for at most 10 seconds. Gaps longer than 10 seconds remain missing so stale prices are not treated as current.

### Window boundaries / leakage rule

A feature evaluated at timestamp t uses only returns in:

[t - window, t)

The left boundary is included and t itself is excluded.

No price data at or after t is used in the feature.

### Simple realized volatility

For valid 1-second proxy prices:

1. Compute log returns:

   r_t = ln(P_t / P_(t-1))

2. Compute sample standard deviation using ddof=1.

3. Annualize using:

   annualized_vol = std(log_returns) * sqrt(31,536,000)

because the underlying return interval is one second.

### EWMA realized volatility

EWMA volatility uses the same 1-second log returns as simple realized volatility.

Squared returns are exponentially weighted so newer observations receive more weight than older observations.

Configured half-lives:

- 5min window: 2.5 minutes
- 15min window: 7.5 minutes
- 1hr window: 30 minutes
- 4hr window: 2 hours
- 24hr window: 12 hours

The EWMA variance estimate is square-rooted and annualized with the same 1-second annualization factor.

### Missing-data / coverage rule

For each window, n_obs stores the number of usable 1-second returns.

Expected full-window observation counts:

- 5min: 300
- 15min: 900
- 1hr: 3,600
- 4hr: 14,400
- 24hr: 86,400

Coverage is defined as:

coverage = n_obs / expected_observations

A minimum coverage threshold of 80% is required.

If coverage is below 80%, both the simple and EWMA volatility values for that window are returned as NaN rather than silently producing a weak estimate.

### Observed ranges on 300-timestamp validation sample

Simple realized volatility:

- 5min: min 8.78%, median 54.13%, max 163.48%
- 15min: min 9.02%, median 55.08%, max 165.02%
- 1hr: min 14.47%, median 56.33%, max 178.01%
- 4hr: min 19.39%, median 56.66%, max 116.06%
- 24hr: min 21.73%, median 57.81%, max 107.09%

EWMA realized volatility:

- 5min: min 8.76%, median 53.64%, max 165.65%
- 15min: min 8.62%, median 54.56%, max 164.56%
- 1hr: min 14.24%, median 56.10%, max 184.41%
- 4hr: min 19.29%, median 55.72%, max 131.78%
- 24hr: min 21.60%, median 57.05%, max 109.51%

### Validation results

Independent 15-minute manual check:

- Manual vol: 0.6397940574
- Function vol: 0.6397940574
- Difference: 0.0000000000

300-timestamp validation sample:

- Negative volatility values: 0
- Non-finite volatility values: 0

NaNs after the 80% coverage rule:

- 5min: 17
- 15min: 9
- 1hr: 9
- 4hr: 9
- 24hr: 0

Longer windows were visibly smoother than shorter windows in the time-series comparison plot.

### EWMA behavior check

EWMA behaved as expected:

- EWMA > simple vol in windows where recent volatility increased.
- EWMA < simple vol in windows where recent volatility decreased.
- EWMA ≈ simple vol in relatively stable windows.

Examples:

- 5min, 2026-08-20 13:14:37 UTC:
  simple = 91.22%, EWMA = 103.22%

- 5min stable example, 2026-07-10 19:35:06 UTC:
  simple = 90.37%, EWMA = 90.38%

- 4hr, 2026-06-25 14:22:04 UTC:
  simple = 116.06%, EWMA = 131.78%

### Function interface

Implementation file:

scripts/realized_vol.py

Single-window simple realized vol:

realized_vol(t, window)

Single-window EWMA realized vol:

ewma_realized_vol(t, window)

All-window optimized feature wrapper:

realized_vol_features(t)

Example:

features = realized_vol_features(
    pd.Timestamp("2026-07-01 12:00:00", tz="UTC")
)

The wrapper computes all five windows from one maximum 24-hour proxy load and returns simple vol, EWMA vol, n_obs, and coverage for each window.

### Open issues / future review

- The 10-second maximum forward-fill limit is currently a chosen baseline and may be revisited later if model performance or data-quality analysis suggests a better threshold.
- The 80% minimum coverage threshold is a baseline rule chosen after examining the 300-timestamp coverage distribution.
- EWMA half-lives may be tuned later, but should not be changed without re-running validation.