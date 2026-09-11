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
