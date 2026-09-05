# Kalshi Quote Notes

## Historical Candlestick Endpoints

KXBTC15M candlesticks are split across two public endpoints:

- `https://external-api.kalshi.com/trade-api/v2/historical/markets/{ticker}/candlesticks`
- `https://external-api.kalshi.com/trade-api/v2/series/KXBTC15M/markets/{ticker}/candlesticks`

Older markets are available from the historical endpoint and newer markets from
the series endpoint. The crossover was confirmed experimentally in late June
2026. The implemented routing boundary is `2026-06-29 00:00 UTC`: markets that
close before it try historical first, and markets at or after it try series
first. Stored rows run historical through June 28 and series from June 29.

The boundary is an efficiency rule, not an assumption of correctness. An HTTP
404 from the preferred endpoint triggers a request to the other endpoint. Rate
limits, server errors, and network errors are retried on the same endpoint and
do not trigger endpoint fallback.

## Response Field Names

Historical OHLC objects use:

```text
open, high, low, close
volume
open_interest
```

Series OHLC objects use:

```text
open_dollars, high_dollars, low_dollars, close_dollars
volume_fp
open_interest_fp
```

The backfill normalizes both responses into the common schema documented in
`schema.md`. Missing or unknown field names and non-null nonnumeric values raise
errors instead of becoming silent nulls. Legitimate null trade-price OHLC values
for no-trade minutes are preserved as missing; bid and ask OHLC values remain
strictly numeric.

## Historical Coverage and Validation

Final validation for `2026-05-26` through `2026-08-24`:

- Settled markets: 8,597
- Markets with at least one candle: 8,591
- Zero-candle markets: 6
- Markets with all 15 candles: 8,586
- Total quote rows: 128,820
- Historical endpoint rows: 48,066
- Series endpoint rows: 80,754
- Bid/ask close violations: 0
- Price values outside `[0, 1]`: 0
- Entire missing close dates: 0 across 91 dates

Candle-count distribution (`candles: markets`):

```text
0: 6
2: 1
3: 1
5: 1
9: 1
11: 1
15: 8,586
```

The strict validator reports `FAIL` for perfect market and 15-candle
completeness because of the verified upstream API gaps below. The bid/ask,
price-range, date-coverage, and endpoint-routing checks found no data-integrity
errors.

## Explicit Coverage Exceptions

The following partial historical markets were independently re-requested:

- `KXBTC15M-26JUN070315-15`: 3 candles
- `KXBTC15M-26JUN102100-00`: 2 candles
- `KXBTC15M-26JUN102115-15`: 5 candles
- `KXBTC15M-26JUN102130-30`: 11 candles

For each ticker, the historical endpoint returned HTTP 200 with the same partial
count and the series endpoint returned HTTP 404.

On July 30, `KXBTC15M-26JUL300515-15` (`09:00-09:15 UTC`) has 9 candles. The
following six markets covering `09:15-10:45 UTC` return HTTP 200 with empty
candlestick arrays from the series endpoint, while historical returns 404:

- `KXBTC15M-26JUL300530-30`
- `KXBTC15M-26JUL300545-45`
- `KXBTC15M-26JUL300600-00`
- `KXBTC15M-26JUL300615-15`
- `KXBTC15M-26JUL300630-30`
- `KXBTC15M-26JUL300645-45`

Full 15-candle coverage resumes with `KXBTC15M-26JUL300700-00`, the
`10:45-11:00 UTC` market. No missing candles were synthesized or filled.

## Manual Alignment Check

`KXBTC15M-26MAY252000-00` was inspected end to end:

- Market window: `2026-05-25 23:45` through `2026-05-26 00:00 UTC`
- Candles: 15, with period ends from `23:46` through `00:00 UTC`
- Strike: `77267.81`
- Expiration value: `77267.70`
- Settlement result: `NO`
- Final `price_close`: `0.01`

Ticker alignment, candle timestamps within the exact market window, and final
quote direction (`0.01 < 0.5` for a NO settlement) were manually confirmed.

## Request Rate and Backoff

The completed backfill was sequential and waited 0.5 seconds after each HTTP
response, limiting it to about 2 requests per second before network latency.
That setting successfully produced the 91 daily quote files and is the only
experimentally supported sustainable rate for this backfill.

HTTP 429, 5xx, timeout, and network failures use exponential retry waits. With
five attempts, waits are 1, 2, 4, and 8 seconds before the final attempt fails.
No persistent run log records whether transient HTTP 429 responses occurred, so
their occurrence in the completed run cannot be verified. Kalshi's maximum
allowed request rate was not established and must not be inferred as 2 requests
per second.

## Historical Candlestick Limitations

Historical candlesticks provide one-minute YES bid/ask top-of-book OHLC and
trade-price OHLC. They do not provide historical bid size, ask size, or full
order-book depth. Quote-aware historical P&L can use observed prices but cannot
verify that an arbitrary fill quantity was executable.

For a decision at time `t`, the leakage-safe quote is the close from the latest
candle whose `period_end_ts <= t`. Candle highs and lows are intraminute extrema
and must not be treated as executable decision-time quotes.

## Live Recorder

`scripts/record_kalshi_quotes.py` polls open KXBTC15M markets about every 10
seconds and stores snapshots in `data/kalshi_quotes_live/YYYY-MM-DD.parquet`.

Stored fields:

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

In the initial stored sample, both top-of-book size fields were populated in all
34 rows and each had 13 distinct values. `yes_bid_size` ranged from `0.0` to
`4084.09`; `yes_ask_size` ranged from `10.02` to `54631.46`. Real top-of-book
size is therefore being captured and varies over time.

`liquidity` was `0.0` in every row of that initial sample. It should not yet be
assumed to represent usable depth. The open-markets response does not contain a
full order-book depth snapshot.

## Result-Labeling Tiers

- **Tier 1 - probability-only:** no market prices are involved.
- **Tier 2 - quote-aware/top-of-book:** historical bid/ask prices are used, but
  historical size and depth are unavailable.
- **Tier 3 - depth-aware/execution-aware:** size, depth, and execution
  constraints are modeled from suitable data.

Every future P&L figure must carry one of these tier labels. An unlabeled P&L
number is not considered a complete project result.
