# Data Schema

## Kalshi Markets

Each row represents one settled KXBTC15M market.

Columns:

- ticker
- strike
- open_time
- close_time
- settlement_result
- settlement_value
- volume
- `settlement_ts` — timestamp when Kalshi recorded the market settlement, in UTC

## BTC Prices

Each row represents BTC price data from one exchange at one timestamp.

Columns:

- timestamp
- exchange
- price
- volume