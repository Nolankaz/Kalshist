## BULLISH

Symbol:
BTCUSD

Endpoint:
GET /trading-api/v1/history/markets/BTCUSD/trades

Historical retention:
About 90 days

Maximum requested time window:
About 7 days

Pagination:
Supported

Historical test:
SUCCESS

Confirmed historical BTCUSD trade data is accessible
for the June 2026 target period.

Verdict:
PRIMARY SOURCE

## GEMINI

Symbol:
BTCUSD

Endpoint:
GET /v1/trades/BTCUSD

Maximum trades per response:
500

Ordering:
Newest trades first.

Pagination/filtering:
timestamp returns trades AFTER the requested timestamp.
since_tid returns trades after a trade ID.

Historical testing:
The API successfully returned some June 15, 2026 trades
when queried close to their timestamps.

However, a request beginning at June 15 00:00 returned the
newest 500 trades after that timestamp, spanning June 16–22.

Because the endpoint has no end_timestamp parameter and caps
responses at 500 newest-first records, reliably enumerating
every trade in an arbitrary historical UTC day is not practical
through this public endpoint.

The current Gemini documentation also states that the public
trade endpoint is limited to seven calendar days of data.

Verdict:
DO NOT USE for the 60–90 day historical backfill.

Possible future use:
Live/recent market data.

## KRAKEN

Symbol:
BTC/USD

Endpoint:
GET /0/public/PostTrade

Maximum trades per request:
1000

Pagination:
Forward using the response's `last_ts`.

Example:
request with `from_ts`
→ receive trades
→ retrieve `last_ts`
→ next request uses `last_ts` as `from_ts`

Timestamp precision:
Sub-second timestamps

Rate limit:
Keep public REST requests around 1 request/second or slower.

Historical retention:
Exact maximum retention is not clearly documented, but historical
BTC/USD trades were successfully retrieved from June 15, 2026.

Historical test:
SUCCESS

Test range:
June 15, 2026
12:00:00 → 12:01:00 UTC

Test result:
Successfully returned BTC/USD trades from the requested historical period.

Example trade:
price = 66197.50
quantity = 0.00150319 BTC
trade_ts = 2026-06-15T12:00:00.208727538Z

Pagination field returned:
last_ts = 2026-06-15T12:00:03.391712752Z

Verdict:
PRIMARY SOURCE

## CRYPTO.COM

Spot symbol:
BTC_USD

Instrument type:
CCY_PAIR

Endpoint:
GET /exchange/v1/public/get-trades

Parameters:
instrument_name
start_ts
end_ts
count

Trade fields:
p  = price
q  = quantity
s  = side
i  = instrument
t  = timestamp in milliseconds
tn = timestamp in nanoseconds

Timestamp precision:
Nanoseconds available

Historical test:
SUCCESS

Test range:
June 15, 2026
12:00:00 → 12:01:00 UTC

Confirmed:
Historical BTC/USD spot trades from June 2026 are
retrievable through the public API.

Example:
price = 66220.98
quantity = 0.03000 BTC
instrument = BTC_USD
timestamp_ms = 1781524859699
timestamp_ns = 1781524859699185235

Important:
Use BTC_USD.
Do NOT use BTCUSD-PERP or BTCUSD dated futures.

Pagination/order:
Returned trades appear near the end of the requested
window first. Full backfill needs to account for ordering
and timestamp pagination rather than assuming the response
starts at start_ts.

Verdict:
PRIMARY SOURCE

## COINBASE

Symbol:
BTC-USD

Endpoint:
GET /products/BTC-USD/trades

Maximum trades per response:
1000

Pagination:
Cursor pagination.

CB-AFTER → older trades
CB-BEFORE → newer trades

Rate limit:
10 requests/sec normal public limit
15 requests/sec burst

Historical retention:
No simple 90-day retention limit documented on this endpoint.

Major disadvantage:
No convenient start/end timestamp for jumping directly to a
historical date. Reaching 60–90 days ago may require paging
through a very large number of trades.

Verdict:
BACKUP SOURCE

## BITSTAMP

Symbol:
btcusd

Public trades endpoint:
GET /api/v2/transactions/btcusd/

Trade history options:
time=minute
time=hour
time=day

Historical arbitrary timestamp lookup:
Not supported by the public transactions endpoint.

Historical OHLC:
GET /api/v2/ohlc/btcusd/

Finest OHLC resolution:
60 seconds

Max OHLC candles:
1000

Rate limit:
400 requests/second
10,000 requests per 10 minutes

Verdict:
BACKUP ONLY.
Useful for 1-minute historical data, but poor fit for
trade-level 60–90 day reconstruction.