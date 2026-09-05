import math
import time
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

from storage import save_daily


SERIES_TICKER = "KXBTC15M"
SERIES_BASE_URL = (f"https://external-api.kalshi.com/trade-api/v2/series/{SERIES_TICKER}/markets")
HISTORICAL_BASE_URL = ("https://external-api.kalshi.com/trade-api/v2/historical/markets")

MARKETS_FOLDER = Path("data/kalshi_markets")
QUOTES_FOLDER = Path("data/kalshi_quotes")

CUTOVER_TS = pd.Timestamp("2026-06-29", tz="UTC")
REQUEST_INTERVAL_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5

OUTPUT_COLUMNS = [
    "ticker",
    "close_date",
    "period_end_ts",
    "yes_bid_open",
    "yes_bid_high",
    "yes_bid_low",
    "yes_bid_close",
    "yes_ask_open",
    "yes_ask_high",
    "yes_ask_low",
    "yes_ask_close",
    "price_open",
    "price_high",
    "price_low",
    "price_close",
    "volume",
    "open_interest",
    "source_endpoint",
]


class MarketFetchError(RuntimeError):
    def __init__(self, message, failed_both_endpoints=False):
        super().__init__(message)
        self.failed_both_endpoints = failed_both_endpoints


def normalize_number(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value for {field_name}: {value!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}")

    return number


def normalize_period_end_ts(value):
    number = normalize_number(value, "end_period_ts")

    if not number.is_integer():
        raise ValueError(f"Invalid end_period_ts: {value!r}")

    return int(number)


def normalize_ohlc(data, endpoint_type, allow_none=False):
    if not isinstance(data, dict):
        raise ValueError(f"Unrecognized {endpoint_type} OHLC fields: {data}")

    if endpoint_type == "historical":
        source_fields = ["open", "high", "low", "close"]
    elif endpoint_type == "series":
        source_fields = [
            "open_dollars",
            "high_dollars",
            "low_dollars",
            "close_dollars",
        ]
    else:
        raise ValueError(f"Unknown endpoint type: {endpoint_type}")

    if not all(field in data for field in source_fields):
        raise ValueError(f"Unrecognized {endpoint_type} OHLC fields: {data}")

    return {
        output_field: (
            None
            if allow_none and data[source_field] is None
            else normalize_number(data[source_field], source_field)
        )
        for output_field, source_field in zip(
            ["open", "high", "low", "close"],
            source_fields,
        )
    }


def normalize_candle(candle, endpoint_type):
    if not isinstance(candle, dict):
        raise ValueError(f"Unrecognized {endpoint_type} candle: {candle}")

    required_sections = ["price", "yes_bid", "yes_ask", "end_period_ts"]

    if not all(field in candle for field in required_sections):
        raise ValueError(f"Unrecognized {endpoint_type} candle fields: {candle}")

    price = normalize_ohlc(candle["price"], endpoint_type, allow_none=True)
    yes_bid = normalize_ohlc(candle["yes_bid"], endpoint_type)
    yes_ask = normalize_ohlc(candle["yes_ask"], endpoint_type)

    if endpoint_type == "historical":
        volume_field = "volume"
        open_interest_field = "open_interest"
    elif endpoint_type == "series":
        volume_field = "volume_fp"
        open_interest_field = "open_interest_fp"
    else:
        raise ValueError(f"Unknown endpoint type: {endpoint_type}")

    if volume_field not in candle or open_interest_field not in candle:
        raise ValueError(f"Unrecognized {endpoint_type} candle fields: {candle}")

    return {
        "period_end_ts": normalize_period_end_ts(candle["end_period_ts"]),
        "yes_bid_open": yes_bid["open"],
        "yes_bid_high": yes_bid["high"],
        "yes_bid_low": yes_bid["low"],
        "yes_bid_close": yes_bid["close"],
        "yes_ask_open": yes_ask["open"],
        "yes_ask_high": yes_ask["high"],
        "yes_ask_low": yes_ask["low"],
        "yes_ask_close": yes_ask["close"],
        "price_open": price["open"],
        "price_high": price["high"],
        "price_low": price["low"],
        "price_close": price["close"],
        "volume": normalize_number(candle[volume_field], volume_field),
        "open_interest": normalize_number(
            candle[open_interest_field],
            open_interest_field,
        ),
    }


def endpoint_url(endpoint_type, ticker):
    if endpoint_type == "historical":
        base_url = HISTORICAL_BASE_URL
    elif endpoint_type == "series":
        base_url = SERIES_BASE_URL
    else:
        raise ValueError(f"Unknown endpoint type: {endpoint_type}")

    return f"{base_url}/{ticker}/candlesticks"


def retry_wait(attempt, message):
    wait_seconds = 2 ** attempt
    print(f"    {message} Retrying in {wait_seconds} seconds...")
    time.sleep(wait_seconds)


def request_candlesticks(
    client,
    endpoint_type,
    ticker,
    start_ts,
    end_ts,
    max_retries=MAX_RETRIES,
):
    url = endpoint_url(endpoint_type, ticker)
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": 1,
    }

    for attempt in range(max_retries):
        try:
            response = client.get(url, params=params)
        except httpx.RequestError as exc:
            if attempt == max_retries - 1:
                raise

            retry_wait(
                attempt,
                f"{ticker} {endpoint_type} request failed: {exc}.",
            )
            continue

        # Keep sequential traffic near two requests per second, including 404s.
        time.sleep(REQUEST_INTERVAL_SECONDS)

        if response.status_code == 404:
            return None

        if response.status_code == 429:
            if attempt == max_retries - 1:
                response.raise_for_status()

            retry_wait(
                attempt,
                f"{ticker} {endpoint_type} was rate limited.",
            )
            continue

        if 500 <= response.status_code < 600:
            if attempt == max_retries - 1:
                response.raise_for_status()

            retry_wait(
                attempt,
                f"{ticker} {endpoint_type} returned HTTP {response.status_code}.",
            )
            continue

        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict) or "candlesticks" not in payload:
            raise ValueError(f"Unrecognized {endpoint_type} response for {ticker}: {payload}")

        candles = payload["candlesticks"]

        if not isinstance(candles, list):
            raise ValueError(f"Unrecognized {endpoint_type} candlesticks for {ticker}: {candles}")

        return candles

    raise RuntimeError(f"{ticker} {endpoint_type} request failed after retries.")


def preferred_endpoint(close_time):
    if close_time < CUTOVER_TS:
        return "historical"

    return "series"


def fetch_market_candles(client, market):
    ticker = market["ticker"]
    close_time = market["close_time"]
    start_ts = int(market["open_time"].timestamp())
    end_ts = int(close_time.timestamp())

    preferred = preferred_endpoint(close_time)
    alternate = "series" if preferred == "historical" else "historical"

    try:
        candles = request_candlesticks(
            client,
            preferred,
            ticker,
            start_ts,
            end_ts,
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise MarketFetchError(
            f"{preferred} endpoint failed without a 404: {exc}"
        ) from exc

    if candles is not None:
        return candles, preferred

    print(f"    {ticker}: {preferred} returned 404; trying {alternate}")

    try:
        candles = request_candlesticks(
            client,
            alternate,
            ticker,
            start_ts,
            end_ts,
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise MarketFetchError(
            f"{preferred} returned 404 and {alternate} failed: {exc}",
            failed_both_endpoints=True,
        ) from exc

    if candles is None:
        raise MarketFetchError(
            f"both {preferred} and {alternate} endpoints returned 404",
            failed_both_endpoints=True,
        )

    return candles, alternate


def normalize_market_candles(candles, endpoint_type, ticker, close_date):
    rows = []

    for candle in candles:
        normalized = normalize_candle(candle, endpoint_type)
        rows.append(
            {
                "ticker": ticker,
                "close_date": close_date,
                **normalized,
                "source_endpoint": endpoint_type,
            }
        )

    return rows


def load_settled_markets():
    files = sorted(MARKETS_FOLDER.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No settled-market Parquet files found in {MARKETS_FOLDER}")

    frames = [pd.read_parquet(path) for path in files]
    markets = pd.concat(frames, ignore_index=True)
    required_columns = {"ticker", "open_time", "close_time"}
    missing = required_columns - set(markets.columns)

    if missing:
        raise ValueError(f"Missing expected Kalshi market columns: {missing}")

    markets = markets[["ticker", "open_time", "close_time"]].copy()

    if markets[["ticker", "open_time", "close_time"]].isna().any().any():
        raise ValueError("Settled-market source contains missing required values.")

    markets["open_time"] = pd.to_datetime(
        markets["open_time"],
        utc=True,
        errors="raise",
    )
    markets["close_time"] = pd.to_datetime(
        markets["close_time"],
        utc=True,
        errors="raise",
    )

    duplicate_tickers = markets.loc[
        markets["ticker"].duplicated(keep=False),
        "ticker",
    ].unique()

    if len(duplicate_tickers):
        raise ValueError(f"Duplicate settled-market tickers found: {duplicate_tickers[:10].tolist()}")

    invalid_windows = markets["close_time"] <= markets["open_time"]

    if invalid_windows.any():
        tickers = markets.loc[invalid_windows, "ticker"].head(10).tolist()
        raise ValueError(f"Markets have invalid open/close windows: {tickers}")

    markets["close_date"] = markets["close_time"].dt.strftime("%Y-%m-%d")
    return markets.sort_values(["close_time", "ticker"]).reset_index(drop=True)


def main():
    markets = load_settled_markets()

    dates_processed = 0
    dates_skipped = 0
    dates_written = 0
    markets_processed = 0
    markets_succeeded = 0
    failed_markets = []
    failed_both_endpoints = []
    empty_candle_markets = []
    total_candle_rows_written = 0
    source_counts = Counter()

    print(
        f"Loaded {len(markets)} settled markets across "
        f"{markets['close_date'].nunique()} UTC close dates."
    )

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for close_date, daily_markets in markets.groupby("close_date", sort=True):
            output_path = QUOTES_FOLDER / f"{close_date}.parquet"

            if output_path.exists():
                dates_skipped += 1
                print(f"Skipping {close_date}: {output_path} already exists")
                continue

            dates_processed += 1
            daily_rows = []
            daily_failures = []
            market_count = len(daily_markets)
            print(f"Processing {close_date}: {market_count} markets")

            for market_number, (_, market) in enumerate(
                daily_markets.iterrows(),
                start=1,
            ):
                ticker = market["ticker"]
                markets_processed += 1

                try:
                    candles, endpoint_type = fetch_market_candles(client, market)
                    rows = normalize_market_candles(
                        candles,
                        endpoint_type,
                        ticker,
                        close_date,
                    )

                    if not candles:
                        empty_response = (
                            f"{ticker} ({close_date}, source={endpoint_type})"
                        )
                        empty_candle_markets.append(empty_response)
                        print(f"  EMPTY CANDLE RESPONSE: {empty_response}")
                except (MarketFetchError, ValueError) as exc:
                    failure = f"{ticker} ({close_date}): {exc}"
                    failed_markets.append(failure)
                    daily_failures.append(failure)

                    if (
                        isinstance(exc, MarketFetchError)
                        and exc.failed_both_endpoints
                    ):
                        failed_both_endpoints.append(failure)
                        print(f"  FAILED BOTH ENDPOINTS: {failure}")
                    else:
                        print(f"  FAILED: {failure}")

                    continue

                daily_rows.extend(rows)
                markets_succeeded += 1

                if (
                    market_number == 1
                    or market_number % 20 == 0
                    or market_number == market_count
                ):
                    print(
                        f"  Markets {market_number}/{market_count}; "
                        f"latest={ticker}, candles={len(rows)}, "
                        f"source={endpoint_type}"
                    )

            if daily_failures:
                print(
                    f"Not saving {close_date}: "
                    f"{len(daily_failures)} market(s) failed"
                )
                continue

            daily_df = pd.DataFrame(daily_rows, columns=OUTPUT_COLUMNS)
            save_daily(daily_df, "kalshi_quotes", close_date)

            dates_written += 1
            total_candle_rows_written += len(daily_df)
            source_counts.update(daily_df["source_endpoint"])
            print(f"Saved {len(daily_df)} candles for {close_date}")

    print()
    print("Backfill complete.")
    print(f"Dates processed: {dates_processed}")
    print(f"Dates skipped: {dates_skipped}")
    print(f"Dates written: {dates_written}")
    print(f"Markets processed: {markets_processed}")
    print(f"Markets succeeded: {markets_succeeded}")
    print(f"Markets failed: {len(failed_markets)}")
    print(
        "Markets that failed both endpoints: "
        f"{len(failed_both_endpoints)}"
    )
    print(
        "Markets returning HTTP 200 with empty candlesticks: "
        f"{len(empty_candle_markets)}"
    )
    print(
        "Total candle rows written during this execution: "
        f"{total_candle_rows_written}"
    )
    print("Rows written during this execution by source_endpoint:")
    print(f"  historical: {source_counts['historical']}")
    print(f"  series: {source_counts['series']}")

    if empty_candle_markets:
        print("Tickers returning HTTP 200 with empty candlesticks:")

        for market in empty_candle_markets:
            print(f"  {market}")

    if failed_both_endpoints:
        print("Tickers that failed both endpoints:")

        for failure in failed_both_endpoints:
            print(f"  {failure}")


if __name__ == "__main__":
    main()
