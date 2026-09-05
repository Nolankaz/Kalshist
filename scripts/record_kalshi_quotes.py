import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from storage import save_daily


BASE_URL = "https://external-api.kalshi.com/trade-api/v2/markets"
SERIES_TICKER = "KXBTC15M"

TABLE_NAME = "kalshi_quotes_live"
OUTPUT_FOLDER = Path("data") / TABLE_NAME

POLL_INTERVAL_SECONDS = 10
FLUSH_INTERVAL_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5

RETRYABLE_STATUS_CODES = {408, 425, 429}

NUMERIC_FIELD_MAP = {
    "yes_bid": "yes_bid_dollars",
    "yes_ask": "yes_ask_dollars",
    "yes_bid_size": "yes_bid_size_fp",
    "yes_ask_size": "yes_ask_size_fp",
    "last_price": "last_price_dollars",
    "liquidity": "liquidity_dollars",
    "volume": "volume_fp",
    "volume_24h": "volume_24h_fp",
    "open_interest": "open_interest_fp",
}

OUTPUT_COLUMNS = [
    "ticker",
    "observed_ts",
    *NUMERIC_FIELD_MAP,
]


class TemporaryRequestError(RuntimeError):
    pass


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


def retry_wait(attempt, message):
    wait_seconds = 2 ** attempt
    print(f"{message} Retrying in {wait_seconds} seconds...")
    time.sleep(wait_seconds)


def request_open_markets(client, max_retries=MAX_RETRIES):
    params = {
        "limit": 1000,
        "status": "open",
        "series_ticker": SERIES_TICKER,
    }

    for attempt in range(max_retries):
        try:
            response = client.get(BASE_URL, params=params)
        except httpx.RequestError as exc:
            if attempt == max_retries - 1:
                raise TemporaryRequestError(
                    f"Kalshi request failed after {max_retries} attempts: {exc}"
                ) from exc

            retry_wait(attempt, f"Kalshi request failed: {exc}.")
            continue

        retryable_status = (
            response.status_code in RETRYABLE_STATUS_CODES
            or 500 <= response.status_code < 600
        )

        if retryable_status:
            if attempt == max_retries - 1:
                raise TemporaryRequestError(
                    "Kalshi request failed after "
                    f"{max_retries} attempts with HTTP {response.status_code}."
                )

            retry_wait(
                attempt,
                f"Kalshi returned HTTP {response.status_code}.",
            )
            continue

        response.raise_for_status()
        observed_ts = datetime.now(timezone.utc)
        payload = response.json()

        if not isinstance(payload, dict) or "markets" not in payload:
            raise ValueError(f"Unrecognized Kalshi markets response: {payload}")

        markets = payload["markets"]

        if not isinstance(markets, list):
            raise ValueError(f"Unrecognized Kalshi markets list: {markets}")

        return markets, observed_ts

    raise TemporaryRequestError("Kalshi request failed after retries.")


def normalize_market(market, observed_ts):
    if not isinstance(market, dict):
        raise ValueError(f"Unrecognized Kalshi market: {market}")

    required_fields = {"ticker", *NUMERIC_FIELD_MAP.values()}
    missing = required_fields - set(market)

    if missing:
        raise ValueError(
            f"Missing expected fields for Kalshi market: {missing}. Market: {market}"
        )

    ticker = market["ticker"]

    if not isinstance(ticker, str) or not ticker:
        raise ValueError(f"Invalid Kalshi market ticker: {ticker!r}")

    row = {
        "ticker": ticker,
        "observed_ts": observed_ts,
    }

    for output_field, source_field in NUMERIC_FIELD_MAP.items():
        row[output_field] = normalize_number(
            market[source_field],
            source_field,
        )

    return row


def normalize_snapshot(markets, observed_ts):
    rows = [normalize_market(market, observed_ts) for market in markets]
    tickers = [row["ticker"] for row in rows]

    if len(tickers) != len(set(tickers)):
        raise ValueError("Kalshi returned duplicate tickers in one markets response.")

    return rows


def flush_buffer(buffer):
    if not buffer:
        return

    buffered_df = pd.DataFrame(buffer, columns=OUTPUT_COLUMNS)
    buffered_df["observed_ts"] = pd.to_datetime(
        buffered_df["observed_ts"],
        utc=True,
        errors="raise",
    )
    buffered_df["observed_date"] = buffered_df["observed_ts"].dt.strftime(
        "%Y-%m-%d"
    )

    for observed_date, daily_buffer in buffered_df.groupby(
        "observed_date",
        sort=True,
    ):
        daily_buffer = daily_buffer.drop(columns=["observed_date"])
        output_path = OUTPUT_FOLDER / f"{observed_date}.parquet"

        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            missing = set(OUTPUT_COLUMNS) - set(existing_df.columns)

            if missing:
                raise ValueError(
                    f"Existing {output_path} is missing columns: {missing}"
                )

            existing_df = existing_df[OUTPUT_COLUMNS].copy()
            existing_df["observed_ts"] = pd.to_datetime(
                existing_df["observed_ts"],
                utc=True,
                errors="raise",
            )
            combined_df = pd.concat(
                [existing_df, daily_buffer],
                ignore_index=True,
            )
        else:
            combined_df = daily_buffer.copy()

        rows_before_deduplication = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=["ticker", "observed_ts"],
            keep="last",
        )
        combined_df = combined_df.sort_values(
            ["observed_ts", "ticker"]
        ).reset_index(drop=True)
        duplicates_removed = rows_before_deduplication - len(combined_df)

        save_daily(combined_df, TABLE_NAME, observed_date)

        print(
            f"Flushed {len(daily_buffer)} buffered rows to {output_path}; "
            f"file rows={len(combined_df)}, duplicates removed={duplicates_removed}"
        )


def main():
    buffer = []
    last_flush_time = time.monotonic()

    print(
        f"Recording open {SERIES_TICKER} markets every "
        f"{POLL_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop."
    )

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            while True:
                poll_started = time.monotonic()

                try:
                    markets, observed_ts = request_open_markets(client)
                except TemporaryRequestError as exc:
                    print(f"Temporary request failure: {exc}")
                else:
                    rows = normalize_snapshot(markets, observed_ts)
                    buffer.extend(rows)
                    print(
                        f"{observed_ts.isoformat()} - "
                        f"open markets={len(markets)}, buffered rows={len(buffer)}"
                    )

                if time.monotonic() - last_flush_time >= FLUSH_INTERVAL_SECONDS:
                    if buffer:
                        flush_buffer(buffer)
                        buffer.clear()

                    last_flush_time = time.monotonic()

                elapsed = time.monotonic() - poll_started
                time.sleep(max(0, POLL_INTERVAL_SECONDS - elapsed))
    except KeyboardInterrupt:
        print("\nStopping recorder...")
    finally:
        if buffer:
            print(f"Flushing {len(buffer)} buffered rows before exit...")
            flush_buffer(buffer)
            buffer.clear()

    print("Recorder stopped.")


if __name__ == "__main__":
    main()