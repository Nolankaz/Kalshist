import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from storage import save_daily


BASE_URL = "https://api.kraken.com/0/public/PostTrade"

EXCHANGE_NAME = "kraken"

START_DATE = date(2026, 5, 27)
END_DATE = date(2026, 8, 25)

COUNT = 1000


def request_page(params, max_retries=5):
    """
    Request one page of Kraken trades.

    Retries temporary errors and rate-limit responses instead
    of silently skipping data.
    """

    for attempt in range(max_retries):
        try:
            response = httpx.get(
                BASE_URL,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                wait_seconds = 2 ** attempt

                print(
                    f"Rate limited. Waiting "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            payload = response.json()

            # Kraken can return HTTP 200 while still reporting
            # an API-level error inside the JSON response.
            errors = payload.get("error", [])

            if errors:
                raise RuntimeError(f"Kraken API error: {errors}")

            return payload

        except (httpx.HTTPError, RuntimeError) as exc:
            if attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Request failed: {exc}. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Kraken request failed after retries.")


def fetch_kraken_trades(day):
    """
    Fetch every BTC/USD trade from Kraken for one UTC day.

    Returns a raw DataFrame with:
        trade_id
        timestamp
        price
        size
    """

    day_start = datetime(
        day.year,
        day.month,
        day.day,
        tzinfo=timezone.utc
    )

    day_end = day_start + timedelta(days=1)

    # Kraken's PostTrade endpoint accepts ISO timestamps.
    current_from = day_start.isoformat().replace("+00:00", "Z")

    end_time = day_end.isoformat().replace("+00:00", "Z")

    all_trades = []

    page_number = 1
    previous_last_ts = None

    while True:
        print(f"  {day} - Fetching page {page_number}...")

        params = {
            "symbol": "BTC/USD",
            "from_ts": current_from,
            "to_ts": end_time,
            "count": COUNT,
        }

        payload = request_page(params)

        result = payload.get("result", {})

        trades = result.get("trades", [])

        last_ts = result.get("last_ts")

        if not trades:
            break

        all_trades.extend(trades)

        if not last_ts:
            raise RuntimeError("Kraken returned trades but no last_ts.")

        # Prevent an infinite loop if Kraken stops advancing.
        last_timestamp = pd.Timestamp(last_ts)

        if (
            previous_last_ts is not None
            and last_timestamp <= previous_last_ts
        ):
            raise RuntimeError(
                "Kraken pagination stopped advancing."
            )

        previous_last_ts = last_timestamp

        # If the last trade reached or passed the end of
        # the requested day, we are done.
        last_timestamp = pd.Timestamp(last_ts)

        if last_timestamp >= pd.Timestamp(day_end):
            break

        last_timestamp = pd.Timestamp(last_ts)

        current_from = (
            last_timestamp
            + pd.Timedelta(nanoseconds=1)
        ).isoformat().replace("+00:00", "Z")

        page_number += 1

        # Kraken public REST guidance is conservative,
        # so stay around 1 request/second.
        time.sleep(1.0)

    if not all_trades:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "timestamp",
                "price",
                "size",
            ]
        )

    df = pd.DataFrame(
        all_trades
    )

    required_columns = {
        "trade_id",
        "trade_ts",
        "price",
        "quantity",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing expected Kraken columns: "
            f"{missing}"
        )

    df = df[
        [
            "trade_id",
            "trade_ts",
            "price",
            "quantity",
        ]
    ].copy()

    df = df.rename(
        columns={
            "trade_ts": "timestamp",
            "quantity": "size",
        }
    )

    # Pagination boundaries may overlap slightly.
    df = df.drop_duplicates(subset=["trade_id"])

    # Kraken gives ISO timestamp strings, so unlike Bullish
    # we do NOT use unit="ms" here.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise"
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="raise"
    )

    df["size"] = pd.to_numeric(
        df["size"],
        errors="raise"
    )

    # Keep only trades belonging to this UTC day.
    df = df[
        (df["timestamp"] >= day_start)
        & (df["timestamp"] < day_end)
    ]

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def bucket_to_1s(trades_df):
    """
    Convert raw trades into one row per second.

    price       = VWAP for that second
    volume      = total BTC traded that second
    trade_count = number of trades that second
    """

    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "exchange",
                "price",
                "volume",
                "trade_count",
            ]
        )

    df = trades_df.copy()

    # Example:
    # 12:00:00.208727538
    #
    # becomes:
    #
    # 12:00:00
    df["second"] = (df["timestamp"].dt.floor("s"))

    # Needed for:
    #
    # VWAP =
    # sum(price * size) / sum(size)
    df["price_x_size"] = (df["price"] * df["size"])

    grouped = (
        df.groupby(
            "second",
            as_index=False
        )
        .agg(
            weighted_price_sum=(
                "price_x_size",
                "sum"
            ),
            volume=(
                "size",
                "sum"
            ),
            trade_count=(
                "trade_id",
                "count"
            ),
        )
    )

    grouped["price"] = (
        grouped["weighted_price_sum"]
        / grouped["volume"]
    )

    grouped["exchange"] = (EXCHANGE_NAME)

    grouped = grouped.rename(
        columns={
            "second": "timestamp"
        }
    )

    grouped = grouped[
        [
            "timestamp",
            "exchange",
            "price",
            "volume",
            "trade_count",
        ]
    ]

    return grouped


def backfill_day(day):
    print()
    print(f"Fetching Kraken {day}...")

    raw_df = fetch_kraken_trades(day)

    print(f"Raw trades: {len(raw_df)}")

    if raw_df.empty:
        print(
            f"No trades found for {day}. "
            "Nothing will be saved."
        )
        return

    bucketed_df = bucket_to_1s(raw_df)

    print(
        f"1-second buckets: "
        f"{len(bucketed_df)}"
    )

    save_daily(
        bucketed_df,
        "btc_prices_1s/kraken",
        day.isoformat()
    )

    print(f"Saved Kraken {day}")


def main():
    current_day = START_DATE

    while current_day <= END_DATE:
        backfill_day(current_day)

        current_day += timedelta(days=1)


if __name__ == "__main__":
    main()