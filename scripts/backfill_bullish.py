import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from storage import save_daily


BASE_URL = (
    "https://api.exchange.bullish.com/"
    "trading-api/v1/history/markets/BTCUSD/trades"
)

EXCHANGE_NAME = "bullish"

START_DATE = date(2026, 5, 28)
END_DATE = date(2026, 8, 25)

PAGE_SIZE = 100


def request_page(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = httpx.get(
                url,
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

            return response.json()

        except httpx.HTTPError as exc:
            if attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Request failed: {exc}. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Bullish request failed after retries.")


def fetch_bullish_trades(day):
    """
    Fetch every BTCUSD trade from Bullish for one UTC day.

    The day is split into 1-hour windows.
    Each hour is fully paginated before moving to the next hour.

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

    all_trades = []

    window_start = day_start

    while window_start < day_end:
        window_end = min(
            window_start + timedelta(hours=1),
            day_end
        )

        print(
            f"  Window: "
            f"{window_start.strftime('%H:%M')} "
            f"to "
            f"{window_end.strftime('%H:%M')}"
        )

        start_ms = int(
            window_start.timestamp() * 1000
        )

        end_ms = int(
            window_end.timestamp() * 1000
        )

        params = {
            "createdAtTimestamp[gte]": start_ms,
            "createdAtTimestamp[lt]": end_ms,
            "_pageSize": PAGE_SIZE,
            "_metaData": "true",
        }

        current_url = BASE_URL
        current_params = params

        page_number = 1

        while True:
            print(
                f"    Fetching page "
                f"{page_number}..."
            )

            payload = request_page(
                current_url,
                params=current_params
            )

            trades = payload.get(
                "data",
                []
            )

            if not trades:
                break

            all_trades.extend(
                trades
            )

            links = payload.get(
                "links",
                {}
            )

            next_link = links.get(
                "next"
            )

            if not next_link:
                break

            if next_link.startswith("/"):
                current_url = (
                    "https://api.exchange.bullish.com"
                    + next_link
                )
            else:
                current_url = next_link

            # next_link already contains its own
            # query parameters.
            current_params = None

            page_number += 1

            time.sleep(0.05)

        window_start = window_end

    if not all_trades:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "timestamp",
                "price",
                "size"
            ]
        )

    df = pd.DataFrame(
        all_trades
    )

    required_columns = {
        "tradeId",
        "createdAtTimestamp",
        "price",
        "quantity",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing expected Bullish "
            f"columns: {missing}"
        )

    df = df[
        [
            "tradeId",
            "createdAtTimestamp",
            "price",
            "quantity",
        ]
    ].copy()

    df = df.rename(
        columns={
            "tradeId": "trade_id",
            "createdAtTimestamp": "timestamp",
            "quantity": "size",
        }
    )

    # Hour boundaries / pagination can overlap.
    df = df.drop_duplicates(
        subset=["trade_id"]
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="raise"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    df["price"] = pd.to_numeric(
        df["price"],
        errors="raise"
    )

    df["size"] = pd.to_numeric(
        df["size"],
        errors="raise"
    )

    # Final safety filter:
    # only keep trades belonging to this UTC day.
    df = df[
        (df["timestamp"] >= day_start)
        & (df["timestamp"] < day_end)
    ]

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    if not df.empty:
        print(
            f"  First trade: "
            f"{df['timestamp'].min()}"
        )

        print(
            f"  Last trade: "
            f"{df['timestamp'].max()}"
        )

    return df


def bucket_to_1s(trades_df):
    """
    Convert raw trades into one row per second.

    price       = volume-weighted average price
    volume      = total BTC traded that second
    trade_count = number of raw trades that second
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

    # Put every raw trade into its UTC second.
    df["second"] = df["timestamp"].dt.floor("s")

    # Needed for VWAP numerator:
    # price * size
    df["price_x_size"] = (df["price"] * df["size"])

    grouped = (
        df.groupby("second", as_index=False)
        .agg(
            weighted_price_sum=("price_x_size", "sum"),
            volume=("size", "sum"),
            trade_count=("trade_id", "count"),
        )
    )

    grouped["price"] = (
        grouped["weighted_price_sum"]
        / grouped["volume"]
    )

    grouped["exchange"] = EXCHANGE_NAME

    grouped = grouped.rename(columns={"second": "timestamp"})

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
    print(f"Fetching Bullish {day}...")

    raw_df = fetch_bullish_trades(day)

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
        "btc_prices_1s/bullish",
        day.isoformat()
    )

    print(f"Saved Bullish {day}")


def main():
    current_day = START_DATE

    while current_day <= END_DATE:
        backfill_day(current_day)

        current_day += timedelta(days=1)


if __name__ == "__main__":
    main()