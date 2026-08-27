import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from storage import save_daily


BASE_URL = "https://api.crypto.com/exchange/v1/public/get-trades"

EXCHANGE_NAME = "crypto_com"
INSTRUMENT = "BTC_USD"

START_DATE = date(2026, 7, 29)
END_DATE = date(2026, 8, 25)

COUNT = 1000


def request_page(params, max_retries=5):
    """
    Request one page of Crypto.com public trades.
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

            # Crypto.com often returns HTTP 200 even if the
            # API-level response contains an error code.
            code = payload.get("code")

            if code != 0:
                raise RuntimeError(
                    f"Crypto.com API error: "
                    f"code={code}, "
                    f"message={payload.get('message')}"
                )

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

    raise RuntimeError( "Crypto.com request failed after retries.")


def fetch_crypto_com_trades(day):
    """
    Fetch all BTC_USD spot trades for one UTC day.

    Crypto.com returns newest trades first, so pagination
    moves BACKWARD through time by changing end_ts.

    Returns raw DataFrame:
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

    # Use nanoseconds so we can move pagination boundaries
    # forward/backward precisely.
    start_ns = int(day_start.timestamp() * 1_000_000_000)

    current_end_ns = int(day_end.timestamp() * 1_000_000_000)

    all_trades = []

    page_number = 1
    previous_oldest_ns = None

    while current_end_ns > start_ns:
        if page_number % 100 == 0:
            print(
                f"{day} - page {page_number:,} - "
                f"{len(all_trades):,} trades fetched"
            )

        params = {
            "instrument_name": INSTRUMENT,
            "start_ts": start_ns,
            "end_ts": current_end_ns,
            "count": COUNT,
        }

        payload = request_page(params)

        trades = (
            payload
            .get("result", {})
            .get("data", [])
        )

        if not trades:
            break

        all_trades.extend(trades)

        # "tn" is Crypto.com's nanosecond timestamp.
        page_timestamps_ns = [
            int(trade["tn"])
            for trade in trades
        ]

        oldest_ns = min(page_timestamps_ns)

        # Prevent an infinite loop if the API returns
        # the same oldest boundary twice.
        if (
            previous_oldest_ns is not None
            and oldest_ns >= previous_oldest_ns
        ):
            raise RuntimeError(
                "Crypto.com pagination stopped advancing."
            )

        previous_oldest_ns = oldest_ns

        # We've reached the beginning of the requested day.
        if oldest_ns <= start_ns:
            break

        # Move end_ts one nanosecond BEFORE the oldest
        # trade we already received.
        current_end_ns = oldest_ns - 1

        page_number += 1

        # Public limit is much higher, but this is intentionally
        # conservative while validating the script.
        time.sleep(0.01)

    if not all_trades:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "timestamp",
                "price",
                "size",
            ]
        )

    df = pd.DataFrame(all_trades)

    required_columns = {
        "d",
        "tn",
        "p",
        "q",
        "i",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing expected Crypto.com columns: "
            f"{missing}"
        )

    # Make sure we did not accidentally download
    # a futures or perpetual instrument.
    wrong_instruments = df[df["i"] != INSTRUMENT]

    if not wrong_instruments.empty:
        raise ValueError(
            "Crypto.com returned trades for an "
            "unexpected instrument."
        )

    df = df[
        [
            "d",
            "tn",
            "p",
            "q",
        ]
    ].copy()

    df = df.rename(
        columns={
            "d": "trade_id",
            "tn": "timestamp",
            "p": "price",
            "q": "size",
        }
    )

    # Protect against pagination overlap.
    df = df.drop_duplicates(subset=["trade_id"])

    # Convert nanosecond timestamp strings/integers first.
    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="raise"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ns",
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

    # Keep only trades from the requested UTC day.
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
    volume      = BTC volume for that second
    trade_count = number of raw trades
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

    df["second"] = (df["timestamp"].dt.floor("s"))

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

    grouped["exchange"] = (EXCHANGE_NAME)

    grouped = grouped.rename(
        columns={"second": "timestamp"}
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
    print(f"Fetching Crypto.com {day}...")

    raw_df = fetch_crypto_com_trades(day)

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
        "btc_prices_1s/crypto_com",
        day.isoformat()
    )

    print(f"Saved Crypto.com {day}")


def main():
    current_day = START_DATE

    while current_day <= END_DATE:
        backfill_day(current_day)

        current_day += timedelta(days=1)


if __name__ == "__main__":
    main()