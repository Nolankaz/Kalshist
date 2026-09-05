import pandas as pd

from storage import load_range


START_DATE = "2026-05-26"
END_DATE = "2026-08-24"

EXPECTED_QUOTE_ROWS = 15


def require_columns(df, columns, table_name):
    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing expected {table_name} columns: {missing}")


def main():
    markets = load_range("kalshi_markets", START_DATE, END_DATE)
    quotes = load_range("kalshi_quotes", START_DATE, END_DATE)

    if markets.empty:
        raise ValueError("No settled Kalshi markets found for the requested range.")

    if quotes.empty:
        raise ValueError("No Kalshi quotes found for the requested range.")

    market_columns = [
        "ticker",
        "open_time",
        "close_time",
        "settlement_result",
        "strike",
        "expiration_value",
    ]
    quote_columns = [
        "ticker",
        "period_end_ts",
        "yes_bid_close",
        "yes_ask_close",
        "price_close",
        "source_endpoint",
    ]

    require_columns(markets, market_columns, "settled-market")
    require_columns(quotes, quote_columns, "quote")

    markets = markets.copy()
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

    quote_counts = quotes.groupby("ticker").size()
    eligible_markets = markets[
        markets["ticker"].map(quote_counts).eq(EXPECTED_QUOTE_ROWS)
    ].sort_values(["close_time", "ticker"])

    if eligible_markets.empty:
        raise ValueError(
            f"No market has exactly {EXPECTED_QUOTE_ROWS} quote rows."
        )

    market = eligible_markets.iloc[0]
    ticker = market["ticker"]
    market_quotes = quotes[quotes["ticker"] == ticker].copy()
    market_quotes = market_quotes.sort_values("period_end_ts").reset_index(
        drop=True
    )
    market_quotes["timestamp"] = pd.to_datetime(
        market_quotes["period_end_ts"],
        unit="s",
        utc=True,
        errors="raise",
    )

    quote_row_count = len(market_quotes)

    print("Kalshi market inspection")
    print(f"Range loaded: {START_DATE} through {END_DATE}")
    print(
        "Selection rule: earliest market with exactly "
        f"{EXPECTED_QUOTE_ROWS} quote rows"
    )

    print("\nMarket metadata")

    for column in market_columns:
        print(f"{column}: {market[column]}")

    if "settlement_value" in markets.columns:
        print(f"settlement_value: {market['settlement_value']}")

    print("\nQuote coverage")
    print(f"Total quote rows for {ticker}: {quote_row_count}")
    print(
        f"Exactly {EXPECTED_QUOTE_ROWS} rows: "
        f"{'YES' if quote_row_count == EXPECTED_QUOTE_ROWS else 'NO'}"
    )

    print("\nChronological quote rows")
    display_columns = [
        "timestamp",
        "yes_bid_close",
        "yes_ask_close",
        "price_close",
        "source_endpoint",
    ]
    print(market_quotes[display_columns].to_string(index=False))

    first_timestamp = market_quotes["timestamp"].iloc[0]
    final_timestamp = market_quotes["timestamp"].iloc[-1]
    open_time = market["open_time"]
    close_time = market["close_time"]
    first_inside_window = open_time <= first_timestamp <= close_time
    final_inside_window = open_time <= final_timestamp <= close_time

    print("\nTimestamp alignment")
    print(f"Market window: {open_time} through {close_time}")
    print(f"First candle timestamp: {first_timestamp}")
    print(
        "First timestamp inside market window: "
        f"{'PASS' if first_inside_window else 'FAIL'}"
    )
    print(f"Final candle timestamp: {final_timestamp}")
    print(
        "Final timestamp inside market window: "
        f"{'PASS' if final_inside_window else 'FAIL'}"
    )

    final_price = market_quotes["price_close"].iloc[-1]
    settlement_result = str(market["settlement_result"]).upper()

    print("\nFinal quote direction")
    print(f"Settlement result: {settlement_result}")
    print(f"Final price_close: {final_price}")

    if pd.isna(final_price):
        print("Directional comparison: unavailable because price_close is missing")
    elif settlement_result == "YES":
        print(
            "Final price_close above 0.5: "
            f"{'YES' if final_price > 0.5 else 'NO'}"
        )
    elif settlement_result == "NO":
        print(
            "Final price_close below 0.5: "
            f"{'YES' if final_price < 0.5 else 'NO'}"
        )
    else:
        print(
            "Directional comparison: unavailable for unrecognized "
            f"settlement_result {settlement_result!r}"
        )


if __name__ == "__main__":
    main()
