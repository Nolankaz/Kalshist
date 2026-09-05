import pandas as pd

from storage import load_range


START_DATE = "2026-05-26"
END_DATE = "2026-08-24"

EXPECTED_CANDLES_PER_MARKET = 15
SAMPLE_SIZE = 10

PRICE_COLUMNS = [
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
]


def require_columns(df, columns, table_name):
    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing expected {table_name} columns: {missing}")


def format_period_end(series):
    return pd.to_datetime(
        series,
        unit="s",
        utc=True,
        errors="raise",
    )


def main():
    markets = load_range("kalshi_markets", START_DATE, END_DATE)
    quotes = load_range("kalshi_quotes", START_DATE, END_DATE)

    if markets.empty:
        raise ValueError("No settled Kalshi markets found for the requested range.")

    if quotes.empty:
        raise ValueError("No Kalshi quotes found for the requested range.")

    require_columns(
        markets,
        ["ticker", "open_time", "close_time"],
        "settled-market",
    )
    require_columns(
        quotes,
        [
            "ticker",
            "close_date",
            "period_end_ts",
            "source_endpoint",
            *PRICE_COLUMNS,
        ],
        "quote",
    )

    markets = markets.copy()
    quotes = quotes.copy()

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
    markets["close_date"] = markets["close_time"].dt.strftime("%Y-%m-%d")

    if markets["ticker"].duplicated().any():
        duplicate_tickers = markets.loc[
            markets["ticker"].duplicated(keep=False),
            "ticker",
        ].unique()
        raise ValueError(f"Duplicate settled-market tickers found: {duplicate_tickers[:10].tolist()}")

    quote_counts = quotes.groupby("ticker").size()
    coverage = markets[["ticker", "close_date", "open_time", "close_time"]].copy()
    coverage["candle_count"] = (coverage["ticker"].map(quote_counts).fillna(0).astype(int))

    zero_candle_markets = coverage[coverage["candle_count"] == 0].copy()
    markets_with_candles = len(coverage) - len(zero_candle_markets)
    unexpected_tickers = sorted(set(quotes["ticker"]) - set(markets["ticker"]))

    print("Kalshi quote validation")
    print(f"Range: {START_DATE} through {END_DATE}")
    print(f"Quote rows loaded: {len(quotes)}")

    print("\nMarket coverage")
    print(f"Total settled markets: {len(coverage)}")
    print(f"Markets with at least one candle: {markets_with_candles}")
    print(f"Markets with zero candles: {len(zero_candle_markets)}")
    print(f"Quote tickers absent from settled markets: {len(unexpected_tickers)}")

    if zero_candle_markets.empty:
        print("Zero-candle tickers: none")
    else:
        print("Zero-candle tickers by close date:")

        for close_date, daily_zeroes in zero_candle_markets.groupby(
            "close_date",
            sort=True,
        ):
            print(f"  {close_date}")

            for row in daily_zeroes.sort_values("open_time").itertuples():
                window = (
                    f"{row.open_time.strftime('%H:%M')}-"
                    f"{row.close_time.strftime('%H:%M')} UTC"
                )
                print(f"    {row.ticker} ({window})")

    if unexpected_tickers:
        print("Unexpected quote tickers:")

        for ticker in unexpected_tickers:
            print(f"  {ticker}")

    market_coverage_pass = (
        zero_candle_markets.empty
        and not unexpected_tickers
    )

    count_distribution = (
        coverage["candle_count"].value_counts().sort_index()
    )
    partial_markets = coverage[
        (coverage["candle_count"] > 0)
        & (coverage["candle_count"] < EXPECTED_CANDLES_PER_MARKET)
    ].copy()

    print("\nCandle-count distribution")
    print(
        f"Markets with {EXPECTED_CANDLES_PER_MARKET} candles: "
        f"{(coverage['candle_count'] == EXPECTED_CANDLES_PER_MARKET).sum()}"
    )
    print("Complete distribution (candles: markets):")

    for candle_count, market_count in count_distribution.items():
        print(f"  {candle_count}: {market_count}")

    if partial_markets.empty:
        print("Markets with partial positive coverage: none")
    else:
        print("Markets with partial positive coverage:")

        for row in partial_markets.sort_values("open_time").itertuples():
            window = (
                f"{row.open_time.strftime('%Y-%m-%d %H:%M')}-"
                f"{row.close_time.strftime('%H:%M')} UTC"
            )
            print(f"  {row.ticker}: {row.candle_count} candles ({window})")

    candle_count_pass = (
        coverage["candle_count"] == EXPECTED_CANDLES_PER_MARKET
    ).all()

    bid_ask_valid = (
        quotes["yes_bid_close"].notna()
        & quotes["yes_ask_close"].notna()
        & (quotes["yes_bid_close"] <= quotes["yes_ask_close"])
    )
    bid_ask_violations = quotes.loc[~bid_ask_valid].copy()
    bid_ask_pass_rate = bid_ask_valid.mean()

    print("\nBid/ask consistency")
    print(f"Rows checked: {len(quotes)}")
    print(f"Violations: {len(bid_ask_violations)}")
    print(f"Pass rate: {bid_ask_pass_rate:.6%}")

    if not bid_ask_violations.empty:
        sample = bid_ask_violations[
            ["ticker", "period_end_ts", "yes_bid_close", "yes_ask_close"]
        ].head(SAMPLE_SIZE).copy()
        sample["timestamp"] = format_period_end(sample["period_end_ts"])
        sample = sample[
            ["ticker", "timestamp", "yes_bid_close", "yes_ask_close"]
        ]
        print("Violation sample:")
        print(sample.to_string(index=False))

    bid_ask_pass = bid_ask_violations.empty

    price_violation_frames = []

    for column in PRICE_COLUMNS:
        numeric_values = pd.to_numeric(quotes[column], errors="coerce")
        violations = quotes[column].notna() & (
            numeric_values.isna()
            | ~numeric_values.between(0, 1, inclusive="both")
        )

        if violations.any():
            offending = quotes.loc[
                violations,
                ["ticker", "period_end_ts", column],
            ].copy()
            offending["timestamp"] = format_period_end(
                offending["period_end_ts"]
            )
            offending["column"] = column
            offending = offending.rename(columns={column: "value"})
            price_violation_frames.append(
                offending[["ticker", "timestamp", "column", "value"]]
            )

    if price_violation_frames:
        price_violations = pd.concat(
            price_violation_frames,
            ignore_index=True,
        )
    else:
        price_violations = pd.DataFrame(
            columns=["ticker", "timestamp", "column", "value"]
        )

    print("\nPrice ranges")
    print(f"Non-null values checked: {quotes[PRICE_COLUMNS].count().sum()}")
    print(f"Violations outside [0, 1]: {len(price_violations)}")

    if not price_violations.empty:
        print("Violations by column:")
        print(price_violations["column"].value_counts().to_string())
        print(f"First {min(SAMPLE_SIZE, len(price_violations))} offending rows:")
        print(price_violations.head(SAMPLE_SIZE).to_string(index=False))

    price_ranges_pass = price_violations.empty

    expected_dates = set(markets["close_date"])
    quote_dates = set(
        pd.to_datetime(
            quotes["close_date"],
            utc=True,
            errors="raise",
        ).dt.strftime("%Y-%m-%d")
    )
    missing_dates = sorted(expected_dates - quote_dates)
    unexpected_dates = sorted(quote_dates - expected_dates)

    print("\nDate coverage")
    print(f"Expected close dates: {len(expected_dates)}")
    print(f"Close dates represented in quotes: {len(quote_dates)}")
    print(f"Entire missing dates: {len(missing_dates)}")

    if missing_dates:
        print("Missing dates:")

        for close_date in missing_dates:
            print(f"  {close_date}")

    if unexpected_dates:
        print("Unexpected quote dates:")

        for close_date in unexpected_dates:
            print(f"  {close_date}")

    date_coverage_pass = not missing_dates and not unexpected_dates

    endpoint_counts = quotes["source_endpoint"].value_counts(dropna=False)
    historical_count = int(endpoint_counts.get("historical", 0))
    series_count = int(endpoint_counts.get("series", 0))

    print("\nEndpoint routing")
    print("source_endpoint value counts:")
    print(endpoint_counts.to_string())
    print(
        "Historical source present: "
        f"{'YES' if historical_count else 'NO - ZERO ROWS'}"
    )
    print(
        "Series source present: "
        f"{'YES' if series_count else 'NO - ZERO ROWS'}"
    )

    endpoint_routing_pass = historical_count > 0 and series_count > 0

    checks = [
        ("Market coverage", market_coverage_pass),
        ("Candle-count completeness", candle_count_pass),
        ("Bid/ask consistency", bid_ask_pass),
        ("Price ranges", price_ranges_pass),
        ("Date coverage", date_coverage_pass),
        ("Endpoint routing", endpoint_routing_pass),
    ]

    print("\nFinal summary")

    for check_name, passed in checks:
        print(f"{check_name}: {'PASS' if passed else 'FAIL'}")

    print(f"Overall: {'PASS' if all(passed for _, passed in checks) else 'FAIL'}")


if __name__ == "__main__":
    main()
