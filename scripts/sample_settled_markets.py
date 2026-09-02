import pandas as pd

from storage import load_range


START_DATE = "2026-05-27"
END_DATE = "2026-08-25"

SAMPLE_SIZE = 100


def main():
    # Load all saved Kalshi markets in the date range.
    df = load_range(
        "kalshi_markets",
        START_DATE,
        END_DATE
    )

    print(f"Loaded {len(df)} total Kalshi markets")

    # Stop if no data was loaded.
    if df.empty:
        print("No Kalshi market data found.")
        return

    # Show the columns so we know exactly what fields exist.
    print("\nColumns:")
    print(df.columns.tolist())

    # Keep only markets with a settlement result.
    settled = df[df["settlement_result"].notna()].copy()

    print(
        f"\nMarkets with settlement_result: "
        f"{len(settled)}"
    )

    # Keep only markets with an actual BTC expiration/reference value.
    if "expiration_value" in settled.columns:
        settled = settled[settled["expiration_value"].notna()].copy()

        print(
            f"Markets with expiration_value: "
            f"{len(settled)}"
        )
    else:
        print(
            "\nWARNING: expiration_value is not "
            "present in the saved Kalshi data."
        )
        return

    # Don't sample more markets than we actually have.
    sample_size = min(SAMPLE_SIZE, len(settled))

    if sample_size == 0:
        print("No usable settled markets found.")
        return

    # Randomly choose a reproducible sample.
    sample = settled.sample(n=sample_size, random_state=42)

    # Sort chronologically for easier inspection.
    if "close_time" in sample.columns:
        sample = sample.sort_values("close_time")

    print(
        f"\nCreated sample of "
        f"{len(sample)} markets"
    )

    # Show the most useful columns.
    columns_to_show = [
        column
        for column in [
            "ticker",
            "strike",
            "close_time",
            "settlement_result",
            "expiration_value",
            "settlement_value"
        ]
        if column in sample.columns
    ]

    print("\nSample:")
    print(sample[columns_to_show].head(10))


if __name__ == "__main__":
    main()