from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_PATH = Path("data/features/market_features.parquet")
ORDINARY_TICKER = "KXBTC15M-26JUL090245-45"
SPECIFIED_TICKER = "KXBTC15M-26JUN102100-00"
DISPLAY_COLUMNS = [
    "ticker",
    "horizon_minutes",
    "open_time",
    "close_time",
    "decision_time",
    "quote_time",
    "quote_age_seconds",
    "spot",
    "strike",
    "log_moneyness",
    "settlement_result",
    "settlement_value",
    "expiration_value",
    "y",
    "y_from_expiration",
    "target_agrees",
]


def load_market_rows(features, ticker):
    rows = features[features["ticker"] == ticker].sort_values("horizon_minutes", ascending=False)

    if len(rows) != 2 or set(rows["horizon_minutes"]) != {5, 10}:
        raise ValueError(f"Expected exactly T-10 and T-5 rows for {ticker}, found {len(rows)} rows")

    return rows


def print_market(label, rows):
    print(f"\n{label}: {rows.iloc[0]['ticker']}")

    for _, row in rows.iterrows():
        print(f"\nT-{row['horizon_minutes']} row")

        for column in DISPLAY_COLUMNS:
            print(f"{column}: {row[column]}")

        minutes_to_close = (row["close_time"] - row["decision_time"]).total_seconds() / 60
        recomputed_quote_age = (row["decision_time"] - row["quote_time"]).total_seconds()
        sign_matches = np.sign(row["log_moneyness"]) == np.sign(row["spot"] - row["strike"])
        expected_y_from_result = int(row["settlement_result"] == "yes")
        expiration_target_matches = "not defined"

        if pd.notna(row["expiration_value"]) and pd.notna(row["strike"]):
            expiration_target_matches = row["y"] == int(row["expiration_value"] > row["strike"])

        print("Derived checks:")
        print(f"minutes from decision_time to close_time: {minutes_to_close}")
        print(f"open_time < decision_time < close_time: {row['open_time'] < row['decision_time'] < row['close_time']}")
        print(f"quote_time <= decision_time: {row['quote_time'] <= row['decision_time']}")
        print(f"recomputed quote age in seconds: {recomputed_quote_age}")
        print(f"sign(log_moneyness) matches spot - strike: {sign_matches}")
        print(f"y matches settlement_result: {row['y'] == expected_y_from_result}")
        print(f"y matches settlement_value: {row['y'] == row['settlement_value']}")
        print(f"y matches expiration_value > strike where defined: {expiration_target_matches}")


def inspect_day7_markets():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_PATH}")

    features = pd.read_parquet(FEATURE_PATH)
    missing_columns = [column for column in DISPLAY_COLUMNS if column not in features.columns]

    if missing_columns:
        raise ValueError(f"Feature table is missing required columns: {missing_columns}")

    ordinary_rows = load_market_rows(features, ORDINARY_TICKER)
    specified_rows = load_market_rows(features, SPECIFIED_TICKER)

    if ordinary_rows[DISPLAY_COLUMNS].isna().any().any():
        raise ValueError(f"Ordinary market {ORDINARY_TICKER} does not have complete requested data")

    print(f"Feature table: {FEATURE_PATH}")
    print_market("Ordinary complete-data market", ordinary_rows)
    print_market("Specified market", specified_rows)


if __name__ == "__main__":
    inspect_day7_markets()
