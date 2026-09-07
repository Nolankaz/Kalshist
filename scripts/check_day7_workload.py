from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realized_vol import EXCHANGES


MARKETS_FOLDER = Path("data/kalshi_markets")
QUOTES_FOLDER = Path("data/kalshi_quotes")
BTC_PRICES_FOLDER = Path("data/btc_prices_1s")
DECISION_HORIZONS = {
    "T-10min": pd.Timedelta(minutes=10),
    "T-5min": pd.Timedelta(minutes=5),
}


def load_daily_files(folder, columns):
    files = sorted(folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {folder}")

    frames = [pd.read_parquet(path, columns=columns) for path in files]
    return pd.concat(frames, ignore_index=True), files


def earliest_usable_btc_timestamp():
    earliest_timestamps = []

    for exchange in EXCHANGES:
        files = sorted((BTC_PRICES_FOLDER / exchange).glob("*.parquet"))

        if not files:
            raise FileNotFoundError(f"No BTC price files found for {exchange}")

        timestamps = pd.read_parquet(files[0], columns=["timestamp"])["timestamp"]
        earliest_timestamps.append(pd.to_datetime(timestamps, utc=True).min())

    return min(earliest_timestamps)


FULL_LOOKBACK_AVAILABLE_FROM = earliest_usable_btc_timestamp() + pd.Timedelta(hours=24)


def latest_quotes_at_horizon(markets, quotes, horizon):
    decisions = markets[["ticker", "close_time"]].copy()
    decisions["decision_time"] = decisions["close_time"] - horizon
    candidates = decisions.merge(quotes[["ticker", "quote_time"]], on="ticker", how="left")
    candidates = candidates[candidates["quote_time"] <= candidates["decision_time"]]
    return candidates.sort_values(["ticker", "quote_time"]).drop_duplicates("ticker", keep="last")


def check_day7_workload():
    markets, market_files = load_daily_files(MARKETS_FOLDER, ["ticker", "close_time"])
    quotes, quote_files = load_daily_files(QUOTES_FOLDER, ["ticker", "period_end_ts"])

    if markets["ticker"].duplicated().any():
        raise ValueError("Settled-market data contains duplicate tickers")

    if quotes.duplicated(["ticker", "period_end_ts"]).any():
        raise ValueError("Quote data contains duplicate ticker and period_end_ts rows")

    if not pd.api.types.is_integer_dtype(quotes["period_end_ts"]):
        raise TypeError(f"period_end_ts must contain Unix seconds, found {quotes['period_end_ts'].dtype}")

    markets["close_time"] = pd.to_datetime(markets["close_time"], utc=True)
    quotes["quote_time"] = pd.to_datetime(quotes["period_end_ts"], unit="s", utc=True)

    available_tickers = {}

    for label, horizon in DECISION_HORIZONS.items():
        latest_quotes = latest_quotes_at_horizon(markets, quotes, horizon)
        available_tickers[label] = set(latest_quotes["ticker"])

    eligible_tickers = set.intersection(*available_tickers.values())
    eligible_markets = markets[markets["ticker"].isin(eligible_tickers)]
    early_eligible_markets = eligible_markets[
        eligible_markets["close_time"] < FULL_LOOKBACK_AVAILABLE_FROM
    ]
    timestamp_count = len(eligible_markets) * len(DECISION_HORIZONS)

    print("Day 7 workload check")
    print(f"Market files: {len(market_files)} ({market_files[0].name} through {market_files[-1].name})")
    print(f"Quote files: {len(quote_files)} ({quote_files[0].name} through {quote_files[-1].name})")
    print("Quote rule: latest candle with period_end_ts <= decision_time")
    print("Decision horizons: T-10min and T-5min")
    print(f"Full 24-hour BTC lookback first available: {FULL_LOOKBACK_AVAILABLE_FROM}")
    print(f"\nTotal settled markets: {len(markets)}")

    for label in DECISION_HORIZONS:
        print(f"Markets with a leakage-safe quote at {label}: {len(available_tickers[label])}")

    print(f"Markets with leakage-safe quotes at both horizons: {len(eligible_markets)}")
    print(f"True Day 7 timestamp count (eligible markets x 2): {timestamp_count}")
    print(
        "Eligible markets before full-lookback availability: "
        f"{len(early_eligible_markets)}"
    )


if __name__ == "__main__":
    check_day7_workload()
