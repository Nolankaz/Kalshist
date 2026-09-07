from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realized_vol import MAX_FFILL_SECONDS, build_vwap_proxy, load_exchange_prices, realized_vol, realized_vol_features


TIMESTAMP = pd.Timestamp("2026-07-01 12:00:00", tz="UTC")
SAMPLE_PATH = Path("data/realized_vol/realized_vol_sample.parquet")


def check_wrapper_divergence():
    single_window_vol = realized_vol(TIMESTAMP, "15min")
    features = realized_vol_features(TIMESTAMP)
    feature_vol = features["15min_vol"]
    absolute_difference = abs(single_window_vol - feature_vol)

    print("Realized volatility wrapper divergence check")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"realized_vol(t, \"15min\"): {single_window_vol:.15f}")
    print(f"realized_vol_features(t)[\"15min_vol\"]: {feature_vol:.15f}")
    print(f"Absolute difference: {absolute_difference:.15f}")
    print(f"15min_n_obs: {features['15min_n_obs']}")


def source_filling_grid_start(timestamp):
    grid_start = (timestamp - pd.Timedelta(hours=24) - pd.Timedelta(seconds=1)).ceil("s")
    history_start = grid_start - pd.Timedelta(seconds=MAX_FFILL_SECONDS)
    history_end = grid_start + pd.Timedelta(seconds=1)
    prices = load_exchange_prices(history_start, history_end)
    proxy = build_vwap_proxy(prices)
    grid = pd.date_range(history_start, grid_start, freq="s", tz="UTC")

    if proxy.empty:
        return grid_start, pd.NaT

    observed_sources = pd.Series(proxy.index, index=proxy.index)
    filled_sources = observed_sources.reindex(grid).ffill(limit=MAX_FFILL_SECONDS)
    return grid_start, filled_sources.loc[grid_start]


def check_batch_left_edge_fill():
    sample = pd.read_parquet(SAMPLE_PATH, columns=["timestamp"])

    if len(sample) != 300:
        raise ValueError(f"Expected 300 sample rows, found {len(sample)}")

    timestamps = pd.to_datetime(sample["timestamp"], utc=True)
    affected = []

    for timestamp in timestamps:
        grid_start, source_timestamp = source_filling_grid_start(timestamp)

        if pd.notna(source_timestamp) and source_timestamp < grid_start:
            source_age_seconds = (grid_start - source_timestamp).total_seconds()
            assert source_age_seconds <= MAX_FFILL_SECONDS
            affected.append({
                "t": timestamp,
                "g0": grid_start,
                "source_timestamp": source_timestamp,
                "source_age_seconds": source_age_seconds,
            })

    affected_count = len(affected)
    unaffected_count = len(sample) - affected_count
    affected_percentage = affected_count / len(sample) * 100

    print("\nBatch-style left-edge forward-fill diagnostic")
    print(f"Sample row count: {len(sample)}")
    print(f"Affected row count: {affected_count}")
    print(f"Unaffected row count: {unaffected_count}")
    print(f"Affected percentage: {affected_percentage:.2f}%")
    print("First 10 affected examples:")

    if affected:
        print(pd.DataFrame(affected).head(10).to_string(index=False))
    else:
        print("None")


if __name__ == "__main__":
    check_wrapper_divergence()
    check_batch_left_edge_fill()
