from pathlib import Path
import sys
import time
import tracemalloc

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_day7_workload import (
    DECISION_HORIZONS,
    MARKETS_FOLDER,
    QUOTES_FOLDER,
    latest_quotes_at_horizon,
    load_daily_files,
)
from scripts.realized_vol import MIN_COVERAGE, WINDOWS, realized_vol_features
from scripts.realized_vol_batch import realized_vol_features_batch


REPRESENTATIVE_DATE = pd.Timestamp("2026-07-15", tz="UTC")
EARLY_DATE = pd.Timestamp("2026-05-27", tz="UTC")
NAIVE_SECONDS_PER_CALL = 0.025165
TOTAL_DAY7_TIMESTAMPS = 17182
TOTAL_DATASET_DAYS = 91
SEAM_TIMESTAMP_COUNT = 5
VOL_TOLERANCE = 1e-10
COVERAGE_TOLERANCE = 1e-12
VOL_COLUMNS = [
    column
    for window in WINDOWS
    for column in (f"{window}_vol", f"{window}_ewma_vol")
]


def load_workload_inputs():
    markets, _ = load_daily_files(MARKETS_FOLDER, ["ticker", "close_time"])
    quotes, _ = load_daily_files(QUOTES_FOLDER, ["ticker", "period_end_ts"])
    markets["close_time"] = pd.to_datetime(markets["close_time"], utc=True)
    quotes["quote_time"] = pd.to_datetime(quotes["period_end_ts"], unit="s", utc=True)
    return markets, quotes


def build_day7_workload(markets, quotes, close_date):
    day_end = close_date + pd.Timedelta(days=1)
    day_markets = markets[
        (markets["close_time"] >= close_date)
        & (markets["close_time"] < day_end)
    ].copy()
    available_tickers = {}

    for label, horizon in DECISION_HORIZONS.items():
        latest_quotes = latest_quotes_at_horizon(day_markets, quotes, horizon)
        available_tickers[label] = set(latest_quotes["ticker"])

    eligible_tickers = set.intersection(*available_tickers.values())
    eligible_markets = day_markets[day_markets["ticker"].isin(eligible_tickers)]
    eligible_markets = eligible_markets.sort_values(["close_time", "ticker"])
    decision_timestamps = [
        market.close_time - horizon
        for market in eligible_markets.itertuples()
        for horizon in DECISION_HORIZONS.values()
    ]
    return eligible_markets, pd.DatetimeIndex(sorted(decision_timestamps), name="timestamp")


def compare_feature_row(timestamp, reference, batch_row):
    failures = []

    for window in WINDOWS:
        n_obs_column = f"{window}_n_obs"
        coverage_column = f"{window}_coverage"

        if int(reference[n_obs_column]) != int(batch_row[n_obs_column]):
            failures.append(
                f"{n_obs_column}: reference={reference[n_obs_column]!r}, batch={batch_row[n_obs_column]!r}"
            )

        coverage_difference = abs(reference[coverage_column] - batch_row[coverage_column])

        if coverage_difference > COVERAGE_TOLERANCE:
            failures.append(
                f"{coverage_column}: reference={reference[coverage_column]!r}, "
                f"batch={batch_row[coverage_column]!r}, difference={coverage_difference!r}"
            )

        for suffix in ("vol", "ewma_vol"):
            column = f"{window}_{suffix}"
            reference_is_nan = pd.isna(reference[column])
            batch_is_nan = pd.isna(batch_row[column])

            if reference_is_nan != batch_is_nan:
                failures.append(
                    f"{column}: NaN mismatch, reference={reference[column]!r}, batch={batch_row[column]!r}"
                )
            elif not reference_is_nan:
                difference = abs(reference[column] - batch_row[column])

                if difference > VOL_TOLERANCE:
                    failures.append(
                        f"{column}: reference={reference[column]!r}, "
                        f"batch={batch_row[column]!r}, difference={difference!r}"
                    )

    print(f"{timestamp}: {'PASS' if not failures else 'FAIL'}")

    for failure in failures:
        print(f"  {failure}")

    return not failures


def check_day_boundary_seam(timestamps, batch_features):
    seam_results = []

    print("\nDay-boundary seam validation")

    for timestamp in timestamps[:SEAM_TIMESTAMP_COUNT]:
        reference = realized_vol_features(timestamp)
        batch_row = batch_features.loc[timestamp]
        seam_results.append(compare_feature_row(timestamp, reference, batch_row))

    return all(seam_results) and len(seam_results) >= SEAM_TIMESTAMP_COUNT


def check_early_data_boundary(markets, quotes):
    _, early_timestamps = build_day7_workload(markets, quotes, EARLY_DATE)
    timestamps_on_early_date = early_timestamps[early_timestamps.normalize() == EARLY_DATE]

    if timestamps_on_early_date.empty:
        raise ValueError(f"No eligible Day 7 decision timestamp found on {EARLY_DATE.date()}")

    timestamp = timestamps_on_early_date[0]
    batch_features = realized_vol_features_batch([timestamp])
    batch_row = batch_features.iloc[0]
    reference = realized_vol_features(timestamp)

    print("\nEarly-data boundary check")
    print(f"Timestamp: {timestamp}")
    print(f"24hr_vol: {batch_row['24hr_vol']}")
    print(f"24hr_ewma_vol: {batch_row['24hr_ewma_vol']}")
    print(f"24hr_n_obs: {int(batch_row['24hr_n_obs'])}")
    print(f"24hr_coverage: {batch_row['24hr_coverage']}")

    insufficient_coverage = batch_row["24hr_coverage"] < MIN_COVERAGE
    volatility_is_nan = pd.isna(batch_row["24hr_vol"]) and pd.isna(batch_row["24hr_ewma_vol"])
    reference_matches = compare_feature_row(timestamp, reference, batch_row)
    boundary_pass = insufficient_coverage and volatility_is_nan and reference_matches
    print(f"Early-data boundary: {'PASS' if boundary_pass else 'FAIL'}")
    return boundary_pass


def benchmark_realized_vol_batch():
    markets, quotes = load_workload_inputs()
    eligible_markets, day_timestamps = build_day7_workload(
        markets,
        quotes,
        REPRESENTATIVE_DATE,
    )

    if day_timestamps.empty:
        raise ValueError(f"No eligible Day 7 timestamps found for {REPRESENTATIVE_DATE.date()}")

    print("Day 6 section 3.2 batch benchmark")
    print(f"Representative UTC close date: {REPRESENTATIVE_DATE.date()}")
    print(f"Eligible markets that day: {len(eligible_markets)}")
    print(f"Decision timestamps: {len(day_timestamps)}")

    tracemalloc.start()
    start_time = time.perf_counter()
    batch_features = realized_vol_features_batch(day_timestamps)
    full_day_runtime = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    seconds_per_timestamp = full_day_runtime / len(day_timestamps)
    timestamps_per_second = len(day_timestamps) / full_day_runtime
    peak_memory_mb = peak_bytes / 1024 ** 2

    print("\nMeasured full-day batch performance")
    print(f"Wall-clock seconds: {full_day_runtime:.6f}")
    print(f"Seconds per timestamp: {seconds_per_timestamp:.9f}")
    print(f"Timestamps per second: {timestamps_per_second:.2f}")
    print(f"Peak Python-tracked memory: {peak_memory_mb:.2f} MB")

    naive_projected_seconds = NAIVE_SECONDS_PER_CALL * TOTAL_DAY7_TIMESTAMPS
    naive_projected_minutes = naive_projected_seconds / 60
    batch_projected_seconds = full_day_runtime * TOTAL_DATASET_DAYS
    batch_projected_minutes = batch_projected_seconds / 60
    speedup_factor = naive_projected_seconds / batch_projected_seconds

    print("\nProjection based on one representative full UTC day")
    print(f"Naive projected total seconds: {naive_projected_seconds:.2f}")
    print(f"Naive projected total minutes: {naive_projected_minutes:.2f}")
    print(f"Batch projected total seconds (full-day runtime x 91): {batch_projected_seconds:.2f}")
    print(f"Batch projected total minutes: {batch_projected_minutes:.2f}")
    print(f"Projected speedup factor: {speedup_factor:.2f}x")

    seam_pass = check_day_boundary_seam(day_timestamps, batch_features)
    early_boundary_pass = check_early_data_boundary(markets, quotes)

    print("\nFinal summary")
    print(f"Full-day runtime: {full_day_runtime:.6f} seconds")
    print(f"Peak Python-tracked memory: {peak_memory_mb:.2f} MB")
    print(f"Projected full Day 7 batch runtime: {batch_projected_seconds:.2f} seconds ({batch_projected_minutes:.2f} minutes)")
    print(f"Projected speedup: {speedup_factor:.2f}x")
    print(f"Day-boundary seam: {'PASS' if seam_pass else 'FAIL'}")
    print(f"Early-data boundary: {'PASS' if early_boundary_pass else 'FAIL'}")

    if not seam_pass or not early_boundary_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    benchmark_realized_vol_batch()
