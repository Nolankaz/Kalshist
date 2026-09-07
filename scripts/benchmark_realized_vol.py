from pathlib import Path
import random
from statistics import mean, median
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realized_vol import EXCHANGES, realized_vol_features


DAY7_TIMESTAMP_COUNT = 17182

START_TIME = pd.Timestamp("2026-05-28 00:00:00", tz="UTC")
BENCHMARK_TIMESTAMP_COUNT = 30
MIN_DISTINCT_DATES = 10
FIRST_REPEAT_DATE_COUNT = 10
RANDOM_SEED = 20260906
BTC_DATA_ROOT = Path("data/btc_prices_1s")


def latest_safe_timestamp():
    latest_by_exchange = {}

    for exchange in EXCHANGES:
        files = sorted((BTC_DATA_ROOT / exchange).glob("*.parquet"))

        if not files:
            raise FileNotFoundError(f"No BTC data found for {exchange}")

        timestamps = pd.read_parquet(files[-1], columns=["timestamp"])["timestamp"]
        latest_by_exchange[exchange] = pd.to_datetime(timestamps, utc=True).max().floor("s")

    return min(latest_by_exchange.values()), latest_by_exchange


def eligible_dates(start_time, end_time):
    dates = list(pd.date_range(start=start_time.normalize(), end=end_time.normalize(), freq="D", tz="UTC"))

    if len(dates) < MIN_DISTINCT_DATES:
        raise ValueError(f"Usable range contains only {len(dates)} UTC dates")

    return dates


def random_timestamp_on_date(rng, date, start_time, end_time):
    lower = max(date, start_time)
    upper = min(date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), end_time)
    seconds = int((upper - lower).total_seconds())
    return lower + pd.Timedelta(seconds=rng.randint(0, seconds))


def sample_benchmark_timestamps(rng, start_time, end_time):
    total_seconds = int((end_time - start_time).total_seconds())

    while True:
        timestamps = set()

        while len(timestamps) < BENCHMARK_TIMESTAMP_COUNT:
            seconds = rng.randint(0, total_seconds)
            timestamps.add(start_time + pd.Timedelta(seconds=seconds))

        if len({timestamp.date() for timestamp in timestamps}) >= MIN_DISTINCT_DATES:
            timestamps = sorted(timestamps)
            rng.shuffle(timestamps)
            return timestamps


def sample_separated_dates(rng, dates, count):
    candidates = dates.copy()
    rng.shuffle(candidates)
    selected = []

    for date in candidates:
        if all(abs((date - other).days) >= 2 for other in selected):
            selected.append(date)

        if len(selected) == count:
            return selected

    raise ValueError(f"Usable range cannot provide {count} dates separated by at least two days")


def time_feature_call(timestamp):
    start = time.perf_counter()
    features = realized_vol_features(timestamp)
    runtime = time.perf_counter() - start
    return features, runtime


def print_runtime_summary(runtimes, timestamps):
    distinct_dates = len({timestamp.date() for timestamp in timestamps})

    print("\nOverall runtime statistics (seconds per call):")
    print(f"Mean: {mean(runtimes):.6f}")
    print(f"Median: {median(runtimes):.6f}")
    print(f"P95: {np.percentile(runtimes, 95):.6f}")
    print(f"Min: {min(runtimes):.6f}")
    print(f"Max: {max(runtimes):.6f}")
    print(f"Number of timestamps: {len(timestamps)}")
    print(f"Number of distinct UTC dates: {distinct_dates}")


def benchmark_realized_vol():
    rng = random.Random(RANDOM_SEED)
    end_time, latest_by_exchange = latest_safe_timestamp()

    if end_time < START_TIME:
        raise ValueError(f"Latest safely covered timestamp {end_time} is before {START_TIME}")

    dates = eligible_dates(START_TIME, end_time)
    pair_dates = sample_separated_dates(rng, dates, FIRST_REPEAT_DATE_COUNT)
    timestamps = sample_benchmark_timestamps(rng, START_TIME, end_time)

    print("Realized volatility reference benchmark")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Usable UTC range: {START_TIME} through {end_time}")
    print("Latest BTC timestamp by exchange:")

    for exchange, latest in latest_by_exchange.items():
        print(f"  {exchange}: {latest}")

    print("\nFirst-touch versus repeat timings:")
    first_touch_runtimes = []
    repeat_runtimes = []
    pair_feature_results = []

    for date in pair_dates:
        first_timestamp = random_timestamp_on_date(rng, date, START_TIME, end_time)
        repeat_timestamp = random_timestamp_on_date(rng, date, START_TIME, end_time)

        while repeat_timestamp == first_timestamp:
            repeat_timestamp = random_timestamp_on_date(rng, date, START_TIME, end_time)

        first_features, first_runtime = time_feature_call(first_timestamp)
        repeat_features, repeat_runtime = time_feature_call(repeat_timestamp)
        pair_feature_results.append((first_features, repeat_features))
        first_touch_runtimes.append(first_runtime)
        repeat_runtimes.append(repeat_runtime)
        print(
            f"  {date.date()}: first {first_timestamp} = {first_runtime:.6f}s; "
            f"repeat {repeat_timestamp} = {repeat_runtime:.6f}s"
        )

    first_touch_mean = mean(first_touch_runtimes)
    first_touch_median = median(first_touch_runtimes)
    repeat_mean = mean(repeat_runtimes)
    repeat_median = median(repeat_runtimes)

    print("\nFirst-touch versus repeat summary (seconds per call):")
    print(f"First-touch mean: {first_touch_mean:.6f}")
    print(f"First-touch median: {first_touch_median:.6f}")
    print(f"Repeat mean: {repeat_mean:.6f}")
    print(f"Repeat median: {repeat_median:.6f}")
    print(f"Mean difference (first-touch - repeat): {first_touch_mean - repeat_mean:.6f}")
    print(f"Median difference (first-touch - repeat): {first_touch_median - repeat_median:.6f}")

    print("\nRandom benchmark timestamp timings:")
    runtimes = []
    feature_results = []

    for timestamp in timestamps:
        features, runtime = time_feature_call(timestamp)
        feature_results.append(features)
        runtimes.append(runtime)
        print(f"  {timestamp}: {runtime:.6f}s")

    print_runtime_summary(runtimes, timestamps)

    if DAY7_TIMESTAMP_COUNT is not None:
        mean_runtime = mean(runtimes)
        projected_seconds = mean_runtime * DAY7_TIMESTAMP_COUNT
        projected_minutes = projected_seconds / 60
        print("\nDay 7 runtime projection:")
        print(f"DAY7_TIMESTAMP_COUNT: {DAY7_TIMESTAMP_COUNT}")
        print(f"Projected seconds: {projected_seconds:.2f}")
        print(f"Projected minutes: {projected_minutes:.2f}")


if __name__ == "__main__":
    benchmark_realized_vol()
