from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realized_vol import (
    EWMA_HALFLIVES,
    EXCHANGES,
    EXPECTED_OBS,
    MAX_FFILL_SECONDS,
    MIN_COVERAGE,
    SECONDS_PER_YEAR,
    WINDOWS,
    build_vwap_proxy,
    load_exchange_prices,
)


FEATURE_COLUMNS = [
    column
    for window in WINDOWS
    for column in (
        f"{window}_vol",
        f"{window}_ewma_vol",
        f"{window}_n_obs",
        f"{window}_coverage",
    )
]
MAX_WINDOW = max(WINDOWS.values())
ONE_SECOND = pd.Timedelta(seconds=1)
ONE_NANOSECOND = pd.Timedelta(nanoseconds=1)
ANNUALIZATION_FACTOR = np.sqrt(SECONDS_PER_YEAR)


def normalize_timestamps(timestamps):
    timestamps = list(timestamps)

    if not timestamps:
        return pd.DatetimeIndex([], tz="UTC", name="timestamp")

    normalized = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True), name="timestamp")

    if normalized.hasnans:
        raise ValueError("timestamps must not contain NaT")

    return normalized


def feature_end_labels(timestamps):
    # The W return labels for [t - W, t) end at this exact integer second.
    return (timestamps - ONE_NANOSECOND).floor("s")


def reference_grid_starts(timestamps):
    return (timestamps - MAX_WINDOW - ONE_SECOND).ceil("s")


def required_grid_bounds(timestamps):
    grid_starts = reference_grid_starts(timestamps)
    grid_start = grid_starts.min() - pd.Timedelta(seconds=MAX_FFILL_SECONDS)
    grid_end = feature_end_labels(timestamps).max()
    return grid_start, grid_end


def load_proxy_for_batch(timestamps):
    grid_start, grid_end = required_grid_bounds(timestamps)
    prices = load_exchange_prices(grid_start, grid_end + ONE_SECOND)
    return build_vwap_proxy(prices), grid_start, grid_end


def build_filled_price_grid(proxy, grid_start, grid_end):
    grid = pd.date_range(grid_start, grid_end, freq="s", tz="UTC")

    if proxy.empty:
        filled_price = pd.Series(np.nan, index=grid, name="price")
        fill_source = pd.Series(pd.NaT, index=grid, dtype="datetime64[ns, UTC]", name="fill_source")
        return filled_price, fill_source

    raw_price = proxy["price"].reindex(grid)
    observed_sources = pd.Series(proxy.index, index=proxy.index, name="fill_source")
    observed_sources = observed_sources.where(proxy["price"].notna())
    raw_fill_source = observed_sources.reindex(grid)

    filled_price = raw_price.ffill(limit=MAX_FFILL_SECONDS)
    fill_source = raw_fill_source.ffill(limit=MAX_FFILL_SECONDS)
    return filled_price, fill_source


def truncated_ewm_sum(values, rho, window_size):
    recursive = np.empty(len(values), dtype=float)
    state = 0.0

    # This stable recurrence avoids exponentially rescaling a cumulative sum.
    for position, value in enumerate(values):
        state = value + rho * state
        recursive[position] = state

    # Subtract the recursively accumulated tail older than the fixed window.
    truncated = recursive.copy()

    if len(values) > window_size:
        truncated[window_size:] -= rho ** window_size * recursive[:-window_size]

    return truncated


def remove_values_from_sample_stats(count, current_mean, current_std, values):
    count = int(count)
    mean = current_mean
    m2 = current_std ** 2 * (count - 1) if count >= 2 else 0.0

    for value in values:
        new_count = count - 1

        if new_count == 0:
            count = 0
            mean = np.nan
            m2 = 0.0
            continue

        new_mean = (count * mean - value) / new_count
        m2 -= (value - mean) * (value - new_mean)
        count = new_count
        mean = new_mean

    if count < 2:
        return count, np.nan

    if m2 < 0 and abs(m2) < 1e-24:
        m2 = 0.0

    return count, np.sqrt(m2 / (count - 1))


def repair_24hr_left_edge(timestamps, end_labels, log_returns, fill_source, counts, means, stds, numerators, denominators, rho):
    counts = counts.copy()
    stds = stds.copy()
    numerators = numerators.copy()
    denominators = denominators.copy()

    for position, (timestamp, end_label) in enumerate(zip(timestamps, end_labels)):
        # The wrapper grid starts at g0, so it cannot use fills sourced before g0.
        # Such a source can affect only the first MAX_FFILL_SECONDS + 1 returns.
        grid_start = (timestamp - MAX_WINDOW - ONE_SECOND).ceil("s")
        window_start = (timestamp - MAX_WINDOW).ceil("s")
        repair_end = min(end_label, grid_start + pd.Timedelta(seconds=MAX_FFILL_SECONDS + 1))
        repair_labels = pd.date_range(window_start, repair_end, freq="s", tz="UTC")

        current_sources = fill_source.reindex(repair_labels).to_numpy()
        previous_sources = fill_source.reindex(repair_labels - ONE_SECOND).to_numpy()
        depends_on_earlier_source = np.array([
            (not pd.isna(current) and current < grid_start)
            or (not pd.isna(previous) and previous < grid_start)
            for current, previous in zip(current_sources, previous_sources)
        ])
        candidate_returns = log_returns.reindex(repair_labels)
        invalid_returns = candidate_returns[depends_on_earlier_source & candidate_returns.notna().to_numpy()]

        if invalid_returns.empty:
            continue

        counts[position], stds[position] = remove_values_from_sample_stats(
            counts[position], means[position], stds[position], invalid_returns.to_numpy()
        )
        ages = (end_label - invalid_returns.index).total_seconds()
        weights = rho ** ages
        numerators[position] -= np.sum(invalid_returns.to_numpy() ** 2 * weights)
        denominators[position] -= np.sum(weights)

    return counts, stds, numerators, denominators


def compute_features_from_price_grid(timestamps, filled_price, fill_source):
    timestamps = normalize_timestamps(timestamps)

    if not filled_price.index.equals(fill_source.index):
        raise ValueError("filled_price and fill_source must use the same grid")

    end_labels = feature_end_labels(timestamps)

    if not end_labels.isin(filled_price.index).all():
        raise ValueError("price grid does not contain every requested feature-end label")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(filled_price / filled_price.shift(1))

    result = pd.DataFrame(index=timestamps)
    return_values = log_returns.to_numpy()
    valid_values = log_returns.notna().to_numpy(dtype=float)
    squared_values = np.where(log_returns.notna(), return_values ** 2, 0.0)

    for window, duration in WINDOWS.items():
        window_size = int(duration.total_seconds())
        rolling = log_returns.rolling(window_size, min_periods=1)
        count_series = log_returns.rolling(window_size).count()
        mean_series = rolling.mean()
        std_series = rolling.std(ddof=1)
        counts = count_series.reindex(end_labels).to_numpy()
        means = mean_series.reindex(end_labels).to_numpy()
        stds = std_series.reindex(end_labels).to_numpy()

        halflife_seconds = EWMA_HALFLIVES[window].total_seconds()
        rho = 2 ** (-1 / halflife_seconds)
        numerator_series = pd.Series(truncated_ewm_sum(squared_values, rho, window_size), index=log_returns.index)
        denominator_series = pd.Series(truncated_ewm_sum(valid_values, rho, window_size), index=log_returns.index)
        numerators = numerator_series.reindex(end_labels).to_numpy()
        denominators = denominator_series.reindex(end_labels).to_numpy()

        if window == "24hr":
            counts, stds, numerators, denominators = repair_24hr_left_edge(
                timestamps,
                end_labels,
                log_returns,
                fill_source,
                counts,
                means,
                stds,
                numerators,
                denominators,
                rho,
            )

        coverage = counts / EXPECTED_OBS[window]
        simple_vol = stds * ANNUALIZATION_FACTOR
        ewma_variance = np.divide(
            numerators,
            denominators,
            out=np.full_like(numerators, np.nan),
            where=denominators > 0,
        )
        ewma_vol = np.sqrt(ewma_variance) * ANNUALIZATION_FACTOR
        insufficient_coverage = coverage < MIN_COVERAGE
        simple_vol[insufficient_coverage] = np.nan
        ewma_vol[insufficient_coverage] = np.nan

        result[f"{window}_vol"] = simple_vol
        result[f"{window}_ewma_vol"] = ewma_vol
        result[f"{window}_n_obs"] = counts.astype("int64")
        result[f"{window}_coverage"] = coverage

    return result[FEATURE_COLUMNS]


def compute_features_from_proxy(timestamps, proxy):
    timestamps = normalize_timestamps(timestamps)
    grid_start, grid_end = required_grid_bounds(timestamps)
    filled_price, fill_source = build_filled_price_grid(proxy, grid_start, grid_end)
    return compute_features_from_price_grid(timestamps, filled_price, fill_source)


def empty_feature_frame():
    frame = pd.DataFrame({
        column: pd.Series(dtype="int64" if column.endswith("_n_obs") else "float64")
        for column in FEATURE_COLUMNS
    })
    frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return frame


def realized_vol_features_batch(timestamps):
    timestamps = normalize_timestamps(timestamps)

    if timestamps.empty:
        return empty_feature_frame()

    requests = pd.DataFrame({
        "position": np.arange(len(timestamps)),
        "timestamp": timestamps,
        "date": timestamps.normalize(),
    })
    values = np.empty((len(timestamps), len(FEATURE_COLUMNS)), dtype=float)

    for _, group in requests.groupby("date", sort=False):
        group_timestamps = pd.DatetimeIndex(group["timestamp"])
        proxy, grid_start, grid_end = load_proxy_for_batch(group_timestamps)
        filled_price, fill_source = build_filled_price_grid(proxy, grid_start, grid_end)
        group_features = compute_features_from_price_grid(group_timestamps, filled_price, fill_source)
        values[group["position"].to_numpy(), :] = group_features.to_numpy(dtype=float)

    result = pd.DataFrame(values, index=timestamps, columns=FEATURE_COLUMNS)

    for window in WINDOWS:
        result[f"{window}_n_obs"] = result[f"{window}_n_obs"].astype("int64")

    return result
