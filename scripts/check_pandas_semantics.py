import numpy as np
import pandas as pd


TOLERANCE = 1e-12
ROLLING_WINDOW = 5


def check_ewm_with_times():
    offsets_seconds = [0, 3, 8, 17, 29, 44, 61, 79, 102, 128, 159, 195, 238, 286, 341, 403, 472, 548, 631, 721]
    index = pd.Timestamp("2026-01-01 00:00:00", tz="UTC") + pd.to_timedelta(offsets_seconds, unit="s")
    series = pd.Series([1.2, -0.4, 2.1, 0.7, 3.3, -1.0, 1.8, 2.6, 0.2, 4.1, 1.5, -0.8, 2.9, 3.7, 0.5, 1.1, -1.4, 2.3, 3.0, 0.9], index=index)
    halflife = pd.Timedelta(seconds=60)

    pandas_result = series.ewm(halflife=halflife, times=series.index).mean().iloc[-1]
    explicit_adjust_result = series.ewm(halflife=halflife, times=series.index, adjust=True).mean().iloc[-1]
    last_timestamp = series.index[-1]
    halflife_seconds = halflife.total_seconds()
    weights = np.array([
        2 ** (-(last_timestamp - timestamp).total_seconds() / halflife_seconds)
        for timestamp in series.index
    ])
    manual_result = np.sum(series.to_numpy() * weights) / np.sum(weights)

    print("EWM with times")
    print(f"Pandas result: {pandas_result:.15f}")
    print(f"Manual result: {manual_result:.15f}")
    print(f"Absolute difference: {abs(pandas_result - manual_result):.3e}")
    print(f"Explicit adjust=True result: {explicit_adjust_result:.15f}")

    assert abs(pandas_result - manual_result) <= TOLERANCE
    assert abs(pandas_result - explicit_adjust_result) <= TOLERANCE


def check_sample_standard_deviation():
    series = pd.Series([2.0, 4.5, -1.0, 3.25, 7.0, 0.5, 5.75])
    pandas_result = series.std(ddof=1)
    values = series.to_numpy()
    manual_result = np.sqrt(np.sum((values - values.mean()) ** 2) / (len(values) - 1))

    print("\nSample standard deviation")
    print(f"Pandas result: {pandas_result:.15f}")
    print(f"Manual result: {manual_result:.15f}")
    print(f"Absolute difference: {abs(pandas_result - manual_result):.3e}")

    assert abs(pandas_result - manual_result) <= TOLERANCE


def build_rolling_series():
    index = pd.date_range("2026-01-01 00:00:00", periods=20, freq="s", tz="UTC")
    values = [1.0, np.nan, 2.5, 4.0, np.nan, 3.5, 6.0, 5.0, np.nan, 7.5, 8.0, 6.5, np.nan, 9.0, 10.0, 8.5, 11.0, np.nan, 12.5, 13.0]
    return pd.Series(values, index=index, name="original_value")


def check_rolling_count(series):
    pandas_count = series.rolling(ROLLING_WINDOW).count()
    manual_count = pd.Series(np.nan, index=series.index, dtype=float)

    for position in range(ROLLING_WINDOW - 1, len(series)):
        window = series.iloc[position - ROLLING_WINDOW + 1:position + 1]
        manual_count.iloc[position] = window.notna().sum()

    results = pd.DataFrame({
        "original_value": series,
        "pandas_rolling_count": pandas_count,
        "manual_rolling_count": manual_count,
    })

    print("\nRolling count with NaNs")
    print(results.to_string())

    pd.testing.assert_series_equal(pandas_count, manual_count, check_names=False)


def check_rolling_standard_deviation(series):
    pandas_std = series.rolling(ROLLING_WINDOW, min_periods=1).std(ddof=1)
    manual_std = pd.Series(np.nan, index=series.index, dtype=float)

    for position in range(len(series)):
        window_start = max(0, position - ROLLING_WINDOW + 1)
        values = series.iloc[window_start:position + 1].dropna().to_numpy()

        if len(values) >= 2:
            manual_std.iloc[position] = np.sqrt(np.sum((values - values.mean()) ** 2) / (len(values) - 1))

    absolute_difference = (pandas_std - manual_std).abs()
    results = pd.DataFrame({
        "pandas_rolling_std": pandas_std,
        "manual_rolling_std": manual_std,
        "absolute_difference": absolute_difference,
    })

    print("\nRolling standard deviation with NaNs")
    print(results.to_string())

    assert pandas_std.isna().equals(manual_std.isna())
    np.testing.assert_allclose(pandas_std.dropna(), manual_std.dropna(), rtol=0, atol=TOLERANCE)


def check_reference_interval_boundaries():
    window = pd.Timedelta(minutes=5)
    cases = [
        (
            pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 11:55:00", tz="UTC"),
            pd.Timestamp("2026-01-01 11:59:59", tz="UTC"),
        ),
        (
            pd.Timestamp("2026-01-01 11:14:14.842809", tz="UTC"),
            pd.Timestamp("2026-01-01 11:09:15", tz="UTC"),
            pd.Timestamp("2026-01-01 11:14:14", tz="UTC"),
        ),
    ]

    print("\nReference interval boundary checks")

    for timestamp, expected_lower, expected_upper in cases:
        window_start = timestamp - window
        lower = window_start.ceil("s")
        upper = (timestamp - pd.Timedelta(nanoseconds=1)).floor("s")
        labels = pd.date_range(lower, upper, freq="s")
        surrounding_grid = pd.date_range(lower - pd.Timedelta(seconds=2), upper + pd.Timedelta(seconds=2), freq="s")
        filtered_labels = surrounding_grid[(surrounding_grid >= window_start) & (surrounding_grid < timestamp)]

        assert lower == expected_lower
        assert upper == expected_upper
        assert len(labels) == 300
        assert (labels >= window_start).all()
        assert (labels < timestamp).all()
        pd.testing.assert_index_equal(filtered_labels, labels)

        print(f"t: {timestamp}")
        print(f"t - W: {window_start}")
        print(f"lower: {lower}")
        print(f"upper: {upper}")
        print(f"number of labels: {len(labels)}")
        print(f"first label: {labels[0]}")
        print(f"last label: {labels[-1]}\n")

    print("PASS: all reference interval boundary checks succeeded.")


def check_pandas_semantics():
    check_ewm_with_times()
    check_sample_standard_deviation()
    rolling_series = build_rolling_series()
    check_rolling_count(rolling_series)
    check_rolling_standard_deviation(rolling_series)
    check_reference_interval_boundaries()
    print("\nPASS: all pandas and NumPy semantics checks succeeded.")


if __name__ == "__main__":
    check_pandas_semantics()
