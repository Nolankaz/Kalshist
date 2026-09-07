from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realized_vol import (
    EWMA_HALFLIVES,
    EXPECTED_OBS,
    MIN_COVERAGE,
    SECONDS_PER_YEAR,
    WINDOWS as REFERENCE_WINDOWS,
    build_vwap_proxy,
    load_exchange_prices,
    make_1s_proxy,
    realized_vol_features,
)
from scripts.realized_vol_batch import compute_features_from_proxy, required_grid_bounds, realized_vol_features_batch


SAMPLE_PATH = Path("data/realized_vol/realized_vol_sample.parquet")
WINDOWS = ["5min", "15min", "1hr", "4hr", "24hr"]
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
VOL_TOLERANCE = 1e-10
COVERAGE_TOLERANCE = 1e-12
EXPECTED_ROWS = 300
MAX_DIAGNOSTIC_ROWS = 10
LEAKAGE_TIMESTAMP = pd.Timestamp("2026-07-15 12:00:00", tz="UTC")
FUTURE_LOAD_END = LEAKAGE_TIMESTAMP + pd.Timedelta(minutes=5)
PAST_PERTURBATION_START = LEAKAGE_TIMESTAMP - pd.Timedelta(minutes=2)
PAST_PERTURBATION_END = LEAKAGE_TIMESTAMP - pd.Timedelta(minutes=1)
VOL_COLUMNS = [column for column in FEATURE_COLUMNS if column.endswith("_vol")]
N_OBS_COLUMNS = [column for column in FEATURE_COLUMNS if column.endswith("_n_obs")]
COVERAGE_COLUMNS = [column for column in FEATURE_COLUMNS if column.endswith("_coverage")]


def format_difference(value):
    if pd.isna(value):
        return "n/a"

    return f"{value:.15e}"


def print_mismatch_examples(column, timestamps, stored_values, candidate_values, mismatch_mask, candidate_name):
    differences = (candidate_values - stored_values).abs()
    examples = pd.DataFrame({
        "timestamp": timestamps,
        "stored_value": stored_values,
        f"{candidate_name}_value": candidate_values,
        "absolute_difference": differences,
    })
    examples = examples.loc[mismatch_mask].head(MAX_DIAGNOSTIC_ROWS)
    print("  First mismatching rows:")
    print(examples.to_string(index=False))


def compare_feature_frames(stored, candidate, timestamps, candidate_name):
    all_columns_pass = True

    for column in FEATURE_COLUMNS:
        stored_values = stored[column].reset_index(drop=True)
        candidate_values = candidate[column].reset_index(drop=True)

        if column.endswith("_n_obs"):
            mismatch_mask = stored_values.ne(candidate_values)
            mismatch_count = int(mismatch_mask.sum())
            max_difference = (candidate_values - stored_values).abs().max()
            column_pass = mismatch_count == 0
            print(
                f"{column}: max_abs_integer_difference={format_difference(max_difference)}, "
                f"mismatch_count={mismatch_count}, {'PASS' if column_pass else 'FAIL'}"
            )
        elif column.endswith("_coverage"):
            nan_mismatch = stored_values.isna().ne(candidate_values.isna())
            both_present = stored_values.notna() & candidate_values.notna()
            differences = (candidate_values - stored_values).abs()
            value_mismatch = both_present & differences.gt(COVERAGE_TOLERANCE)
            mismatch_mask = nan_mismatch | value_mismatch
            mismatch_count = int(mismatch_mask.sum())
            max_difference = differences[both_present].max()
            column_pass = mismatch_count == 0
            print(
                f"{column}: max_abs_difference={format_difference(max_difference)}, "
                f"mismatch_count={mismatch_count}, {'PASS' if column_pass else 'FAIL'}"
            )
        elif column.endswith("_vol"):
            nan_mismatch = stored_values.isna().ne(candidate_values.isna())
            both_present = stored_values.notna() & candidate_values.notna()
            differences = (candidate_values - stored_values).abs()
            value_mismatch = both_present & differences.gt(VOL_TOLERANCE)
            mismatch_mask = nan_mismatch | value_mismatch
            mismatch_count = int(mismatch_mask.sum())
            nan_mismatch_count = int(nan_mismatch.sum())
            max_difference = differences[both_present].max()
            column_pass = mismatch_count == 0 and nan_mismatch_count == 0
            print(
                f"{column}: max_abs_difference={format_difference(max_difference)}, "
                f"mismatch_count={mismatch_count}, nan_mask_mismatch_count={nan_mismatch_count}, "
                f"{'PASS' if column_pass else 'FAIL'}"
            )
        else:
            raise ValueError(f"Unrecognized feature column: {column}")

        if not column_pass:
            print_mismatch_examples(
                column,
                timestamps,
                stored_values,
                candidate_values,
                mismatch_mask,
                candidate_name,
            )
            all_columns_pass = False

    return all_columns_pass


def build_reference_frame(timestamps):
    rows = []

    for timestamp in timestamps:
        rows.append(realized_vol_features(timestamp))

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def print_order_mismatches(stored_timestamps, batch_index):
    order_mismatch = stored_timestamps != batch_index
    examples = pd.DataFrame({
        "stored_timestamp": stored_timestamps[order_mismatch],
        "batch_timestamp": batch_index[order_mismatch],
    }).head(MAX_DIAGNOSTIC_ROWS)
    print("First timestamp-order mismatches:")
    print(examples.to_string(index=False))


def check_realized_vol_batch():
    stored_sample = pd.read_parquet(SAMPLE_PATH)

    if len(stored_sample) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} stored rows, found {len(stored_sample)}")

    missing_columns = set(["timestamp", *FEATURE_COLUMNS]) - set(stored_sample.columns)

    if missing_columns:
        raise ValueError(f"Stored sample is missing columns: {sorted(missing_columns)}")

    stored_timestamps = pd.DatetimeIndex(pd.to_datetime(stored_sample["timestamp"], utc=True), name="timestamp")
    stored_features = stored_sample[FEATURE_COLUMNS].reset_index(drop=True)

    print("STEP 0 - REFERENCE VS STORED")
    reference_features = build_reference_frame(stored_timestamps)
    reference_pass = compare_feature_frames(
        stored_features,
        reference_features,
        stored_timestamps,
        "reference",
    )
    print(f"Reference vs stored: {'PASS' if reference_pass else 'FAIL'}")

    if not reference_pass:
        print("\nFINAL FAIL: reference output does not match the stored sample; batch comparison was not run.")
        raise SystemExit(1)

    print("\nSTEP 1 - BATCH VS STORED")
    batch_features = realized_vol_features_batch(stored_timestamps)
    expected_index = stored_timestamps
    row_count_matches = len(batch_features) == len(stored_features)
    column_order_matches = list(batch_features.columns) == FEATURE_COLUMNS
    timestamp_order_matches = row_count_matches and batch_features.index.equals(expected_index)
    print(f"Batch row count: {len(batch_features)} ({'PASS' if row_count_matches else 'FAIL'})")
    print(f"Batch feature columns and order: {'PASS' if column_order_matches else 'FAIL'}")
    print(f"Batch timestamp order: {'PASS' if timestamp_order_matches else 'FAIL'}")

    if not row_count_matches or not column_order_matches:
        print("\nFINAL FAIL: batch shape or columns prevent a valid column-by-column comparison.")
        raise SystemExit(1)

    if not timestamp_order_matches:
        print_order_mismatches(expected_index, batch_features.index)

    batch_features = batch_features[FEATURE_COLUMNS].reset_index(drop=True)

    print("\nSTEP 2 - DETAILED COLUMN-BY-COLUMN REPORT")
    batch_columns_pass = compare_feature_frames(
        stored_features,
        batch_features,
        stored_timestamps,
        "batch",
    )
    overall_pass = reference_pass and row_count_matches and column_order_matches and timestamp_order_matches and batch_columns_pass

    print("\nOverall summary")
    print(f"Total rows compared: {len(stored_features)}")
    print(f"Total columns compared: {len(FEATURE_COLUMNS)}")
    print(f"FINAL {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        raise SystemExit(1)


def reference_features_from_proxy(timestamp, proxy):
    """Test-only in-memory reconstruction of realized_vol_features()."""
    max_window = max(REFERENCE_WINDOWS.values())
    start_time = timestamp - max_window
    load_start_time = start_time - pd.Timedelta(seconds=1)
    visible_proxy = proxy[(proxy.index >= load_start_time) & (proxy.index < timestamp)]

    if visible_proxy.empty or len(visible_proxy) < 2:
        window_returns = {
            window: pd.Series(dtype=float)
            for window in REFERENCE_WINDOWS
        }
    else:
        price_grid = make_1s_proxy(visible_proxy, load_start_time, timestamp)
        log_returns = np.log(price_grid["price"] / price_grid["price"].shift(1)).dropna()
        window_returns = {
            window: log_returns[
                (log_returns.index >= timestamp - duration)
                & (log_returns.index < timestamp)
            ]
            for window, duration in REFERENCE_WINDOWS.items()
        }

    features = {}

    for window, log_returns in window_returns.items():
        n_obs = len(log_returns)
        coverage = n_obs / EXPECTED_OBS[window]

        if coverage < MIN_COVERAGE:
            simple_vol = np.nan
            ewma_vol = np.nan
        else:
            simple_vol = log_returns.std(ddof=1) * np.sqrt(SECONDS_PER_YEAR)
            squared_returns = log_returns ** 2
            ewma_variance = squared_returns.ewm(
                halflife=EWMA_HALFLIVES[window],
                times=squared_returns.index,
            ).mean().iloc[-1]
            ewma_vol = np.sqrt(ewma_variance) * np.sqrt(SECONDS_PER_YEAR)

        features[f"{window}_vol"] = simple_vol
        features[f"{window}_ewma_vol"] = ewma_vol
        features[f"{window}_n_obs"] = n_obs
        features[f"{window}_coverage"] = coverage

    return pd.DataFrame([features], index=pd.DatetimeIndex([timestamp], name="timestamp"))[FEATURE_COLUMNS]


def changed_columns(baseline, candidate, columns):
    return [column for column in columns if not baseline[column].equals(candidate[column])]


def print_changed_values(baseline, candidate, columns):
    if not columns:
        print("  Changed columns: none")
        return

    print("  Changed columns and values:")

    for column in columns:
        print(f"    {column}: baseline={baseline[column].iloc[0]!r}, perturbed={candidate[column].iloc[0]!r}")


def assert_future_corruption_safe(label, baseline, corrupted):
    try:
        pd.testing.assert_frame_equal(baseline, corrupted, check_exact=True)
    except AssertionError:
        print(f"{label} future-corruption test FAIL")
        print_changed_values(baseline, corrupted, changed_columns(baseline, corrupted, FEATURE_COLUMNS))
        raise

    print(f"{label} future-corruption test PASS")


def assert_past_sensitivity(label, baseline, perturbed):
    changed_vol_columns = changed_columns(baseline, perturbed, VOL_COLUMNS)

    try:
        pd.testing.assert_frame_equal(baseline[N_OBS_COLUMNS], perturbed[N_OBS_COLUMNS], check_exact=True)
        pd.testing.assert_frame_equal(baseline[COVERAGE_COLUMNS], perturbed[COVERAGE_COLUMNS], check_exact=True)
        assert changed_vol_columns, "No volatility or EWMA volatility feature changed"
    except (AssertionError, ValueError):
        print(f"{label} past-sensitivity test FAIL")
        print_changed_values(baseline, perturbed, changed_columns(baseline, perturbed, FEATURE_COLUMNS))
        raise

    print(f"{label} past-sensitivity test PASS")
    print(f"  Changed volatility columns: {', '.join(changed_vol_columns)}")


def check_leakage():
    timestamps = pd.DatetimeIndex([LEAKAGE_TIMESTAMP])
    grid_start, _ = required_grid_bounds(timestamps)
    prices = load_exchange_prices(grid_start, FUTURE_LOAD_END)
    proxy = build_vwap_proxy(prices)

    if proxy.empty:
        raise ValueError("No proxy observations loaded for leakage tests")

    future_mask = proxy.index >= LEAKAGE_TIMESTAMP
    past_mask = (
        (proxy.index >= PAST_PERTURBATION_START)
        & (proxy.index < PAST_PERTURBATION_END)
    )

    if not future_mask.any() or not past_mask.any():
        raise ValueError("Leakage test perturbation ranges contain no proxy observations")

    future_corrupted_proxy = proxy.copy()
    future_corrupted_proxy.loc[future_mask, "price"] *= 1.5
    past_perturbed_proxy = proxy.copy()
    past_perturbed_proxy.loc[past_mask, "price"] *= 1.5

    print("\nDAY 6 SECTION 3.1 - LEAKAGE TESTS")
    print(f"Test timestamp: {LEAKAGE_TIMESTAMP}")
    print(f"Future-corruption rows: {int(future_mask.sum())}")
    print(
        f"Past perturbation: [{PAST_PERTURBATION_START}, {PAST_PERTURBATION_END}), "
        f"rows={int(past_mask.sum())}"
    )

    batch_baseline = compute_features_from_proxy(timestamps, proxy)
    batch_future_corrupted = compute_features_from_proxy(timestamps, future_corrupted_proxy)
    batch_past_perturbed = compute_features_from_proxy(timestamps, past_perturbed_proxy)
    assert_future_corruption_safe("Batch", batch_baseline, batch_future_corrupted)
    assert_past_sensitivity("Batch", batch_baseline, batch_past_perturbed)

    reference_baseline = reference_features_from_proxy(LEAKAGE_TIMESTAMP, proxy)
    reference_future_corrupted = reference_features_from_proxy(LEAKAGE_TIMESTAMP, future_corrupted_proxy)
    reference_past_perturbed = reference_features_from_proxy(LEAKAGE_TIMESTAMP, past_perturbed_proxy)
    production_reference = pd.DataFrame(
        [realized_vol_features(LEAKAGE_TIMESTAMP)],
        index=pd.DatetimeIndex([LEAKAGE_TIMESTAMP], name="timestamp"),
    )[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(reference_baseline, production_reference, check_exact=True)
    assert_future_corruption_safe("Reference", reference_baseline, reference_future_corrupted)
    assert_past_sensitivity("Reference", reference_baseline, reference_past_perturbed)


if __name__ == "__main__":
    check_realized_vol_batch()
    check_leakage()
