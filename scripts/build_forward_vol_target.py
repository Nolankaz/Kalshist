"""Build and prove the quarantined Day 13 forward realized-volatility target."""

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scripts.count_threshold_clearance import assert_safe_columns
from scripts.evaluation_split import EXPECTED_SPLIT_ROWS, SPLIT_RANGES
from scripts.realized_vol import MAX_FFILL_SECONDS, MIN_COVERAGE, SECONDS_PER_YEAR, build_vwap_proxy, load_exchange_prices, make_1s_proxy
from scripts.realized_vol_batch import build_filled_price_grid, feature_end_labels, load_proxy_for_batch, realized_vol_features_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
DERIVED_PATH = PROJECT_ROOT / "data/features/derived_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/targets/forward_vol_target.parquet"

MARKET_COLUMNS = ["ticker", "horizon_minutes", "close_date", "decision_time", "close_time"]
DERIVED_COLUMNS = ["ticker", "horizon_minutes", "split", "is_common"]
KEY_COLUMNS = MARKET_COLUMNS + ["split", "is_common"]
FWD_COLUMNS = ["fwd_window_seconds", "fwd_n_obs", "fwd_coverage", "fwd_rv", "fwd_log_rv"]
OUTPUT_COLUMNS = KEY_COLUMNS + FWD_COLUMNS
ROW_KEY = ["ticker", "horizon_minutes"]
WINDOW_SECONDS = {5: 300, 10: 600}
EXPECTED_ROWS = 14_358
EXPECTED_COMMON = {("train", 10): 4_792, ("train", 5): 4_757, ("validation", 10): 1_685, ("validation", 5): 1_649}
TRAIN_START = SPLIT_RANGES["train"][0]
VALIDATION_END = SPLIT_RANGES["validation"][1]
TEST_START = pd.Timestamp("2026-08-10 00:00:00", tz="UTC")
ANNUALIZATION = np.sqrt(SECONDS_PER_YEAR)
REFERENCE_SAMPLE_SIZE = 120
BOUNDARY_SAMPLE_DAYS = 24
BOUNDARY_ROWS_PER_DAY = 2
VALUE_TOLERANCE = 1e-12


def load_rows():
    assert_safe_columns(MARKET_COLUMNS, "Day 13 market-feature projection")
    markets = pd.read_parquet(MARKET_PATH, columns=MARKET_COLUMNS, filters=[("close_date", "<=", VALIDATION_END)])
    derived = pd.read_parquet(DERIVED_PATH, columns=DERIVED_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(markets.columns) == MARKET_COLUMNS and list(derived.columns) == DERIVED_COLUMNS, "Input projection changed"
    assert_safe_columns(markets.columns, "Day 13 loaded market features")
    assert_safe_columns(derived.columns, "Day 13 loaded derived keys")
    assert len(markets) == len(derived) == EXPECTED_ROWS, f"Train/validation structural count differs from {EXPECTED_ROWS:,}: market={len(markets):,}, derived={len(derived):,}"
    assert not markets.duplicated(ROW_KEY).any() and not derived.duplicated(ROW_KEY).any(), "Duplicate input key"
    rows = markets.merge(derived, on=ROW_KEY, how="outer", validate="one_to_one", indicator=True)
    assert rows["_merge"].eq("both").all() and len(rows) == EXPECTED_ROWS, "Market and derived keys differ"
    rows = rows.drop(columns="_merge")[KEY_COLUMNS]
    assert set(rows["split"]) == {"train", "validation"} and rows["close_date"].between(TRAIN_START, VALIDATION_END).all(), "A forbidden split/date entered"
    assert rows.groupby("split", observed=True).size().to_dict() == {split: EXPECTED_SPLIT_ROWS[split] for split in ("train", "validation")}, "Frozen split counts changed"
    for split in ("train", "validation"):
        assert rows.loc[rows["split"].eq(split), "close_date"].between(*SPLIT_RANGES[split]).all(), f"{split} date disagrees with split"
    assert rows.loc[rows["is_common"]].groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON, "Day 9 common population changed"
    assert pd.api.types.is_bool_dtype(rows["is_common"]), "is_common lost boolean dtype"
    assert set(rows["horizon_minutes"]) == set(WINDOW_SECONDS), "Unexpected horizon"
    for column in ("decision_time", "close_time"):
        assert isinstance(rows[column].dtype, pd.DatetimeTZDtype) and str(rows[column].dt.tz) == "UTC", f"{column} is not UTC-aware"
        assert rows[column].notna().all(), f"{column} contains nulls"
    assert rows["close_time"].max() < TEST_START, "A test-period close entered"
    assert rows["close_time"].dt.strftime("%Y-%m-%d").eq(rows["close_date"]).all(), "close_date disagrees with close_time"
    expected_duration = pd.to_timedelta(rows["horizon_minutes"], unit="min")
    assert (rows["close_time"] - rows["decision_time"]).eq(expected_duration).all(), "Decision/close window changed"
    assert rows.groupby("ticker", observed=True)["horizon_minutes"].nunique().eq(2).all(), "A market lacks a horizon"
    return rows.sort_values(ROW_KEY).reset_index(drop=True)


def forward_window_realized_vol(close_times, window_seconds, filled_price):
    """Place the Day 6 simple-RV estimator at each close time on a filled grid."""
    assert window_seconds in WINDOW_SECONDS.values(), "Unexpected forward window"
    close_times = pd.DatetimeIndex(pd.to_datetime(close_times, utc=True), name="close_time")
    end_labels = feature_end_labels(close_times)
    assert end_labels.isin(filled_price.index).all(), "Filled grid misses a close end label"
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(filled_price / filled_price.shift(1))
    count_series = log_returns.rolling(window_seconds).count()
    std_series = log_returns.rolling(window_seconds, min_periods=1).std(ddof=1)
    counts = count_series.reindex(end_labels).to_numpy(dtype=np.int64)
    coverage = counts / window_seconds
    values = std_series.reindex(end_labels).to_numpy(dtype=float) * ANNUALIZATION
    values[coverage < MIN_COVERAGE] = np.nan
    return pd.DataFrame({"fwd_n_obs": counts, "fwd_coverage": coverage, "fwd_rv": values}, index=close_times)


def build_target(rows):
    parts = []
    for _, day in rows.groupby("close_date", sort=True):
        close_times = pd.DatetimeIndex(day["close_time"])
        assert close_times.max() < TEST_START, "Attempted to load test-period BTC prices"
        proxy, grid_start, grid_end = load_proxy_for_batch(close_times)
        assert grid_end < TEST_START, "Batch grid reaches test period"
        filled_price, _ = build_filled_price_grid(proxy, grid_start, grid_end)
        for horizon, window_seconds in WINDOW_SECONDS.items():
            subset = day.loc[day["horizon_minutes"].eq(horizon)].copy()
            computed = forward_window_realized_vol(subset["close_time"], window_seconds, filled_price)
            subset["fwd_window_seconds"] = window_seconds
            for column in ("fwd_n_obs", "fwd_coverage", "fwd_rv"):
                subset[column] = computed[column].to_numpy()
            parts.append(subset)
    output = pd.concat(parts, ignore_index=True).sort_values(ROW_KEY).reset_index(drop=True)
    output["fwd_log_rv"] = np.log(output["fwd_rv"])
    return output[OUTPUT_COLUMNS]


def prove_300_second_engine_equivalence(output):
    subset = output.loc[output["horizon_minutes"].eq(5)].sort_values(["close_time", "ticker"]).reset_index(drop=True)
    engine = realized_vol_features_batch(subset["close_time"])
    expected_count = engine["5min_n_obs"].to_numpy(dtype=np.int64)
    expected_value = engine["5min_vol"].to_numpy(dtype=float)
    actual_count = subset["fwd_n_obs"].to_numpy(dtype=np.int64)
    actual_value = subset["fwd_rv"].to_numpy(dtype=float)
    assert np.array_equal(actual_count, expected_count), "Proof A: five-minute observation counts differ"
    assert np.array_equal(np.isnan(actual_value), np.isnan(expected_value)), "Proof A: five-minute null masks differ"
    error = float(np.max(np.abs(actual_value[~np.isnan(actual_value)] - expected_value[~np.isnan(expected_value)])))
    assert error <= VALUE_TOLERANCE, f"Proof A: engine value error {error:.17g} exceeds {VALUE_TOLERANCE:g}"
    return len(subset), error


def reference_forward_window(row):
    decision_time = row.decision_time
    close_time = row.close_time
    load_start = decision_time - pd.Timedelta(seconds=MAX_FFILL_SECONDS + 1)
    assert close_time < TEST_START and load_start < close_time, "Reference price request reaches test period"
    prices = load_exchange_prices(load_start, close_time)
    proxy = build_vwap_proxy(prices)
    if proxy.empty:
        proxy = pd.DataFrame({"price": pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))})
    grid = make_1s_proxy(proxy, load_start, close_time)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.log(grid["price"] / grid["price"].shift(1))
    inside = returns.loc[(returns.index >= decision_time) & (returns.index < close_time)].dropna()
    count = len(inside)
    coverage = count / WINDOW_SECONDS[row.horizon_minutes]
    value = float(inside.std(ddof=1) * ANNUALIZATION) if coverage >= MIN_COVERAGE else np.nan
    return count, value


def spread_sample(rows, size, seed):
    ordered = rows.sort_values(["close_time", "ticker"]).reset_index(drop=True)
    assert len(ordered) >= size, "Insufficient rows for reference sample"
    rng = np.random.default_rng(seed)
    positions = [int(rng.choice(group)) for group in np.array_split(np.arange(len(ordered)), size)]
    return ordered.iloc[positions]


def prove_600_second_reference(output):
    sample = spread_sample(output.loc[output["horizon_minutes"].eq(10)], REFERENCE_SAMPLE_SIZE, seed=13)
    errors = []
    for row in sample.itertuples(index=False):
        count, value = reference_forward_window(row)
        assert count == row.fwd_n_obs, f"Proof B: observation count differs for {row.ticker}"
        assert np.isnan(value) == np.isnan(row.fwd_rv), f"Proof B: null mask differs for {row.ticker}"
        if not np.isnan(value):
            errors.append(abs(value - row.fwd_rv))
    error = max(errors, default=0.0)
    assert error <= VALUE_TOLERANCE, f"Proof B: independent-reference value error {error:.17g} exceeds {VALUE_TOLERANCE:g}"
    return len(sample), sample["close_date"].min(), sample["close_date"].max(), error


def prove_boundaries(output):
    eligible = output.loc[output["fwd_rv"].notna()].copy()
    dates = sorted(eligible["close_date"].unique())
    assert len(dates) >= BOUNDARY_SAMPLE_DAYS, "Insufficient days for boundary proof"
    rng = np.random.default_rng(1313)
    date_groups = np.array_split(np.arange(len(dates)), BOUNDARY_SAMPLE_DAYS)
    selected_dates = [dates[int(rng.choice(group))] for group in date_groups]
    outside_checks = 0
    inside_checks = 0
    inside_changed = 0

    for date in selected_dates:
        day = eligible.loc[eligible["close_date"].eq(date)]
        all_close_times = pd.DatetimeIndex(output.loc[output["close_date"].eq(date), "close_time"])
        proxy, grid_start, grid_end = load_proxy_for_batch(all_close_times)
        assert grid_end < TEST_START, "Boundary grid reaches test period"
        filled_price, _ = build_filled_price_grid(proxy, grid_start, grid_end)
        candidates = day.loc[day["close_time"] <= grid_end].sort_values(["close_time", "ticker"])
        if candidates.empty:
            continue
        chosen = candidates.iloc[rng.choice(len(candidates), size=min(BOUNDARY_ROWS_PER_DAY, len(candidates)), replace=False)]
        for row in chosen.itertuples(index=False):
            original = forward_window_realized_vol([row.close_time], row.fwd_window_seconds, filled_price).iloc[0]
            assert original.fwd_n_obs == row.fwd_n_obs and abs(original.fwd_rv - row.fwd_rv) <= VALUE_TOLERANCE, "Boundary baseline differs from builder"

            after = filled_price.copy()
            after.loc[after.index >= row.close_time] *= 1.5
            outside = forward_window_realized_vol([row.close_time], row.fwd_window_seconds, after).iloc[0]
            assert outside.fwd_n_obs == row.fwd_n_obs and np.isnan(outside.fwd_rv) == np.isnan(row.fwd_rv), "Proof C: post-close count/mask changed"
            assert abs(outside.fwd_rv - row.fwd_rv) <= VALUE_TOLERANCE, "Proof C: post-close price changed target"
            outside_checks += 1

            inside = filled_price.copy()
            eligible_labels = inside.index[(inside.index > row.decision_time) & (inside.index < row.close_time) & inside.notna()]
            if len(eligible_labels) == 0:
                continue
            inside.loc[eligible_labels[::31]] *= 1.5
            changed = forward_window_realized_vol([row.close_time], row.fwd_window_seconds, inside).iloc[0]
            assert changed.fwd_n_obs == row.fwd_n_obs, "Proof C: inside price perturbation changed coverage"
            inside_checks += 1
            inside_changed += abs(changed.fwd_rv - row.fwd_rv) > VALUE_TOLERANCE

    assert outside_checks >= BOUNDARY_SAMPLE_DAYS, "Too few post-close boundary checks"
    assert inside_checks >= BOUNDARY_SAMPLE_DAYS, "Too few inside-window boundary checks"
    rate = inside_changed / inside_checks
    assert rate >= 0.95, f"Proof C: only {rate:.1%} of inside-window perturbations changed target"
    return len(selected_dates), outside_checks, inside_checks, inside_changed, rate


def prove_quarantine(output):
    assert list(output.columns) == OUTPUT_COLUMNS, "Output columns changed"
    assert all(column.startswith("fwd_") for column in output.columns if column not in KEY_COLUMNS), "A target value column lacks fwd_ prefix"
    assert OUTPUT_PATH.resolve() not in {MARKET_PATH.resolve(), DERIVED_PATH.resolve()}, "Target path collides with features"
    try:
        assert_safe_columns(FWD_COLUMNS, "intentional Day 13 target rejection")
    except AssertionError as error:
        assert "target/outcome/future-only" in str(error), "Safety guard raised for an unrelated reason"
    else:
        raise AssertionError("Proof D: safe-column guard accepted forward target values")


def assert_output(output):
    prove_quarantine(output)
    assert len(output) == EXPECTED_ROWS and not output.duplicated(ROW_KEY).any(), "Output row count/key changed"
    assert output[ROW_KEY].equals(output[ROW_KEY].sort_values(ROW_KEY).reset_index(drop=True)), "Output key order changed"
    assert set(output["split"]) == {"train", "validation"} and output["close_date"].between(TRAIN_START, VALIDATION_END).all(), "Output includes forbidden split/date"
    assert output["close_time"].max() < TEST_START, "Output includes test-period close"
    assert output.groupby("split", observed=True).size().to_dict() == {split: EXPECTED_SPLIT_ROWS[split] for split in ("train", "validation")}, "Output split count changed"
    assert output.loc[output["is_common"]].groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON, "Output common population changed"
    assert output["horizon_minutes"].map(WINDOW_SECONDS).eq(output["fwd_window_seconds"]).all(), "Horizon/window mapping changed"
    assert (output["close_time"] - output["decision_time"]).eq(pd.to_timedelta(output["fwd_window_seconds"], unit="s")).all(), "Output decision/close duration changed"
    assert output["fwd_n_obs"].notna().all() and pd.api.types.is_integer_dtype(output["fwd_n_obs"]), "Observation counts must be integers"
    assert output["fwd_n_obs"].ge(0).all() and output["fwd_n_obs"].le(output["fwd_window_seconds"]).all(), "Observation count outside window"
    np.testing.assert_array_equal(output["fwd_coverage"].to_numpy(), (output["fwd_n_obs"] / output["fwd_window_seconds"]).to_numpy())
    null_mask = output["fwd_coverage"].lt(MIN_COVERAGE)
    assert output["fwd_rv"].isna().equals(null_mask), "Target null mask differs from frozen coverage gate"
    assert output["fwd_log_rv"].isna().equals(null_mask), "Log-target null mask differs from target"
    positive = output.loc[~null_mask, "fwd_rv"].to_numpy(dtype=float)
    assert np.isfinite(positive).all() and (positive > 0).all(), "Non-null target is not positive and finite"
    np.testing.assert_array_equal(output.loc[~null_mask, "fwd_log_rv"].to_numpy(dtype=float), np.log(positive))
    assert output.groupby("ticker", observed=True)["horizon_minutes"].nunique().eq(2).all(), "Output lost one market horizon"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_and_verify(output):
    assert_output(output)
    prior_hash = None
    if OUTPUT_PATH.exists():
        prior = pd.read_parquet(OUTPUT_PATH, columns=OUTPUT_COLUMNS)
        assert_output(prior)
        pd.testing.assert_frame_equal(prior, output, check_exact=True)
        prior_hash = file_sha256(OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, prefix=".forward_vol_target_", suffix=".parquet", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        output.to_parquet(temporary_path, index=False)
        round_trip = pd.read_parquet(temporary_path, columns=OUTPUT_COLUMNS)
        assert_output(round_trip)
        pd.testing.assert_frame_equal(round_trip, output, check_exact=True)
        new_hash = file_sha256(temporary_path)
        if prior_hash is not None:
            assert prior_hash == new_hash, "Deterministic Parquet file hash changed"
        os.replace(temporary_path, OUTPUT_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    return new_hash, prior_hash is not None


def main():
    rows = load_rows()
    output = build_target(rows)
    assert_output(output)
    proof_a_rows, proof_a_error = prove_300_second_engine_equivalence(output)
    proof_b_rows, proof_b_start, proof_b_end, proof_b_error = prove_600_second_reference(output)
    proof_c_days, outside_checks, inside_checks, inside_changed, inside_rate = prove_boundaries(output)
    prove_quarantine(output)
    fingerprint, compared_prior = write_and_verify(output)
    null_counts = output.loc[output["is_common"]].assign(target_null=output["fwd_rv"].isna()).groupby(["split", "horizon_minutes"], observed=True)["target_null"].sum().to_dict()
    print(f"Structural rows: {len(output):,}; common rows: {int(output['is_common'].sum()):,}")
    print("Common target nulls: " + "; ".join(f"{split} T-{horizon}={null_counts[(split, horizon)]}" for split in ("train", "validation") for horizon in (10, 5)))
    print(f"Proof A: PASS; T-5 rows={proof_a_rows:,}; max absolute value error={proof_a_error:.17g}; counts/null masks exact")
    print(f"Proof B: PASS; T-10 sample={proof_b_rows}; dates={proof_b_start}..{proof_b_end}; max absolute value error={proof_b_error:.17g}; counts/null masks exact")
    print(f"Proof C: PASS; days={proof_c_days}; post-close invariant={outside_checks}/{outside_checks}; inside-window changed={inside_changed}/{inside_checks} ({inside_rate:.1%})")
    print("Proof D: PASS; prefixed values, separate target path, safe-column guard rejects values")
    print("Invariants and exact Parquet round trip: PASS")
    print(f"Deterministic prior-file comparison: {'PASS' if compared_prior else 'first write'}; SHA-256={fingerprint}")
    print(f"Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
