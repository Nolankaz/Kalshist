from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_market_features import FEATURE_COLUMNS, FUTURE_ONLY_COLUMNS, KEY_COLUMNS, TARGET_COLUMNS
from scripts.realized_vol_batch import build_filled_price_grid, compute_features_from_price_grid, feature_end_labels, load_proxy_for_batch
from storage import load_range


FEATURE_PATH = Path("data/features/market_features.parquet")
EXPECTED_ROWS = 17_182
EXPECTED_ROWS_PER_HORIZON = 8_591
LEAKAGE_TEST_DATE = "2026-07-15"
FUTURE_GRID_EXTENSION = pd.Timedelta(minutes=5)
PAST_PERTURBATION_START_OFFSET = pd.Timedelta(minutes=2)
PAST_PERTURBATION_END_OFFSET = pd.Timedelta(minutes=1)
VOLATILITY_COLUMNS = [column for column in FEATURE_COLUMNS if column.endswith(("_vol", "_n_obs", "_coverage"))]
RELATIVE_AUC_REVIEW_MARGIN = 0.10


def load_full_source_range(table_name):
    files = sorted((Path("data") / table_name).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No source files found for {table_name}")

    return load_range(table_name, files[0].stem, files[-1].stem)


def independently_select_latest_quotes(features, quotes):
    decisions = features[["ticker", "decision_time"]].reset_index(names="feature_row")
    source_quotes = quotes[["ticker", "period_end_ts"]].copy()
    decisions["decision_time"] = decisions["decision_time"].astype("datetime64[ns, UTC]")
    source_quotes["source_quote_time"] = pd.to_datetime(source_quotes["period_end_ts"], unit="s", utc=True).astype("datetime64[ns, UTC]")
    decisions = decisions.sort_values("decision_time")
    source_quotes = source_quotes.sort_values("source_quote_time")
    selected = pd.merge_asof(
        decisions,
        source_quotes,
        left_on="decision_time",
        right_on="source_quote_time",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    return selected.sort_values("feature_row")["period_end_ts"].reset_index(drop=True)


def rank_auc(scores, targets):
    valid = scores.notna() & targets.notna()
    valid_scores = scores.loc[valid]
    valid_targets = targets.loc[valid]

    if not valid_targets.isin([0, 1]).all():
        raise ValueError("AUC target must contain only 0 and 1")

    positive_count = int(valid_targets.eq(1).sum())
    negative_count = int(valid_targets.eq(0).sum())

    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires at least one positive and one negative observation")

    ranks = valid_scores.rank(method="average")
    positive_rank_sum = ranks.loc[valid_targets.eq(1)].sum()
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    return float(auc), len(valid_scores)


def brier_score(scores, targets):
    valid = scores.notna() & targets.notna()

    if not valid.any():
        raise ValueError("Brier score requires at least one valid observation")

    return float(np.mean((scores.loc[valid] - targets.loc[valid]) ** 2)), int(valid.sum())


def check_feature_auc(features):
    assert len(VOLATILITY_COLUMNS) == 20, f"Expected 20 volatility columns, found {len(VOLATILITY_COLUMNS)}"
    score_columns = ["quote_mid", "log_moneyness", *VOLATILITY_COLUMNS]
    print("\nDay 7 section 3.2 Test 2 rank-based AUC and market Brier scores")
    print(f"Informational INVESTIGATE flag: AUC exceeds quote_mid by at least {RELATIVE_AUC_REVIEW_MARGIN:.2f}; flags do not fail validation")

    for horizon in (10, 5):
        horizon_rows = features[features["horizon_minutes"] == horizon]
        quote_auc, _ = rank_auc(horizon_rows["quote_mid"], horizon_rows["y"])
        results = []

        for column in score_columns:
            auc, valid_count = rank_auc(horizon_rows[column], horizon_rows["y"])
            difference = auc - quote_auc
            review = "INVESTIGATE" if difference >= RELATIVE_AUC_REVIEW_MARGIN else ""
            results.append({
                "score": column,
                "valid_observations": valid_count,
                "auc": auc,
                "auc_minus_quote_mid": difference,
                "review": review,
            })

        result_frame = pd.DataFrame(results)
        market_brier, brier_count = brier_score(horizon_rows["quote_mid"], horizon_rows["y"])
        print(f"\nT-{horizon}")
        print(result_frame.to_string(index=False, formatters={"auc": "{:.6f}".format, "auc_minus_quote_mid": "{:+.6f}".format}))
        print(f"Market Brier score for T-{horizon}: {market_brier:.6f} (valid observations: {brier_count:,})")


def check_feature_leakage(features):
    date_mask = features["decision_time"].dt.strftime("%Y-%m-%d").eq(LEAKAGE_TEST_DATE)
    test_rows = features.loc[date_mask].sort_values("decision_time").copy()

    if test_rows.empty:
        raise ValueError(f"No decision timestamps found on {LEAKAGE_TEST_DATE}")

    assert test_rows["decision_time"].is_unique, "Leakage-test decision timestamps must be unique"
    timestamps = pd.DatetimeIndex(test_rows["decision_time"], name="timestamp")
    load_timestamps = timestamps.append(pd.DatetimeIndex([timestamps.max() + FUTURE_GRID_EXTENSION]))
    proxy, grid_start, grid_end = load_proxy_for_batch(load_timestamps)
    filled_price, fill_source = build_filled_price_grid(proxy, grid_start, grid_end)
    original_volatility = compute_features_from_price_grid(timestamps, filled_price, fill_source)
    volatility_columns = original_volatility.columns.tolist()
    assert volatility_columns == VOLATILITY_COLUMNS, "Day 6 volatility columns differ from the expected 20-column set"

    stored_volatility = test_rows.set_index("decision_time")[volatility_columns]
    stored_volatility.index.name = "timestamp"
    pd.testing.assert_frame_equal(original_volatility, stored_volatility, check_exact=True)

    end_labels = feature_end_labels(timestamps)
    original_spot = pd.Series(filled_price.reindex(end_labels).to_numpy(), index=timestamps, name="spot")
    stored_spot = test_rows.set_index("decision_time")["spot"]
    stored_spot.index.name = "timestamp"
    pd.testing.assert_series_equal(original_spot, stored_spot, check_exact=True)
    stored_strike = test_rows.set_index("decision_time")["strike"]
    stored_strike.index.name = "timestamp"
    original_log_moneyness = np.log(original_spot / stored_strike)
    original_log_moneyness.name = "log_moneyness"
    stored_log_moneyness = test_rows.set_index("decision_time")["log_moneyness"]
    stored_log_moneyness.index.name = "timestamp"
    pd.testing.assert_series_equal(original_log_moneyness, stored_log_moneyness, check_exact=True)

    for timestamp in timestamps:
        future_mask = filled_price.index >= timestamp
        assert future_mask.any(), f"No future grid cells available at {timestamp}"
        assert filled_price.loc[future_mask].notna().any(), f"No populated future prices available at {timestamp}"
        corrupted_price = filled_price.copy()
        corrupted_price.loc[future_mask] *= 1.5
        corrupted_volatility = compute_features_from_price_grid(pd.DatetimeIndex([timestamp]), corrupted_price, fill_source)
        pd.testing.assert_frame_equal(original_volatility.loc[[timestamp]], corrupted_volatility, check_exact=True)

        corrupted_end_label = feature_end_labels(pd.DatetimeIndex([timestamp]))[0]
        corrupted_spot = pd.Series([corrupted_price.loc[corrupted_end_label]], index=pd.DatetimeIndex([timestamp], name="timestamp"), name="spot")
        pd.testing.assert_series_equal(original_spot.loc[[timestamp]], corrupted_spot, check_exact=True)
        corrupted_log_moneyness = np.log(corrupted_spot / stored_strike.loc[[timestamp]])
        corrupted_log_moneyness.name = "log_moneyness"
        pd.testing.assert_series_equal(original_log_moneyness.loc[[timestamp]], corrupted_log_moneyness, check_exact=True)

    mirror_timestamp = timestamps[-1]
    past_start = mirror_timestamp - PAST_PERTURBATION_START_OFFSET
    past_end = mirror_timestamp - PAST_PERTURBATION_END_OFFSET
    past_mask = (filled_price.index >= past_start) & (filled_price.index < past_end)
    assert past_end < mirror_timestamp, "Past perturbation must end strictly before decision time"
    assert past_mask.any(), "Past perturbation interval contains no grid cells"
    assert filled_price.loc[past_mask].notna().any(), "Past perturbation interval contains no populated prices"
    perturbed_price = filled_price.copy()
    perturbed_price.loc[past_mask] *= 1.5
    perturbed_volatility = compute_features_from_price_grid(pd.DatetimeIndex([mirror_timestamp]), perturbed_price, fill_source)
    original_mirror = original_volatility.loc[[mirror_timestamp]]
    value_columns = [column for column in volatility_columns if column.endswith("_vol")]
    n_obs_columns = [column for column in volatility_columns if column.endswith("_n_obs")]
    coverage_columns = [column for column in volatility_columns if column.endswith("_coverage")]
    changed_value_columns = [column for column in value_columns if not original_mirror[column].equals(perturbed_volatility[column])]
    assert changed_value_columns, "Past price perturbation did not change any volatility value"
    pd.testing.assert_frame_equal(original_mirror[n_obs_columns], perturbed_volatility[n_obs_columns], check_exact=True)
    pd.testing.assert_frame_equal(original_mirror[coverage_columns], perturbed_volatility[coverage_columns], check_exact=True)

    print("\nDay 7 section 3.2 Test 1 leakage validation: PASS")
    print(f"UTC date: {LEAKAGE_TEST_DATE}")
    print(f"Decision timestamps tested: {len(timestamps):,}")
    print("Future-corruption volatility, spot, and log_moneyness mismatches: 0")
    print(f"Past perturbation: [{past_start}, {past_end})")
    print(f"Volatility values changed by past perturbation: {len(changed_value_columns):,}")
    print("Past-perturbation n_obs and coverage mismatches: 0")


def check_market_features():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature table not found: {FEATURE_PATH}")

    features = pd.read_parquet(FEATURE_PATH)
    markets = load_full_source_range("kalshi_markets")
    quotes = load_full_source_range("kalshi_quotes")

    assert len(features) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} rows, found {len(features):,}"
    assert not features.duplicated(["ticker", "horizon_minutes"]).any(), "Feature keys must be unique"
    horizon_counts = features["horizon_minutes"].value_counts().sort_index()
    assert horizon_counts.to_dict() == {5: EXPECTED_ROWS_PER_HORIZON, 10: EXPECTED_ROWS_PER_HORIZON}

    column_groups = [KEY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMNS, FUTURE_ONLY_COLUMNS]
    classified_columns = [column for group in column_groups for column in group]
    assert len(classified_columns) == len(set(classified_columns)), "Column groups must be disjoint"
    assert set(classified_columns) == set(features.columns), "Column groups must classify every feature-table column"
    assert not set(FEATURE_COLUMNS) & set(TARGET_COLUMNS + FUTURE_ONLY_COLUMNS), "Features overlap target or future-only columns"
    assert "volume" not in features.columns, "Raw market-level volume must not appear in the feature table"
    assert "settlement_ts" not in features.columns, "settlement_ts must not appear in the feature table"

    assert features["ticker"].isin(markets["ticker"]).all(), "At least one feature ticker is absent from the market source"
    feature_quote_keys = pd.MultiIndex.from_frame(features[["ticker", "quote_period_end_ts"]])
    source_quote_keys = pd.MultiIndex.from_frame(quotes[["ticker", "period_end_ts"]])
    assert feature_quote_keys.isin(source_quote_keys).all(), "At least one stored quote key is absent from the quote source"

    assert features["y"].eq(features["settlement_value"]).all(), "y differs from settlement_value"
    expiration_target_defined = features["y_from_expiration"].notna()
    target_disagreements = features.loc[expiration_target_defined, "y"].ne(features.loc[expiration_target_defined, "y_from_expiration"]).sum()
    assert target_disagreements == 0, "y differs from y_from_expiration where the latter is defined"
    assert features["quote_spread"].ge(0).all(), "Negative quote spreads found"
    assert features["quote_yes_bid"].between(0, 1).all(), "Quote bids outside [0, 1] found"
    assert features["quote_yes_ask"].between(0, 1).all(), "Quote asks outside [0, 1] found"

    expected_decision_time = features["close_time"] - pd.to_timedelta(features["horizon_minutes"], unit="min")
    assert features["decision_time"].eq(expected_decision_time).all(), "Incorrect decision timestamps found"
    assert features["open_time"].lt(features["decision_time"]).all(), "Decision time must be after open time"
    assert features["decision_time"].lt(features["close_time"]).all(), "Decision time must be before close time"
    future_quote_violations = features["quote_time"].gt(features["decision_time"]).sum()
    assert future_quote_violations == 0, "Future quotes found"

    independently_selected_quotes = independently_select_latest_quotes(features, quotes)
    stored_quote_periods = features["quote_period_end_ts"].reset_index(drop=True)
    quote_mismatches = independently_selected_quotes.ne(stored_quote_periods).sum()
    assert quote_mismatches == 0, "Stored quotes do not match independently selected latest quotes"

    decision_times = pd.DatetimeIndex(features["decision_time"])
    assert (feature_end_labels(decision_times) < decision_times).all(), "Feature windows include decision-time data"

    print("Day 7 section 3.1 market-feature validation: PASS")
    print(f"Rows: {len(features):,}")
    print(f"Rows per horizon: {horizon_counts.to_dict()}")
    print(f"Future-quote violations: {future_quote_violations:,}")
    print(f"Target disagreements: {target_disagreements:,}")
    print(f"Independently re-derived quote mismatches: {quote_mismatches:,}")
    check_feature_leakage(features)
    check_feature_auc(features)


if __name__ == "__main__":
    check_market_features()
