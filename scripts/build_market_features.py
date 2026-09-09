from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.realized_vol_batch import (
    FEATURE_COLUMNS as VOLATILITY_COLUMNS,
    build_filled_price_grid,
    compute_features_from_price_grid,
    feature_end_labels,
    load_proxy_for_batch,
)
from storage import load_range


START_DATE = "2026-05-26"
END_DATE = "2026-08-24"
HORIZON_MINUTES = (10, 5)
EXPECTED_MARKETS = 8_591
EXPECTED_ROWS = 17_182
MINUTES_PER_YEAR = 365 * 24 * 60
OUTPUT_PATH = Path("data/features/market_features.parquet")

KEY_COLUMNS = [
    "ticker",
    "horizon_minutes",
    "open_time",
    "close_time",
    "decision_time",
    "close_date",
]
QUOTE_COLUMNS = [
    "quote_yes_bid",
    "quote_yes_ask",
    "quote_mid",
    "quote_spread",
    "quote_last_price",
    "quote_volume",
    "quote_open_interest",
    "quote_period_end_ts",
    "quote_time",
    "quote_age_seconds",
]
FEATURE_COLUMNS = [
    "spot",
    "spot_age_seconds",
    "strike",
    "log_moneyness",
    "T_years",
    "hour_utc",
    "day_of_week",
] + QUOTE_COLUMNS + VOLATILITY_COLUMNS
TARGET_COLUMNS = ["y", "settlement_result", "settlement_value"]
FUTURE_ONLY_COLUMNS = ["expiration_value", "y_from_expiration", "target_agrees", "fwd_log_return"]
OUTPUT_COLUMNS = KEY_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS + FUTURE_ONLY_COLUMNS


def load_source_tables():
    markets = load_range("kalshi_markets", START_DATE, END_DATE)
    quotes = load_range("kalshi_quotes", START_DATE, END_DATE)

    if markets.empty or quotes.empty:
        raise FileNotFoundError("Market and quote source tables must both be present")

    assert markets["ticker"].is_unique, "Settled-market tickers must be unique"
    assert not quotes.duplicated(["ticker", "period_end_ts"]).any(), "Quote ticker/timestamp keys must be unique"

    if not pd.api.types.is_integer_dtype(quotes["period_end_ts"]):
        raise TypeError(f"period_end_ts must contain Unix seconds, found {quotes['period_end_ts'].dtype}")

    market_columns = [
        "ticker",
        "strike",
        "open_time",
        "close_time",
        "settlement_result",
        "expiration_value",
        "settlement_value",
    ]
    quote_columns = [
        "ticker",
        "period_end_ts",
        "yes_bid_close",
        "yes_ask_close",
        "price_close",
        "volume",
        "open_interest",
    ]
    markets = markets[market_columns].copy()
    quotes = quotes[quote_columns].copy()
    markets["open_time"] = pd.to_datetime(markets["open_time"], utc=True)
    markets["close_time"] = pd.to_datetime(markets["close_time"], utc=True)
    quotes["quote_time"] = pd.to_datetime(quotes["period_end_ts"], unit="s", utc=True)
    return markets, quotes


def build_decision_rows(markets, quotes):
    decisions = pd.concat(
        [markets.assign(horizon_minutes=horizon) for horizon in HORIZON_MINUTES],
        ignore_index=True,
    )
    decisions["decision_time"] = decisions["close_time"] - pd.to_timedelta(decisions["horizon_minutes"], unit="min")

    quote_candidates = decisions[["ticker", "horizon_minutes", "decision_time"]].merge(quotes, on="ticker", how="left")
    quote_candidates = quote_candidates[quote_candidates["quote_time"] <= quote_candidates["decision_time"]]
    selected_quotes = quote_candidates.sort_values(["ticker", "horizon_minutes", "quote_time"])
    selected_quotes = selected_quotes.drop_duplicates(["ticker", "horizon_minutes"], keep="last")

    quote_horizon_counts = selected_quotes.groupby("ticker")["horizon_minutes"].nunique()
    eligible_tickers = quote_horizon_counts[quote_horizon_counts == len(HORIZON_MINUTES)].index
    decisions = decisions[decisions["ticker"].isin(eligible_tickers)].copy()
    selected_quotes = selected_quotes[selected_quotes["ticker"].isin(eligible_tickers)].copy()

    assert decisions["ticker"].nunique() == EXPECTED_MARKETS, "Unexpected eligible-market count"
    assert len(decisions) == EXPECTED_ROWS, "Unexpected decision-row count"
    assert len(selected_quotes) == len(decisions), "Quote fan-out did not collapse to one quote per decision row"
    assert not selected_quotes.duplicated(["ticker", "horizon_minutes"]).any(), "Selected quote keys must be unique"

    selected_quotes = selected_quotes.rename(columns={
        "period_end_ts": "quote_period_end_ts",
        "yes_bid_close": "quote_yes_bid",
        "yes_ask_close": "quote_yes_ask",
        "price_close": "quote_last_price",
        "volume": "quote_volume",
        "open_interest": "quote_open_interest",
    })
    selected_quote_columns = [
        "ticker",
        "horizon_minutes",
        "quote_period_end_ts",
        "quote_yes_bid",
        "quote_yes_ask",
        "quote_last_price",
        "quote_volume",
        "quote_open_interest",
        "quote_time",
    ]
    decisions = decisions.merge(
        selected_quotes[selected_quote_columns],
        on=["ticker", "horizon_minutes"],
        how="left",
        validate="one_to_one",
    )
    assert len(decisions) == EXPECTED_ROWS, "Quote join changed decision-row cardinality"
    assert not decisions.duplicated(["ticker", "horizon_minutes"]).any(), "Decision-row keys must be unique"
    return decisions


def add_btc_features(decisions):
    vol_values = np.full((len(decisions), len(VOLATILITY_COLUMNS)), np.nan,)
    spot_values = np.full(len(decisions), np.nan)
    spot_age_values = np.full(len(decisions), np.nan)
    requests = pd.DataFrame({
        "position": np.arange(len(decisions)),
        "timestamp": decisions["decision_time"],
        "date": decisions["decision_time"].dt.normalize(),
    })

    for _, group in requests.groupby("date", sort=False):
        positions = group["position"].to_numpy()
        timestamps = pd.DatetimeIndex(group["timestamp"])
        proxy, grid_start, grid_end = load_proxy_for_batch(timestamps)
        filled_price, fill_source = build_filled_price_grid(proxy, grid_start, grid_end)
        labels = feature_end_labels(timestamps)
        features = compute_features_from_price_grid(timestamps, filled_price, fill_source)
        group_spot = filled_price.reindex(labels).to_numpy(dtype=float)
        group_sources = pd.DatetimeIndex(fill_source.reindex(labels).array)
        group_spot_age = (labels - group_sources).total_seconds()

        assert np.array_equal(np.isnan(group_spot), pd.isna(group_sources)), "Spot and fill-source null masks differ"
        vol_values[positions, :] = features.to_numpy(dtype=float)
        spot_values[positions] = group_spot
        spot_age_values[positions] = group_spot_age

    decisions[VOLATILITY_COLUMNS] = vol_values
    decisions["spot"] = spot_values
    decisions["spot_age_seconds"] = spot_age_values

    for column in VOLATILITY_COLUMNS:
        if column.endswith("_n_obs"):
            decisions[column] = decisions[column].astype("int64")

    return decisions


def add_derived_columns(features):
    features["close_date"] = features["close_time"].dt.strftime("%Y-%m-%d")
    features["quote_mid"] = (features["quote_yes_bid"] + features["quote_yes_ask"]) / 2
    features["quote_spread"] = features["quote_yes_ask"] - features["quote_yes_bid"]
    features["quote_age_seconds"] = (features["decision_time"] - features["quote_time"]).dt.total_seconds()
    features["log_moneyness"] = np.log(features["spot"] / features["strike"])
    features["T_years"] = features["horizon_minutes"] / MINUTES_PER_YEAR
    features["hour_utc"] = features["decision_time"].dt.hour
    features["day_of_week"] = features["decision_time"].dt.dayofweek

    if not features["settlement_result"].isin(["yes", "no"]).all():
        raise ValueError("settlement_result must contain only yes or no")

    features["y"] = features["settlement_result"].eq("yes").astype("int64")
    valid_expiration_target = features["expiration_value"].notna() & features["strike"].notna()
    features["y_from_expiration"] = np.nan
    features.loc[valid_expiration_target, "y_from_expiration"] = (
        features.loc[valid_expiration_target, "expiration_value"]
        > features.loc[valid_expiration_target, "strike"]
    ).astype(float)
    features["target_agrees"] = pd.Series(pd.NA, index=features.index, dtype="boolean")
    features.loc[valid_expiration_target, "target_agrees"] = (
        features.loc[valid_expiration_target, "y_from_expiration"]
        == features.loc[valid_expiration_target, "y"]
    )
    features["fwd_log_return"] = np.log(features["expiration_value"] / features["spot"])
    return features


def assert_structure(features):
    column_groups = [KEY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMNS, FUTURE_ONLY_COLUMNS]
    flattened_columns = [column for group in column_groups for column in group]
    assert len(flattened_columns) == len(set(flattened_columns)), "Output column groups must be disjoint"
    assert features.columns.tolist() == OUTPUT_COLUMNS, "Every output column must be classified exactly once"
    assert "volume" not in features.columns and "settlement_ts" not in features.columns
    assert len(features) == EXPECTED_ROWS
    assert not features.duplicated(["ticker", "horizon_minutes"]).any()
    assert features["ticker"].nunique() == EXPECTED_MARKETS
    assert features["horizon_minutes"].value_counts().to_dict() == {10: EXPECTED_MARKETS, 5: EXPECTED_MARKETS}
    expected_decision_time = features["close_time"] - pd.to_timedelta(features["horizon_minutes"], unit="min")
    assert features["decision_time"].equals(expected_decision_time)
    assert (features["open_time"] < features["decision_time"]).all()
    assert (features["decision_time"] < features["close_time"]).all()
    assert (features["quote_time"] <= features["decision_time"]).all()
    assert features["quote_time"].equals(pd.to_datetime(features["quote_period_end_ts"], unit="s", utc=True))
    assert (feature_end_labels(pd.DatetimeIndex(features["decision_time"])) < features["decision_time"]).all()
    assert (features["quote_age_seconds"] >= 0).all()
    assert (features["quote_spread"] >= 0).all()
    assert features["quote_yes_bid"].between(0, 1).all()
    assert features["quote_yes_ask"].between(0, 1).all()
    assert features.loc[features["spot"].notna(), "spot_age_seconds"].between(0, 10).all()
    assert features.loc[features["spot"].isna(), "spot_age_seconds"].isna().all()
    assert (features["y"] == features["settlement_value"]).all()
    assert features.loc[features["target_agrees"].notna(), "target_agrees"].all()
    assert features["target_agrees"].notna().sum() == 17_158
    assert features["target_agrees"].isna().sum() == 24


def build_market_features():
    markets, quotes = load_source_tables()
    features = build_decision_rows(markets, quotes)
    features = add_btc_features(features)
    features = add_derived_columns(features)
    features = features[OUTPUT_COLUMNS].sort_values(
        ["close_time", "ticker", "horizon_minutes"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    assert_structure(features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(features):,} rows to {OUTPUT_PATH}")
    print("Day 7 section 2.2 validation summary")
    print(f"Row count: {len(features):,}")
    print(f"Unique (ticker, horizon_minutes) count: {features[['ticker', 'horizon_minutes']].drop_duplicates().shape[0]:,}")
    print(f"Rows per horizon: {features['horizon_minutes'].value_counts().sort_index().to_dict()}")
    print(f"Target mean: {features['y'].mean():.6f}")
    print(f"target_agrees == True: {features['target_agrees'].eq(True).sum():,}")
    print(f"target_agrees == False: {features['target_agrees'].eq(False).sum():,}")
    print(f"target_agrees missing: {features['target_agrees'].isna().sum():,}")
    for horizon in HORIZON_MINUTES:
        horizon_rows = features[features["horizon_minutes"] == horizon]
        median_absolute_log_moneyness_bps = horizon_rows["log_moneyness"].abs().median() * 10_000
        print(f"Median absolute log_moneyness for T-{horizon}: {median_absolute_log_moneyness_bps:.6f} bps")
        print(f"Median quote_mid for T-{horizon}: {horizon_rows['quote_mid'].median():.6f}")
    print(f"Median quote_spread: {features['quote_spread'].median():.6f}")
    print(f"Negative quote spreads: {features['quote_spread'].lt(0).sum():,}")
    print(f"NaN count for spot: {features['spot'].isna().sum():,}")
    print(f"NaN count for 24hr_vol: {features['24hr_vol'].isna().sum():,}")
    print(f"NaN count for quote_yes_bid: {features['quote_yes_bid'].isna().sum():,}")
    print(f"NaN count for quote_yes_ask: {features['quote_yes_ask'].isna().sum():,}")
    print(f"NaN count for quote_last_price: {features['quote_last_price'].isna().sum():,}")
    print("NaN count for every column:")
    print(features.isna().sum().to_string())
    return features


if __name__ == "__main__":
    build_market_features()
