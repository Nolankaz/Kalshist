"""Count outcome-free Stage 0 candidate signals against frozen edge thresholds.

This Day 10 Section 3.3 script uses Tier 2 quote-aware, top-of-book,
size-unaware inputs. It reports candidate-signal counts only: it never loads
outcomes or test rows, does not choose trades, and does not calculate P&L.
"""

from decimal import Decimal
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_calibration import apply_platt
from scripts.analyze_spreads import (
    EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS,
    PREDICTION_COLUMNS,
    PRICE_BUCKET_LABELS,
    assign_price_buckets,
    integer_mils,
)
from scripts.evaluation_split import EXPECTED_SPLIT_ROWS, SPLIT_KEY, SPLIT_RANGES
from scripts.fees import DIRECT_MEMBER, KXBTC15M_FEE_MULTIPLIER, KXBTC15M_FEE_TYPE, calculate_single_fill_fee
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, ROW_KEY, common_prediction_mask


PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
MODEL_MARKET_GAP_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
THRESHOLD_PATH = PROJECT_ROOT / "data/execution/edge_threshold.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/execution/threshold_clearance_summary.parquet"

ALLOWED_SPLITS = ("train", "validation")
HORIZONS = (10, 5)
BASIS_SETTINGS = (1.2, 5.0)
PROBABILITY_VERSIONS = ("raw", "train_fitted_platt")
SELECTED_CANDIDATE = "5min_ewma_vol"
SELECTED_PROBABILITY_COLUMN = f"p_{SELECTED_CANDIDATE}"
CALIBRATED_PROBABILITY_COLUMN = "train_fitted_platt_probability"
PRIMARY_BASIS_BPS = 1.2
CONSERVATIVE_BASIS_BPS = 5.0
MARKET_SUMMARY_HORIZON = 0
PLATT_TOLERANCE = 1e-12
EDGE_TOLERANCE = 1e-15
TIER_LABEL = "Tier 2 inputs, outcome-free: candidate signals, not trades, not P&L."

PREDICTION_COLUMNS_TO_LOAD = ROW_KEY + ["split"] + PREDICTION_COLUMNS
FEATURE_COLUMNS_TO_LOAD = ROW_KEY + [SPLIT_KEY, "quote_yes_bid", "quote_yes_ask", "quote_mid", "quote_age_seconds"]
FORBIDDEN_COLUMNS = {
    "y", "settlement_result", "settlement_value", "expiration_value", "y_from_expiration",
    "target_agrees", "fwd_log_return",
}
EXPECTED_STALE_ROWS = {
    ("train", 10): 2,
    ("train", 5): 3,
    ("validation", 10): 0,
    ("validation", 5): 0,
}
WATERFALL_COUNT_COLUMNS = [
    "n_common", "n_stale_excluded", "n_best_side_ties", "n_positive_gross_best_side",
    "n_positive_net_best_side", "n_threshold_clear",
]
MARKET_COUNT_COLUMNS = [
    "n_unique_markets_threshold_clear", "n_dual_horizon_markets_threshold_clear",
    "n_dual_horizon_same_side", "n_dual_horizon_opposite_side",
    "n_dual_horizon_yes_yes", "n_dual_horizon_no_no",
]


def assert_safe_columns(columns, source):
    loaded = set(columns)
    forbidden = loaded & FORBIDDEN_COLUMNS
    patterned = {
        column for column in loaded
        if column.lower().startswith(("fwd_", "target_")) or "settlement" in column.lower() or "expiration" in column.lower()
    }
    assert not forbidden and not patterned, f"{source} loaded target/outcome/future-only columns: {sorted(forbidden | patterned)}"


def load_common_prediction_rows():
    assert len(PREDICTION_COLUMNS_TO_LOAD) == len(set(PREDICTION_COLUMNS_TO_LOAD)), "Prediction projection has duplicates"
    rows = pd.read_parquet(
        PREDICTIONS_PATH, columns=PREDICTION_COLUMNS_TO_LOAD,
        filters=[("split", "in", list(ALLOWED_SPLITS))],
    )
    assert list(rows.columns) == PREDICTION_COLUMNS_TO_LOAD, "Prediction projection changed"
    assert_safe_columns(rows.columns, "Stage 0 predictions")
    assert rows["split"].isin(ALLOWED_SPLITS).all(), "A forbidden prediction split was loaded"
    assert not rows["split"].eq("test").any(), "Test predictions entered clearance analysis"
    assert not rows.duplicated(ROW_KEY).any(), f"Prediction rows are not unique on {ROW_KEY}"

    common = rows.loc[common_prediction_mask(rows, PREDICTION_COLUMNS)].copy()
    actual_counts = common.groupby(["split", "horizon_minutes"], observed=True).size().to_dict()
    assert actual_counts == EXPECTED_COMMON_ROWS, f"Unexpected common-row counts: {actual_counts}"
    assert common[PREDICTION_COLUMNS].notna().all().all(), "Common rows contain missing predictions"
    return common


def load_quote_rows():
    train_start = SPLIT_RANGES["train"][0]
    validation_end = SPLIT_RANGES["validation"][1]
    rows = pd.read_parquet(
        FEATURES_PATH, columns=FEATURE_COLUMNS_TO_LOAD,
        filters=[(SPLIT_KEY, ">=", train_start), (SPLIT_KEY, "<=", validation_end)],
    )
    assert list(rows.columns) == FEATURE_COLUMNS_TO_LOAD, "Market-feature projection changed"
    assert_safe_columns(rows.columns, "market features")
    assert not rows.duplicated(ROW_KEY).any(), f"Market features are not unique on {ROW_KEY}"

    rows["split"] = pd.Series(pd.NA, index=rows.index, dtype="string")
    for split in ALLOWED_SPLITS:
        start, end = SPLIT_RANGES[split]
        rows.loc[rows[SPLIT_KEY].between(start, end, inclusive="both"), "split"] = split
    assert rows["split"].notna().all(), "A loaded feature row is outside train/validation"
    assert rows["split"].isin(ALLOWED_SPLITS).all() and not rows["split"].eq("test").any(), "Test features entered clearance analysis"
    counts = rows.groupby("split").size().to_dict()
    expected = {split: EXPECTED_SPLIT_ROWS[split] for split in ALLOWED_SPLITS}
    assert counts == expected, f"Unexpected train/validation feature counts: {counts}"
    return rows


def load_legitimate_platt_parameters():
    columns = ["candidate", "horizon_minutes", "fit_split", "parameter_role", "a", "b"]
    rows = pd.read_parquet(
        PLATT_PARAMETERS_PATH, columns=columns,
        filters=[
            ("fit_split", "==", "train"),
            ("parameter_role", "==", "legitimate_train_fit"),
            ("candidate", "==", SELECTED_CANDIDATE),
        ],
    )
    assert list(rows.columns) == columns, "Platt-parameter projection changed"
    assert_safe_columns(rows.columns, "Platt parameters")
    assert len(rows) == 2, f"Expected two legitimate train-fitted Platt rows, found {len(rows)}"
    assert rows["fit_split"].eq("train").all(), "A non-train Platt fit entered clearance analysis"
    assert rows["parameter_role"].eq("legitimate_train_fit").all(), "A non-legitimate Platt fit entered clearance analysis"
    assert rows["candidate"].eq(SELECTED_CANDIDATE).all(), "A different sigma candidate entered clearance analysis"
    assert set(rows["horizon_minutes"]) == set(HORIZONS), "Platt parameter horizons changed"
    assert not rows.duplicated("horizon_minutes").any(), "Platt parameters are not unique by horizon"
    assert np.isfinite(rows[["a", "b"]].to_numpy()).all(), "Platt parameters are non-finite"
    return rows.set_index("horizon_minutes")


def reconstruct_platt_probability(rows, parameters):
    rows[CALIBRATED_PROBABILITY_COLUMN] = np.nan
    for horizon in HORIZONS:
        mask = rows["horizon_minutes"].eq(horizon)
        fit = {"a": float(parameters.loc[horizon, "a"]), "b": float(parameters.loc[horizon, "b"])}
        rows.loc[mask, CALIBRATED_PROBABILITY_COLUMN] = apply_platt(rows.loc[mask, SELECTED_PROBABILITY_COLUMN], fit)
    probability = rows[CALIBRATED_PROBABILITY_COLUMN]
    assert probability.notna().all() and np.isfinite(probability).all(), "Reconstructed Platt probability is invalid"
    assert probability.between(0.0, 1.0, inclusive="both").all(), "Reconstructed Platt probability falls outside [0, 1]"
    return rows


def verify_platt_carry_forward(rows):
    columns = ["ticker", "split", "horizon_minutes", "raw_model_probability", "platt_model_probability"]
    gap = pd.read_parquet(
        MODEL_MARKET_GAP_PATH, columns=columns,
        filters=[("split", "in", list(ALLOWED_SPLITS))],
    )
    assert list(gap.columns) == columns, "Model-market-gap projection changed"
    assert_safe_columns(gap.columns, "model-market-gap artifact")
    assert gap["split"].isin(ALLOWED_SPLITS).all() and not gap["split"].eq("test").any(), "Test rows entered the Platt check"
    assert not gap.duplicated(ROW_KEY).any(), f"Model-market-gap rows are not unique on {ROW_KEY}"

    checked = rows.merge(gap, on=ROW_KEY + ["split"], how="outer", validate="one_to_one", indicator=True)
    assert checked["_merge"].eq("both").all(), "Day 9 model-market-gap keys differ from the common population"
    raw_difference = np.abs(checked[SELECTED_PROBABILITY_COLUMN].to_numpy() - checked["raw_model_probability"].to_numpy())
    platt_difference = np.abs(checked[CALIBRATED_PROBABILITY_COLUMN].to_numpy() - checked["platt_model_probability"].to_numpy())
    maximum_raw_difference = float(raw_difference.max())
    maximum_platt_difference = float(platt_difference.max())
    assert maximum_raw_difference <= PLATT_TOLERANCE, f"Selected raw probability differs by {maximum_raw_difference:.17g}"
    assert maximum_platt_difference <= PLATT_TOLERANCE, f"Train-fitted Platt probability differs by {maximum_platt_difference:.17g}"
    return maximum_platt_difference


def join_quotes(common, features):
    columns = ROW_KEY + ["split", SPLIT_KEY, "quote_yes_bid", "quote_yes_ask", "quote_mid", "quote_age_seconds"]
    rows = common.merge(features[columns], on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert rows["_merge"].eq("both").all(), "A common prediction row did not match its market quote"
    assert len(rows) == len(common), "Quote join changed the common-row count"
    rows = rows.drop(columns="_merge")
    numeric = ["quote_yes_bid", "quote_yes_ask", "quote_mid", "quote_age_seconds"]
    assert rows[numeric].notna().all().all() and np.isfinite(rows[numeric].to_numpy()).all(), "A required quote value is invalid"
    assert rows["quote_yes_bid"].lt(rows["quote_yes_ask"]).all(), "A common row has a non-positive bid/ask spread"
    assert rows[["quote_yes_bid", "quote_yes_ask", "quote_mid"]].ge(0.0).all().all(), "A quote probability is negative"
    assert rows[["quote_yes_bid", "quote_yes_ask", "quote_mid"]].le(1.0).all().all(), "A quote probability exceeds one"
    return rows


def calculate_execution_fees(rows):
    bid_mils = integer_mils(rows["quote_yes_bid"], "quote_yes_bid")
    ask_mils = integer_mils(rows["quote_yes_ask"], "quote_yes_ask")
    assert np.all(ask_mils > bid_mils), "Executable asks must exceed bids"
    ask_prices = [Decimal(int(value)) / 1000 for value in ask_mils]
    no_prices = [Decimal(1000 - int(value)) / 1000 for value in bid_mils]

    yes_results = [
        calculate_single_fill_fee(
            price, 1, "buy", multiplier=KXBTC15M_FEE_MULTIPLIER,
            fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER,
        )
        for price in ask_prices
    ]
    no_results = [
        calculate_single_fill_fee(
            price, 1, "buy", multiplier=KXBTC15M_FEE_MULTIPLIER,
            fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER,
        )
        for price in no_prices
    ]
    assert all(result.executable_price == price for result, price in zip(yes_results, ask_prices)), "YES fees did not use asks"
    assert all(result.executable_price == price for result, price in zip(no_results, no_prices)), "NO fees did not use 1 - bid"
    assert all(result.action == "buy" and result.rounding_mode == DIRECT_MEMBER for result in yes_results + no_results), (
        "A fee did not use a Direct Member one-contract buy"
    )
    assert all(
        result.contracts == 1 and result.fee_type == KXBTC15M_FEE_TYPE and result.multiplier == KXBTC15M_FEE_MULTIPLIER
        for result in yes_results + no_results
    ), "A fee did not use the frozen KXBTC15M one-contract settings"

    rows["fee_yes_probability_units"] = np.array([float(result.net_cash_fee) for result in yes_results])
    rows["fee_no_probability_units"] = np.array([float(result.net_cash_fee) for result in no_results])
    assert rows[["fee_yes_probability_units", "fee_no_probability_units"]].ge(0.0).all().all(), "A fee is negative"
    return rows


def apply_stale_rule_and_buckets(rows):
    rows["signal_eligible"] = rows["quote_age_seconds"].le(60.0)
    stale_counts = rows.loc[~rows["signal_eligible"]].groupby(["split", "horizon_minutes"], observed=True).size()
    stale_counts = stale_counts.reindex(pd.MultiIndex.from_tuples(EXPECTED_STALE_ROWS, names=["split", "horizon_minutes"]), fill_value=0).to_dict()
    assert stale_counts == EXPECTED_STALE_ROWS, f"Unexpected A6 stale-row counts: {stale_counts}"

    rows = assign_price_buckets(rows)
    for horizon in HORIZONS:
        validation = rows.loc[rows["split"].eq("validation") & rows["horizon_minutes"].eq(horizon)]
        counts = validation["price_bucket"].value_counts(sort=False).reindex(PRICE_BUCKET_LABELS, fill_value=0).tolist()
        expected = EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS[horizon]
        assert counts == expected, f"Validation T-{horizon} price-bucket counts changed: {counts}"
    rows["price_bucket"] = rows["price_bucket"].astype("string")
    return rows


def build_probability_rows(rows):
    versions = []
    for version, probability_column in (
        ("raw", SELECTED_PROBABILITY_COLUMN),
        ("train_fitted_platt", CALIBRATED_PROBABILITY_COLUMN),
    ):
        version_rows = rows.copy()
        version_rows["probability_version"] = version
        version_rows["model_probability"] = version_rows[probability_column]
        versions.append(version_rows)
    edge_rows = pd.concat(versions, ignore_index=True)
    assert edge_rows["probability_version"].isin(PROBABILITY_VERSIONS).all(), "An unsupported probability version was created"

    p = edge_rows["model_probability"]
    edge_rows["disagreement"] = p - edge_rows["quote_mid"]
    edge_rows["gross_edge_yes"] = p - edge_rows["quote_yes_ask"]
    edge_rows["gross_edge_no"] = edge_rows["quote_yes_bid"] - p
    edge_rows["net_edge_yes"] = edge_rows["gross_edge_yes"] - edge_rows["fee_yes_probability_units"]
    edge_rows["net_edge_no"] = edge_rows["gross_edge_no"] - edge_rows["fee_no_probability_units"]

    both_gross_positive = edge_rows["gross_edge_yes"].gt(0.0) & edge_rows["gross_edge_no"].gt(0.0)
    assert not both_gross_positive.any(), "A row has positive gross YES and NO edge simultaneously"
    assert (edge_rows["net_edge_yes"] <= edge_rows["gross_edge_yes"] + EDGE_TOLERANCE).all(), "YES net edge exceeds gross edge"
    assert (edge_rows["net_edge_no"] <= edge_rows["gross_edge_no"] + EDGE_TOLERANCE).all(), "NO net edge exceeds gross edge"

    edge_rows["best_side_tie"] = edge_rows["net_edge_yes"].eq(edge_rows["net_edge_no"])
    edge_rows["best_side"] = np.where(edge_rows["net_edge_yes"].ge(edge_rows["net_edge_no"]), "YES", "NO")
    edge_rows["best_net_edge"] = np.where(
        edge_rows["best_side"].eq("YES"), edge_rows["net_edge_yes"], edge_rows["net_edge_no"],
    )
    edge_rows["best_gross_edge"] = np.where(
        edge_rows["best_side"].eq("YES"), edge_rows["gross_edge_yes"], edge_rows["gross_edge_no"],
    )
    return edge_rows


def load_thresholds():
    columns = [
        "horizon_minutes", "basis_bps", "threshold_role", "price_bucket", "basis_term",
        "model_error_term", "required_net_edge", "probability_version", "fee_model_status",
        "order_size", "bucket_merge_fired",
    ]
    rows = pd.read_parquet(THRESHOLD_PATH, columns=columns)
    assert list(rows.columns) == columns, "Threshold projection changed"
    assert_safe_columns(rows.columns, "edge-threshold artifact")
    assert len(rows) == 28, f"Expected 28 threshold rows, found {len(rows)}"
    assert not rows.duplicated(["horizon_minutes", "basis_bps", "price_bucket"]).any(), "Threshold key is not unique"
    assert set(rows["horizon_minutes"]) == set(HORIZONS), "Threshold horizons changed"
    assert set(rows["basis_bps"]) == set(BASIS_SETTINGS), "Threshold basis settings changed"
    assert rows["probability_version"].eq("train_fitted_platt").all(), "Threshold construction probability changed"
    assert rows["fee_model_status"].eq("verified").all(), "Threshold fee model is not verified"
    assert rows["order_size"].eq(1).all(), "Threshold order size is not one contract"
    assert not rows["bucket_merge_fired"].any(), "The frozen threshold artifact unexpectedly merged buckets"
    assert rows.loc[rows["basis_bps"].eq(PRIMARY_BASIS_BPS), "threshold_role"].eq("primary").all(), "Primary threshold is mislabeled"
    assert rows.loc[rows["basis_bps"].eq(CONSERVATIVE_BASIS_BPS), "threshold_role"].eq("conservative_sensitivity").all(), (
        "Conservative sensitivity is mislabeled"
    )
    assert np.allclose(
        rows["required_net_edge"], rows["basis_term"] + rows["model_error_term"], rtol=0.0, atol=EDGE_TOLERANCE,
    ), "Threshold additive identity failed"
    rows = rows.rename(columns={"probability_version": "threshold_probability_version"})
    return rows


def join_thresholds(edge_rows, thresholds):
    rows = edge_rows.merge(
        thresholds, on=["horizon_minutes", "price_bucket"], how="left", validate="many_to_many", indicator=True,
    )
    assert rows["_merge"].eq("both").all(), "A candidate row did not match its frozen threshold"
    assert len(rows) == len(edge_rows) * len(BASIS_SETTINGS), "Threshold join did not create exactly two basis settings per row"
    rows = rows.drop(columns="_merge")
    assert rows.groupby(ROW_KEY + ["split", "probability_version"], observed=True).size().eq(2).all(), (
        "A candidate row does not have exactly two threshold settings"
    )
    rows["threshold_clear"] = rows["signal_eligible"] & rows["best_net_edge"].ge(rows["required_net_edge"])
    clearing = rows.loc[rows["threshold_clear"]]
    assert clearing["best_net_edge"].gt(0.0).all(), "A threshold-clearing row lacks positive best-side net edge"
    assert clearing["signal_eligible"].all(), "An A6-ineligible row cleared a threshold"
    return rows


def waterfall_record(rows, split, horizon, basis_bps, probability_version, breakdown, breakdown_value):
    n_common = len(rows)
    clearing_edges = rows.loc[rows["threshold_clear"], "best_net_edge"]
    role = "primary" if basis_bps == PRIMARY_BASIS_BPS else "conservative_sensitivity"
    probability_role = "diagnostic" if probability_version == "raw" else "shipping"
    denominator = n_common if n_common else np.nan
    return {
        "split": split,
        "horizon_minutes": horizon,
        "basis_bps": basis_bps,
        "probability_version": probability_version,
        "breakdown": breakdown,
        "breakdown_value": breakdown_value,
        "threshold_role": role,
        "probability_role": probability_role,
        "n_common": n_common,
        "n_stale_excluded": int((~rows["signal_eligible"]).sum()),
        "n_best_side_ties": int(rows["best_side_tie"].sum()),
        "n_positive_gross_best_side": int(rows["best_gross_edge"].gt(0.0).sum()),
        "n_positive_net_best_side": int(rows["best_net_edge"].gt(0.0).sum()),
        "n_threshold_clear": int(rows["threshold_clear"].sum()),
        "stale_excluded_share": float((~rows["signal_eligible"]).sum() / denominator),
        "positive_gross_best_side_share": float(rows["best_gross_edge"].gt(0.0).sum() / denominator),
        "positive_net_best_side_share": float(rows["best_net_edge"].gt(0.0).sum() / denominator),
        "threshold_clear_share": float(rows["threshold_clear"].sum() / denominator),
        "median_best_side_net_edge_probability_units": float(clearing_edges.median()) if len(clearing_edges) else np.nan,
        "p90_best_side_net_edge_probability_units": float(clearing_edges.quantile(0.90)) if len(clearing_edges) else np.nan,
        "required_net_edge_min_probability_units": float(rows["required_net_edge"].min()) if n_common else np.nan,
        "required_net_edge_max_probability_units": float(rows["required_net_edge"].max()) if n_common else np.nan,
    }


def build_market_summary(rows):
    records = []
    for split in ALLOWED_SPLITS:
        for basis_bps in BASIS_SETTINGS:
            role = "primary" if basis_bps == PRIMARY_BASIS_BPS else "conservative_sensitivity"
            for probability_version in PROBABILITY_VERSIONS:
                group = rows.loc[
                    rows["split"].eq(split) & rows["basis_bps"].eq(basis_bps)
                    & rows["probability_version"].eq(probability_version)
                ]
                clearing = group.loc[group["threshold_clear"], ["ticker", "horizon_minutes", "best_side"]]
                assert not clearing.duplicated(["ticker", "horizon_minutes"]).any(), "A market/horizon clears more than once"
                side_by_horizon = clearing.pivot(index="ticker", columns="horizon_minutes", values="best_side")
                for horizon in HORIZONS:
                    if horizon not in side_by_horizon:
                        side_by_horizon[horizon] = pd.NA
                dual = side_by_horizon.dropna(subset=list(HORIZONS))
                same = dual[10].eq(dual[5])
                yes_yes = dual[10].eq("YES") & dual[5].eq("YES")
                no_no = dual[10].eq("NO") & dual[5].eq("NO")
                records.append({
                    "split": split,
                    "horizon_minutes": MARKET_SUMMARY_HORIZON,
                    "basis_bps": basis_bps,
                    "probability_version": probability_version,
                    "breakdown": "market_summary",
                    "breakdown_value": "cross_horizon",
                    "threshold_role": role,
                    "probability_role": "diagnostic" if probability_version == "raw" else "shipping",
                    "n_unique_markets_threshold_clear": int(clearing["ticker"].nunique()),
                    "n_dual_horizon_markets_threshold_clear": len(dual),
                    "n_dual_horizon_same_side": int(same.sum()),
                    "n_dual_horizon_opposite_side": int((~same).sum()),
                    "n_dual_horizon_yes_yes": int(yes_yes.sum()),
                    "n_dual_horizon_no_no": int(no_no.sum()),
                })
    return records


def build_summary(rows):
    records = []
    for split in ALLOWED_SPLITS:
        for horizon in HORIZONS:
            for basis_bps in BASIS_SETTINGS:
                for probability_version in PROBABILITY_VERSIONS:
                    group = rows.loc[
                        rows["split"].eq(split) & rows["horizon_minutes"].eq(horizon)
                        & rows["basis_bps"].eq(basis_bps) & rows["probability_version"].eq(probability_version)
                    ]
                    records.append(waterfall_record(group, split, horizon, basis_bps, probability_version, "overall", "all"))
                    for bucket in PRICE_BUCKET_LABELS:
                        subset = group.loc[group["price_bucket"].eq(bucket)]
                        records.append(waterfall_record(subset, split, horizon, basis_bps, probability_version, "price_bucket", bucket))
                    for side in ("YES", "NO"):
                        subset = group.loc[group["best_side"].eq(side)]
                        records.append(waterfall_record(subset, split, horizon, basis_bps, probability_version, "side", side))
    records.extend(build_market_summary(rows))
    summary = pd.DataFrame(records)
    for column in WATERFALL_COUNT_COLUMNS + MARKET_COUNT_COLUMNS:
        if column not in summary:
            summary[column] = pd.NA
        summary[column] = summary[column].astype("Int64")

    key = ["split", "horizon_minutes", "basis_bps", "probability_version", "breakdown", "breakdown_value"]
    assert len(summary) == 168, f"Expected 168 summary rows, found {len(summary)}"
    assert not summary.duplicated(key).any(), f"Summary key is not unique: {key}"
    assert set(summary["breakdown"]) == {"overall", "price_bucket", "side", "market_summary"}, "Breakdown labels changed"
    assert not any(
        token in column.lower() for column in summary.columns
        for token in ("outcome", "settlement", "expiration", "profit", "return", "win_rate", "hit_rate", "pnl")
    ), "A forbidden outcome/P&L-like statistic entered the summary"
    return sort_summary(summary)


def sort_summary(summary):
    split_order = {"train": 0, "validation": 1}
    horizon_order = {10: 0, 5: 1, MARKET_SUMMARY_HORIZON: 2}
    basis_order = {PRIMARY_BASIS_BPS: 0, CONSERVATIVE_BASIS_BPS: 1}
    probability_order = {"train_fitted_platt": 0, "raw": 1}
    breakdown_order = {"overall": 0, "price_bucket": 1, "side": 2, "market_summary": 3}
    value_order = {"all": 0, **{value: index for index, value in enumerate(PRICE_BUCKET_LABELS, start=1)}, "YES": 8, "NO": 9, "cross_horizon": 10}
    temporary = summary.assign(
        _split=summary["split"].map(split_order),
        _horizon=summary["horizon_minutes"].map(horizon_order),
        _basis=summary["basis_bps"].map(basis_order),
        _probability=summary["probability_version"].map(probability_order),
        _breakdown=summary["breakdown"].map(breakdown_order),
        _value=summary["breakdown_value"].map(value_order),
    )
    return temporary.sort_values(
        ["_split", "_horizon", "_basis", "_probability", "_breakdown", "_value"]
    ).drop(columns=["_split", "_horizon", "_basis", "_probability", "_breakdown", "_value"]).reset_index(drop=True)


def validate_waterfalls(summary):
    overall = summary.loc[summary["breakdown"].eq("overall")].copy()
    assert len(overall) == 16, f"Expected 16 overall rows, found {len(overall)}"
    for row in overall.itertuples(index=False):
        expected_common = EXPECTED_COMMON_ROWS[(row.split, row.horizon_minutes)]
        expected_stale = EXPECTED_STALE_ROWS[(row.split, row.horizon_minutes)]
        assert row.n_common == expected_common, f"{row.split} T-{row.horizon_minutes} common denominator changed"
        assert row.n_stale_excluded == expected_stale, f"{row.split} T-{row.horizon_minutes} stale count changed"
        assert row.n_common >= row.n_positive_gross_best_side >= row.n_positive_net_best_side >= row.n_threshold_clear, (
            f"Waterfall nesting failed for {row.split} T-{row.horizon_minutes}, {row.basis_bps:g} bps, {row.probability_version}"
        )

    fields = WATERFALL_COUNT_COLUMNS
    for breakdown in ("price_bucket", "side"):
        grouped = summary.loc[summary["breakdown"].eq(breakdown)].groupby(
            ["split", "horizon_minutes", "basis_bps", "probability_version"], observed=True,
        )[fields].sum().reset_index()
        checked = overall.merge(
            grouped, on=["split", "horizon_minutes", "basis_bps", "probability_version"],
            suffixes=("_overall", f"_{breakdown}"), validate="one_to_one",
        )
        for field in fields:
            assert checked[f"{field}_overall"].equals(checked[f"{field}_{breakdown}"]), f"{breakdown} does not sum to overall {field}"

    market = summary.loc[summary["breakdown"].eq("market_summary")]
    assert len(market) == 8 and market["horizon_minutes"].eq(MARKET_SUMMARY_HORIZON).all(), "Market-summary sentinel rows changed"
    assert (
        market["n_dual_horizon_markets_threshold_clear"]
        == market["n_dual_horizon_same_side"] + market["n_dual_horizon_opposite_side"]
    ).all(), "Dual-horizon side counts do not reconcile"
    assert (
        market["n_dual_horizon_same_side"] == market["n_dual_horizon_yes_yes"] + market["n_dual_horizon_no_no"]
    ).all(), "Same-side dual-horizon counts do not reconcile"


def print_table(title, rows, columns, formatters=None):
    print(f"\n{TIER_LABEL}")
    print(title)
    print(rows[columns].to_string(index=False, formatters=formatters or {}))


def print_console_summary(summary, maximum_platt_difference):
    overall = summary.loc[summary["breakdown"].eq("overall")]
    primary_platt = overall.loc[
        overall["basis_bps"].eq(PRIMARY_BASIS_BPS) & overall["probability_version"].eq("train_fitted_platt")
    ]
    print_table(
        "Primary 1.2 bps / train-fitted Platt waterfall (shipping probability)", primary_platt,
        ["split", "horizon_minutes", "n_common", "n_stale_excluded", "n_positive_gross_best_side", "n_positive_net_best_side", "n_threshold_clear"],
    )

    sensitivity_platt = overall.loc[
        overall["basis_bps"].eq(CONSERVATIVE_BASIS_BPS) & overall["probability_version"].eq("train_fitted_platt")
    ]
    print_table(
        "Conservative 5.0 bps sensitivity / train-fitted Platt clearance", sensitivity_platt,
        ["split", "horizon_minutes", "n_common", "n_threshold_clear", "threshold_clear_share"],
        {"threshold_clear_share": "{:.2%}".format},
    )

    comparison = overall.pivot(
        index=["split", "horizon_minutes", "basis_bps"], columns="probability_version", values="n_threshold_clear",
    ).reset_index()
    print_table(
        "Raw diagnostic versus train-fitted Platt clearance (1.2 primary; 5.0 conservative sensitivity)", comparison,
        ["split", "horizon_minutes", "basis_bps", "raw", "train_fitted_platt"],
    )

    sides = summary.loc[
        summary["breakdown"].eq("side") & summary["basis_bps"].eq(PRIMARY_BASIS_BPS)
        & summary["probability_version"].eq("train_fitted_platt")
    ]
    print_table(
        "Primary 1.2 bps / train-fitted Platt clearing side", sides,
        ["split", "horizon_minutes", "breakdown_value", "n_threshold_clear"],
    )

    print_table(
        "Primary 1.2 bps / train-fitted Platt best-side net edge among clearing rows (probability units)", primary_platt,
        ["split", "horizon_minutes", "n_threshold_clear", "median_best_side_net_edge_probability_units", "p90_best_side_net_edge_probability_units"],
        {
            "median_best_side_net_edge_probability_units": lambda value: "NA" if pd.isna(value) else f"{value:.9f}",
            "p90_best_side_net_edge_probability_units": lambda value: "NA" if pd.isna(value) else f"{value:.9f}",
        },
    )

    markets = summary.loc[
        summary["breakdown"].eq("market_summary") & summary["basis_bps"].eq(PRIMARY_BASIS_BPS)
        & summary["probability_version"].eq("train_fitted_platt")
    ]
    print_table(
        "Primary 1.2 bps / train-fitted Platt distinct-market diagnostics (descriptive; no trade collapse)", markets,
        [
            "split", "n_unique_markets_threshold_clear", "n_dual_horizon_markets_threshold_clear",
            "n_dual_horizon_same_side", "n_dual_horizon_opposite_side",
        ],
    )

    ties = overall.loc[overall["basis_bps"].eq(PRIMARY_BASIS_BPS), ["split", "horizon_minutes", "probability_version", "n_best_side_ties"]]
    print_table(
        "Best-side exact ties (deterministically assigned YES; basis-independent)", ties,
        ["split", "horizon_minutes", "probability_version", "n_best_side_ties"],
    )
    print(f"\nPlatt reproduction maximum absolute difference: {maximum_platt_difference:.17g}")
    print(f"Output rows: {len(summary):,}")
    print(f"Saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


def count_threshold_clearance():
    predictions = load_common_prediction_rows()
    parameters = load_legitimate_platt_parameters()
    predictions = reconstruct_platt_probability(predictions, parameters)
    maximum_platt_difference = verify_platt_carry_forward(predictions)

    features = load_quote_rows()
    rows = join_quotes(predictions, features)
    rows = calculate_execution_fees(rows)
    rows = apply_stale_rule_and_buckets(rows)
    edge_rows = build_probability_rows(rows)
    thresholds = load_thresholds()
    threshold_rows = join_thresholds(edge_rows, thresholds)

    summary = build_summary(threshold_rows)
    validate_waterfalls(summary)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUTPUT_PATH, index=False)
    saved = pd.read_parquet(OUTPUT_PATH, columns=list(summary.columns))
    assert len(saved) == len(summary), "Saved summary row count changed"
    assert list(saved.columns) == list(summary.columns), "Saved summary schema changed"
    print_console_summary(summary, maximum_platt_difference)
    return summary


if __name__ == "__main__":
    count_threshold_clearance()
