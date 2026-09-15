"""Build the preregistered Day 10 Stage 0 net-edge threshold table.

This Section 3.2 script implements the saved ``Day 10 Edge Threshold Rule``
mechanically. It reads validation outcomes only for horizon-level ECE, never
loads test rows, and does not calculate edges, signal clearance, or P&L.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_calibration import apply_platt, build_reliability_table
from scripts.analyze_spreads import (
    EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS,
    PREDICTION_COLUMNS,
    PRICE_BUCKET_EDGES,
    PRICE_BUCKET_LABELS,
    assign_price_buckets,
)
from scripts.check_market_features import rank_auc
from scripts.evaluation_split import EXPECTED_SPLIT_ROWS, SPLIT_KEY, SPLIT_RANGES
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, ROW_KEY, common_prediction_mask


PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
PLATT_SCORES_PATH = PROJECT_ROOT / "data/models/stage0_platt_validation_scores.parquet"
BASIS_SENSITIVITY_PATH = PROJECT_ROOT / "data/models/stage0_basis_probability_sensitivity.parquet"
MODEL_MARKET_GAP_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/execution/edge_threshold.parquet"

VALIDATION_SPLIT = "validation"
HORIZONS = (10, 5)
BASIS_SETTINGS = (1.2, 5.0)
SELECTED_CANDIDATE = "5min_ewma_vol"
SELECTED_PROBABILITY_COLUMN = f"p_{SELECTED_CANDIDATE}"
CALIBRATED_PROBABILITY_COLUMN = "train_fitted_platt_probability"
PROBABILITY_VERSION = "train_fitted_platt"
PRIMARY_BASIS_BPS = 1.2
CONSERVATIVE_BASIS_BPS = 5.0
SPARSE_BUCKET_MINIMUM = 50
FEE_MODEL_STATUS = "verified"
PLATT_TOLERANCE = 1e-12
AUC_TOLERANCE = 1e-12
ADDITION_TOLERANCE = 1e-15


def load_validation_common_predictions():
    columns = ROW_KEY + ["split", "y"] + PREDICTION_COLUMNS
    assert len(columns) == len(set(columns)), "Prediction projection contains duplicate columns"
    rows = pd.read_parquet(PREDICTIONS_PATH, columns=columns, filters=[("split", "==", VALIDATION_SPLIT)])
    assert list(rows.columns) == columns, "Prediction projection changed"
    assert rows["split"].eq(VALIDATION_SPLIT).all(), "A non-validation prediction row was loaded"
    assert not rows["split"].eq("test").any(), "Test rows entered threshold construction"
    assert not rows.duplicated(ROW_KEY).any(), f"Prediction rows are not unique on {ROW_KEY}"
    assert rows["y"].notna().all() and rows["y"].isin([0, 1]).all(), "Validation outcomes are invalid"

    common = rows.loc[common_prediction_mask(rows, PREDICTION_COLUMNS)].copy()
    for horizon in HORIZONS:
        horizon_rows = common.loc[common["horizon_minutes"].eq(horizon)]
        expected = EXPECTED_COMMON_ROWS[(VALIDATION_SPLIT, horizon)]
        assert len(horizon_rows) == expected, f"Expected {expected:,} validation common T-{horizon} rows, found {len(horizon_rows):,}"
        assert horizon_rows[PREDICTION_COLUMNS].notna().all().all(), f"Validation common T-{horizon} has missing predictions"
    assert len(common) == sum(EXPECTED_COMMON_ROWS[(VALIDATION_SPLIT, horizon)] for horizon in HORIZONS)
    return common


def load_legitimate_platt_parameters():
    columns = ["candidate", "horizon_minutes", "fit_split", "parameter_role", "a", "b"]
    parameters = pd.read_parquet(
        PLATT_PARAMETERS_PATH, columns=columns,
        filters=[
            ("fit_split", "==", "train"),
            ("parameter_role", "==", "legitimate_train_fit"),
            ("candidate", "==", SELECTED_CANDIDATE),
        ],
    )
    assert len(parameters) == 2, f"Expected two legitimate train-fitted Platt rows, found {len(parameters)}"
    assert parameters["candidate"].eq(SELECTED_CANDIDATE).all(), "Platt parameters use another sigma candidate"
    assert parameters["fit_split"].eq("train").all(), "A non-train Platt parameter was loaded"
    assert parameters["parameter_role"].eq("legitimate_train_fit").all(), "A non-legitimate Platt parameter was loaded"
    assert set(parameters["horizon_minutes"]) == set(HORIZONS), "Platt parameter horizons changed"
    assert not parameters.duplicated("horizon_minutes").any(), "Platt parameters are not unique by horizon"
    assert np.isfinite(parameters[["a", "b"]].to_numpy()).all() and parameters["b"].gt(0).all(), "Platt parameters are invalid"
    return parameters.set_index("horizon_minutes")


def reconstruct_calibrated_probability(rows, parameters):
    rows[CALIBRATED_PROBABILITY_COLUMN] = np.nan
    for horizon in HORIZONS:
        mask = rows["horizon_minutes"].eq(horizon)
        fit = {"a": float(parameters.loc[horizon, "a"]), "b": float(parameters.loc[horizon, "b"])}
        rows.loc[mask, CALIBRATED_PROBABILITY_COLUMN] = apply_platt(rows.loc[mask, SELECTED_PROBABILITY_COLUMN], fit)
    probabilities = rows[CALIBRATED_PROBABILITY_COLUMN]
    assert probabilities.notna().all() and np.isfinite(probabilities).all(), "Reconstructed Platt probabilities are invalid"
    assert probabilities.between(0.0, 1.0, inclusive="both").all(), "Reconstructed Platt probabilities fall outside [0, 1]"
    return rows


def regression_check_probabilities(rows):
    gap_columns = ROW_KEY[:1] + ["split", "horizon_minutes", "raw_model_probability", "platt_model_probability"]
    gap = pd.read_parquet(MODEL_MARKET_GAP_PATH, columns=gap_columns, filters=[("split", "==", VALIDATION_SPLIT)])
    assert gap["split"].eq(VALIDATION_SPLIT).all() and not gap["split"].eq("test").any(), "Test rows entered the Platt regression check"
    assert not gap.duplicated(ROW_KEY).any(), f"Model-market-gap rows are not unique on {ROW_KEY}"

    checked = rows.merge(gap, on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert checked["_merge"].eq("both").all(), "A validation common row did not match the model-market-gap artifact"
    assert len(checked) == len(gap), "The validation model-market-gap keys differ from the common population"
    raw_difference = np.abs(checked[SELECTED_PROBABILITY_COLUMN].to_numpy() - checked["raw_model_probability"].to_numpy())
    platt_difference = np.abs(checked[CALIBRATED_PROBABILITY_COLUMN].to_numpy() - checked["platt_model_probability"].to_numpy())
    maximum_raw_difference = float(raw_difference.max())
    maximum_platt_difference = float(platt_difference.max())
    assert maximum_raw_difference <= PLATT_TOLERANCE, f"Selected raw probability differs by {maximum_raw_difference:.17g}"
    assert maximum_platt_difference <= PLATT_TOLERANCE, f"Train-fitted Platt probability differs by {maximum_platt_difference:.17g}"

    score_columns = ["candidate", "horizon_minutes", "n", "calibrated_auc"]
    stored_scores = pd.read_parquet(PLATT_SCORES_PATH, columns=score_columns)
    assert len(stored_scores) == 2 and stored_scores["candidate"].eq(SELECTED_CANDIDATE).all(), "Stored Platt score rows changed"
    assert set(stored_scores["horizon_minutes"]) == set(HORIZONS), "Stored Platt AUC horizons changed"
    assert not stored_scores.duplicated("horizon_minutes").any(), "Stored Platt AUC is not unique by horizon"
    stored_scores = stored_scores.set_index("horizon_minutes")

    auc_records = []
    for horizon in HORIZONS:
        horizon_rows = checked.loc[checked["horizon_minutes"].eq(horizon)]
        auc, auc_count = rank_auc(horizon_rows[CALIBRATED_PROBABILITY_COLUMN], horizon_rows["y"])
        stored_auc = float(stored_scores.loc[horizon, "calibrated_auc"])
        expected_count = EXPECTED_COMMON_ROWS[(VALIDATION_SPLIT, horizon)]
        assert auc_count == expected_count == int(stored_scores.loc[horizon, "n"]), f"T-{horizon} AUC population changed"
        auc_difference = abs(auc - stored_auc)
        assert auc_difference <= AUC_TOLERANCE, f"T-{horizon} calibrated AUC differs by {auc_difference:.17g}"
        auc_records.append({
            "horizon_minutes": horizon,
            "calibrated_auc": auc,
            "stored_calibrated_auc": stored_auc,
            "auc_regression_abs_diff": auc_difference,
        })
    return checked.drop(columns="_merge"), maximum_platt_difference, pd.DataFrame(auc_records)


def calculate_model_error_terms(rows, auc_results):
    records = []
    for horizon in HORIZONS:
        horizon_rows = rows.loc[rows["horizon_minutes"].eq(horizon)].copy()
        expected_count = EXPECTED_COMMON_ROWS[(VALIDATION_SPLIT, horizon)]
        reliability = build_reliability_table(horizon_rows, horizon, prediction_column=CALIBRATED_PROBABILITY_COLUMN)
        assert len(reliability) == 10, f"T-{horizon} reliability table does not have 10 deciles"
        assert int(reliability["n"].sum()) == expected_count, f"T-{horizon} reliability counts changed"
        deviations = (reliability["observed_yes_rate"] - reliability["mean_predicted"]).abs()
        ece = float((reliability["n"] / expected_count * deviations).sum())
        max_deviation = float(deviations.max())
        assert np.isfinite(ece) and ece >= 0.0, f"T-{horizon} validation ECE is invalid"
        assert np.isfinite(max_deviation) and max_deviation >= 0.0, f"T-{horizon} maximum decile deviation is invalid"
        auc_row = auc_results.loc[auc_results["horizon_minutes"].eq(horizon)].iloc[0]
        records.append({
            "horizon_minutes": horizon,
            "model_error_term": ece,
            "ece_validation_row_count": expected_count,
            "max_abs_decile_deviation": max_deviation,
            "calibrated_auc": float(auc_row["calibrated_auc"]),
            "stored_calibrated_auc": float(auc_row["stored_calibrated_auc"]),
            "auc_regression_abs_diff": float(auc_row["auc_regression_abs_diff"]),
        })
    return pd.DataFrame(records)


def load_validation_basis_rows():
    columns = ROW_KEY[:1] + ["split", "horizon_minutes", "basis_bps", "probability_version", "max_abs_shift"]
    rows = pd.read_parquet(
        BASIS_SENSITIVITY_PATH, columns=columns,
        filters=[
            ("split", "==", VALIDATION_SPLIT),
            ("probability_version", "==", PROBABILITY_VERSION),
            ("basis_bps", "in", list(BASIS_SETTINGS)),
        ],
    )
    assert rows["split"].eq(VALIDATION_SPLIT).all() and not rows["split"].eq("test").any(), "Test rows entered basis threshold construction"
    assert rows["probability_version"].eq(PROBABILITY_VERSION).all(), "A non-Platt basis sensitivity row was loaded"
    assert set(rows["basis_bps"]) == set(BASIS_SETTINGS), "Basis settings changed"
    assert set(rows["horizon_minutes"]) == set(HORIZONS), "Basis horizons changed"
    assert not rows.duplicated(ROW_KEY + ["basis_bps"]).any(), "Basis rows are not unique at row-key/basis grain"
    assert rows["max_abs_shift"].notna().all() and np.isfinite(rows["max_abs_shift"]).all(), "Basis shifts are non-finite"
    assert rows["max_abs_shift"].ge(0.0).all(), "Basis shifts must be nonnegative"
    expected_rows = sum(EXPECTED_COMMON_ROWS[(VALIDATION_SPLIT, horizon)] for horizon in HORIZONS) * len(BASIS_SETTINGS)
    assert len(rows) == expected_rows, f"Expected {expected_rows:,} validation Platt basis rows, found {len(rows):,}"
    return rows


def join_validation_quote_mid(basis_rows):
    start_date, end_date = SPLIT_RANGES[VALIDATION_SPLIT]
    feature_columns = ROW_KEY + [SPLIT_KEY, "quote_mid"]
    features = pd.read_parquet(
        FEATURES_PATH, columns=feature_columns,
        filters=[(SPLIT_KEY, ">=", start_date), (SPLIT_KEY, "<=", end_date)],
    )
    assert len(features) == EXPECTED_SPLIT_ROWS[VALIDATION_SPLIT], "Validation feature count changed"
    assert features[SPLIT_KEY].between(start_date, end_date, inclusive="both").all(), "A non-validation feature row was loaded"
    assert not features.duplicated(ROW_KEY).any(), f"Validation features are not unique on {ROW_KEY}"
    assert features["quote_mid"].notna().all() and np.isfinite(features["quote_mid"]).all(), "Validation quote_mid is invalid"

    joined = basis_rows.merge(features, on=ROW_KEY, how="left", validate="many_to_one", indicator=True)
    assert joined["_merge"].eq("both").all(), "A basis-sensitivity row did not match exactly one validation feature row"
    assert len(joined) == len(basis_rows), "The quote_mid join changed the basis row count"
    return joined.drop(columns="_merge")


def check_and_assign_price_buckets(rows):
    rows = assign_price_buckets(rows)
    for basis_bps in BASIS_SETTINGS:
        for horizon in HORIZONS:
            subset = rows.loc[rows["basis_bps"].eq(basis_bps) & rows["horizon_minutes"].eq(horizon)]
            counts = subset["price_bucket"].value_counts(sort=False).reindex(PRICE_BUCKET_LABELS, fill_value=0)
            expected_counts = EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS[horizon]
            assert counts.tolist() == expected_counts, f"Validation T-{horizon} bucket counts changed at {basis_bps:g} bps: {counts.tolist()}"
            assert int(counts.min()) >= SPARSE_BUCKET_MINIMUM, f"Validation T-{horizon} has a sparse bucket: {counts.to_dict()}"
    return rows, False


def calculate_basis_terms(rows):
    terms = rows.groupby(
        ["horizon_minutes", "basis_bps", "price_bucket"], observed=False, sort=False,
    )["max_abs_shift"].agg(basis_term="median", basis_validation_row_count="size").reset_index()
    assert len(terms) == len(HORIZONS) * len(BASIS_SETTINGS) * len(PRICE_BUCKET_LABELS), "Basis term shape changed"
    assert terms["basis_validation_row_count"].ge(SPARSE_BUCKET_MINIMUM).all(), "A sparse bucket reached basis aggregation"
    return terms


def build_threshold_table(basis_terms, model_error_terms, maximum_platt_difference, bucket_merge_fired):
    threshold = basis_terms.merge(model_error_terms, on="horizon_minutes", how="left", validate="many_to_one")
    threshold["required_net_edge"] = threshold["basis_term"] + threshold["model_error_term"]
    threshold["threshold_role"] = np.where(
        threshold["basis_bps"].eq(PRIMARY_BASIS_BPS), "primary", "conservative_sensitivity",
    )
    threshold["basis_statistic"] = "median_max_abs_shift"
    threshold["model_error_statistic"] = "validation_10_decile_ece"
    threshold["probability_version"] = PROBABILITY_VERSION
    threshold["fee_model_status"] = FEE_MODEL_STATUS
    threshold["order_size"] = 1
    threshold["sigma_candidate"] = SELECTED_CANDIDATE
    threshold["bucket_merge_fired"] = bucket_merge_fired
    threshold["platt_reproduction_max_abs_diff"] = maximum_platt_difference
    threshold["rule_reference"] = "Day 10 Edge Threshold Rule"
    bucket_bounds = {
        label: (PRICE_BUCKET_EDGES[index], PRICE_BUCKET_EDGES[index + 1])
        for index, label in enumerate(PRICE_BUCKET_LABELS)
    }
    threshold["bucket_low"] = threshold["price_bucket"].map(lambda value: bucket_bounds[value][0]).astype(float)
    threshold["bucket_high"] = threshold["price_bucket"].map(lambda value: bucket_bounds[value][1]).astype(float)

    horizon_order = {horizon: index for index, horizon in enumerate(HORIZONS)}
    bucket_order = {bucket: index for index, bucket in enumerate(PRICE_BUCKET_LABELS)}
    threshold["_horizon_order"] = threshold["horizon_minutes"].map(horizon_order)
    threshold["_bucket_order"] = threshold["price_bucket"].map(bucket_order)
    threshold = threshold.sort_values(["_horizon_order", "basis_bps", "_bucket_order"]).drop(columns=["_horizon_order", "_bucket_order"])
    threshold["price_bucket"] = threshold["price_bucket"].astype("string")

    column_order = [
        "horizon_minutes", "basis_bps", "threshold_role", "price_bucket", "bucket_low", "bucket_high",
        "basis_term", "model_error_term", "required_net_edge", "basis_validation_row_count",
        "ece_validation_row_count", "basis_statistic", "model_error_statistic", "probability_version",
        "fee_model_status", "order_size", "sigma_candidate", "bucket_merge_fired",
        "max_abs_decile_deviation", "platt_reproduction_max_abs_diff", "calibrated_auc",
        "stored_calibrated_auc", "auc_regression_abs_diff", "rule_reference",
    ]
    return threshold[column_order].reset_index(drop=True)


def validate_threshold_table(threshold):
    key = ["horizon_minutes", "basis_bps", "price_bucket"]
    assert len(threshold) == 28, f"Expected 28 threshold rows, found {len(threshold)}"
    assert not threshold.duplicated(key).any(), f"Threshold key {key} is not unique"
    assert set(threshold["horizon_minutes"]) == set(HORIZONS), "Threshold horizons changed"
    assert set(threshold["basis_bps"]) == set(BASIS_SETTINGS), "Threshold basis settings changed"
    assert set(threshold["price_bucket"]) == set(PRICE_BUCKET_LABELS), "Threshold price buckets changed"
    group_sizes = threshold.groupby(["horizon_minutes", "basis_bps"]).size()
    assert group_sizes.eq(7).all() and len(group_sizes) == 4, "Each horizon/basis setting must have seven buckets"
    assert threshold["threshold_role"].eq("primary").sum() == 14, "Expected 14 primary rows"
    assert threshold["threshold_role"].eq("conservative_sensitivity").sum() == 14, "Expected 14 sensitivity rows"
    assert threshold.loc[threshold["basis_bps"].eq(PRIMARY_BASIS_BPS), "threshold_role"].eq("primary").all(), "Primary rows are mislabeled"
    sensitivity_roles = threshold.loc[threshold["basis_bps"].eq(CONSERVATIVE_BASIS_BPS), "threshold_role"]
    assert sensitivity_roles.eq("conservative_sensitivity").all(), "5 bps rows are mislabeled as primary"

    value_columns = ["basis_term", "model_error_term", "required_net_edge"]
    assert np.isfinite(threshold[value_columns].to_numpy()).all(), "Threshold terms contain non-finite values"
    assert threshold[value_columns].ge(0.0).all().all(), "Threshold terms contain negative values"
    expected_sum = threshold["basis_term"] + threshold["model_error_term"]
    assert np.allclose(threshold["required_net_edge"], expected_sum, rtol=0.0, atol=ADDITION_TOLERANCE), "Threshold terms were not added exactly"
    assert threshold.groupby("horizon_minutes")["model_error_term"].nunique().eq(1).all(), "ECE varies within a horizon"
    assert threshold.groupby("horizon_minutes")["ece_validation_row_count"].nunique().eq(1).all(), "ECE population varies within a horizon"
    assert threshold["basis_statistic"].eq("median_max_abs_shift").all(), "Basis statistic label changed"
    assert threshold["model_error_statistic"].eq("validation_10_decile_ece").all(), "Model-error statistic label changed"
    assert threshold["probability_version"].eq(PROBABILITY_VERSION).all(), "Probability version changed"
    assert threshold["fee_model_status"].eq("verified").all(), "Fee model is not verified"
    assert threshold["order_size"].eq(1).all(), "Order size changed"
    assert not threshold["bucket_merge_fired"].any(), "A price-bucket merge unexpectedly fired"


def print_summary(threshold, maximum_platt_difference, bucket_merge_fired):
    horizon_diagnostics = threshold.groupby("horizon_minutes", sort=False).first().reset_index()
    print("Day 10 Section 3.2 — frozen Stage 0 net-edge threshold")
    print("\nValidation ECE and probability regression diagnostics (probability units)")
    columns = [
        "horizon_minutes", "model_error_term", "max_abs_decile_deviation", "ece_validation_row_count",
        "calibrated_auc", "stored_calibrated_auc", "auc_regression_abs_diff",
    ]
    print(horizon_diagnostics[columns].to_string(index=False, formatters={
        "model_error_term": "{:.9f}".format,
        "max_abs_decile_deviation": "{:.9f}".format,
        "calibrated_auc": "{:.12f}".format,
        "stored_calibrated_auc": "{:.12f}".format,
        "auc_regression_abs_diff": "{:.3e}".format,
    }))

    ranges = threshold.groupby(["horizon_minutes", "basis_bps", "threshold_role"], sort=False).agg(
        basis_term_min=("basis_term", "min"),
        basis_term_max=("basis_term", "max"),
        required_net_edge_min=("required_net_edge", "min"),
        required_net_edge_max=("required_net_edge", "max"),
    ).reset_index()
    print("\nBasis-term and required-net-edge ranges (probability units)")
    print(ranges.to_string(index=False, formatters={
        "basis_term_min": "{:.9f}".format,
        "basis_term_max": "{:.9f}".format,
        "required_net_edge_min": "{:.9f}".format,
        "required_net_edge_max": "{:.9f}".format,
    }))
    print(f"\nBucket merge fired: {'YES' if bucket_merge_fired else 'NO'}")
    print(f"Platt reproduction maximum absolute difference: {maximum_platt_difference:.17g}")
    print("AUC regression: PASS at both horizons (tolerance 1e-12)")
    print(f"Output rows: {len(threshold)}")


def build_edge_threshold():
    common = load_validation_common_predictions()
    parameters = load_legitimate_platt_parameters()
    common = reconstruct_calibrated_probability(common, parameters)
    checked, maximum_platt_difference, auc_results = regression_check_probabilities(common)
    model_error_terms = calculate_model_error_terms(checked, auc_results)
    basis_rows = load_validation_basis_rows()
    basis_rows = join_validation_quote_mid(basis_rows)
    basis_rows, bucket_merge_fired = check_and_assign_price_buckets(basis_rows)
    basis_terms = calculate_basis_terms(basis_rows)
    threshold = build_threshold_table(basis_terms, model_error_terms, maximum_platt_difference, bucket_merge_fired)
    validate_threshold_table(threshold)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    threshold.to_parquet(OUTPUT_PATH, index=False)
    assert OUTPUT_PATH.exists(), "Threshold artifact was not saved"
    print_summary(threshold, maximum_platt_difference, bucket_merge_fired)
    print(f"\nSaved threshold table to {OUTPUT_PATH}")
    print("All Section 3.2 assertions passed; no test rows, clearance counts, edges, or P&L were computed.")
    return threshold


if __name__ == "__main__":
    build_edge_threshold()
