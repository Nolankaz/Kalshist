"""Build the amended Day 12 outcome-free features on train and validation rows."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from scripts.analyze_calibration import PLATT_CLIP, platt_feature, stable_sigmoid
from scripts.analyze_spreads import PREDICTION_COLUMNS
from scripts.count_threshold_clearance import assert_safe_columns
from scripts.evaluation_split import EXPECTED_SPLIT_ROWS, SPLIT_KEY, SPLIT_RANGES
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, EXPECTED_HORIZON_ROWS, EXPECTED_SCORING_ROWS, ROW_KEY, common_prediction_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
MODEL_MARKET_GAP_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/features/derived_features.parquet"

ALLOWED_SPLITS = ("train", "validation")
HORIZONS = (10, 5)
SELECTED_CANDIDATE = "5min_ewma_vol"
SELECTED_PROBABILITY_COLUMN = f"p_{SELECTED_CANDIDATE}"
EXPECTED_TOTAL_COMMON = 12_883
EXPECTED_SPLIT_COMMON = {"train": 9_549, "validation": 3_334}
PLATT_REPRODUCTION_TOLERANCE = 1e-12

VOLATILITY_COLUMNS = [
    "5min_vol", "5min_ewma_vol", "15min_vol", "15min_ewma_vol", "1hr_vol",
    "1hr_ewma_vol", "4hr_vol", "4hr_ewma_vol", "24hr_vol", "24hr_ewma_vol",
]
LOG_INPUT_COLUMNS = ["5min_ewma_vol", "5min_vol", "15min_vol", "1hr_vol", "4hr_vol", "24hr_vol"]
FEATURE_INPUT_COLUMNS = ROW_KEY + [SPLIT_KEY, "decision_time", "hour_utc"] + VOLATILITY_COLUMNS
PREDICTION_INPUT_COLUMNS = ROW_KEY + ["split", SELECTED_PROBABILITY_COLUMN, "z_5min_ewma_vol"] + [
    column for column in PREDICTION_COLUMNS if column != SELECTED_PROBABILITY_COLUMN
]
PLATT_PARAMETER_COLUMNS = ["candidate", "horizon_minutes", "fit_split", "parameter_role", "a", "b"]
GAP_INPUT_COLUMNS = ROW_KEY + ["split", "platt_model_probability"]
BASE_FEATURE_COLUMNS = [
    "stage0_logit", "log_sigma", "log_ratio_5m_1h", "log_ratio_15m_4h",
    "log_ratio_1h_24h", "sin_hour", "cos_hour",
]
OUTPUT_COLUMNS = ROW_KEY + [
    "split", SPLIT_KEY, "decision_time", "hour_utc", "is_common", "stage0_logit",
    "z_5min_ewma_vol", "log_sigma", "log_ratio_5m_1h", "log_ratio_15m_4h",
    "log_ratio_1h_24h", "sin_hour", "cos_hour",
]
FEATURE_HYPOTHESES = {
    "stage0_logit": "Baseline carrier; higher Stage 0 log odds should strongly raise fitted YES log odds.",
    "log_sigma": "Absolute volatility level may shift calibration slightly, with no predicted sign.",
    "log_ratio_5m_1h": "Short-window volatility expansion or contraction should have near-zero directional main effect.",
    "log_ratio_15m_4h": "Medium-window volatility expansion or contraction should have near-zero directional main effect.",
    "log_ratio_1h_24h": "Session-versus-day volatility expansion or contraction should have near-zero directional main effect.",
    "sin_hour": "UTC intraday seasonality may have a small cyclical effect with no predicted sign.",
    "cos_hour": "Complementary UTC intraday seasonality may have a small cyclical effect with no predicted sign.",
}


def assert_safe_projection(columns, source):
    """Apply the existing outcome guard and exclude quote and execution fields."""
    assert_safe_columns(columns, source)
    forbidden = []
    for column in columns:
        name = column.lower()
        if (name in {"y", "day_of_week"} or name.startswith(("fwd_", "target_"))
                or any(token in name for token in ("settlement", "expiration", "quote_", "outcome"))
                or re.search(r"(?:^|_)(?:bid|ask|mid|midpoint|spread|price|fee|pnl|payoff|capital)(?:_|$)", name)):
            forbidden.append(column)
    assert not forbidden, f"{source} contains forbidden columns: {sorted(forbidden)}"


def assert_population(rows):
    assert len(rows) == EXPECTED_SCORING_ROWS, f"Expected {EXPECTED_SCORING_ROWS:,} rows, found {len(rows):,}"
    assert not rows.duplicated(ROW_KEY).any(), f"Duplicate row key {ROW_KEY}"
    assert rows["split"].notna().all() and set(rows["split"]) == set(ALLOWED_SPLITS), "Unexpected split"
    assert rows[SPLIT_KEY].notna().all(), "Missing close_date"
    assert rows[SPLIT_KEY].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "A test or out-of-range date entered"
    assert set(rows["horizon_minutes"]) == set(HORIZONS), "Unexpected horizon"
    split_counts = rows.groupby("split", observed=True).size().to_dict()
    assert split_counts == {name: EXPECTED_SPLIT_ROWS[name] for name in ALLOWED_SPLITS}, f"Split counts changed: {split_counts}"
    for split in ALLOWED_SPLITS:
        start, end = SPLIT_RANGES[split]
        split_rows = rows.loc[rows["split"].eq(split)]
        assert split_rows[SPLIT_KEY].between(start, end).all(), f"{split} dates changed"
        horizon_counts = split_rows.groupby("horizon_minutes", observed=True).size().to_dict()
        assert horizon_counts == {horizon: EXPECTED_HORIZON_ROWS[split] for horizon in HORIZONS}, f"{split} horizon counts changed"
    assert pd.api.types.is_bool_dtype(rows["is_common"]), "is_common must be boolean"
    assert int(rows["is_common"].sum()) == EXPECTED_TOTAL_COMMON, "Day 9 common count changed"
    common = rows.loc[rows["is_common"]]
    split_common = common.groupby("split", observed=True).size().to_dict()
    horizon_common = common.groupby(["split", "horizon_minutes"], observed=True).size().to_dict()
    assert split_common == EXPECTED_SPLIT_COMMON, f"Split common counts changed: {split_common}"
    assert horizon_common == EXPECTED_COMMON_ROWS, f"Horizon common counts changed: {horizon_common}"
    assert rows.groupby("ticker", observed=True)["horizon_minutes"].nunique().eq(2).all(), "A ticker lacks a horizon"


def load_safe_inputs():
    assert PLATT_CLIP == 1e-6, "Frozen PLATT_CLIP changed"
    assert len(VOLATILITY_COLUMNS) == len(PREDICTION_COLUMNS) == 10, "Stage 0 volatility candidate count changed"
    assert set(PREDICTION_COLUMNS) == {f"p_{column}" for column in VOLATILITY_COLUMNS}, "Stage 0 probability columns changed"
    assert len(PREDICTION_INPUT_COLUMNS) == len(set(PREDICTION_INPUT_COLUMNS)), "Duplicate prediction projection column"
    assert_safe_projection(FEATURE_INPUT_COLUMNS, "market-feature projection")
    assert_safe_projection(PREDICTION_INPUT_COLUMNS, "Stage 0 prediction projection")
    features = pd.read_parquet(
        FEATURES_PATH, columns=FEATURE_INPUT_COLUMNS,
        filters=[(SPLIT_KEY, ">=", SPLIT_RANGES["train"][0]), (SPLIT_KEY, "<=", SPLIT_RANGES["validation"][1])],
    )
    predictions = pd.read_parquet(
        PREDICTIONS_PATH, columns=PREDICTION_INPUT_COLUMNS,
        filters=[("split", "in", list(ALLOWED_SPLITS))],
    )
    assert list(features.columns) == FEATURE_INPUT_COLUMNS, "Market-feature projection changed"
    assert list(predictions.columns) == PREDICTION_INPUT_COLUMNS, "Stage 0 prediction projection changed"
    assert_safe_projection(features.columns, "loaded market features")
    assert_safe_projection(predictions.columns, "loaded Stage 0 predictions")
    assert len(features) == len(predictions) == EXPECTED_SCORING_ROWS, "Input row count changed"
    assert not features.duplicated(ROW_KEY).any() and not predictions.duplicated(ROW_KEY).any(), "Input key is not unique"
    assert isinstance(features["decision_time"].dtype, pd.DatetimeTZDtype), "decision_time lost timezone"
    assert str(features["decision_time"].dt.tz) == "UTC" and features["decision_time"].notna().all(), "decision_time must be present and UTC"
    assert features["hour_utc"].notna().all() and features["hour_utc"].between(0, 23).all(), "Invalid UTC hour"
    assert features["hour_utc"].eq(features["decision_time"].dt.hour).all(), "UTC hour disagrees with decision_time"
    assert predictions["split"].isin(ALLOWED_SPLITS).all(), "A test prediction entered"
    predictions["is_common"] = common_prediction_mask(predictions, PREDICTION_COLUMNS)
    rows = features.merge(predictions, on=ROW_KEY, how="outer", validate="one_to_one", indicator=True)
    assert rows["_merge"].eq("both").all() and len(rows) == EXPECTED_SCORING_ROWS, "Feature/prediction keys differ"
    rows = rows.drop(columns="_merge")
    rows["split"] = rows["split"].astype("string")
    rows[SPLIT_KEY] = rows[SPLIT_KEY].astype("string")
    assert_population(rows)
    return rows


def build_base_features(rows):
    for column in VOLATILITY_COLUMNS:
        observed = rows[column].dropna().to_numpy(dtype=float)
        assert np.isfinite(observed).all() and (observed > 0).all(), f"{column} has non-positive or non-finite values"
    assert rows.loc[rows["is_common"], VOLATILITY_COLUMNS].notna().all().all(), "A common row lacks a Stage 0 volatility"
    assert rows.loc[rows["is_common"], LOG_INPUT_COLUMNS].gt(0).all().all(), "A common log input is non-positive"
    selected = rows[SELECTED_PROBABILITY_COLUMN]
    assert selected.dropna().between(0, 1).all(), "Selected Stage 0 probability is invalid"
    rows["stage0_logit"] = platt_feature(selected)
    rows["log_sigma"] = np.log(rows["5min_ewma_vol"])
    for name, numerator, denominator in (
        ("log_ratio_5m_1h", "5min_vol", "1hr_vol"),
        ("log_ratio_15m_4h", "15min_vol", "4hr_vol"),
        ("log_ratio_1h_24h", "1hr_vol", "24hr_vol"),
    ):
        rows[name] = np.log(rows[numerator] / rows[denominator])
    angle = 2.0 * np.pi * rows["hour_utc"].to_numpy(dtype=float) / 24.0
    rows["sin_hour"] = np.sin(angle)
    rows["cos_hour"] = np.cos(angle)
    assert np.allclose(rows["sin_hour"] ** 2 + rows["cos_hour"] ** 2, 1.0, rtol=0.0, atol=1e-12), "Hour cycle identity failed"
    return rows


def check_stage0_reproduction(rows):
    assert_safe_projection(PLATT_PARAMETER_COLUMNS, "Platt parameter projection")
    assert_safe_projection(GAP_INPUT_COLUMNS, "Stage 0 baseline projection")
    parameters = pd.read_parquet(
        PLATT_PARAMETERS_PATH, columns=PLATT_PARAMETER_COLUMNS,
        filters=[("candidate", "==", SELECTED_CANDIDATE), ("fit_split", "==", "train"), ("parameter_role", "==", "legitimate_train_fit")],
    )
    frozen = pd.read_parquet(
        MODEL_MARKET_GAP_PATH, columns=GAP_INPUT_COLUMNS,
        filters=[("split", "in", list(ALLOWED_SPLITS))],
    )
    assert list(parameters.columns) == PLATT_PARAMETER_COLUMNS and list(frozen.columns) == GAP_INPUT_COLUMNS, "Stage 0 projection changed"
    assert_safe_projection(parameters.columns, "loaded Platt parameters")
    assert_safe_projection(frozen.columns, "loaded Stage 0 baseline")
    assert len(parameters) == len(HORIZONS) and not parameters.duplicated("horizon_minutes").any(), "Legitimate Platt parameters changed"
    assert set(parameters["horizon_minutes"]) == set(HORIZONS), "Platt horizons changed"
    assert parameters["candidate"].eq(SELECTED_CANDIDATE).all(), "A different Platt candidate was loaded"
    assert parameters["fit_split"].eq("train").all() and parameters["parameter_role"].eq("legitimate_train_fit").all(), "A non-legitimate Platt fit was loaded"
    assert np.isfinite(parameters[["a", "b"]].to_numpy(dtype=float)).all(), "Stored Platt parameters are not finite"
    assert len(frozen) == EXPECTED_TOTAL_COMMON and not frozen.duplicated(ROW_KEY).any(), "Frozen Stage 0 common keys changed"
    assert frozen["split"].isin(ALLOWED_SPLITS).all(), "A test baseline row entered"
    common = rows.loc[rows["is_common"], ROW_KEY + ["split", "stage0_logit"]]
    checked = common.merge(frozen, on=ROW_KEY + ["split"], how="outer", validate="one_to_one", indicator=True)
    assert len(checked) == EXPECTED_TOTAL_COMMON and checked["_merge"].eq("both").all(), "Stage 0 and Day 9 common keys differ"
    maximum_error = 0.0
    parameters = parameters.set_index("horizon_minutes")
    for horizon in HORIZONS:
        horizon_rows = checked.loc[checked["horizon_minutes"].eq(horizon)]
        fit = parameters.loc[horizon]
        reconstructed = stable_sigmoid(float(fit["a"]) + float(fit["b"]) * horizon_rows["stage0_logit"].to_numpy(dtype=float))
        expected = horizon_rows["platt_model_probability"].to_numpy(dtype=float)
        assert np.isfinite(reconstructed).all() and np.isfinite(expected).all(), "Invalid Stage 0 baseline comparison"
        error = float(np.max(np.abs(reconstructed - expected)))
        assert error <= PLATT_REPRODUCTION_TOLERANCE, f"T-{horizon} Platt reproduction error: {error:.17g}"
        maximum_error = max(maximum_error, error)
    return maximum_error


def assert_output(rows):
    assert list(rows.columns) == OUTPUT_COLUMNS, "Derived-feature columns changed"
    assert_safe_projection(rows.columns, "derived-feature output")
    assert isinstance(rows["decision_time"].dtype, pd.DatetimeTZDtype) and str(rows["decision_time"].dt.tz) == "UTC", "Output decision_time lost UTC"
    assert pd.api.types.is_integer_dtype(rows["horizon_minutes"]) and pd.api.types.is_integer_dtype(rows["hour_utc"]), "Horizon/hour dtype changed"
    assert pd.api.types.is_string_dtype(rows["split"]) and pd.api.types.is_string_dtype(rows[SPLIT_KEY]), "Split/date dtype changed"
    assert_population(rows)
    common = rows.loc[rows["is_common"]]
    required = BASE_FEATURE_COLUMNS + ["z_5min_ewma_vol"]
    assert common[required].notna().all().all(), "A common row lacks a frozen feature or diagnostic z"
    assert np.isfinite(common[required].to_numpy(dtype=float)).all(), "A common feature or diagnostic z is non-finite"
    for column in required:
        assert np.isfinite(rows[column].dropna().to_numpy(dtype=float)).all(), f"{column} has a non-finite value"
    assert np.allclose(rows["sin_hour"] ** 2 + rows["cos_hour"] ** 2, 1.0, rtol=0.0, atol=1e-12), "Saved hour cycle identity failed"


def build_derived_features():
    rows = build_base_features(load_safe_inputs())
    maximum_platt_error = check_stage0_reproduction(rows)
    output = rows[OUTPUT_COLUMNS].sort_values(ROW_KEY).reset_index(drop=True)
    assert_output(output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    saved = pd.read_parquet(OUTPUT_PATH, columns=OUTPUT_COLUMNS)
    assert_output(saved)
    pd.testing.assert_frame_equal(saved, output, check_exact=True)

    common = saved.loc[saved["is_common"]]
    counts = common.groupby(["split", "horizon_minutes"], observed=True).size().to_dict()
    print(f"Derived rows: {len(saved):,}; common rows: {len(common):,}")
    print(f"Common by split: train={EXPECTED_SPLIT_COMMON['train']:,}; validation={EXPECTED_SPLIT_COMMON['validation']:,}")
    print("Common by split/horizon: " + "; ".join(f"{split} T-{horizon}={counts[(split, horizon)]:,}" for split in ALLOWED_SPLITS for horizon in HORIZONS))
    print(f"Stage 0 Platt reproduction max absolute error: {maximum_platt_error:.17g}")
    print("Sine/cosine identity: PASS (absolute tolerance 1e-12)")
    print("Forbidden-column and outcome-free checks: PASS")
    print("Parquet round trip: PASS")
    print(f"Written: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("Base-feature hypotheses:")
    for name, hypothesis in FEATURE_HYPOTHESES.items():
        print(f"  {name}: {hypothesis}")
    return saved


if __name__ == "__main__":
    build_derived_features()
