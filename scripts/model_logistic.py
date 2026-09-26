"""Fit the amended Day 12 M1/M2 logistic models and freeze B1 predictions."""

import hashlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.analyze_calibration import CANDIDATE, PREDICTION_COLUMN, fit_platt, platt_feature, stable_sigmoid
from scripts.build_derived_features import OUTPUT_PATH as DERIVED_PATH
from scripts.evaluation_split import SPLIT_KEY, SPLIT_RANGES
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, EXPECTED_SCORING_ROWS, HORIZONS, PREDICTIONS_PATH, ROW_KEY
from scripts.walk_forward import FROZEN_FOLDS, PLATT_PARAMETER_TOLERANCE, PLATT_PROBABILITY_TOLERANCE, run_walk_forward, single_fold_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAP_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
PREDICTION_OUTPUT_PATH = PROJECT_ROOT / "data/models/logistic_oof_predictions.parquet"
COEFFICIENT_OUTPUT_PATH = PROJECT_ROOT / "data/models/logistic_coefficients.parquet"
SELECTION_OUTPUT_PATH = PROJECT_ROOT / "data/models/logistic_fold_selection.parquet"

CONFIG_M1 = "m1_walk_forward"
CONFIG_M2 = "m2_train_only"
CONFIG_B1 = "b1_walk_forward_stage0"
CONFIGURATIONS = (CONFIG_M1, CONFIG_M2, CONFIG_B1)
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
MAX_ITER = 1_000
TOL = 1e-6
B1_C = 1e12
FEATURE_NAMES = [
    "stage0_logit", "log_sigma", "log_ratio_5m_1h", "log_ratio_15m_4h",
    "log_ratio_1h_24h", "sin_hour", "cos_hour",
    "stage0_logit_x_log_ratio_5m_1h", "stage0_logit_x_log_sigma",
]
BASE_FEATURE_NAMES = FEATURE_NAMES[:7]
FEATURE_COLUMNS = ROW_KEY + ["split", SPLIT_KEY, "is_common"] + BASE_FEATURE_NAMES
PREDICTION_COLUMNS = ["configuration"] + ROW_KEY + ["split", SPLIT_KEY, "fold", "probability", "stage0_platt_probability", "y"]
COEFFICIENT_COLUMNS = [
    "configuration", "fold", "horizon_minutes", "feature", "coefficient_standardized", "coefficient_original",
    "fit_mean", "fit_sd", "intercept_standardized", "intercept_original", "selected_C", "n_fit_rows", "n_iter",
]
SELECTION_COLUMNS = [
    "configuration", "fold", "horizon_minutes", "C", "inner_train_rows", "inner_holdout_rows",
    "inner_holdout_start", "inner_holdout_end", "inner_brier", "selected",
]
EXPECTED_PREDICTION_ROWS = {CONFIG_M1: 7_012, CONFIG_M2: 3_334, CONFIG_B1: 7_012}
EXPECTED_HORIZON_PREDICTIONS = {
    CONFIG_M1: {10: 3_531, 5: 3_481},
    CONFIG_M2: {10: 1_685, 5: 1_649},
    CONFIG_B1: {10: 3_531, 5: 3_481},
}


def load_inputs():
    """Load only train/validation common features, targets, and frozen baselines."""
    assert len(FEATURE_NAMES) == 9 and FEATURE_NAMES[:7] == BASE_FEATURE_NAMES, "Amended feature list changed"
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)), "Duplicate feature projection column"
    rows = pd.read_parquet(DERIVED_PATH, columns=FEATURE_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(rows.columns) == FEATURE_COLUMNS and len(rows) == EXPECTED_SCORING_ROWS, "Derived feature projection changed"
    assert rows["split"].isin(("train", "validation")).all(), "A test feature row entered"
    rows = rows.loc[rows["is_common"]].drop(columns="is_common").copy()
    assert len(rows) == sum(EXPECTED_COMMON_ROWS.values()), "Common population count changed"
    assert not rows.duplicated(ROW_KEY).any(), "Duplicate common feature key"
    assert rows.groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON_ROWS, "Common split/horizon counts changed"
    assert rows[SPLIT_KEY].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "A test date entered"
    for split in ("train", "validation"):
        assert rows.loc[rows["split"].eq(split), SPLIT_KEY].between(*SPLIT_RANGES[split]).all(), f"{split} date disagrees with split"
    build_design(rows)

    target_columns = ROW_KEY + ["split", "y"]
    targets = pd.read_parquet(PREDICTIONS_PATH, columns=target_columns, filters=[("split", "in", ["train", "validation"])])
    assert list(targets.columns) == target_columns and len(targets) == EXPECTED_SCORING_ROWS, "Target projection changed"
    assert targets["split"].isin(("train", "validation")).all() and not targets.duplicated(ROW_KEY).any(), "Unsafe or duplicate targets"
    matched = rows[ROW_KEY + ["split"]].merge(targets, on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert len(matched) == len(rows) and matched["_merge"].eq("both").all() and matched["y"].isin((0, 1)).all(), "Common feature/target match changed"
    target_lookup = targets.set_index(ROW_KEY)["y"]

    raw_columns = ROW_KEY + ["split", PREDICTION_COLUMN]
    raw = pd.read_parquet(PREDICTIONS_PATH, columns=raw_columns, filters=[("split", "in", ["train", "validation"])])
    assert list(raw.columns) == raw_columns and len(raw) == EXPECTED_SCORING_ROWS, "Stage 0 raw probability projection changed"
    assert raw["split"].isin(("train", "validation")).all() and not raw.duplicated(ROW_KEY).any(), "Unsafe Stage 0 raw probabilities"
    raw_matched = rows[ROW_KEY + ["split", "stage0_logit"]].merge(raw, on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert len(raw_matched) == len(rows) and raw_matched["_merge"].eq("both").all(), "Common Stage 0 raw keys changed"
    assert np.allclose(raw_matched["stage0_logit"], platt_feature(raw_matched[PREDICTION_COLUMN]), atol=1e-12, rtol=0), "Stage 0 logit changed"
    raw_lookup = raw.set_index(ROW_KEY)[PREDICTION_COLUMN]

    gap_columns = ROW_KEY + ["split", "platt_model_probability"]
    gap = pd.read_parquet(GAP_PATH, columns=gap_columns, filters=[("split", "in", ["train", "validation"])])
    assert list(gap.columns) == gap_columns and len(gap) == len(rows), "Frozen Stage 0 baseline projection changed"
    assert gap["split"].isin(("train", "validation")).all() and not gap.duplicated(ROW_KEY).any(), "Unsafe Stage 0 baseline"
    baseline = rows[ROW_KEY + ["split", "stage0_logit"]].merge(gap, on=ROW_KEY + ["split"], how="outer", validate="one_to_one", indicator=True)
    assert len(baseline) == len(rows) and baseline["_merge"].eq("both").all(), "Frozen Stage 0 baseline keys changed"
    values = baseline["platt_model_probability"].to_numpy(dtype=float)
    assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), "Invalid frozen Stage 0 probability"
    parameter_columns = ["candidate", "horizon_minutes", "fit_split", "parameter_role", "a", "b"]
    parameters = pd.read_parquet(PLATT_PARAMETERS_PATH, columns=parameter_columns, filters=[
        ("candidate", "==", CANDIDATE), ("fit_split", "==", "train"), ("parameter_role", "==", "legitimate_train_fit")
    ])
    assert list(parameters.columns) == parameter_columns and len(parameters) == len(HORIZONS), "Legitimate train Platt fit changed"
    assert set(parameters["horizon_minutes"]) == set(HORIZONS) and not parameters.duplicated("horizon_minutes").any(), "Platt horizon changed"
    for horizon in HORIZONS:
        fit = parameters.loc[parameters["horizon_minutes"].eq(horizon)].iloc[0]
        subset = baseline.loc[baseline["horizon_minutes"].eq(horizon)]
        reproduced = stable_sigmoid(float(fit["a"]) + float(fit["b"]) * subset["stage0_logit"].to_numpy(dtype=float))
        assert np.max(np.abs(reproduced - subset["platt_model_probability"].to_numpy(dtype=float))) <= 1e-12, "Carried baseline is not legitimate train Platt"
    return rows, target_lookup, raw_lookup, gap


def build_design(rows):
    """Build the nine raw, unstandardized columns in their frozen order."""
    assert set(BASE_FEATURE_NAMES).issubset(rows.columns), "Missing amended base features"
    raw = rows[BASE_FEATURE_NAMES].to_numpy(dtype=float)
    interactions = np.column_stack((raw[:, 0] * raw[:, 2], raw[:, 0] * raw[:, 1]))
    design = np.column_stack((raw, interactions))
    feature_names = FEATURE_NAMES.copy()
    assert feature_names == [
        "stage0_logit", "log_sigma", "log_ratio_5m_1h", "log_ratio_15m_4h", "log_ratio_1h_24h",
        "sin_hour", "cos_hour", "stage0_logit_x_log_ratio_5m_1h", "stage0_logit_x_log_sigma",
    ], "Frozen feature order changed"
    assert design.shape == (len(rows), 9) and np.isfinite(design).all(), "Raw design lost rows or contains non-finite values"
    return design, feature_names


def fit_scaler(x_fit):
    """Fit population-SD scaling on the supplied fit frame only (ddof=0)."""
    mean = x_fit.mean(axis=0)
    sd = x_fit.std(axis=0, ddof=0)
    assert np.isfinite(mean).all() and np.isfinite(sd).all() and (sd > 0).all(), "Invalid fit-only scaler"
    return mean, sd


def transform_design(x, mean, sd):
    transformed = (x - mean) / sd
    assert transformed.shape == x.shape and np.isfinite(transformed).all(), "Invalid standardized design"
    return transformed


def fit_regularized_model(x, y, c):
    assert c in C_GRID and set(np.unique(y)) == {0, 1}, "Invalid C or target classes"
    model = LogisticRegression(penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=MAX_ITER, tol=TOL, C=c)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning, module="sklearn.linear_model._logistic")
        model.fit(x, y)
    assert int(model.n_iter_[0]) < MAX_ITER, f"Logistic fit reached max_iter at C={c}"
    return model


def select_c(fit_rows, fit_y, configuration, fold, horizon):
    dates = sorted(fit_rows[SPLIT_KEY].unique())
    assert len(dates) > 7, "Outer fit has fewer than eight distinct dates"
    holdout_dates = dates[-7:]
    holdout_mask = fit_rows[SPLIT_KEY].isin(holdout_dates).to_numpy()
    inner_train = fit_rows.loc[~holdout_mask]
    inner_holdout = fit_rows.loc[holdout_mask]
    assert len(inner_train) + len(inner_holdout) == len(fit_rows), "Inner split lost or duplicated rows"
    assert inner_train[SPLIT_KEY].max() < inner_holdout[SPLIT_KEY].min(), "Inner train/holdout temporal order changed"
    assert sorted(inner_holdout[SPLIT_KEY].unique()) == holdout_dates, "Inner holdout is not the last seven dates"
    x_train, names_train = build_design(inner_train)
    x_holdout, names_holdout = build_design(inner_holdout)
    assert names_train == names_holdout == FEATURE_NAMES, "Inner feature order changed"
    mean, sd = fit_scaler(x_train)
    x_train_std = transform_design(x_train, mean, sd)
    x_holdout_std = transform_design(x_holdout, mean, sd)
    y_train, y_holdout = fit_y[~holdout_mask], fit_y[holdout_mask]
    records = []
    for c in C_GRID:
        model = fit_regularized_model(x_train_std, y_train, c)
        probabilities = model.predict_proba(x_holdout_std)[:, 1]
        brier = float(np.mean((probabilities - y_holdout) ** 2))
        assert np.isfinite(brier), "Non-finite inner-holdout Brier"
        records.append({
            "configuration": configuration, "fold": fold.number, "horizon_minutes": horizon, "C": c,
            "inner_train_rows": len(inner_train), "inner_holdout_rows": len(inner_holdout),
            "inner_holdout_start": holdout_dates[0], "inner_holdout_end": holdout_dates[-1],
            "inner_brier": brier, "selected": False,
        })
    # No tolerance was frozen for Brier ties; exact ties choose the smaller C.
    winner = min(records, key=lambda record: (record["inner_brier"], record["C"]))
    winner["selected"] = True
    assert winner["C"] in C_GRID and sum(record["selected"] for record in records) == 1, "C selection failed"
    return float(winner["C"]), records


def coefficient_records(configuration, fold, horizon, names, beta_std, beta_original, mean, sd, intercept_std, intercept_original, selected_c, n_fit, n_iter):
    assert len(names) == len(beta_std) == len(beta_original) == len(mean) == len(sd), "Coefficient representation length changed"
    return [{
        "configuration": configuration, "fold": fold.number, "horizon_minutes": horizon, "feature": name,
        "coefficient_standardized": float(beta_std[index]), "coefficient_original": float(beta_original[index]),
        "fit_mean": float(mean[index]), "fit_sd": float(sd[index]),
        "intercept_standardized": float(intercept_std), "intercept_original": float(intercept_original),
        "selected_C": float(selected_c), "n_fit_rows": int(n_fit), "n_iter": int(n_iter),
    } for index, name in enumerate(names)]


def fit_outer_logistic(fit_rows, score_rows, fit_y, configuration, fold, horizon):
    selected_c, selection = select_c(fit_rows, fit_y, configuration, fold, horizon)
    x_fit, names_fit = build_design(fit_rows)
    x_score, names_score = build_design(score_rows)
    assert names_fit == names_score == FEATURE_NAMES, "Outer feature order changed"
    mean, sd = fit_scaler(x_fit)
    x_fit_std = transform_design(x_fit, mean, sd)
    x_score_std = transform_design(x_score, mean, sd)
    model = fit_regularized_model(x_fit_std, fit_y, selected_c)
    repeated = fit_regularized_model(x_fit_std, fit_y, selected_c)
    probabilities = model.predict_proba(x_score_std)[:, 1]
    assert np.max(np.abs(model.intercept_ - repeated.intercept_)) <= 1e-10, "Final intercept is not deterministic"
    assert np.max(np.abs(model.coef_ - repeated.coef_)) <= 1e-10, "Final coefficients are not deterministic"
    assert np.max(np.abs(probabilities - repeated.predict_proba(x_score_std)[:, 1])) <= 1e-10, "Final predictions are not deterministic"
    beta_std = model.coef_[0]
    intercept_std = float(model.intercept_[0])
    beta_original = beta_std / sd
    intercept_original = intercept_std - float(np.dot(beta_std, mean / sd))
    for x_raw, x_standardized in ((x_fit, x_fit_std), (x_score, x_score_std)):
        standard_scores = model.decision_function(x_standardized)
        original_scores = intercept_original + x_raw @ beta_original
        assert np.max(np.abs(standard_scores - original_scores)) <= 1e-10, "Original-unit decision score disagrees"
        assert np.max(np.abs(stable_sigmoid(standard_scores) - stable_sigmoid(original_scores))) <= 1e-12, "Original-unit probability disagrees"
    coefficients = coefficient_records(configuration, fold, horizon, FEATURE_NAMES, beta_std, beta_original, mean, sd, intercept_std, intercept_original, selected_c, len(fit_rows), model.n_iter_[0])
    return probabilities, {"coefficients": coefficients, "selection": selection}


def fit_b1(fit_rows, score_rows, fit_y, raw_lookup, fold, horizon):
    fit_index = pd.MultiIndex.from_frame(fit_rows[ROW_KEY])
    score_index = pd.MultiIndex.from_frame(score_rows[ROW_KEY])
    raw_fit = raw_lookup.loc[fit_index].to_numpy(dtype=float)
    raw_score = raw_lookup.loc[score_index].to_numpy(dtype=float)
    x_fit = fit_rows["stage0_logit"].to_numpy(dtype=float).reshape(-1, 1)
    x_score = score_rows["stage0_logit"].to_numpy(dtype=float).reshape(-1, 1)
    assert np.allclose(x_fit[:, 0], platt_feature(raw_fit), atol=1e-12, rtol=0), "B1 fit logit changed"
    assert np.allclose(x_score[:, 0], platt_feature(raw_score), atol=1e-12, rtol=0), "B1 score logit changed"
    model = LogisticRegression(fit_intercept=True, penalty="l2", C=B1_C, solver="newton-cholesky", max_iter=10_000, tol=1e-12)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning, module="sklearn.linear_model._logistic")
        model.fit(x_fit, fit_y)
    platt = fit_platt(raw_fit, fit_y)
    probabilities = model.predict_proba(x_score)[:, 1]
    independently_platt = stable_sigmoid(platt["a"] + platt["b"] * platt_feature(raw_score))
    intercept_error = abs(float(model.intercept_[0]) - platt["a"])
    slope_error = abs(float(model.coef_[0, 0]) - platt["b"])
    probability_error = float(np.max(np.abs(probabilities - independently_platt)))
    assert model.coef_[0, 0] > 0 and platt["b"] > 0, "B1 Stage 0 slope is not positive"
    assert intercept_error <= PLATT_PARAMETER_TOLERANCE and slope_error <= PLATT_PARAMETER_TOLERANCE, f"B1 Platt parameters disagree at fold {fold.number} T-{horizon}"
    assert probability_error <= PLATT_PROBABILITY_TOLERANCE, f"B1 Platt probabilities disagree at fold {fold.number} T-{horizon}"
    coefficients = coefficient_records(CONFIG_B1, fold, horizon, ["stage0_logit"], model.coef_[0], model.coef_[0], np.array([0.0]), np.array([1.0]), model.intercept_[0], model.intercept_[0], B1_C, len(fit_rows), model.n_iter_[0])
    return probabilities, {"coefficients": coefficients, "selection": [], "platt_errors": (intercept_error, slope_error, probability_error)}


def run_configuration(name, schedule, rows, target_lookup, raw_lookup, baseline):
    assert name in CONFIGURATIONS, "Unknown configuration"

    def fit_predict(fit_rows, score_rows, *, fold, horizon):
        fit_index = pd.MultiIndex.from_frame(fit_rows[ROW_KEY])
        fit_y = target_lookup.loc[fit_index].to_numpy(dtype=int)
        assert len(fit_y) == len(fit_rows) and set(np.unique(fit_y)) == {0, 1}, "Invalid outer fit targets"
        if name == CONFIG_B1:
            return fit_b1(fit_rows, score_rows, fit_y, raw_lookup, fold, horizon)
        return fit_outer_logistic(fit_rows, score_rows, fit_y, name, fold, horizon)

    oof, fit_records = run_walk_forward(rows, fit_predict, folds=schedule)
    oof.insert(0, "configuration", name)
    meta = rows[ROW_KEY + ["split"]].merge(baseline, on=ROW_KEY + ["split"], how="outer", validate="one_to_one", indicator=True)
    assert len(meta) == len(rows) and meta["_merge"].eq("both").all(), "Baseline metadata keys changed"
    meta = meta.drop(columns="_merge").rename(columns={"platt_model_probability": "stage0_platt_probability"})
    meta = meta.merge(target_lookup.rename("y").reset_index(), on=ROW_KEY, how="left", validate="one_to_one", indicator=True)
    assert meta["_merge"].eq("both").all(), "Prediction metadata lacks target"
    meta = meta.drop(columns="_merge")
    predictions = oof.merge(meta, on=ROW_KEY, how="left", validate="one_to_one", indicator=True)
    assert len(predictions) == len(oof) and predictions["_merge"].eq("both").all(), "OOF metadata merge changed keys"
    predictions = predictions.drop(columns="_merge")[PREDICTION_COLUMNS]
    coefficients = pd.DataFrame([record for fit in fit_records for record in fit["metadata"]["coefficients"]], columns=COEFFICIENT_COLUMNS)
    selection = pd.DataFrame([record for fit in fit_records for record in fit["metadata"]["selection"]], columns=SELECTION_COLUMNS)
    assert len(predictions) == EXPECTED_PREDICTION_ROWS[name], f"{name} prediction count changed"
    assert predictions.groupby("horizon_minutes").size().to_dict() == EXPECTED_HORIZON_PREDICTIONS[name], f"{name} horizon count changed"
    assert not predictions.duplicated(["configuration"] + ROW_KEY).any(), f"{name} duplicate prediction key"
    if name == CONFIG_M2:
        assert predictions["split"].eq("validation").all(), "M2 scored a non-validation row"
        assert {fit["horizon_minutes"]: (fit["fit_rows"], fit["score_rows"]) for fit in fit_records} == {
            horizon: (EXPECTED_COMMON_ROWS[("train", horizon)], EXPECTED_COMMON_ROWS[("validation", horizon)]) for horizon in HORIZONS
        }, "M2 fit/score horizon counts changed"
    return predictions, coefficients, selection, fit_records


def assert_predictions(predictions):
    assert list(predictions.columns) == PREDICTION_COLUMNS and len(predictions) == 17_358, "Prediction schema or count changed"
    assert not predictions.duplicated(["configuration"] + ROW_KEY).any(), "Duplicate prediction key"
    assert set(predictions["configuration"]) == set(CONFIGURATIONS), "Prediction configuration changed"
    assert predictions.groupby("configuration").size().to_dict() == EXPECTED_PREDICTION_ROWS, "Prediction configuration counts changed"
    assert set(predictions["horizon_minutes"]) == set(HORIZONS), "Prediction horizon changed"
    assert predictions["split"].isin(("train", "validation")).all(), "A test prediction row entered"
    assert predictions[SPLIT_KEY].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "A test prediction date entered"
    assert predictions["y"].isin((0, 1)).all(), "Invalid prediction target"
    for column in ("probability", "stage0_platt_probability"):
        values = predictions[column].to_numpy(dtype=float)
        assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), f"Invalid {column}"
    assert pd.api.types.is_integer_dtype(predictions["fold"]) and pd.api.types.is_integer_dtype(predictions["horizon_minutes"]), "Prediction fold/horizon dtype changed"


def assert_coefficients(coefficients):
    assert list(coefficients.columns) == COEFFICIENT_COLUMNS and len(coefficients) == 138, "Coefficient schema or count changed"
    assert not coefficients.duplicated(["configuration", "fold", "horizon_minutes", "feature"]).any(), "Duplicate coefficient key"
    assert set(coefficients["configuration"]) == set(CONFIGURATIONS), "Coefficient configuration changed"
    assert np.isfinite(coefficients[["coefficient_standardized", "coefficient_original", "fit_mean", "fit_sd", "intercept_standardized", "intercept_original", "selected_C"]].to_numpy(dtype=float)).all(), "Non-finite coefficient record"
    assert coefficients["fit_sd"].gt(0).all() and coefficients["n_iter"].lt(10_000).all(), "Invalid coefficient scaler or iteration"
    for (configuration, fold, horizon), group in coefficients.groupby(["configuration", "fold", "horizon_minutes"], sort=False):
        assert group["feature"].tolist() == (["stage0_logit"] if configuration == CONFIG_B1 else FEATURE_NAMES), f"Feature order changed at {configuration}/{fold}/T-{horizon}"
        assert group["selected_C"].nunique() == 1 and group["n_fit_rows"].nunique() == 1, "Fit metadata differs across feature rows"
        if configuration == CONFIG_B1:
            assert group["coefficient_original"].gt(0).all() and group["selected_C"].eq(B1_C).all(), "B1 coefficient or C changed"
        else:
            assert group["selected_C"].isin(C_GRID).all() and group["n_iter"].lt(MAX_ITER).all(), "Regularized fit failed convergence"


def assert_selection(selection):
    assert list(selection.columns) == SELECTION_COLUMNS and len(selection) == 98, "C-selection schema or count changed"
    assert not selection.duplicated(["configuration", "fold", "horizon_minutes", "C"]).any(), "Duplicate C-selection key"
    assert set(selection["configuration"]) == {CONFIG_M1, CONFIG_M2} and selection["C"].isin(C_GRID).all(), "C-selection configuration/grid changed"
    assert selection["selected"].dtype == bool and np.isfinite(selection["inner_brier"].to_numpy(dtype=float)).all(), "Invalid C selection"
    for _, group in selection.groupby(["configuration", "fold", "horizon_minutes"], sort=False):
        assert group["C"].tolist() == list(C_GRID) and int(group["selected"].sum()) == 1, "C grid or winner changed"
        assert group["inner_train_rows"].nunique() == group["inner_holdout_rows"].nunique() == 1, "Inner counts changed across C"
        assert group["inner_holdout_start"].nunique() == group["inner_holdout_end"].nunique() == 1, "Inner date range changed across C"
        winner = min(group.itertuples(index=False), key=lambda row: (row.inner_brier, row.C))
        assert bool(group.loc[group["C"].eq(winner.C), "selected"].iloc[0]), "C winner violates Brier/tie rule"


def prediction_fingerprint(predictions):
    assert not predictions.duplicated(["configuration"] + ROW_KEY).any(), "Cannot fingerprint duplicate predictions"
    ordered = predictions[["configuration"] + ROW_KEY + ["probability"]].sort_values(["configuration"] + ROW_KEY)
    lines = []
    for row in ordered.itertuples(index=False):
        assert all("|" not in text and "\n" not in text for text in (row.configuration, row.ticker)), "Unencodable fingerprint key"
        assert np.isfinite(row.probability), "Cannot fingerprint non-finite probability"
        lines.append(f"{row.configuration}|{row.ticker}|{int(row.horizon_minutes)}|{float(row.probability):.10f}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def write_and_verify(frame, path, checker):
    checker(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    saved = pd.read_parquet(path)
    checker(saved)
    pd.testing.assert_frame_equal(saved, frame, check_exact=True)
    return saved


def main():
    rows, targets, raw, baseline = load_inputs()
    results = [
        run_configuration(CONFIG_M1, FROZEN_FOLDS, rows, targets, raw, baseline),
        run_configuration(CONFIG_M2, single_fold_schedule(SPLIT_RANGES["train"], SPLIT_RANGES["validation"]), rows, targets, raw, baseline),
        run_configuration(CONFIG_B1, FROZEN_FOLDS, rows, targets, raw, baseline),
    ]
    predictions = pd.concat([result[0] for result in results], ignore_index=True)[PREDICTION_COLUMNS]
    coefficients = pd.concat([result[1] for result in results], ignore_index=True)[COEFFICIENT_COLUMNS]
    selection = pd.concat([result[2] for result in results[:2]], ignore_index=True)[SELECTION_COLUMNS]
    for frame, string_columns in (
        (predictions, ["configuration", "ticker", "split", SPLIT_KEY]),
        (coefficients, ["configuration", "feature"]),
        (selection, ["configuration", "inner_holdout_start", "inner_holdout_end"]),
    ):
        frame[string_columns] = frame[string_columns].astype("str")
    assert_predictions(predictions)
    assert_coefficients(coefficients)
    assert_selection(selection)
    fingerprint = prediction_fingerprint(predictions)
    assert prediction_fingerprint(predictions.sample(frac=1, random_state=11)) == fingerprint, "Fingerprint depends on row order"
    saved_predictions = write_and_verify(predictions, PREDICTION_OUTPUT_PATH, assert_predictions)
    write_and_verify(coefficients, COEFFICIENT_OUTPUT_PATH, assert_coefficients)
    write_and_verify(selection, SELECTION_OUTPUT_PATH, assert_selection)
    assert prediction_fingerprint(saved_predictions) == fingerprint, "Prediction fingerprint changed after round trip"
    selected = selection.loc[selection["selected"], ["configuration", "fold", "horizon_minutes", "C"]]
    assert len(selected) == 14, "Expected 14 selected outer fits"
    print(f"Model feature count: {len(FEATURE_NAMES)}")
    print(f"M1 rows: {EXPECTED_PREDICTION_ROWS[CONFIG_M1]:,}; M2 rows: {EXPECTED_PREDICTION_ROWS[CONFIG_M2]:,}; B1 rows: {EXPECTED_PREDICTION_ROWS[CONFIG_B1]:,}; total: {len(predictions):,}")
    print("Selected C (configuration/fold/T-horizon): " + "; ".join(f"{row.configuration}/{row.fold}/T-{row.horizon_minutes}={row.C:g}" for row in selected.itertuples(index=False)))
    print("Convergence: PASS")
    print("Determinism: PASS")
    print("Fit-only standardization: PASS")
    print("B1 Platt equivalence: PASS")
    print(f"Prediction SHA-256 fingerprint: {fingerprint}")
    print("Parquet round trips: PASS (predictions, coefficients, C selection)")
    print(f"Outputs: {PREDICTION_OUTPUT_PATH.relative_to(PROJECT_ROOT)}; {COEFFICIENT_OUTPUT_PATH.relative_to(PROJECT_ROOT)}; {SELECTION_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return fingerprint


if __name__ == "__main__":
    main()
