"""Fit Day 13 HAR/N1 and freeze unscored forecasts and Stage 0 probabilities."""

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.analyze_calibration import PLATT_CLIP, fit_platt, platt_feature, stable_sigmoid
from scripts.count_threshold_clearance import assert_safe_columns
from scripts.evaluation_split import SPLIT_RANGES
from scripts.model_logistic import prediction_fingerprint
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, EXPECTED_SCORING_ROWS, HORIZONS, ROW_KEY
from scripts.stage0_baseline import stage0_probability
from scripts.walk_forward import FROZEN_FOLDS, common_fold_counts, run_walk_forward, single_fold_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED_PATH = PROJECT_ROOT / "data/features/derived_features.parquet"
MARKET_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
TARGET_PATH = PROJECT_ROOT / "data/targets/forward_vol_target.parquet"
STAGE0_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
GAP_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
B1_PATH = PROJECT_ROOT / "data/models/logistic_oof_predictions.parquet"
FORECAST_PATH = PROJECT_ROOT / "data/models/har_forecasts.parquet"
PROBABILITY_PATH = PROJECT_ROOT / "data/models/har_probabilities.parquet"
COEFFICIENT_PATH = PROJECT_ROOT / "data/models/har_coefficients.parquet"

CONFIG_H1 = "h1_walk_forward"
CONFIG_H2 = "h2_train_only"
CONFIGURATIONS = (CONFIG_H1, CONFIG_H2)
DERIVED_COLUMNS = ROW_KEY + ["split", "close_date", "is_common"]
MARKET_COLUMNS = ROW_KEY + ["close_date", "log_moneyness", "T_years", "5min_vol", "1hr_vol", "24hr_vol", "15min_vol", "5min_ewma_vol"]
FEATURE_COLUMNS = DERIVED_COLUMNS[:-1] + MARKET_COLUMNS[3:]
TARGET_COLUMNS = ROW_KEY + ["split", "fwd_log_rv"]
STAGE0_COLUMNS = ROW_KEY + ["split", "y", "p_5min_ewma_vol"]
GAP_COLUMNS = ROW_KEY + ["split", "platt_model_probability"]
B1_COLUMNS = ["configuration"] + ROW_KEY + ["split", "close_date", "fold", "probability", "stage0_platt_probability", "y"]
SCORE_KEY = ["configuration"] + ROW_KEY
FORECAST_COLUMNS = SCORE_KEY + ["split", "close_date", "fold", "log_rv_har", "log_rv_n1", "log_rv_naive_s0", "log_rv_naive_15"]
PROBABILITY_COLUMNS = SCORE_KEY + ["split", "close_date", "fold", "probability", "probability_n1_control", "stage0_platt_probability", "b1_probability", "y"]
COEFFICIENT_COLUMNS = ["configuration", "fold", "horizon_minutes", "model", "term", "coefficient", "classical_se_understated_autocorrelation", "n_fit_rows", "n_fit_rows_dropped_null_target", "in_sample_r2", "ols_cross_check_error"]
HAR_TERMS = ["intercept", "ln_5min_vol", "ln_1hr_vol", "ln_24hr_vol"]
N1_TERMS = ["intercept", "ln_5min_ewma_vol"]
PLATT_TERMS = ["intercept", "stage0_raw_logit"]
EXPECTED_SCORE_ROWS = {CONFIG_H1: 7_012, CONFIG_H2: 3_334}
EXPECTED_HORIZON_SCORE_ROWS = {CONFIG_H1: {10: 3_531, 5: 3_481}, CONFIG_H2: {10: 1_685, 5: 1_649}}
EXPECTED_H2_FIT_ROWS = {10: 4_765, 5: 4_695}
EXPECTED_TRAIN_NULLS = {10: 27, 5: 62}
EXPECTED_B1_FINGERPRINT = "78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512"
RAW_PROOF_TOLERANCE = 1e-12
H2_PROOF_TOLERANCE = 1e-10
H1_PROOF_TOLERANCE = 1e-8
OLS_PROOF_TOLERANCE = 1e-10
SECOND_RUN_COEFFICIENT_TOLERANCE = 1e-12


def keys(frame):
    return set(map(tuple, frame[ROW_KEY].itertuples(index=False, name=None)))


def assert_same_keys(left, right, label):
    assert len(left) == len(right) and not left.duplicated(ROW_KEY).any() and not right.duplicated(ROW_KEY).any(), f"{label}: duplicate or unequal key counts"
    assert keys(left) == keys(right), f"{label}: row keys differ"


def load_inputs():
    assert PLATT_CLIP == 1e-6, "Frozen Platt clipping changed"
    assert_safe_columns(MARKET_COLUMNS, "Day 13 market feature projection")
    derived = pd.read_parquet(DERIVED_PATH, columns=DERIVED_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    market = pd.read_parquet(MARKET_PATH, columns=MARKET_COLUMNS, filters=[("close_date", "<=", SPLIT_RANGES["validation"][1])])
    assert list(derived.columns) == DERIVED_COLUMNS and list(market.columns) == MARKET_COLUMNS, "Safe feature projection changed"
    assert_safe_columns(market.columns, "Day 13 loaded market features")
    assert len(derived) == len(market) == EXPECTED_SCORING_ROWS, "Full train/validation feature count changed"
    assert not derived.duplicated(ROW_KEY).any() and not market.duplicated(ROW_KEY).any(), "Duplicate feature key"
    joined = derived.merge(market, on=ROW_KEY + ["close_date"], how="outer", validate="one_to_one", indicator=True)
    assert len(joined) == EXPECTED_SCORING_ROWS and joined["_merge"].eq("both").all(), "Derived/market feature keys or close dates differ"
    joined = joined.drop(columns="_merge")
    common = joined.loc[joined["is_common"], FEATURE_COLUMNS].copy().sort_values(ROW_KEY).reset_index(drop=True)
    assert list(common.columns) == FEATURE_COLUMNS and not any(column.startswith("fwd_") for column in common.columns), "A target entered the feature frame"
    assert len(common) == sum(EXPECTED_COMMON_ROWS.values()), "Day 9 common count changed"
    assert common.groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON_ROWS, "Day 9 split/horizon common counts changed"
    assert common["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "A test date entered features"
    assert common["split"].isin(("train", "validation")).all(), "A test split entered features"
    for split in ("train", "validation"):
        assert common.loc[common["split"].eq(split), "close_date"].between(*SPLIT_RANGES[split]).all(), f"{split} label/date disagreement"
    common_fold_counts(common)
    for horizon in HORIZONS:
        har_design(common.loc[common["horizon_minutes"].eq(horizon)])
        n1_design(common.loc[common["horizon_minutes"].eq(horizon)])
    assert np.isfinite(common["T_years"].to_numpy(dtype=float)).all() and common["T_years"].gt(0).all(), "Invalid T_years"
    assert np.isfinite(common["log_moneyness"].to_numpy(dtype=float)).all(), "Invalid log_moneyness"

    targets = pd.read_parquet(TARGET_PATH, columns=TARGET_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(targets.columns) == TARGET_COLUMNS and len(targets) == EXPECTED_SCORING_ROWS, "Target projection or row count changed"
    assert targets["split"].isin(("train", "validation")).all() and not targets.duplicated(ROW_KEY).any(), "Unsafe target lookup"
    assert_same_keys(derived, targets, "Derived/target population")
    target_splits = derived[ROW_KEY + ["split"]].merge(targets[ROW_KEY + ["split"]], on=ROW_KEY + ["split"], how="outer", validate="one_to_one", indicator=True)
    assert target_splits["_merge"].eq("both").all(), "Target split disagrees with derived keys"
    target_lookup = targets.set_index(ROW_KEY)["fwd_log_rv"]
    common_target_keys = pd.MultiIndex.from_frame(common[ROW_KEY])
    common_targets = target_lookup.loc[common_target_keys]
    null_counts = common.assign(target_null=common_targets.isna().to_numpy()).groupby(["split", "horizon_minutes"], observed=True)["target_null"].sum().to_dict()
    assert {h: null_counts[("train", h)] for h in HORIZONS} == EXPECTED_TRAIN_NULLS, f"Builder-authoritative train null counts changed: {null_counts}"
    assert null_counts[("validation", 10)] == 41 and null_counts[("validation", 5)] == 61, "Builder-authoritative validation null counts changed"
    assert np.isfinite(common_targets.dropna().to_numpy(dtype=float)).all(), "Non-finite forward target"

    stage0 = pd.read_parquet(STAGE0_PATH, columns=STAGE0_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(stage0.columns) == STAGE0_COLUMNS and len(stage0) == EXPECTED_SCORING_ROWS, "Stage 0 projection or row count changed"
    assert stage0["split"].isin(("train", "validation")).all(), "A test Stage 0 row entered"
    assert_same_keys(derived, stage0, "Derived/Stage 0 population")
    frozen = common[ROW_KEY + ["split"]].merge(stage0, on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert frozen["_merge"].eq("both").all() and frozen["y"].isin((0, 1)).all(), "Missing Stage 0 outcome for common rows"
    assert frozen["p_5min_ewma_vol"].notna().all(), "Missing frozen raw Stage 0 probability"
    outcome_lookup = frozen.set_index(ROW_KEY)["y"]
    raw_lookup = frozen.set_index(ROW_KEY)["p_5min_ewma_vol"]

    gap = pd.read_parquet(GAP_PATH, columns=GAP_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(gap.columns) == GAP_COLUMNS and len(gap) == len(common), "Frozen Stage 0 gap population changed"
    assert_same_keys(common, gap, "Common/frozen train-Platt Stage 0 population")
    assert gap["split"].isin(("train", "validation")).all(), "A test Stage 0 baseline entered"
    baseline_lookup = gap.set_index(ROW_KEY)["platt_model_probability"]
    assert np.isfinite(baseline_lookup.to_numpy(dtype=float)).all(), "Invalid frozen Stage 0 baseline"

    b1_all = pd.read_parquet(B1_PATH, columns=B1_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(b1_all.columns) == B1_COLUMNS and len(b1_all) == 17_358, "Day 12 prediction artifact changed"
    assert b1_all["split"].isin(("train", "validation")).all(), "A test Day 12 prediction entered"
    assert prediction_fingerprint(b1_all) == EXPECTED_B1_FINGERPRINT, "Frozen Day 12 prediction fingerprint changed"
    b1 = b1_all.loc[b1_all["configuration"].eq("b1_walk_forward_stage0")].copy()
    assert len(b1) == EXPECTED_SCORE_ROWS[CONFIG_H1] and not b1.duplicated(ROW_KEY).any(), "B1 row population changed"
    b1_lookup = b1.set_index(ROW_KEY)["probability"]
    return common, target_lookup, outcome_lookup, raw_lookup, baseline_lookup, b1_lookup


def assert_design_rows(rows):
    assert len(rows) and rows["horizon_minutes"].nunique() == 1, "Design must contain one horizon"
    assert not rows.duplicated(ROW_KEY).any(), "Duplicate design key"
    assert list(rows.columns) == FEATURE_COLUMNS, "Design rows include an unexpected column or cross-horizon join"
    assert not any(column.startswith("fwd_") for column in rows.columns), "A forward target entered design rows"


def log_design(rows, source_columns, names):
    assert_design_rows(rows)
    assert names[0] == "intercept" and len(names) == len(source_columns) + 1, "Design term order changed"
    assert not any(name.startswith("fwd_") for name in names), "A forward target entered a design matrix"
    values = rows[source_columns].to_numpy(dtype=float)
    assert np.isfinite(values).all() and (values > 0).all(), "Volatility inputs must be finite and positive"
    design = np.column_stack((np.ones(len(rows)), np.log(values)))
    assert design.shape == (len(rows), len(names)) and np.isfinite(design).all(), "Invalid log-vol design"
    return design, names.copy()


def har_design(rows):
    return log_design(rows, ["5min_vol", "1hr_vol", "24hr_vol"], HAR_TERMS)


def n1_design(rows):
    return log_design(rows, ["5min_ewma_vol"], N1_TERMS)


def fit_ols(design, response):
    response = np.asarray(response, dtype=float)
    assert design.ndim == 2 and len(response) == len(design) and len(design) > design.shape[1], "Invalid OLS fit shape"
    assert np.isfinite(design).all() and np.isfinite(response).all(), "Non-finite OLS fit input"
    assert np.array_equal(design[:, 0], np.ones(len(design))), "OLS intercept is missing"
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    assert rank == design.shape[1], "OLS design is rank-deficient"
    independent = LinearRegression(fit_intercept=False).fit(design, response)
    error = float(np.max(np.abs(coefficients - independent.coef_)))
    assert error <= OLS_PROOF_TOLERANCE, f"Independent OLS coefficients differ by {error:.17g}"
    residual = response - design @ coefficients
    rss = float(residual @ residual)
    tss = float(np.sum((response - response.mean()) ** 2))
    assert tss > 0, "Cannot compute in-sample R² with constant response"
    r2 = 1.0 - rss / tss
    residual_variance = rss / (len(response) - design.shape[1])
    classical_se = np.sqrt(np.diag(np.linalg.pinv(design.T @ design)) * residual_variance)
    assert np.isfinite(coefficients).all() and np.isfinite(classical_se).all(), "Invalid OLS coefficient/SE"
    return {"coefficients": coefficients, "classical_se": classical_se, "n_fit_rows": len(response), "in_sample_r2": r2, "cross_check_error": error}


def feed_through(rows, sigma):
    assert_design_rows(rows)
    sigma = np.asarray(sigma, dtype=float)
    assert sigma.shape == (len(rows),) and np.isfinite(sigma).all() and (sigma > 0).all(), "Invalid forecast sigma"
    probability, _ = stage0_probability(rows["log_moneyness"].to_numpy(dtype=float), sigma, rows["T_years"].to_numpy(dtype=float))
    assert np.isfinite(probability).all() and ((probability > 0) & (probability < 1)).all(), "Raw Stage 0 probability is outside (0, 1)"
    return probability


def platt_feed_through(fit_rows, score_rows, fit_sigma, score_sigma, outcome_lookup):
    fit_raw = feed_through(fit_rows, fit_sigma)
    score_raw = feed_through(score_rows, score_sigma)
    fit_y = outcome_lookup.loc[pd.MultiIndex.from_frame(fit_rows[ROW_KEY])].to_numpy(dtype=int)
    assert set(np.unique(fit_y)) == {0, 1}, "Platt fit needs both outcome classes"
    platt = fit_platt(fit_raw, fit_y)
    assert platt["b"] > 0, "Non-positive Platt slope"
    calibrated = stable_sigmoid(platt["a"] + platt["b"] * platt_feature(score_raw))
    assert np.isfinite(calibrated).all() and ((calibrated > 0) & (calibrated < 1)).all(), "Invalid calibrated probability"
    return calibrated, platt


def callback_factory(mode, target_lookup, outcome_lookup):
    assert mode in ("plumbing", "har"), "Unknown callback mode"

    def fit_predict(fit_rows, score_rows, *, fold, horizon):
        assert_design_rows(fit_rows)
        assert_design_rows(score_rows)
        assert fit_rows["horizon_minutes"].eq(horizon).all() and score_rows["horizon_minutes"].eq(horizon).all(), "Cross-horizon callback input"
        if mode == "plumbing":
            fit_sigma = fit_rows["5min_ewma_vol"].to_numpy(dtype=float)
            score_sigma = score_rows["5min_ewma_vol"].to_numpy(dtype=float)
            probability, platt = platt_feed_through(fit_rows, score_rows, fit_sigma, score_sigma, outcome_lookup)
            return probability, {"platt_a": platt["a"], "platt_b": platt["b"]}

        fit_keys = pd.MultiIndex.from_frame(fit_rows[ROW_KEY])
        fit_target = target_lookup.loc[fit_keys].to_numpy(dtype=float)
        valid = np.isfinite(fit_target)
        assert np.isnan(fit_target[~valid]).all(), "A non-null fit target is non-finite"
        fit_for_ols = fit_rows.loc[valid].copy()
        response = fit_target[valid]
        dropped = len(fit_rows) - len(fit_for_ols)
        har_fit, har_terms = har_design(fit_for_ols)
        n1_fit, n1_terms = n1_design(fit_for_ols)
        har = fit_ols(har_fit, response)
        assert har["in_sample_r2"] < 0.9, f"HAR leakage canary fired: fold {fold.number}, T-{horizon}, R²={har['in_sample_r2']:.8g}"
        n1 = fit_ols(n1_fit, response)
        har_fit_all, _ = har_design(fit_rows)
        har_score, _ = har_design(score_rows)
        n1_fit_all, _ = n1_design(fit_rows)
        n1_score, _ = n1_design(score_rows)
        log_har_fit = har_fit_all @ har["coefficients"]
        log_har_score = har_score @ har["coefficients"]
        log_n1_fit = n1_fit_all @ n1["coefficients"]
        log_n1_score = n1_score @ n1["coefficients"]
        assert all(np.isfinite(values).all() for values in (log_har_fit, log_har_score, log_n1_fit, log_n1_score)), "Non-finite log-vol forecast"
        probability_har, platt_har = platt_feed_through(fit_rows, score_rows, np.exp(log_har_fit), np.exp(log_har_score), outcome_lookup)
        probability_n1, platt_n1 = platt_feed_through(fit_rows, score_rows, np.exp(log_n1_fit), np.exp(log_n1_score), outcome_lookup)
        naive_s0 = np.log(score_rows["5min_ewma_vol"].to_numpy(dtype=float))
        naive_15 = np.log(score_rows["15min_vol"].to_numpy(dtype=float))
        assert np.isfinite(naive_s0).all() and np.isfinite(naive_15).all(), "Non-finite naive log-vol forecast"
        forecasts = score_rows[ROW_KEY + ["split", "close_date"]].copy()
        forecasts["fold"] = fold.number
        forecasts["log_rv_har"] = log_har_score
        forecasts["log_rv_n1"] = log_n1_score
        forecasts["log_rv_naive_s0"] = naive_s0
        forecasts["log_rv_naive_15"] = naive_15
        coefficients = []
        for model_name, fit, terms in (("har", har, har_terms), ("n1", n1, n1_terms)):
            for index, term in enumerate(terms):
                coefficients.append({"fold": fold.number, "horizon_minutes": horizon, "model": model_name, "term": term,
                                     "coefficient": float(fit["coefficients"][index]), "classical_se_understated_autocorrelation": float(fit["classical_se"][index]),
                                     "n_fit_rows": fit["n_fit_rows"], "n_fit_rows_dropped_null_target": dropped,
                                     "in_sample_r2": fit["in_sample_r2"], "ols_cross_check_error": fit["cross_check_error"]})
        for model_name, platt in (("platt_har", platt_har), ("platt_n1", platt_n1)):
            for term, value in zip(PLATT_TERMS, (platt["a"], platt["b"])):
                coefficients.append({"fold": fold.number, "horizon_minutes": horizon, "model": model_name, "term": term,
                                     "coefficient": value, "classical_se_understated_autocorrelation": np.nan,
                                     "n_fit_rows": len(fit_rows), "n_fit_rows_dropped_null_target": 0,
                                     "in_sample_r2": np.nan, "ols_cross_check_error": np.nan})
        metadata = {"forecasts": forecasts, "probability_n1": probability_n1, "coefficients": coefficients,
                    "har_fit_rows": har["n_fit_rows"], "dropped_null_target": dropped}
        return probability_har, metadata

    return fit_predict


def aligned_max_error(actual, expected, left_name, right_name):
    assert_same_keys(actual, expected, f"{left_name}/{right_name}")
    checked = actual[ROW_KEY + [left_name]].merge(expected[ROW_KEY + [right_name]], on=ROW_KEY, validate="one_to_one")
    assert len(checked) == len(actual), "Proof join lost a row"
    return float(np.max(np.abs(checked[left_name].to_numpy(dtype=float) - checked[right_name].to_numpy(dtype=float))))


def prove_plumbing(common, outcome_lookup, raw_lookup, baseline_lookup, b1_lookup, target_lookup):
    raw_parts = []
    for horizon in HORIZONS:
        subset = common.loc[common["horizon_minutes"].eq(horizon)]
        part = subset[ROW_KEY].copy()
        part["recomputed_raw"] = feed_through(subset, subset["5min_ewma_vol"].to_numpy(dtype=float))
        raw_parts.append(part)
    raw = pd.concat(raw_parts, ignore_index=True)
    frozen_raw = raw_lookup.rename("frozen_raw").reset_index()
    proof1 = aligned_max_error(raw, frozen_raw, "recomputed_raw", "frozen_raw")
    assert proof1 <= RAW_PROOF_TOLERANCE, f"Raw Stage 0 plumbing proof failed: {proof1:.17g}"

    h2, h2_records = run_walk_forward(common, callback_factory("plumbing", target_lookup, outcome_lookup), single_fold_schedule(SPLIT_RANGES["train"], SPLIT_RANGES["validation"]))
    assert len(h2_records) == 2 and len(h2) == EXPECTED_SCORE_ROWS[CONFIG_H2], "H2 plumbing schedule changed"
    frozen_h2 = baseline_lookup.rename("frozen_stage0").reset_index()
    frozen_h2 = frozen_h2.loc[frozen_h2[ROW_KEY].apply(tuple, axis=1).isin(keys(h2))]
    proof2 = aligned_max_error(h2, frozen_h2, "probability", "frozen_stage0")
    assert proof2 <= H2_PROOF_TOLERANCE, f"H2 frozen Stage 0 plumbing proof failed: {proof2:.17g}"

    h1, h1_records = run_walk_forward(common, callback_factory("plumbing", target_lookup, outcome_lookup), FROZEN_FOLDS)
    assert len(h1_records) == 12 and len(h1) == EXPECTED_SCORE_ROWS[CONFIG_H1], "H1 plumbing schedule changed"
    frozen_b1 = b1_lookup.rename("frozen_b1").reset_index()
    proof3 = aligned_max_error(h1, frozen_b1, "probability", "frozen_b1")
    assert proof3 <= H1_PROOF_TOLERANCE, f"H1 B1 plumbing proof failed: {proof3:.17g}"
    return proof1, proof2, proof3


def run_configuration(configuration, folds, common, target_lookup, outcome_lookup):
    oof, fit_records = run_walk_forward(common, callback_factory("har", target_lookup, outcome_lookup), folds)
    assert len(oof) == EXPECTED_SCORE_ROWS[configuration], f"{configuration} score row count changed"
    assert oof.groupby("horizon_minutes").size().to_dict() == EXPECTED_HORIZON_SCORE_ROWS[configuration], f"{configuration} horizon score counts changed"
    assert len(fit_records) == len(folds) * len(HORIZONS), "Fit metadata count changed"
    forecast_parts = []
    coefficient_records = []
    n1_parts = []
    fit_counts = {}
    maximum_r2 = -np.inf
    maximum_cross_check = 0.0
    for record in fit_records:
        metadata = record["metadata"]
        fold, horizon = record["fold"], record["horizon_minutes"]
        part = metadata["forecasts"].copy()
        assert len(part) == record["score_rows"] and part["fold"].eq(fold).all() and part["horizon_minutes"].eq(horizon).all(), "Forecast metadata does not match harness record"
        forecast_parts.append(part)
        n1 = part[ROW_KEY].copy()
        n1["probability_n1_control"] = metadata["probability_n1"]
        n1_parts.append(n1)
        coefficient_records.extend(metadata["coefficients"])
        fit_counts[(fold, horizon)] = metadata["har_fit_rows"]
        assert metadata["har_fit_rows"] + metadata["dropped_null_target"] == record["fit_rows"], "Null-target fit exclusion changed"
        for coefficient in metadata["coefficients"]:
            if coefficient["model"] == "har":
                maximum_r2 = max(maximum_r2, coefficient["in_sample_r2"])
            if coefficient["model"] in ("har", "n1"):
                maximum_cross_check = max(maximum_cross_check, coefficient["ols_cross_check_error"])
            if coefficient["model"].startswith("platt") and coefficient["term"] == "stage0_raw_logit":
                assert coefficient["coefficient"] > 0, "Non-positive fitted Platt slope"
    forecasts = pd.concat(forecast_parts, ignore_index=True)
    n1 = pd.concat(n1_parts, ignore_index=True)
    assert_same_keys(forecasts, oof, f"{configuration} forecast/harness keys")
    assert_same_keys(n1, oof, f"{configuration} N1/harness keys")
    checked = forecasts[ROW_KEY + ["split", "close_date", "fold"]].merge(oof, on=ROW_KEY + ["close_date", "fold"], validate="one_to_one", suffixes=("", "_harness"))
    assert len(checked) == len(oof) and checked["split"].isin(("train", "validation")).all(), "Forecast/harness rows differ"
    assert not forecasts.duplicated(ROW_KEY).any(), "Duplicate forecast key"
    forecasts.insert(0, "configuration", configuration)
    probabilities = oof.merge(n1, on=ROW_KEY, how="left", validate="one_to_one")
    probabilities = probabilities.merge(forecasts[ROW_KEY + ["split"]], on=ROW_KEY, how="left", validate="one_to_one")
    probabilities.insert(0, "configuration", configuration)
    coefficients = pd.DataFrame(coefficient_records)
    coefficients.insert(0, "configuration", configuration)
    return forecasts, probabilities, coefficients, fit_counts, maximum_r2, maximum_cross_check


def forecast_fingerprint(forecasts):
    assert not forecasts.duplicated(SCORE_KEY).any(), "Cannot fingerprint duplicate forecasts"
    ordered = forecasts.sort_values(SCORE_KEY)
    lines = []
    for row in ordered.itertuples(index=False):
        assert "|" not in row.configuration and "|" not in row.ticker and "\n" not in row.ticker, "Unencodable forecast key"
        values = (row.log_rv_har, row.log_rv_n1, row.log_rv_naive_s0, row.log_rv_naive_15)
        assert np.isfinite(values).all(), "Cannot fingerprint non-finite forecast"
        lines.append(f"{row.configuration}|{row.ticker}|{int(row.horizon_minutes)}|" + "|".join(f"{value:.10f}" for value in values))
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def assert_forecasts(frame):
    assert list(frame.columns) == FORECAST_COLUMNS and len(frame) == sum(EXPECTED_SCORE_ROWS.values()), "Forecast schema/count changed"
    assert not frame.duplicated(SCORE_KEY).any() and set(frame["configuration"]) == set(CONFIGURATIONS), "Forecast key/configuration changed"
    assert not any(column.startswith("fwd_") for column in frame.columns), "Forward target leaked into forecast artifact"
    assert frame["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "Test date in forecasts"
    values = frame[["log_rv_har", "log_rv_n1", "log_rv_naive_s0", "log_rv_naive_15"]].to_numpy(dtype=float)
    assert np.isfinite(values).all() and np.isfinite(np.exp(values[:, :2])).all() and (np.exp(values[:, :2]) > 0).all(), "Invalid forecast sigma"
    for config in CONFIGURATIONS:
        rows = frame.loc[frame["configuration"].eq(config)]
        assert len(rows) == EXPECTED_SCORE_ROWS[config] and rows.groupby("horizon_minutes").size().to_dict() == EXPECTED_HORIZON_SCORE_ROWS[config], f"{config} forecast population changed"
        if config == CONFIG_H2:
            assert rows["split"].eq("validation").all() and rows["fold"].eq(1).all(), "H2 score rows changed"
        else:
            assert rows["fold"].between(1, 6).all(), "H1 fold IDs changed"


def assert_probabilities(frame):
    assert list(frame.columns) == PROBABILITY_COLUMNS and len(frame) == sum(EXPECTED_SCORE_ROWS.values()), "Probability schema/count changed"
    assert not frame.duplicated(SCORE_KEY).any() and set(frame["configuration"]) == set(CONFIGURATIONS), "Probability key/configuration changed"
    assert frame["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "Test date in probabilities"
    for column in ("probability", "probability_n1_control", "stage0_platt_probability"):
        values = frame[column].to_numpy(dtype=float)
        assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), f"Invalid {column}"
    assert frame["y"].isin((0, 1)).all(), "Invalid carried outcome"
    h1 = frame.loc[frame["configuration"].eq(CONFIG_H1)]
    h2 = frame.loc[frame["configuration"].eq(CONFIG_H2)]
    assert h1["b1_probability"].notna().all() and h2["b1_probability"].isna().all(), "B1 must exist only on H1 rows"
    assert ((h1["b1_probability"] > 0) & (h1["b1_probability"] < 1)).all(), "Invalid B1 probability"
    for config, rows in ((CONFIG_H1, h1), (CONFIG_H2, h2)):
        assert len(rows) == EXPECTED_SCORE_ROWS[config] and rows.groupby("horizon_minutes").size().to_dict() == EXPECTED_HORIZON_SCORE_ROWS[config], f"{config} probability population changed"


def assert_coefficients(frame):
    assert list(frame.columns) == COEFFICIENT_COLUMNS and len(frame) == 140, "Coefficient schema/count changed"
    assert not frame.duplicated(["configuration", "fold", "horizon_minutes", "model", "term"]).any(), "Duplicate coefficient key"
    assert set(frame["model"]) == {"har", "n1", "platt_har", "platt_n1"}, "Coefficient model names changed"
    for _, group in frame.groupby(["configuration", "fold", "horizon_minutes"], observed=True):
        for model, expected_terms in (("har", HAR_TERMS), ("n1", N1_TERMS), ("platt_har", PLATT_TERMS), ("platt_n1", PLATT_TERMS)):
            rows = group.loc[group["model"].eq(model)]
            assert rows["term"].tolist() == expected_terms, f"{model} coefficient terms/order changed"
            assert rows["n_fit_rows"].nunique() == 1 and rows["n_fit_rows"].iloc[0] > 0, "Inconsistent fit-row counts"
            assert np.isfinite(rows["coefficient"].to_numpy(dtype=float)).all(), "Non-finite coefficient"
            if model in ("har", "n1"):
                assert np.isfinite(rows["classical_se_understated_autocorrelation"].to_numpy(dtype=float)).all(), "Missing classical OLS SE"
                assert rows["ols_cross_check_error"].le(OLS_PROOF_TOLERANCE).all(), "OLS cross-check failed"
                if model == "har":
                    assert rows["in_sample_r2"].lt(0.9).all(), "HAR leakage canary fired"
            else:
                assert rows["classical_se_understated_autocorrelation"].isna().all() and rows["in_sample_r2"].isna().all(), "Platt must not carry classical OLS statistics"
                assert rows.loc[rows["term"].eq("stage0_raw_logit"), "coefficient"].gt(0).all(), "Non-positive Platt slope"


def stage_and_write(frames):
    staged = []
    prior_exists = all(path.exists() for _, path, _ in frames)
    assert all(path.exists() for _, path, _ in frames) or not any(path.exists() for _, path, _ in frames), "Partial prior Day 13 artifact set"
    try:
        for frame, path, checker in frames:
            checker(frame)
            if prior_exists:
                previous = pd.read_parquet(path)
                checker(previous)
                pd.testing.assert_frame_equal(previous, frame, check_exact=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}_", suffix=".parquet", delete=False) as handle:
                temp_path = Path(handle.name)
            staged.append((temp_path, path))
            frame.to_parquet(temp_path, index=False)
            saved = pd.read_parquet(temp_path)
            checker(saved)
            pd.testing.assert_frame_equal(saved, frame, check_exact=True)
        for temp_path, path in staged:
            os.replace(temp_path, path)
    finally:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)
    return prior_exists


def main():
    common, target_lookup, outcome_lookup, raw_lookup, baseline_lookup, b1_lookup = load_inputs()
    proof1, proof2, proof3 = prove_plumbing(common, outcome_lookup, raw_lookup, baseline_lookup, b1_lookup, target_lookup)
    results = [
        run_configuration(CONFIG_H1, FROZEN_FOLDS, common, target_lookup, outcome_lookup),
        run_configuration(CONFIG_H2, single_fold_schedule(SPLIT_RANGES["train"], SPLIT_RANGES["validation"]), common, target_lookup, outcome_lookup),
    ]
    forecasts = pd.concat([result[0] for result in results], ignore_index=True)[FORECAST_COLUMNS].sort_values(SCORE_KEY).reset_index(drop=True)
    probabilities = pd.concat([result[1] for result in results], ignore_index=True)
    probability_keys = probabilities[SCORE_KEY].copy()
    assert len(probability_keys) == len(forecasts) and not probability_keys.duplicated().any(), "Forecast/probability key counts differ"
    assert set(map(tuple, probability_keys.itertuples(index=False, name=None))) == set(map(tuple, forecasts[SCORE_KEY].itertuples(index=False, name=None))), "Forecast/probability keys differ"
    metadata = probabilities[SCORE_KEY].merge(forecasts[SCORE_KEY + ["split", "close_date", "fold"]], on=SCORE_KEY, validate="one_to_one")
    assert len(metadata) == len(probabilities), "Forecast/probability metadata mismatch"
    probabilities["stage0_platt_probability"] = baseline_lookup.loc[pd.MultiIndex.from_frame(probabilities[ROW_KEY])].to_numpy(dtype=float)
    b1_keys = pd.MultiIndex.from_frame(probabilities[ROW_KEY])
    probabilities["b1_probability"] = np.where(probabilities["configuration"].eq(CONFIG_H1), b1_lookup.reindex(b1_keys).to_numpy(dtype=float), np.nan)
    probabilities["y"] = outcome_lookup.loc[b1_keys].to_numpy(dtype=int)
    probabilities = probabilities[PROBABILITY_COLUMNS].sort_values(SCORE_KEY).reset_index(drop=True)
    coefficients = pd.concat([result[2] for result in results], ignore_index=True)[COEFFICIENT_COLUMNS]
    coefficients = coefficients.sort_values(["configuration", "fold", "horizon_minutes"], kind="stable").reset_index(drop=True)
    for frame, columns in ((forecasts, ["configuration", "ticker", "split", "close_date"]),
                           (probabilities, ["configuration", "ticker", "split", "close_date"]),
                           (coefficients, ["configuration", "model", "term"])):
        frame[columns] = frame[columns].astype("str")
    assert_forecasts(forecasts)
    assert_probabilities(probabilities)
    assert_coefficients(coefficients)
    assert results[1][3][(1, 10)] == EXPECTED_H2_FIT_ROWS[10] and results[1][3][(1, 5)] == EXPECTED_H2_FIT_ROWS[5], "H2 builder-authoritative HAR fit counts changed"
    maximum_r2 = max(result[4] for result in results)
    maximum_cross_check = max(result[5] for result in results)
    assert maximum_r2 < 0.9 and maximum_cross_check <= OLS_PROOF_TOLERANCE, "OLS/canary invariant failed"
    forecast_hash = forecast_fingerprint(forecasts)
    probability_hash = prediction_fingerprint(probabilities)
    assert forecast_fingerprint(forecasts.sample(frac=1, random_state=13)) == forecast_hash, "Forecast fingerprint depends on row order"
    assert prediction_fingerprint(probabilities.sample(frac=1, random_state=13)) == probability_hash, "Probability fingerprint depends on row order"
    deterministic_prior = stage_and_write(((forecasts, FORECAST_PATH, assert_forecasts), (probabilities, PROBABILITY_PATH, assert_probabilities), (coefficients, COEFFICIENT_PATH, assert_coefficients)))
    saved_forecasts = pd.read_parquet(FORECAST_PATH)
    saved_probabilities = pd.read_parquet(PROBABILITY_PATH)
    assert forecast_fingerprint(saved_forecasts) == forecast_hash and prediction_fingerprint(saved_probabilities) == probability_hash, "Saved fingerprint changed"
    print(f"Plumbing Proof 1 raw Stage 0 max error: {proof1:.17g} (PASS)")
    print(f"Plumbing Proof 2 H2 Stage 0 max error: {proof2:.17g} (PASS)")
    print(f"Plumbing Proof 3 H1 B1 max error: {proof3:.17g} (PASS)")
    print(f"Maximum independent OLS coefficient error: {maximum_cross_check:.17g} (PASS)")
    print("H2 HAR fit rows: " + "; ".join(f"T-{horizon}={results[1][3][(1, horizon)]:,}" for horizon in HORIZONS))
    print("H1 HAR fit rows: " + "; ".join(f"fold {fold} T-{horizon}={results[0][3][(fold, horizon)]:,}" for fold in range(1, 7) for horizon in HORIZONS))
    print(f"Forecast/probability rows: H1={EXPECTED_SCORE_ROWS[CONFIG_H1]:,}; H2={EXPECTED_SCORE_ROWS[CONFIG_H2]:,}; total={len(forecasts):,}")
    print("Positive Platt slopes: PASS")
    print(f"Maximum HAR in-sample R²: {maximum_r2:.17g} (canary PASS)")
    print(f"Forecast SHA-256 fingerprint: {forecast_hash}")
    print(f"HAR-fed probability SHA-256 fingerprint: {probability_hash}")
    print(f"Deterministic prior comparison: {'PASS (all frames exact)' if deterministic_prior else 'first write'}")
    print("Artifact schemas, safe designs, population, keys, and Parquet round trips: PASS")


if __name__ == "__main__":
    main()
