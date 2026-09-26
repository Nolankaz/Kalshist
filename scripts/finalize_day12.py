"""Read-only Day 12 verdict, artifact integrity, and deterministic fit check."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts import model_logistic as model
from scripts.backtest_model import book_stats, model_basis_rows, paired_daily
from scripts.build_derived_features import OUTPUT_COLUMNS as DERIVED_COLUMNS, assert_output as assert_derived
from scripts.build_model_trades import MODEL_TRADE_KEY, model_decision_fingerprint
from scripts.build_stage0_trades import TRADE_KEY, decision_fingerprint
from scripts.compare_models import OUTPUT_COLUMNS as COMPARISON_COLUMNS, validate_comparison
from scripts.evaluation_split import SPLIT_RANGES
from scripts.score_stage0 import ROW_KEY
from scripts.walk_forward import FROZEN_FOLDS, SCHEDULE_COLUMNS, single_fold_schedule


PREDICTION_FINGERPRINT = "78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512"
STAGE0_FINGERPRINT = "f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96"
FIXED_DECISION_FINGERPRINT = "72a50a6494e75f2de221756e22aba66053f9e7a597b2f809ec606a0274c828a2"
FINAL_DECISION_FINGERPRINT = "1b8df668ae4d6e9ce77dc08abb11f1ec710361c8f2258bbf2505ba0c7306dc18"
ROOT = model.PROJECT_ROOT
FEATURES = [
    "stage0_logit", "log_sigma", "log_ratio_5m_1h", "log_ratio_15m_4h", "log_ratio_1h_24h",
    "sin_hour", "cos_hour", "stage0_logit_x_log_ratio_5m_1h", "stage0_logit_x_log_sigma",
]


def require_close(actual, expected, label, tolerance=5e-10):
    assert abs(float(actual) - expected) <= tolerance, f"STOP: {label}: {actual} != {expected}"


def require_validation_dates(rows, label):
    assert rows["split"].eq("validation").all(), f"STOP: {label} contains another split"
    assert rows["close_date"].between(*SPLIT_RANGES["validation"]).all(), f"STOP: {label} contains another date"


def check_frozen_artifacts():
    assert model.FEATURE_NAMES == FEATURES and model.C_GRID == (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    assert model.MAX_ITER == 1_000 and model.TOL == 1e-6
    assert len(FROZEN_FOLDS) == 6 and FROZEN_FOLDS[-1].score_end == SPLIT_RANGES["validation"][1]

    derived = pd.read_parquet(ROOT / "data/features/derived_features.parquet")
    assert list(derived.columns) == DERIVED_COLUMNS
    assert_derived(derived)
    schedule = pd.read_parquet(ROOT / "data/models/walk_forward_schedule.parquet")
    assert list(schedule.columns) == SCHEDULE_COLUMNS and len(schedule) == 12
    assert not schedule.duplicated(["fold", "horizon_minutes"]).any()
    assert schedule[["fit_start", "fit_end", "score_start", "score_end"]].stack().le(SPLIT_RANGES["validation"][1]).all()
    for fold in FROZEN_FOLDS:
        part = schedule.loc[schedule["fold"].eq(fold.number)]
        assert len(part) == 2 and set(part["horizon_minutes"]) == {10, 5}
        for field in ("fit_start", "fit_end", "score_start", "score_end"):
            assert part[field].eq(getattr(fold, field)).all()

    predictions = pd.read_parquet(model.PREDICTION_OUTPUT_PATH)
    coefficients = pd.read_parquet(model.COEFFICIENT_OUTPUT_PATH)
    selection = pd.read_parquet(model.SELECTION_OUTPUT_PATH)
    model.assert_predictions(predictions)
    model.assert_coefficients(coefficients)
    model.assert_selection(selection)
    assert model.prediction_fingerprint(predictions) == PREDICTION_FINGERPRINT
    assert selection["inner_holdout_end"].le(SPLIT_RANGES["validation"][1]).all()

    comparison = pd.read_parquet(ROOT / "data/models/model_comparison.parquet")
    assert list(comparison.columns) == COMPARISON_COLUMNS
    validate_comparison(comparison)
    assert set(comparison["configuration"]) == set(model.CONFIGURATIONS)

    threshold = pd.read_parquet(ROOT / "data/models/model_edge_threshold.parquet")
    from scripts.backtest_model import THRESHOLD_COLUMNS
    assert list(threshold.columns) == THRESHOLD_COLUMNS and len(threshold) == 14
    assert not threshold.duplicated(["horizon_minutes", "basis_bps", "threshold_role", "price_bucket"]).any()
    assert set(threshold["horizon_minutes"]) == {10, 5}
    assert threshold["basis_bps"].eq(1.2).all() and threshold["threshold_role"].eq("model_own_bar").all()
    assert threshold["probability_version"].eq(model.CONFIG_M2).all()
    assert np.isfinite(threshold[["basis_term", "model_error_term", "required_net_edge"]].to_numpy(dtype=float)).all()
    assert np.allclose(threshold["required_net_edge"], threshold["basis_term"] + threshold["model_error_term"], atol=1e-12, rtol=0)
    basis, reconstruction_error, independent_shift = model_basis_rows(predictions)
    require_close(reconstruction_error, 1.1102230246251565e-16, "unperturbed M2 reconstruction", 1e-16)
    assert independent_shift == 0.0
    for row in threshold.itertuples(index=False):
        matched = basis.loc[basis["horizon_minutes"].eq(row.horizon_minutes) & basis["price_bucket"].eq(row.price_bucket)]
        assert len(matched) == row.basis_validation_row_count
        require_close(row.basis_term, matched["max_abs_shift"].median(), "model-own basis term", 1e-12)
        ece = comparison.loc[comparison["configuration"].eq(model.CONFIG_M2) & comparison["comparison_window"].eq("validation_primary") & comparison["horizon_minutes"].eq(row.horizon_minutes) & comparison["breakdown"].eq("overall") & comparison["metric"].eq("ece")]
        assert len(ece) == 1
        require_close(row.model_error_term, ece.iloc[0]["model_value"], "model-own ECE term", 1e-12)

    decisions = pd.read_parquet(ROOT / "data/backtest/model_trade_decisions.parquet")
    results = pd.read_parquet(ROOT / "data/backtest/model_results_validation.parquet")
    from scripts.backtest import RESULT_COLUMNS
    from scripts.build_model_trades import MODEL_COLUMNS
    assert list(decisions.columns) == MODEL_COLUMNS and list(results.columns) == RESULT_COLUMNS
    assert len(decisions) == len(results) == 2_246
    for frame, label in ((decisions, "decisions"), (results, "results")):
        assert not frame.duplicated(MODEL_TRADE_KEY).any()
        require_validation_dates(frame, label)
        assert set(frame["threshold_role"]) == {"stage0_fixed_bar", "model_own_bar"}
        assert set(frame["probability_version"]) == {model.CONFIG_M1, model.CONFIG_M2}
        assert frame["basis_bps"].eq(1.2).all()
        assert np.isfinite(frame[["model_probability", "net_edge", "required_net_edge", "fee"]].to_numpy(dtype=float)).all()
    assert decisions.groupby(["probability_version", "threshold_role"]).size().to_dict() == {
        (model.CONFIG_M1, "stage0_fixed_bar"): 756,
        (model.CONFIG_M2, "stage0_fixed_bar"): 728,
        (model.CONFIG_M2, "model_own_bar"): 762,
    }
    pd.testing.assert_frame_equal(results[MODEL_COLUMNS], decisions, check_exact=True)
    assert np.isfinite(results[["gross_pnl", "net_pnl", "surprise"]].to_numpy(dtype=float)).all()
    assert np.allclose(results["net_pnl"], results["gross_pnl"] - results["fee"], atol=1e-12, rtol=0)
    assert model_decision_fingerprint(decisions) == FINAL_DECISION_FINGERPRINT
    assert model_decision_fingerprint(decisions.loc[decisions["threshold_role"].eq("stage0_fixed_bar")]) == FIXED_DECISION_FINGERPRINT

    stage0_decisions = pd.read_parquet(ROOT / "data/backtest/stage0_trade_decisions.parquet")
    assert len(stage0_decisions) == 2_977 and not stage0_decisions.duplicated(TRADE_KEY).any()
    assert stage0_decisions["split"].isin(("train", "validation")).all()
    assert decision_fingerprint(stage0_decisions) == STAGE0_FINGERPRINT
    stage0 = pd.read_parquet(ROOT / "data/backtest/stage0_results_validation.parquet")
    stage0 = stage0.loc[stage0["basis_bps"].eq(1.2)].copy()
    require_validation_dates(stage0, "Stage 0 primary")
    assert len(stage0) == 719 and not stage0.duplicated(TRADE_KEY).any()
    require_close(stage0["net_pnl"].sum(), -12.7339, "Stage 0 primary net P&L", 1e-12)
    require_close(stage0["net_pnl"].mean(), -0.017710570, "Stage 0 primary net/trade")

    for path in (
        ROOT / "data/models/plots/logistic_vs_stage0_reliability.png",
        ROOT / "data/backtest/plots/model_vs_stage0_daily_net_pnl.png",
    ):
        assert path.is_file() and path.stat().st_size > 0, f"Missing Day 12 plot: {path}"
    return predictions, coefficients, selection, comparison, threshold, decisions, results, stage0, reconstruction_error, independent_shift


def check_verdict(comparison, results, stage0):
    brier = comparison.loc[
        comparison["configuration"].eq(model.CONFIG_M2)
        & comparison["comparison_window"].eq("validation_primary")
        & comparison["breakdown"].eq("overall")
        & comparison["metric"].eq("brier")
    ].set_index("horizon_minutes")
    assert set(brier.index) == {0, 10, 5}
    expected = {
        10: (0.189066840, 0.189792806, -0.000725966, 0.000889613),
        5: (0.116485755, 0.116474357, 0.000011398, 0.000554122),
    }
    for horizon, values in expected.items():
        row = brier.loc[horizon]
        for field, value in zip(("model_value", "stage0_value", "difference", "paired_se"), values):
            require_close(row[field], value, f"T-{horizon} {field}")
        assert bool(row["meaningful"]) == (abs(row["difference"]) > 2 * row["paired_se"]) == False
    pooled = brier.loc[0]
    assert not bool(pooled["meaningful"]) and abs(pooled["difference"]) <= 2 * pooled["paired_se"]
    require_close(pooled["difference"], -0.000361265, "pooled Brier difference")

    m2 = results.loc[results["probability_version"].eq(model.CONFIG_M2) & results["threshold_role"].eq("stage0_fixed_bar")]
    m1 = results.loc[results["probability_version"].eq(model.CONFIG_M1)]
    own = results.loc[results["threshold_role"].eq("model_own_bar")]
    assert len(m2) == 728 and len(m1) == 756 and len(own) == 762
    for label, rows, pnl, surprise in (
        ("M2 fixed", m2, -18.226600, -0.137604935),
        ("M1 fixed", m1, -14.634200, -0.130720142),
        ("M2 own", own, -19.556200, -0.136544091),
    ):
        require_close(rows["net_pnl"].sum(), pnl, f"{label} net P&L", 1e-12)
        require_close(rows["surprise"].mean(), surprise, f"{label} surprise")
    require_close(m2["net_pnl"].mean(), -0.025036538, "M2 fixed net/trade")
    require_close(own["net_pnl"].mean(), -0.025664304, "M2 own net/trade")
    _, paired = paired_daily(m2, stage0)
    for field, value in (("mean", -0.261557143), ("se", 0.321544483), ("two_se", 0.643088966),
                         ("lower", -0.904646109), ("upper", 0.381531823)):
        require_close(paired[field], value, f"paired {field}")

    better_probability = ((brier.loc[10, "difference"] < -2 * brier.loc[10, "paired_se"])
                          and (brier.loc[5, "difference"] < -2 * brier.loc[5, "paired_se"])) or (
                              pooled["difference"] < -2 * pooled["paired_se"]
                              and not any(brier.loc[h, "difference"] > 2 * brier.loc[h, "paired_se"] for h in (10, 5)))
    worse_probability = any(brier.loc[h, "difference"] > 2 * brier.loc[h, "paired_se"] for h in (10, 5))
    interval_positive = paired["lower"] > 0
    interval_negative = paired["upper"] < 0
    interval_crosses_zero = paired["lower"] <= 0 <= paired["upper"]
    brier_within_two_se = all(abs(brier.loc[h, "difference"]) <= 2 * brier.loc[h, "paired_se"] for h in (10, 5))
    categories = {
        "A — improves on Stage 0": better_probability and interval_positive,
        "B — better probabilities, no economic improvement": better_probability and not interval_positive,
        "C — no meaningful difference; did not beat Stage 0": brier_within_two_se and interval_crosses_zero,
        "D — worse": worse_probability or interval_negative,
    }
    assert sum(categories.values()) == 1, f"STOP: verdict is not unique: {categories}"
    verdict = next(label for label, applies in categories.items() if applies)
    assert verdict == "C — no meaningful difference; did not beat Stage 0"
    return brier, pooled, paired, categories, verdict, book_stats(m2), book_stats(own)


def check_reproducibility(predictions, coefficients, selection):
    rows, targets, raw, baseline = model.load_inputs()  # Parquet projections filter train/validation before loading.
    rerun = [
        model.run_configuration(model.CONFIG_M1, FROZEN_FOLDS, rows, targets, raw, baseline),
        model.run_configuration(model.CONFIG_M2, single_fold_schedule(SPLIT_RANGES["train"], SPLIT_RANGES["validation"]), rows, targets, raw, baseline),
        model.run_configuration(model.CONFIG_B1, FROZEN_FOLDS, rows, targets, raw, baseline),
    ]
    repeated_predictions = pd.concat([part[0] for part in rerun], ignore_index=True)[model.PREDICTION_COLUMNS]
    repeated_coefficients = pd.concat([part[1] for part in rerun], ignore_index=True)[model.COEFFICIENT_COLUMNS]
    repeated_selection = pd.concat([part[2] for part in rerun[:2]], ignore_index=True)[model.SELECTION_COLUMNS]
    prediction_key = ["configuration"] + ROW_KEY
    coefficient_key = ["configuration", "fold", "horizon_minutes", "feature"]
    selection_key = ["configuration", "fold", "horizon_minutes", "C"]
    rerun_pred = repeated_predictions.sort_values(prediction_key).reset_index(drop=True)
    saved_pred = predictions.sort_values(prediction_key).reset_index(drop=True)
    rerun_coef = repeated_coefficients.sort_values(coefficient_key).reset_index(drop=True)
    saved_coef = coefficients.sort_values(coefficient_key).reset_index(drop=True)
    rerun_selected = repeated_selection.loc[repeated_selection["selected"], selection_key].sort_values(selection_key).reset_index(drop=True)
    saved_selected = selection.loc[selection["selected"], selection_key].sort_values(selection_key).reset_index(drop=True)
    assert len(rerun_selected) == len(saved_selected) == 14
    assert np.array_equal(rerun_selected.to_numpy(dtype=object), saved_selected.to_numpy(dtype=object)), "STOP: selected C changed"
    errors = {}
    for column in ("coefficient_standardized", "coefficient_original", "fit_mean", "fit_sd", "intercept_standardized", "intercept_original"):
        errors[column] = float(np.max(np.abs(rerun_coef[column].to_numpy(dtype=float) - saved_coef[column].to_numpy(dtype=float))))
        assert errors[column] <= 1e-10, f"STOP: {column} reproducibility failed: {errors[column]}"
    prediction_error = float(np.max(np.abs(rerun_pred["probability"].to_numpy(dtype=float) - saved_pred["probability"].to_numpy(dtype=float))))
    assert prediction_error <= 1e-12, f"STOP: prediction reproducibility failed: {prediction_error}"
    assert model.prediction_fingerprint(repeated_predictions) == model.prediction_fingerprint(predictions) == PREDICTION_FINGERPRINT
    return errors, prediction_error


def main():
    predictions, coefficients, selection, comparison, threshold, decisions, results, stage0, reconstruction_error, independent_shift = check_frozen_artifacts()
    brier, pooled, paired, categories, verdict, m2_stats, own_stats = check_verdict(comparison, results, stage0)
    errors, prediction_error = check_reproducibility(predictions, coefficients, selection)
    print(f"Official verdict: {verdict}")
    print(f"Category predicates: {categories}")
    for horizon in (10, 5):
        row = brier.loc[horizon]
        print(f"T-{horizon}: model Brier={row.model_value:.12f}; Stage 0 Brier={row.stage0_value:.12f}; difference={row.difference:+.12f}; paired SE={row.paired_se:.12f}; 2SE={2*row.paired_se:.12f}; meaningful={bool(row.meaningful)}")
    print(f"Pooled: model Brier={pooled.model_value:.12f}; Stage 0 Brier={pooled.stage0_value:.12f}; difference={pooled.difference:+.12f}; paired SE={pooled.paired_se:.12f}; meaningful={bool(pooled.meaningful)}")
    print("Paired 21-day M2 minus Stage 0: " + "; ".join(f"{field}={paired[field]:.12f}" for field in ("mean", "sample_sd", "se", "two_se", "lower", "upper")))
    print("M2 fixed-bar: " + "; ".join(f"{field}={m2_stats[field]}" for field in ("n_trades", "gross_pnl", "fees", "net_pnl", "net_pnl_per_trade", "mean_net_edge", "market_fair_expected_total_pnl", "mean_surprise", "cluster_lower_2se", "cluster_upper_2se")))
    print("M2 model-own: " + "; ".join(f"{field}={own_stats[field]}" for field in ("n_trades", "net_pnl", "net_pnl_per_trade", "mean_surprise", "cluster_lower_2se", "cluster_upper_2se")))
    print(f"Model-own basis reconstruction max error: {reconstruction_error:.17g}; spot-independent shift: {independent_shift:.17g}")
    print(f"Selected C reproduction: PASS ({len(selection.loc[selection['selected']])} of 14 exact)")
    print(f"Coefficient max absolute difference: {errors['coefficient_standardized']:.17g}; all coefficient/scaler/intercept differences: {errors}")
    print(f"Prediction max absolute difference: {prediction_error:.17g}; fingerprint: {PREDICTION_FINGERPRINT} PASS")
    print(f"Stage 0 selector fingerprint: {STAGE0_FINGERPRINT} PASS")
    print(f"Fixed-bar model decision fingerprint: {FIXED_DECISION_FINGERPRINT} PASS")
    print(f"Final model decision fingerprint: {FINAL_DECISION_FINGERPRINT} PASS")
    print("PASS: Day 12 test split remains untouched")


if __name__ == "__main__":
    main()
