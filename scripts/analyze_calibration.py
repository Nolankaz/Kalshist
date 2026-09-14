import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kalshist-matplotlib"))
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_market_features import brier_score, rank_auc
from scripts.score_stage0 import binary_log_loss


INPUT_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/stage0_reliability_validation.parquet"
PLOT_PATH = PROJECT_ROOT / "data/models/plots/stage0_5min_ewma_vol_reliability_validation.png"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
PLATT_SCORES_PATH = PROJECT_ROOT / "data/models/stage0_platt_validation_scores.parquet"
PLATT_PLOT_PATH = PROJECT_ROOT / "data/models/plots/stage0_5min_ewma_vol_platt_reliability_validation.png"
CANDIDATE = "5min_ewma_vol"
PREDICTION_COLUMN = f"p_{CANDIDATE}"
HORIZONS = (10, 5)
EXPECTED_COMMON_ROWS = {10: 1_685, 5: 1_649}
EXPECTED_ALL_COMMON_ROWS = {("train", 10): 4_792, ("train", 5): 4_757, ("validation", 10): 1_685, ("validation", 5): 1_649}
WILSON_Z = 1.96
PLATT_CLIP = 1e-6
MAX_NEWTON_ITERATIONS = 10
GRADIENT_TOLERANCE = 1e-8


def wilson_interval(successes, observations):
    p_hat = successes / observations
    denominator = 1.0 + WILSON_Z ** 2 / observations
    center = (p_hat + WILSON_Z ** 2 / (2.0 * observations)) / denominator
    half_width = WILSON_Z / denominator * np.sqrt(p_hat * (1.0 - p_hat) / observations + WILSON_Z ** 2 / (4.0 * observations ** 2))
    return center - half_width, center + half_width


def load_common_validation_rows():
    validation = pd.read_parquet(INPUT_PATH, filters=[("split", "==", "validation")])
    prediction_columns = [column for column in validation.columns if column.startswith("p_")]
    assert len(prediction_columns) == 10, f"Expected 10 p_* columns, found {len(prediction_columns)}: {prediction_columns}"
    assert PREDICTION_COLUMN in prediction_columns, f"Selected probability column is missing: {PREDICTION_COLUMN}"
    assert validation["split"].eq("validation").all(), "Reliability input must be validation-only"
    assert not validation["split"].eq("test").any(), "Test rows entered reliability analysis"

    common_rows = {}
    for horizon in HORIZONS:
        horizon_rows = validation.loc[validation["horizon_minutes"].eq(horizon)]
        rows = horizon_rows.loc[horizon_rows[prediction_columns].notna().all(axis=1)].copy()
        assert len(rows) == EXPECTED_COMMON_ROWS[horizon], (
            f"Expected {EXPECTED_COMMON_ROWS[horizon]:,} validation T-{horizon} common rows, found {len(rows):,}"
        )
        assert rows["split"].eq("validation").all(), f"T-{horizon} reliability rows must be validation-only"
        assert not rows["split"].eq("test").any(), f"Test rows entered T-{horizon} reliability analysis"
        assert rows["y"].notna().all(), f"T-{horizon} common rows contain null y values"
        assert rows["y"].isin([0, 1]).all(), f"T-{horizon} common rows contain invalid y values"
        assert rows[PREDICTION_COLUMN].notna().all(), f"T-{horizon} common rows contain null {PREDICTION_COLUMN} values"
        common_rows[horizon] = rows
    return common_rows


def build_reliability_table(rows, horizon, prediction_column=PREDICTION_COLUMN):
    binned = rows[[prediction_column, "y"]].copy()
    binned["decile"] = pd.qcut(binned[prediction_column], 10, labels=False) + 1
    table = binned.groupby("decile", observed=True).agg(
        n=("y", "size"),
        mean_predicted=(prediction_column, "mean"),
        yes_count=("y", "sum"),
    ).reset_index()
    table["observed_yes_rate"] = table["yes_count"] / table["n"]
    table["wilson_low"], table["wilson_high"] = wilson_interval(table["yes_count"], table["n"])
    table.insert(0, "horizon_minutes", horizon)
    table.insert(0, "split", "validation")
    table.insert(0, "candidate", CANDIDATE)
    table = table.drop(columns="yes_count")

    assert len(table) == 10, f"T-{horizon} must contain exactly 10 populated quantile bins"
    assert table["decile"].tolist() == list(range(1, 11)), f"T-{horizon} deciles must be numbered 1 through 10"
    assert table["n"].gt(0).all(), f"T-{horizon} contains an empty reliability bin"
    assert table["n"].sum() == EXPECTED_COMMON_ROWS[horizon], f"T-{horizon} reliability bins do not cover every common row"
    assert table["mean_predicted"].is_monotonic_increasing, f"T-{horizon} mean predictions are not monotone across quantile bins"
    assert table[["wilson_low", "wilson_high"]].ge(0).all().all(), f"T-{horizon} Wilson interval falls below zero"
    assert table[["wilson_low", "wilson_high"]].le(1).all().all(), f"T-{horizon} Wilson interval exceeds one"
    assert table["observed_yes_rate"].ge(table["wilson_low"]).all(), f"T-{horizon} observed rate falls below its Wilson interval"
    assert table["observed_yes_rate"].le(table["wilson_high"]).all(), f"T-{horizon} observed rate exceeds its Wilson interval"
    return table


def observed_inversions(table):
    changes = table["observed_yes_rate"].diff()
    return [f"D{int(table.loc[index - 1, 'decile'])}->D{int(table.loc[index, 'decile'])}" for index in changes.index[changes.lt(0)]]


def calibration_diagnostic(table):
    gaps = table["observed_yes_rate"] - table["mean_predicted"]
    low_gap = float(gaps.iloc[:3].mean())
    high_gap = float(gaps.iloc[-3:].mean())
    if low_gap < 0 and high_gap > 0:
        direction = "under-confident"
    elif low_gap > 0 and high_gap < 0:
        direction = "over-confident"
    else:
        direction = "mixed"

    inversions = observed_inversions(table)
    broadly_monotone = len(inversions) <= 1
    outside_wilson = (~table["mean_predicted"].between(table["wilson_low"], table["wilson_high"])).sum()
    uncertainty = "generally larger than" if outside_wilson > len(table) / 2 else "not generally larger than"
    return direction, broadly_monotone, inversions, int(outside_wilson), uncertainty


def plot_reliability(tables):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        table = tables[horizon]
        lower_error = table["observed_yes_rate"] - table["wilson_low"]
        upper_error = table["wilson_high"] - table["observed_yes_rate"]
        axis.errorbar(table["mean_predicted"], table["observed_yes_rate"], yerr=np.vstack([lower_error, upper_error]), fmt="o-", capsize=3, label="Validation deciles")
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
        axis.set_title(f"T-{horizon}")
        axis.set_xlabel("Mean predicted probability")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0].set_ylabel("Observed YES frequency")
    fig.suptitle("Stage 0 Reliability — 5min_ewma_vol (Validation Common Rows)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_calibration():
    common_rows = load_common_validation_rows()
    tables = {horizon: build_reliability_table(common_rows[horizon], horizon) for horizon in HORIZONS}
    reliability = pd.concat(tables.values(), ignore_index=True)
    assert len(reliability) == 20, f"Expected 20 reliability rows, found {len(reliability)}"
    assert not reliability.duplicated(["candidate", "split", "horizon_minutes", "decile"]).any(), "Reliability-table key must be unique"
    assert reliability["candidate"].eq(CANDIDATE).all(), "Reliability output contains another sigma candidate"
    assert reliability["split"].eq("validation").all(), "Reliability output contains a non-validation split"

    print(f"Selected Stage 0 candidate: {CANDIDATE}")
    print(f"Validation common rows: T-10 = {len(common_rows[10]):,}; T-5 = {len(common_rows[5]):,}")
    diagnostics = {}
    for horizon in HORIZONS:
        table = tables[horizon]
        print(f"\nValidation common-row reliability: T-{horizon}")
        print(table[["decile", "n", "mean_predicted", "observed_yes_rate", "wilson_low", "wilson_high"]].to_string(index=False, formatters={
            "mean_predicted": "{:.6f}".format,
            "observed_yes_rate": "{:.6f}".format,
            "wilson_low": "{:.6f}".format,
            "wilson_high": "{:.6f}".format,
        }))
        diagnostics[horizon] = calibration_diagnostic(table)
        direction, broadly_monotone, inversions, outside_wilson, uncertainty = diagnostics[horizon]
        print(f"Observed frequencies monotone: {table['observed_yes_rate'].is_monotonic_increasing}; inversions: {inversions or 'none'}")
        print(f"Diagnostic: {'broadly monotone' if broadly_monotone else 'not broadly monotone'}, {direction}; {outside_wilson}/10 diagonal deviations are outside Wilson intervals and are {uncertainty} the Wilson uncertainty bands.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reliability.to_parquet(OUTPUT_PATH, index=False)
    plot_reliability(tables)
    print(f"\nSaved reliability table to {OUTPUT_PATH}")
    print(f"Saved reliability plot to {PLOT_PATH}")
    return reliability


def stable_sigmoid(values):
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def platt_feature(raw_probabilities):
    clipped = np.clip(np.asarray(raw_probabilities, dtype=float), PLATT_CLIP, 1.0 - PLATT_CLIP)
    return np.log(clipped) - np.log1p(-clipped)


def platt_log_likelihood(design, targets, theta):
    linear_predictor = design @ theta
    return float(np.sum(targets * linear_predictor - np.logaddexp(0.0, linear_predictor)))


def fit_platt(raw_probabilities, targets):
    x = platt_feature(raw_probabilities)
    targets = np.asarray(targets, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    theta = np.array([0.0, 1.0])
    trace = []

    for iteration in range(MAX_NEWTON_ITERATIONS + 1):
        calibrated = stable_sigmoid(design @ theta)
        log_likelihood = platt_log_likelihood(design, targets, theta)
        gradient = design.T @ (targets - calibrated)
        gradient_norm = float(np.linalg.norm(gradient))
        trace.append({"iteration": iteration, "log_likelihood": log_likelihood, "gradient_norm": gradient_norm, "a": theta[0], "b": theta[1]})
        if gradient_norm < GRADIENT_TOLERANCE:
            break
        if iteration == MAX_NEWTON_ITERATIONS:
            raise AssertionError(f"Platt fit did not converge in {MAX_NEWTON_ITERATIONS} iterations; gradient norm={gradient_norm:.3e}")

        weights = calibrated * (1.0 - calibrated)
        information = design.T @ (weights[:, None] * design)
        newton_step = np.linalg.solve(information, gradient)
        step_scale = 1.0
        for _ in range(50):
            proposed_theta = theta + step_scale * newton_step
            proposed_log_likelihood = platt_log_likelihood(design, targets, proposed_theta)
            if proposed_log_likelihood >= log_likelihood - 1e-12:
                theta = proposed_theta
                break
            step_scale *= 0.5
        else:
            raise AssertionError("Platt Newton step failed backtracking")

    trace = pd.DataFrame(trace)
    assert int(trace["iteration"].iloc[-1]) <= MAX_NEWTON_ITERATIONS, "Platt fit exceeded the iteration limit"
    assert trace["gradient_norm"].iloc[-1] < GRADIENT_TOLERANCE, "Platt fit did not reach the gradient tolerance"
    assert trace["log_likelihood"].diff().dropna().ge(-1e-10).all(), "Accepted Platt step decreased log-likelihood"
    assert np.isfinite(theta).all(), "Platt parameters must be finite"
    assert theta[1] > 0, "Platt slope must be positive to preserve ranking"
    return {
        "a": float(theta[0]),
        "b": float(theta[1]),
        "iterations": int(trace["iteration"].iloc[-1]),
        "final_log_likelihood": float(trace["log_likelihood"].iloc[-1]),
        "final_gradient_norm": float(trace["gradient_norm"].iloc[-1]),
        "trace": trace,
    }


def apply_platt(raw_probabilities, fit):
    return stable_sigmoid(fit["a"] + fit["b"] * platt_feature(raw_probabilities))


def load_train_validation_common_rows():
    rows = pd.read_parquet(INPUT_PATH, filters=[("split", "in", ["train", "validation"])])
    prediction_columns = [column for column in rows.columns if column.startswith("p_")]
    assert len(prediction_columns) == 10, f"Expected 10 p_* columns, found {len(prediction_columns)}: {prediction_columns}"
    assert PREDICTION_COLUMN in prediction_columns, f"Selected probability column is missing: {PREDICTION_COLUMN}"
    assert rows["split"].isin(["train", "validation"]).all(), "Calibration input contains an unsupported split"
    assert not rows["split"].eq("test").any(), "Test rows entered calibration analysis"

    common_rows = {}
    for (split, horizon), expected_rows in EXPECTED_ALL_COMMON_ROWS.items():
        subset = rows.loc[rows["split"].eq(split) & rows["horizon_minutes"].eq(horizon)]
        common = subset.loc[subset[prediction_columns].notna().all(axis=1)].copy()
        assert len(common) == expected_rows, f"Expected {expected_rows:,} {split} T-{horizon} common rows, found {len(common):,}"
        assert common["split"].eq(split).all(), f"{split} T-{horizon} common rows contain another split"
        assert not common["split"].eq("test").any(), f"Test rows entered {split} T-{horizon} calibration"
        assert common["y"].notna().all(), f"{split} T-{horizon} common rows contain null y values"
        assert common["y"].isin([0, 1]).all(), f"{split} T-{horizon} common rows contain invalid y values"
        assert common[PREDICTION_COLUMN].notna().all(), f"{split} T-{horizon} common rows contain null raw probabilities"
        common_rows[(split, horizon)] = common
    return common_rows


def probability_metrics(probabilities, targets):
    probabilities = pd.Series(np.asarray(probabilities, dtype=float), index=targets.index)
    brier, brier_n = brier_score(probabilities, targets)
    log_loss, _ = binary_log_loss(probabilities, targets)
    auc, auc_n = rank_auc(probabilities, targets)
    assert brier_n == len(targets) and auc_n == len(targets), "Calibration metrics must use every supplied row"
    return brier, log_loss, auc


def evaluate_train_fit(validation_rows, fit, horizon):
    raw = validation_rows[PREDICTION_COLUMN]
    targets = validation_rows["y"]
    calibrated = apply_platt(raw, fit)
    assert len(calibrated) == EXPECTED_ALL_COMMON_ROWS[("validation", horizon)], f"T-{horizon} validation row count changed"
    assert np.isfinite(calibrated).all(), f"T-{horizon} calibrated probabilities contain non-finite values"
    assert ((calibrated > 0) & (calibrated < 1)).all(), f"T-{horizon} calibrated probabilities must lie strictly inside (0, 1)"
    assert fit["b"] > 0, f"T-{horizon} Platt slope must be positive"

    raw_order = np.argsort(raw.to_numpy(), kind="stable")
    calibrated_order = np.argsort(calibrated, kind="stable")
    assert np.array_equal(raw_order, calibrated_order), f"T-{horizon} calibration reordered validation examples"
    raw_brier, raw_log_loss, raw_auc = probability_metrics(raw, targets)
    calibrated_brier, calibrated_log_loss, calibrated_auc = probability_metrics(calibrated, targets)
    assert abs(raw_auc - calibrated_auc) <= 1e-9, f"T-{horizon} AUC changed after monotone calibration"
    return calibrated, {
        "candidate": CANDIDATE,
        "horizon_minutes": horizon,
        "n": len(validation_rows),
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": raw_brier - calibrated_brier,
        "raw_log_loss": raw_log_loss,
        "calibrated_log_loss": calibrated_log_loss,
        "log_loss_improvement": raw_log_loss - calibrated_log_loss,
        "raw_auc": raw_auc,
        "calibrated_auc": calibrated_auc,
    }


def parameter_record(horizon, fit_split, parameter_role, n, fit):
    return {
        "candidate": CANDIDATE,
        "horizon_minutes": horizon,
        "fit_split": fit_split,
        "parameter_role": parameter_role,
        "n": n,
        "a": fit["a"],
        "b": fit["b"],
        "iterations": fit["iterations"],
        "final_log_likelihood": fit["final_log_likelihood"],
        "final_gradient_norm": fit["final_gradient_norm"],
    }


def plot_platt_reliability(raw_tables, calibrated_tables):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        for table, label, color in [(raw_tables[horizon], "Raw Stage 0", "C0"), (calibrated_tables[horizon], "Train-fitted Platt", "C1")]:
            lower_error = table["observed_yes_rate"] - table["wilson_low"]
            upper_error = table["wilson_high"] - table["observed_yes_rate"]
            axis.errorbar(table["mean_predicted"], table["observed_yes_rate"], yerr=np.vstack([lower_error, upper_error]), fmt="o-", capsize=3, label=label, color=color)
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
        axis.set_title(f"T-{horizon}")
        axis.set_xlabel("Mean predicted probability")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0].set_ylabel("Observed YES frequency")
    fig.suptitle("Stage 0 Raw vs Train-Fitted Platt Reliability (Validation Common Rows)")
    fig.tight_layout()
    PLATT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLATT_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_platt_calibration():
    common_rows = load_train_validation_common_rows()
    print(f"Selected Stage 0 candidate: {CANDIDATE}")
    print("Common rows: " + "; ".join(f"{split} T-{horizon} = {len(common_rows[(split, horizon)]):,}" for split, horizon in EXPECTED_ALL_COMMON_ROWS))

    train_fits = {}
    parameter_rows = []
    for horizon in HORIZONS:
        train_rows = common_rows[("train", horizon)]
        fit = fit_platt(train_rows[PREDICTION_COLUMN], train_rows["y"])
        train_fits[horizon] = fit
        parameter_rows.append(parameter_record(horizon, "train", "legitimate_train_fit", len(train_rows), fit))
        print(f"\nTrain T-{horizon} Newton/IRLS convergence")
        print(fit["trace"].to_string(index=False, formatters={
            "log_likelihood": "{:.9f}".format,
            "gradient_norm": "{:.3e}".format,
            "a": "{:.9f}".format,
            "b": "{:.9f}".format,
        }))

    train_parameters = pd.DataFrame(parameter_rows)
    print("\nLegitimate train-fitted Platt parameters")
    print(train_parameters[["horizon_minutes", "n", "a", "b", "iterations", "final_log_likelihood", "final_gradient_norm"]].to_string(index=False, formatters={
        "a": "{:.9f}".format,
        "b": "{:.9f}".format,
        "final_log_likelihood": "{:.9f}".format,
        "final_gradient_norm": "{:.3e}".format,
    }))

    calibrated_probabilities = {}
    validation_score_rows = []
    ceiling_fits = {}
    for horizon in HORIZONS:
        validation_rows = common_rows[("validation", horizon)]
        calibrated, score_row = evaluate_train_fit(validation_rows, train_fits[horizon], horizon)
        calibrated_probabilities[horizon] = calibrated

        ceiling_fit = fit_platt(validation_rows[PREDICTION_COLUMN], validation_rows["y"])
        ceiling_fits[horizon] = ceiling_fit
        ceiling_probabilities = apply_platt(validation_rows[PREDICTION_COLUMN], ceiling_fit)
        ceiling_brier, ceiling_log_loss, _ = probability_metrics(ceiling_probabilities, validation_rows["y"])
        score_row["validation_fit_ceiling_brier"] = ceiling_brier
        score_row["validation_fit_ceiling_log_loss"] = ceiling_log_loss
        validation_score_rows.append(score_row)
        parameter_rows.append(parameter_record(horizon, "validation", "validation_in_sample_ceiling", len(validation_rows), ceiling_fit))

    validation_scores = pd.DataFrame(validation_score_rows)
    parameters = pd.DataFrame(parameter_rows)
    assert len(parameters) == 4, f"Expected four Platt parameter rows, found {len(parameters)}"
    assert len(validation_scores) == 2, f"Expected two validation score rows, found {len(validation_scores)}"
    print("\nValidation: raw versus legitimate train-fitted calibration")
    print(validation_scores.to_string(index=False, formatters={
        "raw_brier": "{:.9f}".format,
        "calibrated_brier": "{:.9f}".format,
        "brier_improvement": "{:+.9f}".format,
        "raw_log_loss": "{:.9f}".format,
        "calibrated_log_loss": "{:.9f}".format,
        "log_loss_improvement": "{:+.9f}".format,
        "raw_auc": "{:.9f}".format,
        "calibrated_auc": "{:.9f}".format,
        "validation_fit_ceiling_brier": "{:.9f}".format,
        "validation_fit_ceiling_log_loss": "{:.9f}".format,
    }))
    print("Validation AUC unchanged within 1e-9 at both horizons: PASS")

    ceiling_parameters = parameters.loc[parameters["parameter_role"].eq("validation_in_sample_ceiling")]
    print("\nValidation-fitted in-sample ceiling parameters (diagnostic only; not shippable)")
    print(ceiling_parameters[["horizon_minutes", "n", "a", "b", "iterations", "final_gradient_norm"]].to_string(index=False, formatters={
        "a": "{:.9f}".format,
        "b": "{:.9f}".format,
        "final_gradient_norm": "{:.3e}".format,
    }))
    for horizon in HORIZONS:
        relation = "> 1, consistent with stretching probabilities away from 0.5" if train_fits[horizon]["b"] > 1 else "<= 1"
        print(f"Train T-{horizon} slope b={train_fits[horizon]['b']:.6f} ({relation})")

    raw_tables = {}
    calibrated_tables = {}
    for horizon in HORIZONS:
        validation_rows = common_rows[("validation", horizon)].copy()
        validation_rows["p_train_calibrated"] = calibrated_probabilities[horizon]
        raw_tables[horizon] = build_reliability_table(validation_rows, horizon)
        calibrated_tables[horizon] = build_reliability_table(validation_rows, horizon, "p_train_calibrated")
    plot_platt_reliability(raw_tables, calibrated_tables)

    PLATT_PARAMETERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    parameters.to_parquet(PLATT_PARAMETERS_PATH, index=False)
    validation_scores.to_parquet(PLATT_SCORES_PATH, index=False)
    print(f"\nSaved Platt parameters to {PLATT_PARAMETERS_PATH}")
    print(f"Saved validation calibration scores to {PLATT_SCORES_PATH}")
    print(f"Saved raw-versus-Platt reliability plot to {PLATT_PLOT_PATH}")
    return parameters, validation_scores


if __name__ == "__main__":
    analyze_platt_calibration()
