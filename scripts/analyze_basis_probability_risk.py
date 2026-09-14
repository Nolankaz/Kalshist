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

from scripts.analyze_calibration import apply_platt
from scripts.evaluation_split import SPLIT_RANGES
from scripts.stage0_baseline import stage0_probability


PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
SENSITIVITY_PATH = PROJECT_ROOT / "data/models/stage0_basis_probability_sensitivity.parquet"
SUMMARY_PATH = PROJECT_ROOT / "data/models/stage0_basis_probability_summary.parquet"
SETTLEMENT_PATH = PROJECT_ROOT / "data/models/settlement_proximity_summary.parquet"
PLOT_PATH = PROJECT_ROOT / "data/models/plots/stage0_basis_probability_sensitivity.png"
SELECTED_SIGMA = "5min_ewma_vol"
PREDICTION_COLUMN = f"p_{SELECTED_SIGMA}"
ROW_KEY = ["ticker", "horizon_minutes"]
HORIZONS = (10, 5)
BASIS_MAGNITUDES = (1.2, 5.0)
EXPECTED_ROWS = 14_358
EXPECTED_COMMON_ROWS = {("train", 10): 4_792, ("train", 5): 4_757, ("validation", 10): 1_685, ("validation", 5): 1_649}


def load_inputs():
    assert SELECTED_SIGMA == "5min_ewma_vol", f"Unexpected selected sigma: {SELECTED_SIGMA}"
    predictions = pd.read_parquet(PREDICTIONS_PATH, filters=[("split", "in", ["train", "validation"])])
    prediction_columns = [column for column in predictions.columns if column.startswith("p_")]
    assert len(prediction_columns) == 10, f"Expected 10 p_* columns, found {len(prediction_columns)}: {prediction_columns}"
    assert PREDICTION_COLUMN in prediction_columns, f"Selected prediction column is missing: {PREDICTION_COLUMN}"
    assert len(predictions) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} train/validation predictions, found {len(predictions):,}"
    assert predictions["split"].isin(["train", "validation"]).all(), "Prediction input contains an unsupported split"
    assert not predictions["split"].eq("test").any(), "Test rows entered basis-probability analysis"
    assert not predictions.duplicated(ROW_KEY).any(), f"Prediction row key {ROW_KEY} must be unique"

    first_date = SPLIT_RANGES["train"][0]
    last_date = SPLIT_RANGES["validation"][1]
    feature_columns = ROW_KEY + [SELECTED_SIGMA, "strike", "expiration_value", "quote_mid"]
    features = pd.read_parquet(FEATURES_PATH, columns=feature_columns, filters=[("close_date", ">=", first_date), ("close_date", "<=", last_date)])
    assert len(features) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} train/validation feature rows, found {len(features):,}"
    assert not features.duplicated(ROW_KEY).any(), f"Feature row key {ROW_KEY} must be unique"

    rows = predictions.merge(features, on=ROW_KEY, how="left", validate="one_to_one", indicator=True)
    assert len(rows) == EXPECTED_ROWS, "Feature merge changed the train/validation row count"
    assert rows["_merge"].eq("both").all(), "Some predictions did not match train/validation features"
    rows = rows.drop(columns="_merge")
    for split, (start_date, end_date) in SPLIT_RANGES.items():
        if split in {"train", "validation"}:
            assert rows.loc[rows["split"].eq(split), "close_date"].between(start_date, end_date).all(), f"{split} rows violate frozen split dates"

    parameters = pd.read_parquet(PLATT_PARAMETERS_PATH, filters=[("fit_split", "==", "train"), ("parameter_role", "==", "legitimate_train_fit")])
    assert len(parameters) == 2, f"Expected two legitimate train-fit Platt rows, found {len(parameters)}"
    assert parameters["candidate"].eq(SELECTED_SIGMA).all(), "Platt parameters use a different sigma candidate"
    assert parameters["fit_split"].eq("train").all(), "Non-train Platt parameters entered analysis"
    assert parameters["parameter_role"].eq("legitimate_train_fit").all(), "Validation-fit ceiling parameters entered analysis"
    assert set(parameters["horizon_minutes"]) == set(HORIZONS), "Platt parameter horizons are incomplete"
    assert parameters["b"].gt(0).all(), "Platt slopes must be positive"
    return rows, prediction_columns, parameters.set_index("horizon_minutes")


def build_common_rows(rows, prediction_columns):
    common_rows = {}
    for (split, horizon), expected_rows in EXPECTED_COMMON_ROWS.items():
        subset = rows.loc[rows["split"].eq(split) & rows["horizon_minutes"].eq(horizon)]
        common = subset.loc[subset[prediction_columns].notna().all(axis=1)].copy()
        assert len(common) == expected_rows, f"Expected {expected_rows:,} {split} T-{horizon} common rows, found {len(common):,}"
        assert common["split"].eq(split).all() and not common["split"].eq("test").any(), f"Invalid split in {split} T-{horizon} common rows"
        assert common["y"].notna().all() and common["y"].isin([0, 1]).all(), f"Invalid y in {split} T-{horizon} common rows"
        assert common[SELECTED_SIGMA].notna().all() and common[SELECTED_SIGMA].gt(0).all(), f"Invalid selected sigma in {split} T-{horizon}"
        assert common["T_years"].notna().all() and common["T_years"].gt(0).all(), f"Invalid T_years in {split} T-{horizon}"
        reconstructed, _ = stage0_probability(common["log_moneyness"], common[SELECTED_SIGMA], common["T_years"])
        max_error = float(np.max(np.abs(reconstructed - common[PREDICTION_COLUMN].to_numpy())))
        assert max_error <= 1e-15, f"Reconstructed {PREDICTION_COLUMN} differs by up to {max_error:.3e} for {split} T-{horizon}"
        common["reconstructed_p_raw"] = reconstructed
        common_rows[(split, horizon)] = common
    return common_rows


def validate_probabilities(p_raw, p_plus, p_minus, label):
    for name, values in [("raw", p_raw), ("plus", p_plus), ("minus", p_minus)]:
        assert np.isfinite(values).all(), f"{label} {name} probabilities contain non-finite values"
        assert ((values >= 0) & (values <= 1)).all(), f"{label} {name} probabilities fall outside [0, 1]"
    assert np.all(p_plus >= p_raw - 1e-15), f"Positive basis decreased probability for {label}"
    assert np.all(p_minus <= p_raw + 1e-15), f"Negative basis increased probability for {label}"


def sensitivity_frame(rows, basis_bps, probability_version, p_raw, p_plus, p_minus, stage0_probabilities):
    result = rows[["ticker", "split", "horizon_minutes", "log_moneyness", "T_years", SELECTED_SIGMA]].copy()
    result = result.rename(columns={SELECTED_SIGMA: "sigma"})
    result["basis_bps"] = basis_bps
    result["probability_version"] = probability_version
    result["stage0_p_raw"], result["stage0_p_plus"], result["stage0_p_minus"] = stage0_probabilities
    result["p_raw"] = p_raw
    result["p_plus"] = p_plus
    result["p_minus"] = p_minus
    result["shift_plus"] = result["p_plus"] - result["p_raw"]
    result["shift_minus"] = result["p_minus"] - result["p_raw"]
    result["abs_shift_plus"] = result["shift_plus"].abs()
    result["abs_shift_minus"] = result["shift_minus"].abs()
    result["max_abs_shift"] = result[["abs_shift_plus", "abs_shift_minus"]].max(axis=1)
    result["probability_span"] = result["p_plus"] - result["p_minus"]
    result["raw_model_market_abs_gap"] = (rows[PREDICTION_COLUMN] - rows["quote_mid"]).abs().to_numpy()
    return result


def build_probability_sensitivity(common_rows, parameters):
    frames = []
    for (split, horizon), rows in common_rows.items():
        sigma = rows[SELECTED_SIGMA].to_numpy(dtype=float)
        time_to_expiry = rows["T_years"].to_numpy(dtype=float)
        log_moneyness = rows["log_moneyness"].to_numpy(dtype=float)
        p_raw = rows["reconstructed_p_raw"].to_numpy(dtype=float)
        fit = {"a": float(parameters.loc[horizon, "a"]), "b": float(parameters.loc[horizon, "b"])}

        for basis_bps in BASIS_MAGNITUDES:
            log_plus = log_moneyness + np.log1p(basis_bps / 10_000.0)
            log_minus = log_moneyness + np.log1p(-basis_bps / 10_000.0)
            p_plus, _ = stage0_probability(log_plus, sigma, time_to_expiry)
            p_minus, _ = stage0_probability(log_minus, sigma, time_to_expiry)
            validate_probabilities(p_raw, p_plus, p_minus, f"{split} T-{horizon} raw {basis_bps} bps")
            stage0_probabilities = (p_raw, p_plus, p_minus)
            frames.append(sensitivity_frame(rows, basis_bps, "raw", p_raw, p_plus, p_minus, stage0_probabilities))

            p_cal_raw = apply_platt(p_raw, fit)
            p_cal_plus = apply_platt(p_plus, fit)
            p_cal_minus = apply_platt(p_minus, fit)
            validate_probabilities(p_cal_raw, p_cal_plus, p_cal_minus, f"{split} T-{horizon} train-fitted Platt {basis_bps} bps")
            frames.append(sensitivity_frame(rows, basis_bps, "train_fitted_platt", p_cal_raw, p_cal_plus, p_cal_minus, stage0_probabilities))

    sensitivity = pd.concat(frames, ignore_index=True)
    expected_rows = sum(EXPECTED_COMMON_ROWS.values()) * len(BASIS_MAGNITUDES) * 2
    assert len(sensitivity) == expected_rows, f"Expected {expected_rows:,} sensitivity rows, found {len(sensitivity):,}"
    assert sensitivity["split"].isin(["train", "validation"]).all(), "Sensitivity output contains an unsupported split"
    assert not sensitivity["split"].eq("test").any(), "Sensitivity output contains test rows"
    return sensitivity


def summarize_sensitivity(sensitivity):
    summary = sensitivity.groupby(["split", "horizon_minutes", "basis_bps", "probability_version"], sort=False).agg(
        n=("max_abs_shift", "size"),
        mean_max_abs_shift=("max_abs_shift", "mean"),
        median_max_abs_shift=("max_abs_shift", "median"),
        p75_max_abs_shift=("max_abs_shift", lambda values: values.quantile(0.75)),
        p90_max_abs_shift=("max_abs_shift", lambda values: values.quantile(0.90)),
        p95_max_abs_shift=("max_abs_shift", lambda values: values.quantile(0.95)),
        max_max_abs_shift=("max_abs_shift", "max"),
        median_probability_span=("probability_span", "median"),
        p95_probability_span=("probability_span", lambda values: values.quantile(0.95)),
        median_raw_model_market_abs_gap=("raw_model_market_abs_gap", "median"),
    ).reset_index()
    assert len(summary) == 16, f"Expected 16 sensitivity summary rows, found {len(summary)}"
    return summary


def build_settlement_summary(rows):
    market_columns = ["ticker", "split", "strike", "expiration_value"]
    market_rows = rows[market_columns]
    for column in ["split", "strike", "expiration_value"]:
        assert market_rows.groupby("ticker")[column].nunique(dropna=False).le(1).all(), f"{column} differs between horizon rows"
    markets = market_rows.drop_duplicates("ticker").copy()
    assert markets["ticker"].is_unique, "Settlement-proximity analysis must contain one row per ticker"
    valid = markets["strike"].notna() & markets["strike"].gt(0) & markets["expiration_value"].notna() & markets["expiration_value"].gt(0)
    markets = markets.loc[valid].copy()
    assert markets["strike"].gt(0).all(), "Strike must be positive before settlement-distance logs"
    assert markets["expiration_value"].gt(0).all(), "Expiration value must be positive before settlement-distance logs"
    markets["settlement_log_distance"] = np.abs(np.log(markets["expiration_value"] / markets["strike"]))
    markets["settlement_distance_bps"] = markets["settlement_log_distance"] * 10_000.0

    summary_rows = []
    for split in ["train", "validation", "train_validation"]:
        values = markets["settlement_distance_bps"] if split == "train_validation" else markets.loc[markets["split"].eq(split), "settlement_distance_bps"]
        summary_rows.append({
            "split": split,
            "n_markets": len(values),
            "median_settlement_distance_bps": values.median(),
            "p25_settlement_distance_bps": values.quantile(0.25),
            "p75_settlement_distance_bps": values.quantile(0.75),
            "p90_settlement_distance_bps": values.quantile(0.90),
            "p95_settlement_distance_bps": values.quantile(0.95),
            "share_within_1_2_bps": values.le(1.2).mean(),
            "share_within_5_0_bps": values.le(5.0).mean(),
        })
    summary = pd.DataFrame(summary_rows)
    assert summary["n_markets"].gt(0).all(), "Settlement-proximity summary contains an empty split"
    return summary


def plot_sensitivity(summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for row_index, split in enumerate(["train", "validation"]):
        for column_index, horizon in enumerate(HORIZONS):
            axis = axes[row_index, column_index]
            subset = summary.loc[summary["split"].eq(split) & summary["horizon_minutes"].eq(horizon)]
            for version, label, color in [("raw", "Raw", "C0"), ("train_fitted_platt", "Train-fitted Platt", "C1")]:
                version_rows = subset.loc[subset["probability_version"].eq(version)].sort_values("basis_bps")
                axis.plot(version_rows["basis_bps"], version_rows["median_max_abs_shift"] * 100, marker="o", color=color, label=f"{label} median")
                axis.plot(version_rows["basis_bps"], version_rows["p95_max_abs_shift"] * 100, marker="o", linestyle="--", color=color, label=f"{label} p95")
            axis.set_title(f"{split.title()} T-{horizon}")
            axis.set_xlabel("Basis magnitude (bps)")
            axis.set_ylabel("Maximum absolute probability shift (pp)")
            axis.grid(alpha=0.25)
            axis.legend()
    fig.suptitle("Stage 0 Basis-Probability Sensitivity")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_sensitivity_summary(summary, probability_version, title):
    table = summary.loc[summary["probability_version"].eq(probability_version)].copy()
    percentage_columns = ["mean_max_abs_shift", "median_max_abs_shift", "p75_max_abs_shift", "p90_max_abs_shift", "p95_max_abs_shift", "max_max_abs_shift"]
    table[percentage_columns] *= 100.0
    print(f"\n{title}")
    print(table[["split", "horizon_minutes", "basis_bps", "n", *percentage_columns]].to_string(index=False, formatters={column: "{:.4f}".format for column in percentage_columns}))


def analyze_basis_probability_risk():
    rows, prediction_columns, parameters = load_inputs()
    common_rows = build_common_rows(rows, prediction_columns)
    sensitivity = build_probability_sensitivity(common_rows, parameters)
    summary = summarize_sensitivity(sensitivity)
    settlement_summary = build_settlement_summary(rows)

    print(f"Selected sigma: {SELECTED_SIGMA}; calibration source: legitimate train-fitted Platt parameters")
    print("Common rows: " + "; ".join(f"{split} T-{horizon} = {len(common_rows[(split, horizon)]):,}" for split, horizon in EXPECTED_COMMON_ROWS))
    print_sensitivity_summary(summary, "raw", "Raw Stage 0 maximum absolute probability shift (percentage points)")
    print_sensitivity_summary(summary, "train_fitted_platt", "Train-fitted Platt maximum absolute probability shift (percentage points)")

    medians = summary.pivot(index=["split", "horizon_minutes", "basis_bps"], columns="probability_version", values="median_max_abs_shift").reset_index()
    print("\nMedian maximum absolute shifts (percentage points)")
    medians["raw"] *= 100.0
    medians["train_fitted_platt"] *= 100.0
    print(medians.to_string(index=False, formatters={"raw": "{:.4f}".format, "train_fitted_platt": "{:.4f}".format}))

    print("\nSettlement proximity to strike")
    print(settlement_summary.to_string(index=False, formatters={
        "median_settlement_distance_bps": "{:.4f}".format,
        "p25_settlement_distance_bps": "{:.4f}".format,
        "p75_settlement_distance_bps": "{:.4f}".format,
        "p90_settlement_distance_bps": "{:.4f}".format,
        "p95_settlement_distance_bps": "{:.4f}".format,
        "share_within_1_2_bps": "{:.2%}".format,
        "share_within_5_0_bps": "{:.2%}".format,
    }))

    raw_summary = summary.loc[summary["probability_version"].eq("raw")]
    t5_more_sensitive = all(
        raw_summary.loc[raw_summary["split"].eq(split) & raw_summary["horizon_minutes"].eq(5) & raw_summary["basis_bps"].eq(basis), "median_max_abs_shift"].item()
        > raw_summary.loc[raw_summary["split"].eq(split) & raw_summary["horizon_minutes"].eq(10) & raw_summary["basis_bps"].eq(basis), "median_max_abs_shift"].item()
        for split in ["train", "validation"] for basis in BASIS_MAGNITUDES
    )
    validation_ratios = raw_summary.loc[raw_summary["split"].eq("validation")].copy()
    validation_ratios["gap_ratio"] = validation_ratios["median_max_abs_shift"] / validation_ratios["median_raw_model_market_abs_gap"]
    print(f"\nInterpretation: T-5 median raw sensitivity is {'higher' if t5_more_sensitive else 'not consistently higher'} than T-10 across both splits and basis magnitudes.")
    for row in validation_ratios.itertuples(index=False):
        print(f"Validation T-{row.horizon_minutes} at {row.basis_bps:g} bps: median raw shift is {row.gap_ratio:.1%} of the median absolute model-market probability gap.")
    small_basis_ratios = validation_ratios.loc[validation_ratios["basis_bps"].eq(1.2), "gap_ratio"]
    large_basis_ratios = validation_ratios.loc[validation_ratios["basis_bps"].eq(5.0), "gap_ratio"]
    print(f"At 1.2 bps, basis sensitivity is smaller but non-negligible relative to model-market gaps ({small_basis_ratios.min():.1%}–{small_basis_ratios.max():.1%} of the median gap).")
    print(f"At 5 bps, basis sensitivity is material and approximately comparable to the model-market gap ({large_basis_ratios.min():.1%}–{large_basis_ratios.max():.1%} of the median gap).")
    print("Probability sensitivity and settlement proximity are reported as separate channels; no label-flip probability was constructed.")

    SENSITIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_parquet(SENSITIVITY_PATH, index=False)
    summary.to_parquet(SUMMARY_PATH, index=False)
    settlement_summary.to_parquet(SETTLEMENT_PATH, index=False)
    plot_sensitivity(summary)
    print(f"\nSaved row-level sensitivity to {SENSITIVITY_PATH}")
    print(f"Saved sensitivity summary to {SUMMARY_PATH}")
    print(f"Saved settlement proximity summary to {SETTLEMENT_PATH}")
    print(f"Saved sensitivity plot to {PLOT_PATH}")
    return sensitivity, summary, settlement_summary


if __name__ == "__main__":
    analyze_basis_probability_risk()
