import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kalshist-matplotlib"))
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_calibration import apply_platt
from scripts.evaluation_split import SPLIT_RANGES


PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
BASIS_SUMMARY_PATH = PROJECT_ROOT / "data/models/stage0_basis_probability_summary.parquet"
STAGE0_SCORES_PATH = PROJECT_ROOT / "data/models/stage0_scores.parquet"
PLATT_SCORES_PATH = PROJECT_ROOT / "data/models/stage0_platt_validation_scores.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
SUMMARY_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap_summary.parquet"
PLOT_PATH = PROJECT_ROOT / "data/models/plots/stage0_model_market_gap_validation.png"
SELECTED_MODEL = "5min_ewma_vol"
PREDICTION_COLUMN = f"p_{SELECTED_MODEL}"
ROW_KEY = ["ticker", "horizon_minutes"]
SPLITS = ("train", "validation")
HORIZONS = (10, 5)
MODEL_VERSIONS = ("raw", "train_fitted_platt")
BASIS_MAGNITUDES = (1.2, 5.0)
EXPECTED_COMMON_ROWS = {("train", 10): 4_792, ("train", 5): 4_757, ("validation", 10): 1_685, ("validation", 5): 1_649}


def load_inputs():
    assert SELECTED_MODEL == "5min_ewma_vol", f"Unexpected selected model: {SELECTED_MODEL}"
    prediction_columns = [column for column in pq.read_schema(PREDICTIONS_PATH).names if column.startswith("p_")]
    assert len(prediction_columns) == 10, f"Expected 10 p_* columns, found {len(prediction_columns)}: {prediction_columns}"
    assert PREDICTION_COLUMN in prediction_columns, f"Selected prediction column is missing: {PREDICTION_COLUMN}"

    columns = ROW_KEY + ["split", *prediction_columns]
    predictions = pd.read_parquet(PREDICTIONS_PATH, columns=columns, filters=[("split", "in", list(SPLITS))])
    assert predictions["split"].isin(SPLITS).all(), "Prediction input contains an unsupported split"
    assert not predictions["split"].eq("test").any(), "Test rows entered model-market analysis"
    assert not predictions.duplicated(ROW_KEY).any(), f"Prediction row key {ROW_KEY} must be unique"

    first_date = SPLIT_RANGES["train"][0]
    last_date = SPLIT_RANGES["validation"][1]
    features = pd.read_parquet(FEATURES_PATH, columns=ROW_KEY + ["close_date", "quote_mid"], filters=[("close_date", ">=", first_date), ("close_date", "<=", last_date)])
    assert not features.duplicated(ROW_KEY).any(), f"Feature row key {ROW_KEY} must be unique"
    assert features["close_date"].between(first_date, last_date).all(), "Features fall outside the train/validation date range"
    rows = predictions.merge(features, on=ROW_KEY, how="left", validate="one_to_one", indicator=True)
    assert len(rows) == len(predictions), "Feature merge changed the train/validation row count"
    assert rows["_merge"].eq("both").all(), "Some predictions did not match market features"
    rows = rows.drop(columns=["_merge", "close_date"])

    parameters = pd.read_parquet(PLATT_PARAMETERS_PATH, filters=[("fit_split", "==", "train"), ("parameter_role", "==", "legitimate_train_fit")])
    assert len(parameters) == 2, f"Expected exactly two legitimate train-fit Platt rows, found {len(parameters)}"
    assert parameters["candidate"].eq(SELECTED_MODEL).all(), "Platt parameters use a different Stage 0 model"
    assert parameters["fit_split"].eq("train").all(), "Non-train Platt parameters entered analysis"
    assert parameters["parameter_role"].eq("legitimate_train_fit").all(), "Validation-fitted ceiling parameters entered analysis"
    assert set(parameters["horizon_minutes"]) == set(HORIZONS), "Platt parameter horizons are incomplete"
    assert parameters["b"].gt(0).all(), "Platt slopes must be positive"
    assert np.isfinite(parameters[["a", "b"]].to_numpy(dtype=float)).all(), "Platt parameters must be finite"
    return rows, prediction_columns, parameters.set_index("horizon_minutes")


def build_gap_rows(rows, prediction_columns, parameters):
    common_frames = []
    for (split, horizon), expected_rows in EXPECTED_COMMON_ROWS.items():
        subset = rows.loc[rows["split"].eq(split) & rows["horizon_minutes"].eq(horizon)]
        common = subset.loc[subset[prediction_columns].notna().all(axis=1)].copy()
        assert len(common) == expected_rows, f"Expected {expected_rows:,} {split} T-{horizon} common rows, found {len(common):,}"
        assert common["split"].eq(split).all() and not common["split"].eq("test").any(), f"Invalid split in {split} T-{horizon} common rows"
        common_frames.append(common)

    common = pd.concat(common_frames, ignore_index=True)
    assert not common.duplicated(ROW_KEY).any(), f"Common-row key {ROW_KEY} must be unique"
    assert not common["split"].eq("test").any(), "Test rows entered model-market analysis"
    assert np.isfinite(common[[*prediction_columns, "quote_mid"]].to_numpy(dtype=float)).all(), "Raw or market probabilities contain non-finite values"
    assert common[[*prediction_columns, "quote_mid"]].ge(0).all().all() and common[[*prediction_columns, "quote_mid"]].le(1).all().all(), "Raw or market probabilities fall outside [0, 1]"

    common["market_probability"] = common["quote_mid"]
    common["raw_model_probability"] = common[PREDICTION_COLUMN]
    common["platt_model_probability"] = np.nan
    for horizon in HORIZONS:
        mask = common["horizon_minutes"].eq(horizon)
        fit = {"a": float(parameters.loc[horizon, "a"]), "b": float(parameters.loc[horizon, "b"])}
        common.loc[mask, "platt_model_probability"] = apply_platt(common.loc[mask, "raw_model_probability"], fit)

    probability_columns = ["market_probability", "raw_model_probability", "platt_model_probability"]
    assert np.isfinite(common[probability_columns].to_numpy(dtype=float)).all(), "Gap probabilities contain non-finite values"
    assert common[probability_columns].ge(0).all().all() and common[probability_columns].le(1).all().all(), "Gap probabilities fall outside [0, 1]"
    for horizon in HORIZONS:
        horizon_rows = common.loc[common["horizon_minutes"].eq(horizon)]
        raw_ranks = horizon_rows["raw_model_probability"].rank(method="average")
        platt_ranks = horizon_rows["platt_model_probability"].rank(method="average")
        assert raw_ranks.equals(platt_ranks), f"Platt calibration changed model ordering within T-{horizon}"

    common["signed_gap_raw"] = common["raw_model_probability"] - common["market_probability"]
    common["abs_gap_raw"] = common["signed_gap_raw"].abs()
    common["signed_gap_platt"] = common["platt_model_probability"] - common["market_probability"]
    common["abs_gap_platt"] = common["signed_gap_platt"].abs()
    output_columns = ROW_KEY[:1] + ["split", "horizon_minutes", "market_probability", "raw_model_probability", "platt_model_probability", "signed_gap_raw", "abs_gap_raw", "signed_gap_platt", "abs_gap_platt"]
    output = common[output_columns].copy()
    assert not output.duplicated(ROW_KEY).any(), f"Output row key {ROW_KEY} must be unique"
    return output


def spearman_correlation(left, right):
    correlation = left.rank(method="average").corr(right.rank(method="average"))
    assert np.isfinite(correlation), "Spearman correlation must be finite"
    return float(correlation)


def summarize_gaps(rows):
    summary_rows = []
    for split in SPLITS:
        for horizon in HORIZONS:
            subset = rows.loc[rows["split"].eq(split) & rows["horizon_minutes"].eq(horizon)]
            for version, probability_column, signed_column, absolute_column in [
                ("raw", "raw_model_probability", "signed_gap_raw", "abs_gap_raw"),
                ("train_fitted_platt", "platt_model_probability", "signed_gap_platt", "abs_gap_platt"),
            ]:
                absolute = subset[absolute_column]
                signed = subset[signed_column]
                summary_rows.append({
                    "split": split,
                    "horizon_minutes": horizon,
                    "model_version": version,
                    "n": len(subset),
                    "mean_abs_gap": absolute.mean(),
                    "median_abs_gap": absolute.median(),
                    "p75_abs_gap": absolute.quantile(0.75),
                    "p90_abs_gap": absolute.quantile(0.90),
                    "p95_abs_gap": absolute.quantile(0.95),
                    "max_abs_gap": absolute.max(),
                    "mean_signed_gap": signed.mean(),
                    "median_signed_gap": signed.median(),
                    "spearman_model_market": spearman_correlation(subset[probability_column], subset["market_probability"]),
                })
    summary = pd.DataFrame(summary_rows)
    assert len(summary) == 8, f"Expected 8 summary rows, found {len(summary)}"
    assert not summary.duplicated(["split", "horizon_minutes", "model_version"]).any(), "Summary key must be unique"
    assert np.isfinite(summary["spearman_model_market"]).all(), "Summary contains a non-finite Spearman correlation"
    return summary


def build_version_comparison(summary):
    indexed = summary.set_index(["split", "horizon_minutes", "model_version"])
    rows = []
    for split in SPLITS:
        for horizon in HORIZONS:
            raw = indexed.loc[(split, horizon, "raw")]
            platt = indexed.loc[(split, horizon, "train_fitted_platt")]
            rows.append({
                "split": split,
                "horizon_minutes": horizon,
                "raw_median_abs_gap": raw["median_abs_gap"],
                "platt_median_abs_gap": platt["median_abs_gap"],
                "median_abs_gap_change": platt["median_abs_gap"] - raw["median_abs_gap"],
                "raw_mean_signed_gap": raw["mean_signed_gap"],
                "platt_mean_signed_gap": platt["mean_signed_gap"],
                "raw_spearman": raw["spearman_model_market"],
                "platt_spearman": platt["spearman_model_market"],
            })
    return pd.DataFrame(rows)


def build_basis_comparison(summary):
    basis = pd.read_parquet(BASIS_SUMMARY_PATH)
    basis = basis.loc[basis["split"].eq("validation") & basis["horizon_minutes"].isin(HORIZONS) & basis["basis_bps"].isin(BASIS_MAGNITUDES) & basis["probability_version"].isin(MODEL_VERSIONS)].copy()
    assert len(basis) == 8, f"Expected 8 validation basis-summary rows, found {len(basis)}"
    assert not basis.duplicated(["horizon_minutes", "basis_bps", "probability_version"]).any(), "Basis-summary key must be unique"
    gap = summary.loc[summary["split"].eq("validation"), ["horizon_minutes", "model_version", "n", "median_abs_gap"]]
    comparison = basis.merge(gap, left_on=["horizon_minutes", "probability_version"], right_on=["horizon_minutes", "model_version"], how="left", validate="many_to_one", suffixes=("_basis", "_gap"))
    assert comparison["n_basis"].eq(comparison["n_gap"]).all(), "Basis and gap summaries use different validation row counts"
    raw = comparison.loc[comparison["probability_version"].eq("raw")]
    assert np.allclose(raw["median_raw_model_market_abs_gap"], raw["median_abs_gap"], rtol=0.0, atol=1e-15), "Stored raw gap medians differ from the rebuilt common population"
    comparison["basis_shift_to_gap_ratio"] = comparison["median_max_abs_shift"] / comparison["median_abs_gap"]
    assert np.isfinite(comparison["basis_shift_to_gap_ratio"]).all(), "Basis-to-gap ratio must be finite"
    return comparison.sort_values(["horizon_minutes", "basis_bps", "probability_version"], ascending=[False, True, True])


def load_brier_context():
    stage0 = pd.read_parquet(STAGE0_SCORES_PATH, filters=[("split", "==", "validation"), ("population", "==", "common"), ("candidate", "==", SELECTED_MODEL)])
    platt = pd.read_parquet(PLATT_SCORES_PATH, filters=[("candidate", "==", SELECTED_MODEL)])
    assert len(stage0) == 2 and len(platt) == 2, "Expected two validation/common Brier rows from each stored score artifact"
    assert stage0["split"].eq("validation").all() and stage0["population"].eq("common").all(), "Brier context must be validation/common only"
    assert set(stage0["horizon_minutes"]) == set(HORIZONS) and set(platt["horizon_minutes"]) == set(HORIZONS), "Brier horizons are incomplete"
    context = stage0[["horizon_minutes", "n", "brier", "brier_market"]].merge(platt[["horizon_minutes", "n", "raw_brier", "calibrated_brier"]], on="horizon_minutes", validate="one_to_one", suffixes=("_stage0", "_platt"))
    assert context["n_stage0"].eq(context["n_platt"]).all(), "Stored Brier artifacts use different validation row counts"
    assert np.allclose(context["brier"], context["raw_brier"], rtol=0.0, atol=1e-15), "Stored raw Brier values disagree"
    context = context.rename(columns={"n_stage0": "n", "brier": "raw_brier", "raw_brier": "raw_brier_platt_artifact", "brier_market": "market_brier", "calibrated_brier": "platt_brier"})
    return context[["horizon_minutes", "n", "raw_brier", "platt_brier", "market_brier"]].sort_values("horizon_minutes", ascending=False)


def plot_validation_gaps(rows):
    validation = rows.loc[rows["split"].eq("validation")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        subset = validation.loc[validation["horizon_minutes"].eq(horizon)]
        axis.scatter(subset["market_probability"], subset["raw_model_probability"], s=10, alpha=0.22, label="Raw", color="C0")
        axis.scatter(subset["market_probability"], subset["platt_model_probability"], s=10, alpha=0.22, label="Train-fitted Platt", color="C1")
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Model = market")
        axis.set_title(f"Validation T-{horizon}")
        axis.set_xlabel("Market probability (quote_mid)")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Model probability")
    fig.suptitle("Stage 0 Model-Market Probability Gap")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_results(summary, version_comparison, basis_comparison, brier_context):
    percentage_columns = ["raw_median_abs_gap", "platt_median_abs_gap", "median_abs_gap_change", "raw_mean_signed_gap", "platt_mean_signed_gap"]
    display = version_comparison.copy()
    display[percentage_columns] *= 100.0
    print("Raw versus train-fitted Platt model-market disagreement (percentage points)")
    print(display.to_string(index=False, formatters={column: "{:+.4f}".format if "signed" in column or "change" in column else "{:.4f}".format for column in percentage_columns} | {"raw_spearman": "{:.6f}".format, "platt_spearman": "{:.6f}".format}))

    basis_display = basis_comparison[["horizon_minutes", "probability_version", "basis_bps", "median_abs_gap", "median_max_abs_shift", "basis_shift_to_gap_ratio"]].copy()
    basis_display[["median_abs_gap", "median_max_abs_shift"]] *= 100.0
    print("\nValidation basis sensitivity relative to model-market disagreement")
    print(basis_display.to_string(index=False, formatters={"median_abs_gap": "{:.4f}".format, "median_max_abs_shift": "{:.4f}".format, "basis_shift_to_gap_ratio": "{:.1%}".format}))

    print("\nStored validation/common Brier benchmark")
    print(brier_context.to_string(index=False, formatters={"raw_brier": "{:.6f}".format, "platt_brier": "{:.6f}".format, "market_brier": "{:.6f}".format}))

    validation = summary.loc[summary["split"].eq("validation")]
    rank_min, rank_max = validation["spearman_model_market"].min(), validation["spearman_model_market"].max()
    gap_min, gap_max = validation["median_abs_gap"].min(), validation["median_abs_gap"].max()
    changes = version_comparison.loc[version_comparison["split"].eq("validation"), "median_abs_gap_change"]
    direction = "reduced" if changes.lt(0).all() else "increased" if changes.gt(0).all() else "changed in mixed directions"
    small = basis_comparison.loc[basis_comparison["basis_bps"].eq(1.2), "basis_shift_to_gap_ratio"]
    large = basis_comparison.loc[basis_comparison["basis_bps"].eq(5.0), "basis_shift_to_gap_ratio"]
    market_beats_both = (brier_context["market_brier"] < brier_context[["raw_brier", "platt_brier"]].min(axis=1)).all()
    print(f"\nConclusion: validation model and market rankings are similar (Spearman {rank_min:.3f} to {rank_max:.3f}), while median absolute disagreement is {gap_min:.1%} to {gap_max:.1%} probability.")
    print(f"Train-fitted Platt calibration {direction} the validation median absolute gap; the table reports the actual horizon-specific changes.")
    print(f"Median 1.2 bp basis sensitivity is {small.min():.1%} to {small.max():.1%} of the corresponding gap; median 5 bp sensitivity is {large.min():.1%} to {large.max():.1%}.")
    print(f"The market quote_mid {'retains' if market_beats_both else 'does not retain'} lower validation/common Brier than both raw and train-fitted Platt Stage 0 at T-10 and T-5.")
    print("Model-market disagreement is not the same as tradable edge; this analysis does not use it as an edge estimate or construct a label-flip probability.")


def analyze_model_market_gap():
    rows, prediction_columns, parameters = load_inputs()
    gap_rows = build_gap_rows(rows, prediction_columns, parameters)
    summary = summarize_gaps(gap_rows)
    version_comparison = build_version_comparison(summary)
    basis_comparison = build_basis_comparison(summary)
    brier_context = load_brier_context()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gap_rows.to_parquet(OUTPUT_PATH, index=False)
    summary.to_parquet(SUMMARY_PATH, index=False)
    plot_validation_gaps(gap_rows)
    print_results(summary, version_comparison, basis_comparison, brier_context)
    print(f"\nSaved {len(gap_rows):,} row-level gaps to {OUTPUT_PATH}")
    print(f"Saved {len(summary)} summary rows to {SUMMARY_PATH}")
    print(f"Saved validation diagnostic plot to {PLOT_PATH}")
    print("No test rows or row-level outcome columns were read.")
    return gap_rows, summary


if __name__ == "__main__":
    analyze_model_market_gap()
