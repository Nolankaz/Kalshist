"""Score frozen Day 13 log-volatility forecasts under the pre-registered rule."""

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.analyze_backtest import clustered_mean_se
from scripts.evaluation_split import SPLIT_RANGES
from scripts.model_har_rv import (
    CONFIG_H1, CONFIG_H2, EXPECTED_HORIZON_SCORE_ROWS, EXPECTED_SCORE_ROWS,
    FORECAST_COLUMNS, FORECAST_PATH, SCORE_KEY, assert_forecasts, forecast_fingerprint,
)
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, ROW_KEY
from scripts.walk_forward import FROZEN_FOLDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = PROJECT_ROOT / "data/targets/forward_vol_target.parquet"
DERIVED_PATH = PROJECT_ROOT / "data/features/derived_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/har_vol_comparison.parquet"
EXPECTED_FORECAST_FINGERPRINT = "d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee07ee0b31aa472eeaa50acc9"
TARGET_COLUMNS = ROW_KEY + ["split", "close_date", "fwd_log_rv"]
DERIVED_COLUMNS = ROW_KEY + ["split", "close_date", "is_common"]
MODEL_COLUMNS = {"har": "log_rv_har", "n1": "log_rv_n1", "naive_s0": "log_rv_naive_s0", "naive_15": "log_rv_naive_15"}
BENCHMARKS = ("naive_s0", "n1", "naive_15")
POPULATIONS = (
    (CONFIG_H2, "h2_validation", None),
    (CONFIG_H1, "h1_all_oof", None),
    (CONFIG_H1, "h1_folds_1_3", (1, 2, 3)),
    (CONFIG_H1, "h1_folds_4_6", (4, 5, 6)),
)
METRICS = ("mse", "mae", "mean_residual", "r2_vs_naive", "fit_mean_r2_inflated_by_regime_shift")
OUTPUT_COLUMNS = ["configuration", "population", "horizon_minutes", "model", "benchmark", "metric", "value", "difference", "naive_se", "clustered_se", "design_effect", "meaningful", "n", "n_clusters", "verdict"]
EXPECTED_H2_TARGET_ROWS = {10: 1_644, 5: 1_588}
EXPECTED_H1_TARGET_ROWS = {10: 3_473, 5: 3_380}
EXPECTED_H1_FOLD_COUNTS = {1: {10: 644, 5: 640}, 2: {10: 616, 5: 613}, 3: {10: 586, 5: 579}, 4: {10: 556, 5: 549}, 5: {10: 594, 5: 585}, 6: {10: 535, 5: 515}}
EXPECTED_H1_FIT_TARGET_ROWS = {1: {10: 2_936, 5: 2_903}, 2: {10: 3_573, 5: 3_529}, 3: {10: 4_184, 5: 4_125}, 4: {10: 4_765, 5: 4_695}, 5: {10: 5_310, 5: 5_226}, 6: {10: 5_891, 5: 5_797}}
VERDICT_WORDING = {
    "V4": "V4 — worse",
    "V1": "V1 — structure adds skill",
    "V2": "V2 — level/scale correction only",
    "V3": "V3 — no meaningful difference: Stage 0's trailing-σ assumption was already adequate as a volatility forecast.",
}


def key_set(frame, columns):
    assert not frame.duplicated(columns).any(), f"Duplicate keys in {columns}"
    return set(map(tuple, frame[columns].itertuples(index=False, name=None)))


def load_frozen_forecasts_and_common():
    """Complete all forecast/fingerprint checks before reading a target value."""
    assert pq.ParquetFile(FORECAST_PATH).metadata.num_rows == 10_346, "Frozen forecast file row count changed"
    forecasts = pd.read_parquet(FORECAST_PATH, columns=FORECAST_COLUMNS, filters=[("close_date", "<=", SPLIT_RANGES["validation"][1])])
    assert_forecasts(forecasts)
    assert len(forecasts) == 10_346 and forecasts["split"].isin(("train", "validation")).all(), "Forecast includes an unapproved split/date"
    assert forecast_fingerprint(forecasts) == EXPECTED_FORECAST_FINGERPRINT, "Frozen forecast fingerprint changed"
    assert not any(column.startswith("fwd_") for column in forecasts.columns), "Target column in forecasts"
    assert forecasts.groupby(["configuration", "horizon_minutes"], observed=True).size().to_dict() == {
        (config, horizon): count for config, by_horizon in EXPECTED_HORIZON_SCORE_ROWS.items() for horizon, count in by_horizon.items()
    }, "Forecast horizon populations changed"

    common = pd.read_parquet(DERIVED_PATH, columns=DERIVED_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(common.columns) == DERIVED_COLUMNS and common["split"].isin(("train", "validation")).all(), "Derived structural projection changed"
    assert common["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "Test date in structural rows"
    assert not common.duplicated(ROW_KEY).any(), "Duplicate derived row key"
    common = common.loc[common["is_common"], ROW_KEY + ["split", "close_date"]].copy()
    assert common.groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON_ROWS, "Day 9 common population changed"
    for configuration in (CONFIG_H1, CONFIG_H2):
        scored = forecasts.loc[forecasts["configuration"].eq(configuration)]
        assert len(scored) == EXPECTED_SCORE_ROWS[configuration], "Configuration count changed"
        if configuration == CONFIG_H2:
            expected = common.loc[common["split"].eq("validation")]
            assert scored["fold"].eq(1).all(), "H2 fold changed"
        else:
            expected_parts = []
            for fold in FROZEN_FOLDS:
                section = common.loc[common["close_date"].between(fold.score_start, fold.score_end)].copy()
                assert section.groupby("horizon_minutes").size().to_dict() == EXPECTED_H1_FOLD_COUNTS[fold.number], f"Fold {fold.number} score count changed"
                expected_parts.append(section.assign(fold=fold.number))
                actual_fold = scored.loc[scored["fold"].eq(fold.number)]
                assert key_set(actual_fold, ROW_KEY) == key_set(section, ROW_KEY), f"Fold {fold.number} score keys changed"
            expected = pd.concat(expected_parts, ignore_index=True)
        assert key_set(scored, ROW_KEY) == key_set(expected, ROW_KEY), f"{configuration} common score keys changed"
        checked = scored[ROW_KEY + ["split", "close_date"]].merge(expected[ROW_KEY + ["split", "close_date"]], on=ROW_KEY + ["split", "close_date"], how="outer", validate="one_to_one", indicator=True)
        assert len(checked) == len(scored) and checked["_merge"].eq("both").all(), f"{configuration} split/date changed"
        if configuration == CONFIG_H1:
            fold_checked = scored[ROW_KEY + ["fold"]].merge(expected[ROW_KEY + ["fold"]], on=ROW_KEY + ["fold"], how="outer", validate="one_to_one", indicator=True)
            assert len(fold_checked) == len(scored) and fold_checked["_merge"].eq("both").all(), "H1 fold IDs changed"
    return forecasts, common


def load_target_and_align(forecasts, common):
    assert pq.ParquetFile(TARGET_PATH).metadata.num_rows == 14_358, "Target structural row count changed"
    target = pd.read_parquet(TARGET_PATH, columns=TARGET_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(target.columns) == TARGET_COLUMNS and len(target) == 14_358, "Target projection/population changed"
    assert not target.duplicated(ROW_KEY).any() and target["split"].isin(("train", "validation")).all(), "Unsafe target keys/split"
    assert target["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "Test target date"
    structural = common.merge(target, on=ROW_KEY + ["split", "close_date"], how="left", validate="one_to_one", indicator=True)
    assert len(structural) == len(common) and structural["_merge"].eq("both").all(), "Common target structural rows missing"
    structural = structural.drop(columns="_merge")
    aligned = []
    for configuration in (CONFIG_H1, CONFIG_H2):
        rows = forecasts.loc[forecasts["configuration"].eq(configuration)].copy()
        joined = rows.merge(target, on=ROW_KEY, how="left", validate="one_to_one", suffixes=("", "_target"), indicator=True)
        assert len(joined) == len(rows) and joined["_merge"].eq("both").all(), f"{configuration}: missing target keys"
        assert joined["split"].eq(joined["split_target"]).all() and joined["close_date"].eq(joined["close_date_target"]).all(), f"{configuration}: target split/date mismatch"
        joined = joined.drop(columns=["split_target", "close_date_target", "_merge"])
        aligned.append(joined)
    scored = pd.concat(aligned, ignore_index=True)
    assert len(scored) == len(forecasts) and key_set(scored, SCORE_KEY) == key_set(forecasts, SCORE_KEY), "Scoring alignment lost keys"
    for horizon, expected in EXPECTED_H2_TARGET_ROWS.items():
        section = scored.loc[scored["configuration"].eq(CONFIG_H2) & scored["horizon_minutes"].eq(horizon)]
        assert section["fwd_log_rv"].notna().sum() == expected, f"H2 T-{horizon} target-bearing count changed"
    for horizon, expected in EXPECTED_H1_TARGET_ROWS.items():
        section = scored.loc[scored["configuration"].eq(CONFIG_H1) & scored["horizon_minutes"].eq(horizon)]
        assert section["fwd_log_rv"].notna().sum() == expected, f"H1 T-{horizon} target-bearing count changed"
    return scored, structural


def fit_window_means(structural):
    """Reference each score row to its own horizon/fold fit-window target mean."""
    reference = {}
    for horizon in (10, 5):
        train = structural.loc[structural["split"].eq("train") & structural["horizon_minutes"].eq(horizon), "fwd_log_rv"].dropna()
        assert len(train) == EXPECTED_H1_FIT_TARGET_ROWS[4][horizon], "H2 train target-bearing fit count changed"
        reference[(CONFIG_H2, 1, horizon)] = float(train.mean())
        for fold in FROZEN_FOLDS:
            values = structural.loc[structural["horizon_minutes"].eq(horizon) & structural["close_date"].between(fold.fit_start, fold.fit_end), "fwd_log_rv"].dropna()
            assert len(values) == EXPECTED_H1_FIT_TARGET_ROWS[fold.number][horizon], f"Fold {fold.number} T-{horizon} fit target count changed"
            reference[(CONFIG_H1, fold.number, horizon)] = float(values.mean())
        assert reference[(CONFIG_H2, 1, horizon)] == reference[(CONFIG_H1, 4, horizon)], "Fold 4/H2 fit mean changed"
    return reference


def select_populations(scored):
    for configuration, population, folds in POPULATIONS:
        rows = scored.loc[scored["configuration"].eq(configuration) & scored["fwd_log_rv"].notna()]
        if folds is not None:
            rows = rows.loc[rows["fold"].isin(folds)]
        assert len(rows) and rows["split"].isin(("train", "validation")).all(), f"{population}: empty or unsafe population"
        if population == "h1_folds_1_3":
            assert rows["split"].eq("train").all(), "Folds 1-3 are not train-date rows"
        if population in ("h2_validation", "h1_folds_4_6"):
            assert rows["split"].eq("validation").all(), f"{population} is not validation-date"
        if population == "h2_validation":
            assert rows.groupby("horizon_minutes").size().to_dict() == EXPECTED_H2_TARGET_ROWS, "H2 target-bearing counts changed"
        if population == "h1_all_oof":
            assert rows.groupby("horizon_minutes").size().to_dict() == EXPECTED_H1_TARGET_ROWS, "H1 target-bearing counts changed"
        yield configuration, population, rows


def assign_verdict(comparisons):
    primary = comparisons["naive_s0"]
    control = comparisons["n1"]
    if primary["meaningful"] and primary["difference"] > 0:
        return "V4"
    if primary["meaningful"] and primary["difference"] < 0:
        if control["meaningful"] and control["difference"] < 0:
            return "V1"
        return "V2"
    return "V3"


def score(scored, reference):
    output = []
    for configuration, population, population_rows in select_populations(scored):
        for horizon in (10, 5):
            rows = population_rows.loc[population_rows["horizon_minutes"].eq(horizon)].copy()
            assert len(rows) > 1 and not rows.duplicated(ROW_KEY).any(), f"{population} T-{horizon}: missing/duplicate keys"
            target = rows["fwd_log_rv"].to_numpy(dtype=float)
            assert np.isfinite(target).all(), "Non-finite target-bearing score value"
            fit_means = np.array([reference[(configuration, int(fold), horizon)] for fold in rows["fold"]], dtype=float)
            fit_mean_denominator = float(np.mean(np.square(target - fit_means)))
            assert fit_mean_denominator > 0, "Fit-mean R² reference is degenerate"
            losses = {}
            metrics = {}
            n_clusters = int(rows["close_date"].nunique())
            if population == "h2_validation":
                assert n_clusters == 21, f"H2 T-{horizon}: expected 21 close-date clusters"
            for model, column in MODEL_COLUMNS.items():
                prediction = rows[column].to_numpy(dtype=float)
                assert np.isfinite(prediction).all(), f"{model}: non-finite forecast"
                error = prediction - target
                losses[model] = np.square(error)
                metrics[model] = {"mse": float(np.mean(losses[model])), "mae": float(np.mean(np.abs(error))),
                                  "mean_residual": float(np.mean(-error))}
            naive_mse = metrics["naive_s0"]["mse"]
            assert naive_mse > 0, "Naive-S0 MSE is degenerate"
            for model in MODEL_COLUMNS:
                metrics[model]["r2_vs_naive"] = 1.0 - metrics[model]["mse"] / naive_mse
                metrics[model]["fit_mean_r2_inflated_by_regime_shift"] = 1.0 - metrics[model]["mse"] / fit_mean_denominator
                for metric in METRICS:
                    output.append({"configuration": configuration, "population": population, "horizon_minutes": horizon,
                                   "model": model, "benchmark": "", "metric": metric, "value": metrics[model][metric],
                                   "difference": np.nan, "naive_se": np.nan, "clustered_se": np.nan, "design_effect": np.nan,
                                   "meaningful": pd.NA, "n": len(rows), "n_clusters": n_clusters, "verdict": ""})
            assert abs(metrics["naive_s0"]["r2_vs_naive"]) <= 1e-15, "Naive-S0 R²_vs_naive must be zero"
            comparisons = {}
            for benchmark in BENCHMARKS:
                differential = losses["har"] - losses[benchmark]
                checked = clustered_mean_se(differential, rows["close_date"])
                mean_difference = float(checked["mean_net_pnl_per_trade"])
                assert checked["n_trades"] == len(rows) and checked["n_clusters"] == n_clusters, "Cluster helper population changed"
                assert abs(mean_difference - (metrics["har"]["mse"] - metrics[benchmark]["mse"])) <= 1e-12, "Paired mean/MSE identity failed"
                meaningful = bool(abs(mean_difference) > 2 * checked["se_cluster"])
                comparisons[benchmark] = {"difference": mean_difference, "meaningful": meaningful}
                output.append({"configuration": configuration, "population": population, "horizon_minutes": horizon,
                               "model": "har", "benchmark": benchmark, "metric": "paired_squared_log_loss_difference",
                               "value": mean_difference, "difference": mean_difference, "naive_se": checked["se_naive"],
                               "clustered_se": checked["se_cluster"], "design_effect": checked["design_effect"],
                               "meaningful": meaningful, "n": len(rows), "n_clusters": n_clusters, "verdict": ""})
            if population == "h2_validation":
                verdict = assign_verdict(comparisons)
                for item in output[-len(BENCHMARKS):]:
                    if item["benchmark"] == "naive_s0":
                        item["verdict"] = verdict
    result = pd.DataFrame(output, columns=OUTPUT_COLUMNS)
    result["meaningful"] = result["meaningful"].astype("boolean")
    return result.sort_values(OUTPUT_COLUMNS[:6]).reset_index(drop=True)


def assert_output(result):
    assert list(result.columns) == OUTPUT_COLUMNS and len(result) == len(POPULATIONS) * 2 * (len(MODEL_COLUMNS) * len(METRICS) + len(BENCHMARKS)), "Comparison schema/count changed"
    assert not result.duplicated(OUTPUT_COLUMNS[:6]).any(), "Duplicate comparison key"
    assert set(result["configuration"]) == {CONFIG_H1, CONFIG_H2} and set(result["population"]) == {item[1] for item in POPULATIONS}, "Comparison populations changed"
    assert set(result["horizon_minutes"]) == {10, 5}, "Comparison horizons changed"
    assert result["n"].gt(0).all() and result["n_clusters"].gt(1).all(), "Invalid comparison population size"
    assert np.isfinite(result["value"].to_numpy(dtype=float)).all(), "Non-finite metric"
    pure = result["benchmark"].eq("")
    assert pure.sum() == len(POPULATIONS) * 2 * len(MODEL_COLUMNS) * len(METRICS), "Per-model metrics missing"
    assert result.loc[pure, ["difference", "naive_se", "clustered_se", "design_effect"]].isna().all().all(), "Pure metrics contain fake comparisons"
    assert result.loc[pure, "meaningful"].isna().all() and result.loc[pure, "verdict"].eq("").all(), "Pure metric comparison flags changed"
    paired = result.loc[~pure]
    assert paired["model"].eq("har").all() and set(paired["benchmark"]) == set(BENCHMARKS), "Paired comparisons changed"
    assert paired["metric"].eq("paired_squared_log_loss_difference").all(), "Paired metric changed"
    assert paired["meaningful"].notna().all() and paired[["difference", "naive_se", "clustered_se", "design_effect"]].notna().all().all(), "Incomplete paired comparison"
    assert (paired["value"] == paired["difference"]).all(), "Paired value/difference changed"
    h2 = paired.loc[paired["population"].eq("h2_validation")]
    assert h2["n_clusters"].eq(21).all(), "H2 cluster count changed"
    assert h2.loc[h2["benchmark"].eq("naive_s0"), "verdict"].isin(VERDICT_WORDING).all(), "H2 V verdict missing"
    assert paired.loc[~(paired["population"].eq("h2_validation") & paired["benchmark"].eq("naive_s0")), "verdict"].eq("").all(), "Verdict outside H2 primary comparison"


def write_and_verify(result):
    assert_output(result)
    prior = None
    if OUTPUT_PATH.exists():
        prior = pd.read_parquet(OUTPUT_PATH)
        assert_output(prior)
        pd.testing.assert_frame_equal(prior, result, check_exact=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, prefix=".har_vol_comparison_", suffix=".parquet", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        result.to_parquet(temporary_path, index=False)
        round_trip = pd.read_parquet(temporary_path)
        assert_output(round_trip)
        pd.testing.assert_frame_equal(round_trip, result, check_exact=True)
        os.replace(temporary_path, OUTPUT_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    return prior is not None


def main():
    forecasts, common = load_frozen_forecasts_and_common()
    print(f"Frozen forecast fingerprint: {EXPECTED_FORECAST_FINGERPRINT} (PASS before target load)")
    scored, structural = load_target_and_align(forecasts, common)
    reference = fit_window_means(structural)
    result = score(scored, reference)
    deterministic_prior = write_and_verify(result)
    print("Frozen forecast population, target alignment, fit means, 21 H2 clusters, paired MSE identities, and round trip: PASS")
    print("H2 target-bearing counts: T-10=1,644; T-5=1,588")
    for horizon in (10, 5):
        section = result.loc[result["population"].eq("h2_validation") & result["horizon_minutes"].eq(horizon)]
        verdict = section.loc[section["benchmark"].eq("naive_s0") & section["metric"].eq("paired_squared_log_loss_difference"), "verdict"].item()
        print(f"T-{horizon}: {VERDICT_WORDING[verdict]}")
    print(f"Comparison rows: {len(result)}; deterministic prior comparison: {'PASS' if deterministic_prior else 'first write'}")
    print(f"Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
