"""Score frozen Day 12 predictions against Stage 0 on identical row sets."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kalshist-matplotlib"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_calibration import build_reliability_table, probability_metrics, wilson_interval
from scripts.analyze_model_market_gap import PLATT_SCORES_PATH
from scripts.analyze_spreads import PRICE_BUCKET_LABELS, assign_price_buckets
from scripts.build_derived_features import OUTPUT_PATH as DERIVED_PATH
from scripts.check_market_features import brier_score
from scripts.evaluation_split import SPLIT_KEY, SPLIT_RANGES
from scripts.model_logistic import CONFIG_B1, CONFIG_M1, CONFIG_M2, GAP_PATH, PREDICTION_COLUMNS, PREDICTION_OUTPUT_PATH, prediction_fingerprint
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, HORIZONS, LOG_LOSS_EPSILON, ROW_KEY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/model_comparison.parquet"
PLOT_PATH = PROJECT_ROOT / "data/models/plots/logistic_vs_stage0_reliability.png"
FROZEN_FINGERPRINT = "78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512"
FROZEN_CANDIDATE = "5min_ewma_vol"
ABS_Z_EDGES = [0.0, 0.25, 0.5, 1.0, np.inf]
ABS_Z_LABELS = ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 1.0)", "[1.0, infinity)"]
METRICS = ("brier", "log_loss", "auc", "ece")
OUTPUT_COLUMNS = [
    "configuration", "comparison_window", "horizon_minutes", "breakdown", "breakdown_value", "metric",
    "model_value", "stage0_value", "market_value", "difference", "paired_se", "meaningful", "n", "thin",
]
WINDOWS = {
    CONFIG_M2: ("validation_primary",),
    CONFIG_M1: ("validation_primary", "train_secondary", "all_oof_descriptive"),
    CONFIG_B1: ("validation_primary", "train_secondary", "all_oof_descriptive"),
}
EXPECTED_WINDOW_ROWS = {"validation_primary": 3_334, "train_secondary": 3_678, "all_oof_descriptive": 7_012}
EXPECTED_COMPARISON_ROWS = 217


def load_frozen_predictions():
    rows = pd.read_parquet(PREDICTION_OUTPUT_PATH, columns=PREDICTION_COLUMNS)
    assert list(rows.columns) == PREDICTION_COLUMNS and len(rows) == 17_358, "Frozen prediction schema/count changed"
    assert prediction_fingerprint(rows) == FROZEN_FINGERPRINT, "STOP: Day 12 prediction fingerprint changed"
    assert not rows.duplicated(["configuration"] + ROW_KEY).any(), "Duplicate frozen prediction key"
    assert set(rows["configuration"]) == set(WINDOWS), "Frozen configurations changed"
    assert set(rows["horizon_minutes"]) == set(HORIZONS), "Frozen horizons changed"
    assert rows["split"].isin(("train", "validation")).all(), "A test split entered"
    assert rows[SPLIT_KEY].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "A test date entered"
    assert rows["y"].isin((0, 1)).all(), "Frozen target is not binary"
    for column in ("probability", "stage0_platt_probability"):
        values = rows[column].to_numpy(dtype=float)
        assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), f"Invalid {column}"
    counts = rows.groupby(["configuration", "split", "horizon_minutes"], observed=True).size().to_dict()
    for configuration in (CONFIG_M1, CONFIG_B1):
        for split in ("train", "validation"):
            for horizon in HORIZONS:
                expected = 1_846 if split == "train" and horizon == 10 else 1_832 if split == "train" else EXPECTED_COMMON_ROWS[(split, horizon)]
                assert counts[(configuration, split, horizon)] == expected, f"{configuration}/{split}/T-{horizon} count changed"
    for horizon in HORIZONS:
        assert counts[(CONFIG_M2, "validation", horizon)] == EXPECTED_COMMON_ROWS[("validation", horizon)], "M2 validation horizon count changed"
    assert rows.loc[rows["configuration"].eq(CONFIG_M2), "split"].eq("validation").all(), "M2 contains train scores"
    return rows


def join_context(predictions):
    """Verify carried Stage 0 values and join benchmark-only quote/z columns."""
    first, last = SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]
    baseline_columns = ROW_KEY + ["split", "platt_model_probability"]
    baseline = pd.read_parquet(GAP_PATH, columns=baseline_columns, filters=[("split", "in", ["train", "validation"])])
    assert len(baseline) == sum(EXPECTED_COMMON_ROWS.values()), "Frozen Stage 0 common population changed"
    assert baseline.groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON_ROWS, "Frozen Stage 0 split/horizon keys changed"
    assert baseline["split"].isin(("train", "validation")).all() and not baseline.duplicated(ROW_KEY).any(), "Invalid frozen Stage 0 keys"
    rows = predictions.merge(baseline, on=ROW_KEY + ["split"], how="left", validate="many_to_one", indicator=True)
    assert len(rows) == len(predictions) and rows["_merge"].eq("both").all(), "Model/Stage 0 key sets differ"
    assert np.max(np.abs(rows["stage0_platt_probability"] - rows["platt_model_probability"])) <= 1e-12, "Carried Stage 0 values changed"
    rows = rows.drop(columns=["_merge", "platt_model_probability"])

    z_columns = ROW_KEY + ["split", SPLIT_KEY, "is_common", "z_5min_ewma_vol"]
    derived = pd.read_parquet(DERIVED_PATH, columns=z_columns, filters=[("split", "in", ["train", "validation"])])
    assert derived["split"].isin(("train", "validation")).all(), "A test derived-feature row entered"
    derived = derived.loc[derived["is_common"]].drop(columns="is_common")
    assert len(derived) == sum(EXPECTED_COMMON_ROWS.values()) and not derived.duplicated(ROW_KEY).any(), "Derived common keys changed"
    rows = rows.merge(derived, on=ROW_KEY + ["split", SPLIT_KEY], how="left", validate="many_to_one", indicator=True)
    assert len(rows) == len(predictions) and rows["_merge"].eq("both").all(), "Model/derived key sets differ"
    rows = rows.drop(columns="_merge")
    assert np.isfinite(rows["z_5min_ewma_vol"].to_numpy(dtype=float)).all(), "Frozen z is not finite"
    rows["abs_z_bucket"] = pd.cut(rows["z_5min_ewma_vol"].abs(), bins=ABS_Z_EDGES, labels=ABS_Z_LABELS, right=False).astype("str")
    assert rows["abs_z_bucket"].isin(ABS_Z_LABELS).all(), "A frozen |z| bucket is missing"

    market_columns = ROW_KEY + [SPLIT_KEY, "quote_mid"]
    market = pd.read_parquet(MARKET_FEATURES_PATH, columns=market_columns, filters=[(SPLIT_KEY, ">=", first), (SPLIT_KEY, "<=", last)])
    assert len(market) == 14_358 and not market.duplicated(ROW_KEY).any(), "Train/validation market projection changed"
    assert market[SPLIT_KEY].between(first, last).all(), "A test market row entered"
    rows = rows.merge(market, on=ROW_KEY + [SPLIT_KEY], how="left", validate="many_to_one", indicator=True)
    assert len(rows) == len(predictions) and rows["_merge"].eq("both").all(), "Market benchmark join changed row keys"
    rows = rows.drop(columns="_merge")
    assert np.isfinite(rows["quote_mid"].to_numpy(dtype=float)).all() and rows["quote_mid"].between(0, 1).all(), "Invalid market probability"
    rows = assign_price_buckets(rows)
    assert rows["price_bucket"].isin(PRICE_BUCKET_LABELS).all(), "A frozen price bucket is missing"
    assert prediction_fingerprint(rows) == FROZEN_FINGERPRINT, "Context join changed frozen predictions"
    return rows


def window_rows(rows, configuration, window):
    assert window in WINDOWS[configuration], "Unsupported comparison window"
    subset = rows.loc[rows["configuration"].eq(configuration)]
    if window == "validation_primary":
        subset = subset.loc[subset["split"].eq("validation")]
        if configuration != CONFIG_M2:
            assert set(subset["fold"]) == {4, 5, 6}, "Validation fold membership changed"
    elif window == "train_secondary":
        subset = subset.loc[subset["split"].eq("train")]
        assert set(subset["fold"]) == {1, 2, 3}, "Train fold membership changed"
    else:
        assert set(subset["fold"]) == {1, 2, 3, 4, 5, 6}, "All-OOF fold membership changed"
    assert len(subset) == EXPECTED_WINDOW_ROWS[window], f"{configuration}/{window} row count changed"
    assert not subset.duplicated(ROW_KEY).any(), "Window contains duplicate market/horizon keys"
    return subset.copy()


def assert_identical_keys(rows, baseline_keys):
    model_keys = set(map(tuple, rows[ROW_KEY].itertuples(index=False, name=None)))
    stage0_keys = set(map(tuple, baseline_keys[ROW_KEY].itertuples(index=False, name=None)))
    assert len(model_keys) == len(rows) and model_keys == stage0_keys, "Model and Stage 0 row-key sets differ"


def reliability_table(rows, probability_column, horizon):
    """Use the frozen validation helper; extend its same qcut/Wilson rule to other windows."""
    if len(rows) == EXPECTED_COMMON_ROWS[("validation", horizon)] and rows["split"].eq("validation").all():
        return build_reliability_table(rows, horizon, prediction_column=probability_column)
    binned = rows[[probability_column, "y"]].copy()
    binned["decile"] = pd.qcut(binned[probability_column], 10, labels=False) + 1
    table = binned.groupby("decile", observed=True).agg(n=("y", "size"), mean_predicted=(probability_column, "mean"), yes_count=("y", "sum")).reset_index()
    table["observed_yes_rate"] = table["yes_count"] / table["n"]
    table["wilson_low"], table["wilson_high"] = wilson_interval(table["yes_count"], table["n"])
    assert table["decile"].tolist() == list(range(1, 11)) and table["n"].sum() == len(rows), "Ten-decile coverage changed"
    return table.drop(columns="yes_count")


def score_population(rows, probability_column, horizon):
    assert len(rows) and rows["horizon_minutes"].eq(horizon).all(), "Scoring population horizon changed"
    probabilities = rows[probability_column].reset_index(drop=True)
    targets = rows["y"].reset_index(drop=True)
    brier, log_loss, auc = probability_metrics(probabilities, targets)
    reliability = reliability_table(rows, probability_column, horizon)
    ece = float((reliability["n"] / len(rows) * (reliability["observed_yes_rate"] - reliability["mean_predicted"]).abs()).sum())
    assert np.isfinite([brier, log_loss, auc, ece]).all(), "Non-finite probability metric"
    return {"brier": brier, "log_loss": log_loss, "auc": auc, "ece": ece}, reliability


def paired_difference(model_probability, stage0_probability, targets):
    model = np.asarray(model_probability, dtype=float)
    baseline = np.asarray(stage0_probability, dtype=float)
    y = np.asarray(targets, dtype=float)
    assert len(model) == len(baseline) == len(y) and len(y) > 0, "Paired row count changed"
    differences = (model - y) ** 2 - (baseline - y) ** 2
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else np.nan
    meaningful = bool(abs(mean) > 2 * se) if np.isfinite(se) else pd.NA
    return mean, se, meaningful


def brier_only(rows, probability_column):
    score, n = brier_score(rows[probability_column].reset_index(drop=True), rows["y"].reset_index(drop=True))
    assert n == len(rows) and np.isfinite(score), "Brier calculation dropped rows"
    return float(score)


def metric_record(configuration, window, horizon, breakdown, value, metric, model_value, stage0_value, market_value, difference, paired_se, meaningful, n):
    return {
        "configuration": configuration, "comparison_window": window, "horizon_minutes": horizon,
        "breakdown": breakdown, "breakdown_value": value, "metric": metric,
        "model_value": model_value, "stage0_value": stage0_value, "market_value": market_value,
        "difference": difference, "paired_se": paired_se, "meaningful": meaningful,
        "n": n, "thin": n < 30,
    }


def score_brier_row(rows, configuration, window, horizon, breakdown, value):
    model_brier = brier_only(rows, "probability")
    stage0_brier = brier_only(rows, "stage0_platt_probability")
    market_brier = brier_only(rows, "quote_mid")
    difference, se, meaningful = paired_difference(rows["probability"], rows["stage0_platt_probability"], rows["y"])
    assert abs(difference - (model_brier - stage0_brier)) <= 1e-12, "Paired Brier difference changed"
    return metric_record(configuration, window, horizon, breakdown, value, "brier", model_brier, stage0_brier, market_brier, difference, se, meaningful, len(rows))


def score_window(rows, configuration, window, baseline_keys):
    subset = window_rows(rows, configuration, window)
    expected_stage0 = baseline_keys.merge(subset[ROW_KEY], on=ROW_KEY, how="inner", validate="one_to_one")
    assert_identical_keys(subset, expected_stage0)
    records = []
    reliability = {}
    for horizon in HORIZONS:
        group = subset.loc[subset["horizon_minutes"].eq(horizon)].copy()
        assert len(group) > 0, "Empty comparison horizon"
        model_scores, model_reliability = score_population(group, "probability", horizon)
        stage0_scores, stage0_reliability = score_population(group, "stage0_platt_probability", horizon)
        reliability[horizon] = (model_reliability, stage0_reliability)
        brier_record = score_brier_row(group, configuration, window, horizon, "overall", "overall")
        assert abs(brier_record["model_value"] - model_scores["brier"]) <= 1e-15, "Model Brier paths disagree"
        assert abs(brier_record["stage0_value"] - stage0_scores["brier"]) <= 1e-15, "Stage 0 Brier paths disagree"
        records.append(brier_record)
        for metric in ("ece", "log_loss", "auc"):
            records.append(metric_record(configuration, window, horizon, "overall", "overall", metric, model_scores[metric], stage0_scores[metric], np.nan, np.nan, np.nan, pd.NA, len(group)))
        for breakdown, column, labels in (("abs_z", "abs_z_bucket", ABS_Z_LABELS), ("price_bucket", "price_bucket", PRICE_BUCKET_LABELS)):
            assert group[column].isin(labels).all(), f"Unexpected {breakdown} label"
            total = 0
            for label in labels:
                bucket = group.loc[group[column].eq(label)]
                if bucket.empty:
                    continue
                total += len(bucket)
                records.append(score_brier_row(bucket, configuration, window, horizon, breakdown, label))
            assert total == len(group), f"{breakdown} buckets do not cover parent population"
    records.append(score_brier_row(subset, configuration, window, 0, "overall", "overall"))
    return records, reliability


def verify_stage0_validation(comparison, frozen_scores):
    expected_keys = set(map(tuple, frozen_scores[["candidate", "horizon_minutes"]].itertuples(index=False, name=None)))
    assert expected_keys == {(FROZEN_CANDIDATE, 10), (FROZEN_CANDIDATE, 5)}, "Frozen Stage 0 validation rows changed"
    for horizon in HORIZONS:
        frozen = frozen_scores.loc[frozen_scores["horizon_minutes"].eq(horizon)].iloc[0]
        assert int(frozen["n"]) == EXPECTED_COMMON_ROWS[("validation", horizon)], "Frozen validation count changed"
        rows = comparison.loc[
            comparison["configuration"].eq(CONFIG_M2) & comparison["comparison_window"].eq("validation_primary")
            & comparison["horizon_minutes"].eq(horizon) & comparison["breakdown"].eq("overall")
            & comparison["metric"].eq("brier")
        ]
        assert len(rows) == 1 and abs(float(rows["stage0_value"].iloc[0]) - float(frozen["calibrated_brier"])) <= 1e-12, f"Frozen Stage 0 T-{horizon} Brier did not reproduce"


def plot_reliability(rows):
    m2 = window_rows(rows, CONFIG_M2, "validation_primary")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    colors = {"M2": "#1665a5", "Stage 0": "#e3821c", "Market mid": "#528b57"}
    for axis, horizon in zip(axes, HORIZONS):
        horizon_rows = m2.loc[m2["horizon_minutes"].eq(horizon)]
        for label, column in (("M2", "probability"), ("Stage 0", "stage0_platt_probability"), ("Market mid", "quote_mid")):
            table = build_reliability_table(horizon_rows, horizon, prediction_column=column)
            lower = table["observed_yes_rate"] - table["wilson_low"]
            upper = table["wilson_high"] - table["observed_yes_rate"]
            axis.errorbar(table["mean_predicted"], table["observed_yes_rate"], yerr=np.vstack((lower, upper)), marker="o", markersize=3, linewidth=1.5, capsize=2, color=colors[label], label=label)
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", label="Perfect calibration")
        axis.set(title=f"T-{horizon} · validation common", xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed YES frequency")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Frozen M2, Stage 0, and market reliability")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    assert PLOT_PATH.exists() and PLOT_PATH.stat().st_size > 0, "Reliability plot was not saved"


def validate_comparison(comparison):
    assert list(comparison.columns) == OUTPUT_COLUMNS and len(comparison) == EXPECTED_COMPARISON_ROWS, "Comparison schema/count changed"
    key = ["configuration", "comparison_window", "horizon_minutes", "breakdown", "breakdown_value", "metric"]
    assert not comparison.duplicated(key).any(), "Duplicate comparison key"
    assert set(comparison["configuration"]) == set(WINDOWS), "Comparison configurations changed"
    assert set(comparison["comparison_window"]) == set(EXPECTED_WINDOW_ROWS), "Comparison windows changed"
    assert set(comparison["horizon_minutes"]) == {0, 10, 5}, "Comparison horizons changed"
    assert comparison["metric"].isin(METRICS).all() and comparison["breakdown"].isin(("overall", "abs_z", "price_bucket")).all(), "Unknown comparison metric/breakdown"
    assert comparison["n"].gt(0).all() and pd.api.types.is_bool_dtype(comparison["thin"]), "Invalid cell size/thin dtype"
    assert comparison["thin"].eq(comparison["n"].lt(30)).all(), "Thin-cell rule changed"
    assert np.isfinite(comparison[["model_value", "stage0_value"]].to_numpy(dtype=float)).all(), "Non-finite model/Stage 0 result"
    brier = comparison.loc[comparison["metric"].eq("brier")]
    assert np.isfinite(brier[["market_value", "difference"]].to_numpy(dtype=float)).all(), "Non-finite Brier context"
    assert brier["paired_se"].notna().eq(brier["n"].gt(1)).all(), "Paired SE availability changed"
    assert brier.loc[brier["n"].gt(1), "paired_se"].ge(0).all(), "Invalid paired SE"
    assert comparison.loc[comparison["metric"].ne("brier"), ["difference", "paired_se", "market_value"]].isna().all().all(), "Secondary metric has fabricated paired result"
    assert pd.api.types.is_bool_dtype(comparison["meaningful"]), "Meaningful dtype changed"


def print_summary(comparison):
    print(f"Frozen prediction fingerprint: {FROZEN_FINGERPRINT} (PASS)")
    print("PRIMARY M2 VALIDATION — paired Brier first")
    for horizon in HORIZONS:
        overall = comparison.loc[comparison["configuration"].eq(CONFIG_M2) & comparison["comparison_window"].eq("validation_primary") & comparison["horizon_minutes"].eq(horizon) & comparison["breakdown"].eq("overall")]
        b = overall.loc[overall["metric"].eq("brier")].iloc[0]
        ece = overall.loc[overall["metric"].eq("ece")].iloc[0]
        log_loss = overall.loc[overall["metric"].eq("log_loss")].iloc[0]
        auc = overall.loc[overall["metric"].eq("auc")].iloc[0]
        print(f"T-{horizon}: n={b.n:,}; model Brier={b.model_value:.9f}; Stage 0 Brier={b.stage0_value:.9f}; difference={b.difference:+.9f}; paired SE={b.paired_se:.9f}; 2SE={2*b.paired_se:.9f}; meaningful={bool(b.meaningful)}")
        print(f"  ECE model={ece.model_value:.9f}, Stage 0={ece.stage0_value:.9f}; log loss model={log_loss.model_value:.9f}, Stage 0={log_loss.stage0_value:.9f}")
    pooled = comparison.loc[comparison["configuration"].eq(CONFIG_M2) & comparison["comparison_window"].eq("validation_primary") & comparison["horizon_minutes"].eq(0) & comparison["breakdown"].eq("overall") & comparison["metric"].eq("brier")].iloc[0]
    print(f"Pooled M2 validation: n={pooled.n:,}; model Brier={pooled.model_value:.9f}; Stage 0 Brier={pooled.stage0_value:.9f}; difference={pooled.difference:+.9f}; paired SE={pooled.paired_se:.9f}; meaningful={bool(pooled.meaningful)}")
    print("T-10 M2 has lower Brier and higher ECE than Stage 0; these describe different aspects of the same frozen probabilities.")
    print("M2 VALIDATION |z| BREAKDOWN — Brier model / Stage 0 / paired difference")
    for horizon in HORIZONS:
        cells = comparison.loc[comparison["configuration"].eq(CONFIG_M2) & comparison["comparison_window"].eq("validation_primary") & comparison["horizon_minutes"].eq(horizon) & comparison["breakdown"].eq("abs_z")]
        for row in cells.itertuples(index=False):
            print(f"T-{horizon} {row.breakdown_value}: n={row.n:,}, {row.model_value:.9f} / {row.stage0_value:.9f} / {row.difference:+.9f}, thin={row.thin}")
    print("M2 VALIDATION MARKET CONTEXT")
    for horizon in HORIZONS:
        overall = comparison.loc[comparison["configuration"].eq(CONFIG_M2) & comparison["comparison_window"].eq("validation_primary") & comparison["horizon_minutes"].eq(horizon) & comparison["breakdown"].eq("overall") & comparison["metric"].eq("brier")].iloc[0]
        print(f"T-{horizon}: market Brier={overall.market_value:.9f}")
    print("M2 VALIDATION QUOTE-MID BUCKETS — Brier model / Stage 0 / paired difference")
    for horizon in HORIZONS:
        cells = comparison.loc[comparison["configuration"].eq(CONFIG_M2) & comparison["comparison_window"].eq("validation_primary") & comparison["horizon_minutes"].eq(horizon) & comparison["breakdown"].eq("price_bucket")]
        for row in cells.itertuples(index=False):
            print(f"T-{horizon} {row.breakdown_value}: n={row.n:,}, {row.model_value:.9f} / {row.stage0_value:.9f} / {row.difference:+.9f}, thin={row.thin}")
    print("M2 VALIDATION AUC — ranking diagnostic")
    for horizon in HORIZONS:
        auc = comparison.loc[comparison["configuration"].eq(CONFIG_M2) & comparison["comparison_window"].eq("validation_primary") & comparison["horizon_minutes"].eq(horizon) & comparison["breakdown"].eq("overall") & comparison["metric"].eq("auc")].iloc[0]
        print(f"T-{horizon}: model={auc.model_value:.9f}, Stage 0={auc.stage0_value:.9f}")
    print("M1 AND B1 CONTEXT — Brier model / frozen Stage 0 / paired difference")
    for configuration in (CONFIG_M1, CONFIG_B1):
        for window in WINDOWS[configuration]:
            for horizon in HORIZONS:
                row = comparison.loc[comparison["configuration"].eq(configuration) & comparison["comparison_window"].eq(window) & comparison["horizon_minutes"].eq(horizon) & comparison["breakdown"].eq("overall") & comparison["metric"].eq("brier")].iloc[0]
                caveat = " [Stage 0 calibrator in-sample]" if window == "train_secondary" else ""
                print(f"{configuration}/{window}/T-{horizon}: n={row.n:,}, {row.model_value:.9f} / {row.stage0_value:.9f} / {row.difference:+.9f}, SE={row.paired_se:.9f}, meaningful={bool(row.meaningful)}{caveat}")
    print(f"Stage 0 frozen validation Brier reproduction: PASS (absolute tolerance 1e-12)")
    print("Identical row keys, paired SE, log-loss clipping, ten-decile ECE, and thin-cell checks: PASS")
    print("Parquet round trip: PASS")
    print(f"Outputs: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}; {PLOT_PATH.relative_to(PROJECT_ROOT)}")


def main():
    predictions = load_frozen_predictions()  # Fingerprint gate precedes every metric.
    assert LOG_LOSS_EPSILON == 1e-15, "Frozen log-loss clip changed"
    rows = join_context(predictions)
    baseline_keys = rows.loc[rows["configuration"].eq(CONFIG_M1), ROW_KEY].drop_duplicates()
    assert len(baseline_keys) == 7_012, "Frozen Stage 0 scored-key universe changed"
    m2 = window_rows(rows, CONFIG_M2, "validation_primary")
    m1_validation = window_rows(rows, CONFIG_M1, "validation_primary")
    assert set(map(tuple, m2[ROW_KEY].itertuples(index=False, name=None))) == set(map(tuple, m1_validation[ROW_KEY].itertuples(index=False, name=None))), "M2 and M1 validation common keys differ"
    records = []
    for configuration in (CONFIG_M2, CONFIG_M1, CONFIG_B1):
        for window in WINDOWS[configuration]:
            window_records, _ = score_window(rows, configuration, window, baseline_keys)
            records.extend(window_records)
    comparison = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    for column in ("configuration", "comparison_window", "breakdown", "breakdown_value", "metric"):
        comparison[column] = comparison[column].astype("str")
    comparison["meaningful"] = comparison["meaningful"].astype("boolean")
    validate_comparison(comparison)
    frozen_columns = ["candidate", "horizon_minutes", "n", "calibrated_brier"]
    frozen = pd.read_parquet(PLATT_SCORES_PATH, columns=frozen_columns, filters=[("candidate", "==", FROZEN_CANDIDATE)])
    assert list(frozen.columns) == frozen_columns and len(frozen) == 2, "Frozen Stage 0 validation score projection changed"
    verify_stage0_validation(comparison, frozen)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(OUTPUT_PATH, index=False)
    saved = pd.read_parquet(OUTPUT_PATH)
    validate_comparison(saved)
    pd.testing.assert_frame_equal(saved, comparison, check_exact=True)
    plot_reliability(rows)
    assert prediction_fingerprint(pd.read_parquet(PREDICTION_OUTPUT_PATH, columns=PREDICTION_COLUMNS)) == FROZEN_FINGERPRINT, "Frozen predictions changed after scoring"
    print_summary(saved)
    return saved


if __name__ == "__main__":
    main()
