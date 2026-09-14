from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_market_features import brier_score, rank_auc


PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/stage0_scores.parquet"
ROW_KEY = ["ticker", "horizon_minutes"]
SCORING_SPLITS = ("train", "validation")
HORIZONS = (10, 5)
BASE_RATES = {"train": 0.490377, "validation": 0.489662}
LOG_LOSS_EPSILON = 1e-15
EXPECTED_ROWS = 17_182
EXPECTED_SCORING_ROWS = 14_358
EXPECTED_SCORE_ROWS = 80
EXPECTED_HORIZON_ROWS = {"train": 5_196, "validation": 1_983}
EXPECTED_COMMON_ROWS = {
    ("train", 10): 4_792,
    ("train", 5): 4_757,
    ("validation", 10): 1_685,
    ("validation", 5): 1_649,
}
FROZEN_TRAIN_FIELDS = (
    "n", "brier", "log_loss", "auc", "auc_log_moneyness", "brier_constant", "brier_market", "n_saturated"
)
FROZEN_TRAIN_RESULTS = {
    ("5min_vol", 10): (4910, 0.203782, 0.594685, 0.777383, 0.774910, 0.249916, 0.186683, 0),
    ("5min_vol", 5): (4863, 0.143036, 0.451740, 0.911452, 0.908648, 0.249893, 0.111385, 0),
    ("5min_ewma_vol", 10): (4910, 0.203371, 0.593816, 0.777906, 0.774910, 0.249916, 0.186683, 0),
    ("5min_ewma_vol", 5): (4863, 0.142541, 0.450212, 0.911306, 0.908648, 0.249893, 0.111385, 0),
    ("15min_vol", 10): (4917, 0.203380, 0.593614, 0.777829, 0.774925, 0.249911, 0.186724, 0),
    ("15min_vol", 5): (4890, 0.144111, 0.454602, 0.911502, 0.908293, 0.249884, 0.111855, 0),
    ("15min_ewma_vol", 10): (4917, 0.203447, 0.593876, 0.777901, 0.774925, 0.249911, 0.186724, 0),
    ("15min_ewma_vol", 5): (4890, 0.144007, 0.454444, 0.911490, 0.908293, 0.249884, 0.111855, 0),
    ("1hr_vol", 10): (4929, 0.203722, 0.594287, 0.777499, 0.774443, 0.249903, 0.186719, 0),
    ("1hr_vol", 5): (4903, 0.144864, 0.456210, 0.911879, 0.908232, 0.249890, 0.111623, 0),
    ("1hr_ewma_vol", 10): (4929, 0.203712, 0.594330, 0.777596, 0.774443, 0.249903, 0.186719, 0),
    ("1hr_ewma_vol", 5): (4903, 0.144760, 0.456113, 0.911977, 0.908232, 0.249890, 0.111623, 0),
    ("4hr_vol", 10): (4927, 0.204097, 0.595227, 0.777414, 0.774671, 0.249923, 0.186665, 0),
    ("4hr_vol", 5): (4906, 0.146026, 0.458845, 0.910908, 0.908050, 0.249900, 0.111804, 0),
    ("4hr_ewma_vol", 10): (4927, 0.204046, 0.595119, 0.777559, 0.774671, 0.249923, 0.186665, 0),
    ("4hr_ewma_vol", 5): (4906, 0.145827, 0.458483, 0.911225, 0.908050, 0.249900, 0.111804, 0),
    ("24hr_vol", 10): (4888, 0.204406, 0.595768, 0.777078, 0.774943, 0.249935, 0.186556, 0),
    ("24hr_vol", 5): (4869, 0.147465, 0.461946, 0.909606, 0.908044, 0.249913, 0.111580, 1),
    ("24hr_ewma_vol", 10): (4888, 0.204388, 0.595758, 0.777183, 0.774943, 0.249935, 0.186556, 0),
    ("24hr_ewma_vol", 5): (4869, 0.147336, 0.461732, 0.909729, 0.908044, 0.249913, 0.111580, 1),
}


def common_prediction_mask(rows, prediction_columns):
    """Select rows with predictions from every Stage 0 candidate."""
    missing_columns = [column for column in prediction_columns if column not in rows.columns]
    assert not missing_columns, f"Missing Stage 0 probability columns: {missing_columns}"
    return rows[prediction_columns].notna().all(axis=1)


def binary_log_loss(probabilities, targets):
    clipped = probabilities.clip(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
    loss = -np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))
    return float(loss), int(probabilities.ne(clipped).sum())


def load_stage0_rows():
    prediction_keys = pd.read_parquet(PREDICTIONS_PATH, columns=ROW_KEY + ["split"])
    assert len(prediction_keys) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} prediction rows, found {len(prediction_keys):,}"
    assert not prediction_keys.duplicated(ROW_KEY).any(), f"Prediction row key {ROW_KEY} must be unique"

    features = pd.read_parquet(FEATURES_PATH, columns=ROW_KEY + ["quote_mid"])
    assert len(features) == len(prediction_keys), f"Expected {len(prediction_keys):,} market-feature rows, found {len(features):,}"
    assert not features.duplicated(ROW_KEY).any(), f"Market-feature row key {ROW_KEY} must be unique"

    row_count = len(prediction_keys)
    merged_keys = prediction_keys.merge(features, on=ROW_KEY, how="left", validate="one_to_one", indicator=True)
    assert len(merged_keys) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} merged rows, found {len(merged_keys):,}"
    assert len(merged_keys) == row_count, f"Merge changed row count from {row_count:,} to {len(merged_keys):,}"
    assert merged_keys["_merge"].eq("both").all(), "Some prediction rows did not match market features on the full row key"
    merged_keys = merged_keys.drop(columns="_merge")
    assert not merged_keys.duplicated(ROW_KEY).any(), f"Merged row key {ROW_KEY} must be unique"
    assert merged_keys["quote_mid"].notna().all(), f"quote_mid contains {merged_keys['quote_mid'].isna().sum():,} NaN values"

    scoring_rows = pd.read_parquet(PREDICTIONS_PATH, filters=[("split", "in", list(SCORING_SPLITS))])
    prediction_columns = [column for column in scoring_rows.columns if column.startswith("p_")]
    assert len(prediction_columns) == 10, f"Expected 10 p_* columns, found {len(prediction_columns)}: {prediction_columns}"
    assert len(scoring_rows) == EXPECTED_SCORING_ROWS, f"Expected {EXPECTED_SCORING_ROWS:,} scoring rows, found {len(scoring_rows):,}"
    assert scoring_rows["split"].isin(SCORING_SPLITS).all(), "Scoring rows contain an unsupported split"
    assert not scoring_rows["split"].eq("test").any(), "Test rows must be excluded before outcomes are read"
    assert scoring_rows["y"].notna().all(), "Train/validation scoring rows contain null y values"
    assert scoring_rows["y"].isin([0, 1]).all(), "Train/validation scoring rows contain invalid y values"
    scoring_rows = scoring_rows.merge(merged_keys[ROW_KEY + ["quote_mid"]], on=ROW_KEY, how="left", validate="one_to_one")
    assert len(scoring_rows) == EXPECTED_SCORING_ROWS, "Adding quote_mid changed the train/validation row count"
    assert scoring_rows["quote_mid"].notna().all(), "Train/validation scoring rows contain null quote_mid values"
    return scoring_rows, prediction_columns


def build_common_populations():
    scoring_rows, prediction_columns = load_stage0_rows()
    assert scoring_rows["split"].isin(SCORING_SPLITS).all(), "Scoring population contains an unsupported split"
    assert not scoring_rows["split"].eq("test").any(), "Test rows must be excluded from scoring machinery"

    populations = {}
    summary_rows = []
    for split, horizon in EXPECTED_COMMON_ROWS:
        subset = scoring_rows.loc[scoring_rows["split"].eq(split) & scoring_rows["horizon_minutes"].eq(horizon)]
        mask = common_prediction_mask(subset, prediction_columns)
        common = subset.loc[mask].copy()
        actual_rows = len(common)
        expected_rows = EXPECTED_COMMON_ROWS[(split, horizon)]
        assert actual_rows == expected_rows, f"Expected {expected_rows:,} common rows for {split} T-{horizon}, found {actual_rows:,}"
        assert common["y"].notna().all(), f"{split} T-{horizon} common rows contain null y values"
        assert common["y"].isin([0, 1]).all(), f"{split} T-{horizon} common rows contain y values outside {{0, 1}}"
        assert common[prediction_columns].notna().all().all(), f"{split} T-{horizon} common rows contain null predictions"
        assert not common["split"].eq("test").any(), f"{split} T-{horizon} common rows contain test data"

        candidate_counts = subset[prediction_columns].notna().sum()
        populations[(split, horizon)] = common
        summary_rows.append({
            "split": split,
            "horizon": f"T-{horizon}",
            "common_rows": actual_rows,
            "total_rows": len(subset),
            "coverage_pct": 100.0 * actual_rows / len(subset),
            "min_candidate_non_null": int(candidate_counts.min()),
            "max_candidate_non_null": int(candidate_counts.max()),
        })

    summary = pd.DataFrame(summary_rows)
    print("Stage 0 common-row population (train/validation only)")
    print(summary.to_string(index=False, formatters={"coverage_pct": "{:.2f}%".format}))
    return scoring_rows, populations, summary, prediction_columns


def score_population(split, horizon, candidate, population, probability_column, rows, total_rows):
    assert split in BASE_RATES, f"No frozen base rate for split: {split}"
    assert population in {"candidate", "common"}, f"Unsupported population: {population}"
    assert len(rows) > 0, f"Cannot score an empty {split} T-{horizon} {candidate} {population} population"
    assert rows["split"].eq(split).all(), f"{split} T-{horizon} {candidate} rows contain another split"
    assert not rows["split"].eq("test").any(), "Test rows entered Stage 0 scoring"
    assert rows["horizon_minutes"].eq(horizon).all(), f"{split} T-{horizon} {candidate} rows contain another horizon"
    assert rows["y"].notna().all(), f"{split} T-{horizon} {candidate} {population} rows contain null y values"
    assert rows["y"].isin([0, 1]).all(), f"{split} T-{horizon} {candidate} {population} rows contain invalid y values"
    assert rows[probability_column].notna().all(), f"{split} T-{horizon} {candidate} {population} rows contain null predictions"
    assert rows[probability_column].between(0, 1).all(), f"{split} T-{horizon} {candidate} probabilities fall outside [0, 1]"
    assert rows["quote_mid"].notna().all(), f"{split} T-{horizon} {candidate} {population} rows contain null quote_mid"

    n = len(rows)
    probabilities = rows[probability_column]
    targets = rows["y"]
    brier, brier_n = brier_score(probabilities, targets)
    log_loss, n_clipped = binary_log_loss(probabilities, targets)
    auc, auc_n = rank_auc(probabilities, targets)
    auc_log_moneyness, log_moneyness_n = rank_auc(rows["log_moneyness"], targets)

    constant = pd.Series(BASE_RATES[split], index=rows.index)
    brier_constant, constant_n = brier_score(constant, targets)
    subset_base_rate = float(targets.mean())
    # The exact decomposition uses Bernoulli variance at the subset rate, plus squared bias from the frozen rate.
    constant_identity = subset_base_rate * (1.0 - subset_base_rate) + (BASE_RATES[split] - subset_base_rate) ** 2
    assert abs(brier_constant - constant_identity) <= 1e-6, f"Constant Brier identity failed for {split} T-{horizon} {candidate}"

    brier_market, market_brier_n = brier_score(rows["quote_mid"], targets)
    log_loss_market, _ = binary_log_loss(rows["quote_mid"], targets)
    metric_counts = {brier_n, auc_n, log_moneyness_n, constant_n, market_brier_n}
    assert metric_counts == {n}, f"Metrics did not use the same {n:,}-row population for {split} T-{horizon} {candidate}"
    assert not np.isclose(auc, auc_log_moneyness, rtol=0.0, atol=np.finfo(float).eps), (
        f"Stage 0 and log-moneyness AUC are identical to machine precision for {split} T-{horizon} {candidate} {population}"
    )

    return {
        "split": split,
        "horizon_minutes": horizon,
        "candidate": candidate,
        "population": population,
        "n": n,
        "coverage": n / total_rows,
        "brier": brier,
        "log_loss": log_loss,
        "n_clipped": n_clipped,
        "n_saturated": int(probabilities.isin([0.0, 1.0]).sum()),
        "auc": auc,
        "auc_log_moneyness": auc_log_moneyness,
        "brier_constant": brier_constant,
        "brier_market": brier_market,
        "log_loss_market": log_loss_market,
        "subset_base_rate": subset_base_rate,
    }


def score_split_population(split, population, scoring_rows, common_populations, prediction_columns):
    results = []
    for horizon in HORIZONS:
        subset = scoring_rows.loc[scoring_rows["split"].eq(split) & scoring_rows["horizon_minutes"].eq(horizon)]
        total_rows = len(subset)
        assert total_rows == EXPECTED_HORIZON_ROWS[split], f"Expected {EXPECTED_HORIZON_ROWS[split]:,} {split} T-{horizon} rows, found {total_rows:,}"
        for probability_column in prediction_columns:
            candidate = probability_column.removeprefix("p_")
            rows = subset.loc[subset[probability_column].notna()] if population == "candidate" else common_populations[(split, horizon)]
            results.append(score_population(split, horizon, candidate, population, probability_column, rows, total_rows))
    return pd.DataFrame(results)


def assert_frozen_train_results(train_candidate_scores):
    actual_keys = set(zip(train_candidate_scores["candidate"], train_candidate_scores["horizon_minutes"]))
    assert len(train_candidate_scores) == 20, f"Expected 20 train candidate score rows, found {len(train_candidate_scores)}"
    assert actual_keys == set(FROZEN_TRAIN_RESULTS), "Train candidate/horizon keys differ from the frozen Day 8 table"

    indexed = train_candidate_scores.set_index(["candidate", "horizon_minutes"])
    integer_fields = {"n", "n_saturated"}
    for key, expected_values in FROZEN_TRAIN_RESULTS.items():
        for field, expected in zip(FROZEN_TRAIN_FIELDS, expected_values):
            actual = indexed.loc[key, field]
            if field in integer_fields:
                assert int(actual) == expected, f"Frozen train mismatch for {key} {field}: expected {expected}, found {actual}"
            else:
                rounded = round(float(actual), 6)
                assert abs(rounded - expected) <= 1e-9, (
                    f"Frozen train mismatch for {key} {field}: expected {expected:.6f}, found {actual:.12f}"
                )


def print_train_summary(train_scores):
    summary = train_scores.groupby(["horizon_minutes", "population"], sort=False).agg(
        candidates=("candidate", "size"), n_min=("n", "min"), n_max=("n", "max"),
        coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
        brier_min=("brier", "min"), brier_max=("brier", "max"),
    ).reset_index()
    print("\nTrain score summary")
    print(summary.to_string(index=False, formatters={
        "coverage_min": "{:.2%}".format,
        "coverage_max": "{:.2%}".format,
        "brier_min": "{:.6f}".format,
        "brier_max": "{:.6f}".format,
    }))


def print_validation_scores(validation_scores, prediction_columns):
    candidate_order = {column.removeprefix("p_"): order for order, column in enumerate(prediction_columns)}
    table = validation_scores.assign(candidate_order=validation_scores["candidate"].map(candidate_order))
    table = table.sort_values(["horizon_minutes", "candidate_order", "population"], ascending=[False, True, False])
    columns = [
        "candidate", "horizon_minutes", "population", "n", "coverage", "brier", "log_loss", "auc",
        "brier_constant", "brier_market", "n_clipped", "n_saturated",
    ]
    print("\nValidation Stage 0 scores")
    print(table[columns].to_string(index=False, formatters={
        "coverage": "{:.2%}".format,
        "brier": "{:.6f}".format,
        "log_loss": "{:.6f}".format,
        "auc": "{:.6f}".format,
        "brier_constant": "{:.6f}".format,
        "brier_market": "{:.6f}".format,
    }))


def score_stage0():
    scoring_rows, common_populations, _, prediction_columns = build_common_populations()

    train_candidate_scores = score_split_population("train", "candidate", scoring_rows, common_populations, prediction_columns)
    assert_frozen_train_results(train_candidate_scores)
    print("\nTrain regression test: PASS (all 20 frozen Day 8 candidate/horizon rows)")

    train_common_scores = score_split_population("train", "common", scoring_rows, common_populations, prediction_columns)
    train_scores = pd.concat([train_candidate_scores, train_common_scores], ignore_index=True)
    print_train_summary(train_scores)

    validation_candidate_scores = score_split_population("validation", "candidate", scoring_rows, common_populations, prediction_columns)
    validation_common_scores = score_split_population("validation", "common", scoring_rows, common_populations, prediction_columns)
    validation_scores = pd.concat([validation_candidate_scores, validation_common_scores], ignore_index=True)
    assert validation_scores["n_clipped"].eq(0).all(), "Validation contains clipped Stage 0 probabilities"
    assert validation_scores["n_saturated"].eq(0).all(), "Validation contains saturated Stage 0 probabilities"
    print_validation_scores(validation_scores, prediction_columns)

    scores = pd.concat([train_scores, validation_scores], ignore_index=True)
    assert len(scores) == EXPECTED_SCORE_ROWS, f"Expected {EXPECTED_SCORE_ROWS} score rows, found {len(scores)}"
    assert not scores.duplicated(["split", "horizon_minutes", "candidate", "population"]).any(), "Score-table key must be unique"
    assert scores["split"].isin(SCORING_SPLITS).all(), "Score table contains an unsupported split"
    assert not scores["split"].eq("test").any(), "Score table contains test results"
    assert scores["n"].gt(0).all(), "Score table contains an empty population"
    assert scores["coverage"].between(0, 1).all(), "Score table contains invalid coverage"
    common_scores = scores.loc[scores["population"].eq("common")]
    for key, expected_rows in EXPECTED_COMMON_ROWS.items():
        actual_counts = set(common_scores.loc[common_scores["split"].eq(key[0]) & common_scores["horizon_minutes"].eq(key[1]), "n"])
        assert actual_counts == {expected_rows}, f"Output common-row count mismatch for {key}: {actual_counts}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(scores)} score rows to {OUTPUT_PATH}")
    return scores


if __name__ == "__main__":
    score_stage0()
