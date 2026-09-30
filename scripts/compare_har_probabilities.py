"""Score the frozen Day 13 sigma swap without changing its models or populations."""

import hashlib
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

from scripts.analyze_calibration import PLATT_CLIP, build_reliability_table, fit_platt, platt_feature, stable_sigmoid, wilson_interval
from scripts.analyze_spreads import PRICE_BUCKET_LABELS, assign_price_buckets
from scripts.compare_models import ABS_Z_EDGES, ABS_Z_LABELS, brier_only, paired_difference, score_population
from scripts.evaluation_split import SPLIT_RANGES
from scripts.model_har_rv import CONFIG_H1, CONFIG_H2, EXPECTED_HORIZON_SCORE_ROWS, PROBABILITY_COLUMNS, PROBABILITY_PATH, SCORE_KEY, assert_probabilities
from scripts.model_logistic import prediction_fingerprint
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, ROW_KEY
from scripts.stage0_baseline import stage0_probability


DERIVED_PATH = PROJECT_ROOT / "data/features/derived_features.parquet"
MARKET_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
TARGET_PATH = PROJECT_ROOT / "data/targets/forward_vol_target.parquet"
STAGE0_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
GAP_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
B1_PATH = PROJECT_ROOT / "data/models/logistic_oof_predictions.parquet"
FROZEN_SCORES_PATH = PROJECT_ROOT / "data/models/stage0_platt_validation_scores.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/har_probability_comparison.parquet"
PLOT_PATH = PROJECT_ROOT / "data/models/plots/har_vs_stage0_reliability.png"
FROZEN_FINGERPRINT = "e864a6377c8d520b04eae3dfb6664feaacdb5c35cb9708a6c0755b741d17fa0b"
FROZEN_B1_FINGERPRINT = "78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512"
FROZEN_CANDIDATE = "5min_ewma_vol"
ORACLE_VALIDATION_ROWS = {10: 1_644, 5: 1_588}
NEAR_ZERO = 1e-12
KEY = ["configuration", "comparison_window", "horizon_minutes", "breakdown", "breakdown_value", "metric"]
OUTPUT_COLUMNS = KEY + ["model_value", "stage0_value", "market_value", "oracle_value", "difference", "paired_se", "meaningful", "n", "thin", "verdict", "numerator", "denominator", "unstable"]
P_WORDING = {
    "P4": "P4 — worse",
    "P1": "P1 — improves",
    "P2": "P2 — mixed",
    "P3": "P3 — no meaningful difference: Forecast σ did not improve Stage 0's probabilities; Stage 0 remains the baseline of record.",
}


def keys(rows):
    assert not rows.duplicated(ROW_KEY).any(), "Duplicate market/horizon key"
    return set(map(tuple, rows[ROW_KEY].itertuples(index=False, name=None)))


def exact_keys(left, right, label):
    assert len(left) == len(right) and keys(left) == keys(right), f"{label}: paired keys differ"


def load_frozen_probabilities():
    assert pq.ParquetFile(PROBABILITY_PATH).metadata.num_rows == 10_346, "HAR probability artifact row count changed"
    rows = pd.read_parquet(PROBABILITY_PATH, columns=PROBABILITY_COLUMNS, filters=[("close_date", "<=", SPLIT_RANGES["validation"][1])])
    assert_probabilities(rows)
    assert prediction_fingerprint(rows) == FROZEN_FINGERPRINT, "STOP: frozen HAR probability fingerprint changed"
    assert len(rows) == 10_346 and not rows.duplicated(SCORE_KEY).any(), "Frozen HAR probability keys changed"
    assert rows.groupby(["configuration", "horizon_minutes"], observed=True).size().to_dict() == {
        (configuration, horizon): count for configuration, counts in EXPECTED_HORIZON_SCORE_ROWS.items() for horizon, count in counts.items()
    }, "H1/H2 horizon populations changed"
    assert rows["split"].isin(("train", "validation")).all() and rows["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "Test split/date entered"
    assert rows.loc[rows["configuration"].eq(CONFIG_H2), "split"].eq("validation").all(), "H2 contains a non-validation row"
    assert set(rows.loc[rows["configuration"].eq(CONFIG_H1), "fold"]) == set(range(1, 7)), "H1 fold IDs changed"
    assert rows["y"].isin((0, 1)).all(), "Non-binary outcome"
    h1 = rows.loc[rows["configuration"].eq(CONFIG_H1)]
    h2 = rows.loc[rows["configuration"].eq(CONFIG_H2)]
    assert len(h1) == 7_012 and len(h2) == 3_334, "H1/H2 total count changed"
    assert h1["b1_probability"].notna().all() and h2["b1_probability"].isna().all(), "B1 presence changed"
    for column in ("probability", "probability_n1_control", "stage0_platt_probability"):
        values = rows[column].to_numpy(dtype=float)
        assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), f"Invalid {column}"
    values = h1["b1_probability"].to_numpy(dtype=float)
    assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), "Invalid B1 probability"
    return rows


def join_context(probabilities):
    derived_columns = ROW_KEY + ["split", "close_date", "is_common", "z_5min_ewma_vol"]
    derived = pd.read_parquet(DERIVED_PATH, columns=derived_columns, filters=[("split", "in", ["train", "validation"])])
    assert list(derived.columns) == derived_columns and len(derived) == 14_358 and not derived.duplicated(ROW_KEY).any(), "Derived projection/population changed"
    assert derived["split"].isin(("train", "validation")).all(), "Test derived row loaded"
    common = derived.loc[derived["is_common"], ROW_KEY + ["split", "close_date", "z_5min_ewma_vol"]].copy()
    assert common.groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON_ROWS, "Day 9 common population changed"
    assert np.isfinite(common["z_5min_ewma_vol"].to_numpy(dtype=float)).all(), "Frozen Stage 0 z is invalid"
    market_columns = ROW_KEY + ["close_date", "quote_mid", "log_moneyness", "T_years"]
    market = pd.read_parquet(MARKET_PATH, columns=market_columns, filters=[("close_date", "<=", SPLIT_RANGES["validation"][1])])
    assert list(market.columns) == market_columns and len(market) == 14_358 and not market.duplicated(ROW_KEY).any(), "Market projection/population changed"
    assert market["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), "Test market row loaded"
    context = common.merge(market, on=ROW_KEY + ["close_date"], how="left", validate="one_to_one", indicator=True)
    assert len(context) == len(common) and context["_merge"].eq("both").all(), "Common/market keys changed"
    context = context.drop(columns="_merge")
    assert np.isfinite(context[["quote_mid", "log_moneyness", "T_years"]].to_numpy(dtype=float)).all(), "Invalid market context"
    assert context["quote_mid"].between(0, 1).all() and context["T_years"].gt(0).all(), "Invalid price/time context"
    context["abs_z_bucket"] = pd.cut(context["z_5min_ewma_vol"].abs(), bins=ABS_Z_EDGES, labels=ABS_Z_LABELS, right=False).astype("str")
    assert context["abs_z_bucket"].isin(ABS_Z_LABELS).all(), "Frozen Stage 0 z buckets changed"
    context = assign_price_buckets(context)
    assert context["price_bucket"].isin(PRICE_BUCKET_LABELS).all(), "Frozen price buckets changed"
    joined = []
    for configuration in (CONFIG_H1, CONFIG_H2):
        part = probabilities.loc[probabilities["configuration"].eq(configuration)]
        expected = context.loc[context["split"].eq("validation")] if configuration == CONFIG_H2 else context.loc[context["close_date"].between("2026-06-29", "2026-08-09")]
        exact_keys(part, expected, f"{configuration}/context")
        combined = part.merge(context, on=ROW_KEY + ["split", "close_date"], how="left", validate="one_to_one", indicator=True)
        assert len(combined) == len(part) and combined["_merge"].eq("both").all(), "Probability/context structural join changed"
        joined.append(combined.drop(columns="_merge"))
    rows = pd.concat(joined, ignore_index=True)
    assert len(rows) == len(probabilities) and prediction_fingerprint(rows) == FROZEN_FINGERPRINT, "Context join changed frozen probabilities"
    return rows, context


def verify_baselines(rows):
    h2 = rows.loc[rows["configuration"].eq(CONFIG_H2)].copy()
    gap_columns = ROW_KEY + ["split", "platt_model_probability"]
    gap = pd.read_parquet(GAP_PATH, columns=gap_columns, filters=[("split", "==", "validation")])
    exact_keys(h2, gap, "H2/frozen Stage 0")
    checked = h2.merge(gap, on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert checked["_merge"].eq("both").all(), "Frozen Stage 0 rows missing"
    carry_error = float(np.max(np.abs(checked["stage0_platt_probability"] - checked["platt_model_probability"])))
    assert carry_error <= 1e-12, "Carried frozen Stage 0 probability changed"
    frozen_columns = ["candidate", "horizon_minutes", "n", "calibrated_brier"]
    frozen = pd.read_parquet(FROZEN_SCORES_PATH, columns=frozen_columns, filters=[("candidate", "==", FROZEN_CANDIDATE)])
    assert len(frozen) == 2 and set(frozen["horizon_minutes"]) == {10, 5}, "Frozen Stage 0 score rows changed"
    errors = []
    for horizon in (10, 5):
        section = h2.loc[h2["horizon_minutes"].eq(horizon)]
        expected = frozen.loc[frozen["horizon_minutes"].eq(horizon)].iloc[0]
        assert len(section) == int(expected["n"]) == EXPECTED_COMMON_ROWS[("validation", horizon)], "Frozen Stage 0 validation count changed"
        error = abs(brier_only(section, "stage0_platt_probability") - float(expected["calibrated_brier"]))
        assert error <= 1e-12, f"STOP: frozen Stage 0 T-{horizon} Brier reproduction failed"
        errors.append(error)
    if 0 in set(frozen["horizon_minutes"]):
        pooled = frozen.loc[frozen["horizon_minutes"].eq(0)].iloc[0]
        error = abs(brier_only(h2, "stage0_platt_probability") - float(pooled["calibrated_brier"]))
        assert error <= 1e-12, "STOP: frozen pooled Stage 0 Brier reproduction failed"
        errors.append(error)
    b1_columns = ["configuration"] + ROW_KEY + ["split", "close_date", "fold", "probability", "stage0_platt_probability", "y"]
    b1_all = pd.read_parquet(B1_PATH, columns=b1_columns, filters=[("split", "in", ["train", "validation"])])
    assert len(b1_all) == 17_358 and prediction_fingerprint(b1_all) == FROZEN_B1_FINGERPRINT, "Frozen Day 12 B1 source changed"
    b1 = b1_all.loc[b1_all["configuration"].eq("b1_walk_forward_stage0")]
    h1 = rows.loc[rows["configuration"].eq(CONFIG_H1)]
    exact_keys(h1, b1, "H1/frozen B1")
    aligned = h1.merge(b1[ROW_KEY + ["split", "close_date", "fold", "probability", "y"]], on=ROW_KEY + ["split", "close_date", "fold", "y"], how="left", validate="one_to_one", indicator=True)
    assert aligned["_merge"].eq("both").all(), "B1 structure/outcome mismatch"
    assert np.max(np.abs(aligned["b1_probability"] - aligned["probability_y"])) == 0, "Carried B1 probability changed"
    return max(errors), carry_error


def record(configuration, window, horizon, breakdown, bucket, metric, n, *, model=np.nan, stage0=np.nan, market=np.nan, oracle=np.nan,
           difference=np.nan, paired_se=np.nan, meaningful=pd.NA, verdict="", numerator=np.nan, denominator=np.nan, unstable=pd.NA):
    return {"configuration": configuration, "comparison_window": window, "horizon_minutes": horizon, "breakdown": breakdown,
            "breakdown_value": bucket, "metric": metric, "model_value": model, "stage0_value": stage0, "market_value": market,
            "oracle_value": oracle, "difference": difference, "paired_se": paired_se, "meaningful": meaningful,
            "n": n, "thin": n < 30, "verdict": verdict, "numerator": numerator, "denominator": denominator, "unstable": unstable}


def paired_brier(rows, model_column, baseline_column, configuration, window, horizon, breakdown="overall", bucket="overall", market=False):
    assert not rows.empty and not rows.duplicated(ROW_KEY).any(), "Paired Brier contains empty/duplicate keys"
    model_keys = rows.loc[rows[model_column].notna(), ROW_KEY]
    baseline_keys = rows.loc[rows[baseline_column].notna(), ROW_KEY]
    exact_keys(model_keys, baseline_keys, "Paired Brier model/baseline")
    model_brier = brier_only(rows, model_column)
    stage0_brier = brier_only(rows, baseline_column)
    difference, se, meaningful = paired_difference(rows[model_column], rows[baseline_column], rows["y"])
    assert abs(difference - (model_brier - stage0_brier)) <= 1e-12, "Paired Brier/MSE identity failed"
    market_brier = brier_only(rows, "quote_mid") if market else np.nan
    return record(configuration, window, horizon, breakdown, bucket, "brier", len(rows), model=model_brier, stage0=stage0_brier,
                  market=market_brier, difference=difference, paired_se=se, meaningful=meaningful)


def p_category(by_horizon):
    states = {h: (bool(row["meaningful"]), float(row["difference"])) for h, row in by_horizon.items()}
    if any(meaningful and difference > 0 for meaningful, difference in states.values()):
        return "P4"
    wins = sum(meaningful and difference < 0 for meaningful, difference in states.values())
    return "P1" if wins == 2 else "P2" if wins == 1 else "P3"


def validate_reliability(table, rows, probability_column, horizon):
    assert len(table) == 10 and table["decile"].tolist() == list(range(1, 11)), "Reliability deciles changed"
    assert int(table["n"].sum()) == len(rows) and not rows.duplicated(ROW_KEY).any(), "Reliability lost/duplicated rows"
    assert rows["horizon_minutes"].eq(horizon).all() and rows["y"].isin((0, 1)).all(), "Reliability row population changed"
    assert rows[probability_column].between(0, 1).all(), "Reliability probability outside [0,1]"
    assert table["n"].gt(0).all() and table["observed_yes_rate"].between(0, 1).all(), "Invalid reliability outcomes"
    yes = table["observed_yes_rate"].to_numpy(dtype=float) * table["n"].to_numpy(dtype=float)
    assert np.allclose(yes, np.rint(yes), rtol=0, atol=1e-10) and int(np.rint(yes).sum()) == int(rows["y"].sum()), "Reliability outcome counts changed"
    low, high = wilson_interval(np.rint(yes).astype(int), table["n"].to_numpy(dtype=int))
    assert np.allclose(low, table["wilson_low"], rtol=0, atol=1e-12) and np.allclose(high, table["wilson_high"], rtol=0, atol=1e-12), "Wilson interval mismatch"
    assert ((0 <= low) & (low <= table["observed_yes_rate"]) & (table["observed_yes_rate"] <= high) & (high <= 1)).all(), "Invalid Wilson bounds"


def score_h2(rows):
    h2 = rows.loc[rows["configuration"].eq(CONFIG_H2)].copy()
    assert len(h2) == 3_334 and h2["split"].eq("validation").all(), "H2 primary population changed"
    results, reliability, primary, control = [], {}, {}, {}
    for horizon in (10, 5):
        group = h2.loc[h2["horizon_minutes"].eq(horizon)].copy()
        assert len(group) == EXPECTED_COMMON_ROWS[("validation", horizon)], "H2 horizon count changed"
        scores = {}
        for name, column in (("har", "probability"), ("stage0", "stage0_platt_probability"), ("market", "quote_mid"), ("n1", "probability_n1_control")):
            scores[name], table = score_population(group, column, horizon)
            validate_reliability(table, group, column, horizon)
            if name in ("har", "stage0", "market"):
                reliability[(horizon, name)] = table
        paired = paired_brier(group, "probability", "stage0_platt_probability", CONFIG_H2, "validation_primary", horizon, market=True)
        assert abs(paired["model_value"] - scores["har"]["brier"]) <= 1e-15 and abs(paired["stage0_value"] - scores["stage0"]["brier"]) <= 1e-15, "H2 Brier paths differ"
        assert abs(paired["market_value"] - scores["market"]["brier"]) <= 1e-15, "Market Brier paths differ"
        results.append(paired)
        primary[horizon] = paired
        n1 = paired_brier(group, "probability_n1_control", "stage0_platt_probability", "h2_n1_control", "validation_primary", horizon)
        assert abs(n1["model_value"] - scores["n1"]["brier"]) <= 1e-15, "N1 Brier paths differ"
        control[horizon] = n1
        results.append(n1)
        for metric in ("log_loss", "auc", "ece"):
            results.append(record(CONFIG_H2, "validation_primary", horizon, "overall", "overall", metric, len(group),
                                  model=scores["har"][metric], stage0=scores["stage0"][metric], market=scores["market"][metric]))
            results.append(record("h2_n1_control", "validation_primary", horizon, "overall", "overall", metric, len(group),
                                  model=scores["n1"][metric], stage0=scores["stage0"][metric]))
        numerator = paired["stage0_value"] - paired["model_value"]
        denominator = paired["stage0_value"] - paired["market_value"]
        unstable = abs(denominator) <= NEAR_ZERO
        closure = numerator / denominator if not unstable else np.nan
        results.append(record(CONFIG_H2, "validation_primary", horizon, "overall", "overall", "market_gap_closure", len(group),
                              model=closure, stage0=paired["stage0_value"], market=paired["market_value"], numerator=numerator,
                              denominator=denominator, unstable=unstable))
        results.append(record("h2_n1_control", "validation_primary", horizon, "overall", "overall", "har_minus_n1_brier", len(group),
                              model=paired["model_value"], stage0=n1["model_value"], difference=paired["model_value"] - n1["model_value"]))
        for breakdown, column, labels in (("abs_z", "abs_z_bucket", ABS_Z_LABELS), ("price_bucket", "price_bucket", PRICE_BUCKET_LABELS)):
            assert group[column].isin(labels).all(), f"{breakdown} does not partition H2"
            total = 0
            for label in labels:
                cell = group.loc[group[column].eq(label)]
                if cell.empty:
                    continue
                total += len(cell)
                results.append(paired_brier(cell, "probability", "stage0_platt_probability", CONFIG_H2, "validation_primary", horizon, breakdown, label, market=True))
            assert total == len(group), f"{breakdown} bucket counts do not sum to T-{horizon}"
    pooled = paired_brier(h2, "probability", "stage0_platt_probability", CONFIG_H2, "validation_primary", 0, market=True)
    results.append(pooled)
    results.append(paired_brier(h2, "probability_n1_control", "stage0_platt_probability", "h2_n1_control", "validation_primary", 0))
    verdict = p_category(primary)
    pooled["verdict"] = verdict
    for row in primary.values():
        row["verdict"] = verdict
    return results, reliability, verdict, control


def score_h1(rows):
    h1 = rows.loc[rows["configuration"].eq(CONFIG_H1)].copy()
    assert len(h1) == 7_012 and set(h1["fold"]) == set(range(1, 7)), "H1 OOF population changed"
    output, verdicts = [], {}
    for window, section in (("all_oof_descriptive", h1), ("validation_folds_4_6", h1.loc[h1["fold"].between(4, 6)])):
        expected = 7_012 if window == "all_oof_descriptive" else 3_334
        assert len(section) == expected and not section.duplicated(ROW_KEY).any(), "H1 window keys/count changed"
        if window == "validation_folds_4_6":
            assert section["split"].eq("validation").all() and set(section["fold"]) == {4, 5, 6}, "H1 validation folds changed"
        horizon_rows = {}
        for horizon in (10, 5):
            group = section.loc[section["horizon_minutes"].eq(horizon)]
            expected_horizon = EXPECTED_HORIZON_SCORE_ROWS[CONFIG_H1][horizon] if window == "all_oof_descriptive" else EXPECTED_COMMON_ROWS[("validation", horizon)]
            assert len(group) == expected_horizon, "H1 horizon population changed"
            paired = paired_brier(group, "probability", "b1_probability", "h1_walk_forward_vs_b1", window, horizon)
            horizon_rows[horizon] = paired
            output.append(paired)
        pooled = paired_brier(section, "probability", "b1_probability", "h1_walk_forward_vs_b1", window, 0)
        verdicts[window] = p_category(horizon_rows)
        pooled["verdict"] = verdicts[window]
        for row in horizon_rows.values():
            row["verdict"] = verdicts[window]
        output.append(pooled)
    return output, verdicts


def score_oracle(rows, context):
    target_columns = ROW_KEY + ["split", "close_date", "fwd_rv"]
    assert pq.ParquetFile(TARGET_PATH).metadata.num_rows == 14_358, "Oracle target structural count changed"
    target = pd.read_parquet(TARGET_PATH, columns=target_columns, filters=[("split", "in", ["train", "validation"])])
    assert list(target.columns) == target_columns and len(target) == 14_358 and not target.duplicated(ROW_KEY).any(), "Oracle target projection/keys changed"
    assert target["split"].isin(("train", "validation")).all() and target["close_date"].le(SPLIT_RANGES["validation"][1]).all(), "Test target row loaded"
    oracle_context = context.merge(target, on=ROW_KEY + ["split", "close_date"], how="left", validate="one_to_one", indicator=True)
    assert len(oracle_context) == len(context) and oracle_context["_merge"].eq("both").all(), "Oracle target/context join changed"
    oracle_context = oracle_context.drop(columns="_merge")
    stage0_columns = ROW_KEY + ["split", "y"]
    outcomes = pd.read_parquet(STAGE0_PATH, columns=stage0_columns, filters=[("split", "==", "train")])
    train_common = oracle_context.loc[oracle_context["split"].eq("train")]
    train_outcomes = outcomes.merge(train_common[ROW_KEY + ["split"]], on=ROW_KEY + ["split"], how="inner", validate="one_to_one")
    exact_keys(train_common, train_outcomes, "Oracle train outcomes")
    oracle_context = oracle_context.merge(train_outcomes[ROW_KEY + ["y"]], on=ROW_KEY, how="left", validate="one_to_one")
    assert oracle_context.loc[oracle_context["split"].eq("train"), "y"].isin((0, 1)).all(), "Oracle train outcome missing"
    h2 = rows.loc[rows["configuration"].eq(CONFIG_H2)]
    results = []
    for horizon in (10, 5):
        fit = oracle_context.loc[oracle_context["split"].eq("train") & oracle_context["horizon_minutes"].eq(horizon) & oracle_context["fwd_rv"].notna()]
        validation = oracle_context.loc[oracle_context["split"].eq("validation") & oracle_context["horizon_minutes"].eq(horizon) & oracle_context["fwd_rv"].notna()]
        expected_fit = 4_765 if horizon == 10 else 4_695
        assert len(fit) == expected_fit and len(validation) == ORACLE_VALIDATION_ROWS[horizon], "Oracle target-bearing population changed"
        h2_subset = h2.loc[h2["horizon_minutes"].eq(horizon)].merge(validation[ROW_KEY], on=ROW_KEY, how="inner", validate="one_to_one")
        exact_keys(h2_subset, validation, "Oracle/H2 exact validation subset")
        for subset in (fit, validation):
            sigma = subset["fwd_rv"].to_numpy(dtype=float)
            assert np.isfinite(sigma).all() and (sigma > 0).all(), "Oracle sigma invalid"
        raw_fit, _ = stage0_probability(fit["log_moneyness"].to_numpy(dtype=float), fit["fwd_rv"].to_numpy(dtype=float), fit["T_years"].to_numpy(dtype=float))
        raw_score, _ = stage0_probability(validation["log_moneyness"].to_numpy(dtype=float), validation["fwd_rv"].to_numpy(dtype=float), validation["T_years"].to_numpy(dtype=float))
        assert np.isfinite(raw_fit).all() and np.isfinite(raw_score).all() and ((raw_fit > 0) & (raw_fit < 1)).all() and ((raw_score > 0) & (raw_score < 1)).all(), "Oracle raw probability invalid"
        platt = fit_platt(raw_fit, fit["y"].to_numpy(dtype=int))
        assert platt["b"] > 0 and PLATT_CLIP == 1e-6, "Oracle Platt rule changed"
        oracle_probability = stable_sigmoid(platt["a"] + platt["b"] * platt_feature(raw_score))
        assert np.isfinite(oracle_probability).all() and ((oracle_probability > 0) & (oracle_probability < 1)).all(), "Oracle calibrated probability invalid"
        oracle_scored = validation[ROW_KEY].copy()
        oracle_scored["oracle_probability"] = oracle_probability
        aligned = h2_subset.merge(oracle_scored, on=ROW_KEY, how="left", validate="one_to_one", indicator=True)
        assert len(aligned) == len(validation) and aligned["_merge"].eq("both").all(), "Oracle/HAR/Stage 0 keys differ"
        oracle_brier = brier_only(aligned, "oracle_probability")
        stage0_brier = brier_only(aligned, "stage0_platt_probability")
        har_brier = brier_only(aligned, "probability")
        oracle_gap = stage0_brier - oracle_brier
        har_improvement = stage0_brier - har_brier
        unstable = abs(oracle_gap) <= NEAR_ZERO
        headroom = har_improvement / oracle_gap if not unstable else np.nan
        config, window, n = "oracle_headroom", "validation_target_bearing", horizon
        results.append(record(config, window, n, "overall", "overall", "brier", len(aligned), model=har_brier, stage0=stage0_brier, oracle=oracle_brier))
        results.append(record(config, window, n, "overall", "overall", "oracle_gap", len(aligned), model=oracle_gap, stage0=stage0_brier, oracle=oracle_brier))
        results.append(record(config, window, n, "overall", "overall", "har_improvement", len(aligned), model=har_improvement, stage0=stage0_brier))
        results.append(record(config, window, n, "overall", "overall", "headroom_captured", len(aligned), model=headroom,
                              numerator=har_improvement, denominator=oracle_gap, unstable=unstable))
    return results


def assert_output(output):
    assert list(output.columns) == OUTPUT_COLUMNS and len(output) == 58 and not output.duplicated(KEY).any(), "Probability comparison schema/key/count changed"
    assert set(output["configuration"]) == {CONFIG_H2, "h2_n1_control", "h1_walk_forward_vs_b1", "oracle_headroom"}, "Comparison configurations changed"
    assert output["n"].gt(0).all() and output["thin"].eq(output["n"].lt(30)).all(), "Thin/count invariant failed"
    assert output["meaningful"].dtype == pd.BooleanDtype() and output["unstable"].dtype == pd.BooleanDtype(), "Nullable flags changed"
    brier = output.loc[output["metric"].eq("brier") & output["configuration"].ne("oracle_headroom")]
    assert np.isfinite(brier[["model_value", "stage0_value", "difference"]].to_numpy(dtype=float)).all(), "Non-finite Brier comparison"
    assert brier.loc[brier["n"].gt(1), "paired_se"].notna().all() and brier.loc[brier["n"].gt(1), "meaningful"].notna().all(), "Paired SE/flag missing"
    assert brier.loc[brier["n"].eq(1), ["paired_se", "meaningful"]].isna().all().all(), "Singleton has fabricated SE"
    assert output.loc[output["configuration"].eq("oracle_headroom"), ["paired_se", "difference", "meaningful"]].isna().all().all(), "Oracle contains a paired verdict"
    assert output.loc[output["configuration"].eq("oracle_headroom"), "breakdown"].eq("overall").all(), "Oracle contains row-level results"
    assert not any(column in output.columns for column in ("ticker", "probability", "oracle_probability", "fwd_rv")), "Row-level oracle value in artifact"
    for horizon in (10, 5):
        overall = output.loc[output["configuration"].eq(CONFIG_H2) & output["comparison_window"].eq("validation_primary") & output["horizon_minutes"].eq(horizon) & output["breakdown"].eq("overall") & output["metric"].eq("brier")]
        assert len(overall) == 1 and int(overall["n"].iloc[0]) == EXPECTED_COMMON_ROWS[("validation", horizon)], "H2 overall Brier row changed"
        for breakdown, labels in (("abs_z", ABS_Z_LABELS), ("price_bucket", PRICE_BUCKET_LABELS)):
            cells = output.loc[output["configuration"].eq(CONFIG_H2) & output["horizon_minutes"].eq(horizon) & output["breakdown"].eq(breakdown)]
            assert set(cells["breakdown_value"]) == set(labels) and int(cells["n"].sum()) == EXPECTED_COMMON_ROWS[("validation", horizon)], f"{breakdown} artifact partition changed"
    assert len(output.loc[output["configuration"].eq("oracle_headroom")]) == 8, "Oracle metric row count changed"
    assert output.loc[output["configuration"].eq("oracle_headroom") & output["metric"].eq("brier"), "n"].sort_values().tolist() == [1_588, 1_644], "Oracle subset count changed"
    h2_verdicts = output.loc[output["configuration"].eq(CONFIG_H2) & output["metric"].eq("brier") & output["breakdown"].eq("overall"), "verdict"]
    assert len(h2_verdicts) == 3 and h2_verdicts.nunique() == 1 and h2_verdicts.iloc[0] in P_WORDING, "H2 P verdict missing/inconsistent"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_comparison(output):
    assert_output(output)
    prior_hash = None
    if OUTPUT_PATH.exists():
        previous = pd.read_parquet(OUTPUT_PATH)
        assert_output(previous)
        pd.testing.assert_frame_equal(previous, output, check_exact=True)
        prior_hash = sha256(OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, prefix=".har_probability_comparison_", suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        output.to_parquet(temporary, index=False)
        saved = pd.read_parquet(temporary)
        assert_output(saved)
        pd.testing.assert_frame_equal(saved, output, check_exact=True)
        if prior_hash is not None:
            assert sha256(temporary) == prior_hash, "Deterministic comparison file hash changed"
        os.replace(temporary, OUTPUT_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return prior_hash is not None


def plot_reliability(tables):
    assert set(tables) == {(h, m) for h in (10, 5) for m in ("har", "stage0", "market")}, "Reliability table set changed"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    colors = {"har": "#1665a5", "stage0": "#e3821c", "market": "#528b57"}
    names = {"har": "HAR-fed Stage 0", "stage0": "Frozen Stage 0", "market": "Market mid"}
    for axis, horizon in zip(axes, (10, 5)):
        for model in ("har", "stage0", "market"):
            table = tables[(horizon, model)]
            lower = table["observed_yes_rate"] - table["wilson_low"]
            upper = table["wilson_high"] - table["observed_yes_rate"]
            axis.errorbar(table["mean_predicted"], table["observed_yes_rate"], yerr=np.vstack((lower, upper)),
                          marker="o", markersize=3, linewidth=1.5, capsize=2, color=colors[model], label=names[model])
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", label="Perfect calibration")
        axis.set(title=f"T-{horizon} · H2 validation common", xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed YES frequency")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("HAR-fed Stage 0, frozen Stage 0, and market reliability")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=PLOT_PATH.parent, prefix=".har_vs_stage0_reliability_", suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, dpi=160, bbox_inches="tight")
        assert temporary.stat().st_size > 0, "Reliability plot is empty"
        if PLOT_PATH.exists():
            assert sha256(temporary) == sha256(PLOT_PATH), "Deterministic reliability plot hash changed"
        os.replace(temporary, PLOT_PATH)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def main():
    probabilities = load_frozen_probabilities()
    print(f"Frozen HAR probability fingerprint: {FROZEN_FINGERPRINT} (PASS before scoring)")
    rows, context = join_context(probabilities)
    stage0_error, carried_error = verify_baselines(rows)
    print(f"Frozen Stage 0 validation Brier max error: {stage0_error:.17g}; carried probability max error: {carried_error:.17g} (PASS before HAR scoring)")
    primary, reliability, verdict, _ = score_h2(rows)
    h1, h1_verdicts = score_h1(rows)
    oracle = score_oracle(rows, context)
    output = pd.DataFrame(primary + h1 + oracle, columns=OUTPUT_COLUMNS)
    for column in ("configuration", "comparison_window", "breakdown", "breakdown_value", "metric", "verdict"):
        output[column] = output[column].astype("str")
    output["meaningful"] = output["meaningful"].astype("boolean")
    output["unstable"] = output["unstable"].astype("boolean")
    output = output.sort_values(KEY).reset_index(drop=True)
    deterministic_prior = write_comparison(output)
    plot_reliability(reliability)
    print(f"H2 P verdict: {P_WORDING[verdict]}")
    print("H1 secondary verdicts: " + "; ".join(f"{window}={P_WORDING[category]}" for window, category in h1_verdicts.items()))
    print(f"Comparison rows: {len(output)}; deterministic prior comparison: {'PASS' if deterministic_prior else 'first write'}")
    print("Paired keys/rules, oracle subset, bucket partitions, reliability tables/Wilson intervals, and exact round trip: PASS")
    print(f"Written: {OUTPUT_PATH}; {PLOT_PATH}")


if __name__ == "__main__":
    main()
