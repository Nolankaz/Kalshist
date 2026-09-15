"""Analyze KXBTC15M quote costs on frozen train and validation rows only.

This Day 10 Section 2.2 analysis never loads test rows, outcomes, targets, or
future-only fields. Fees use the frozen Direct Member, fresh-order, one-fill
execution model in ``scripts.fees``. Quantiles use NumPy's linear method.
"""

import os
from decimal import Decimal
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
from scripts.evaluation_split import EXPECTED_SPLIT_ROWS, SPLIT_KEY, SPLIT_RANGES
from scripts.fees import DIRECT_MEMBER, KXBTC15M_FEE_MULTIPLIER, KXBTC15M_FEE_TYPE, calculate_single_fill_fee, is_on_tick_grid
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, EXPECTED_HORIZON_ROWS, ROW_KEY, common_prediction_mask


FEATURES_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
MODEL_MARKET_GAP_PATH = PROJECT_ROOT / "data/models/stage0_model_market_gap.parquet"
PLATT_PARAMETERS_PATH = PROJECT_ROOT / "data/models/stage0_platt_parameters.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/execution/spread_summary.parquet"
PRICE_PLOT_PATH = PROJECT_ROOT / "data/execution/plots/spread_by_price_bucket.png"
HOUR_PLOT_PATH = PROJECT_ROOT / "data/execution/plots/spread_by_hour.png"

ALLOWED_SPLITS = ("train", "validation")
HORIZONS = (10, 5)
POPULATIONS = ("all_rows", "day9_common")
FEATURE_COLUMNS = [
    "ticker", "horizon_minutes", SPLIT_KEY, "hour_utc", "quote_yes_bid", "quote_yes_ask",
    "quote_mid", "quote_spread", "quote_age_seconds",
]
PREDICTION_COLUMNS = [
    "p_5min_vol", "p_5min_ewma_vol", "p_15min_vol", "p_15min_ewma_vol", "p_1hr_vol",
    "p_1hr_ewma_vol", "p_4hr_vol", "p_4hr_ewma_vol", "p_24hr_vol", "p_24hr_ewma_vol",
]
FORBIDDEN_COLUMNS = {
    "y", "settlement_result", "settlement_value", "expiration_value", "y_from_expiration",
    "target_agrees", "fwd_log_return",
}
PRICE_BUCKET_EDGES = [0.00, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 1.00]
PRICE_BUCKET_LABELS = ["p00_10", "p10_25", "p25_40", "p40_60", "p60_75", "p75_90", "p90_100"]
PRICE_BUCKET_DISPLAY = ["[0,.10)", "[.10,.25)", "[.25,.40)", "[.40,.60)", "[.60,.75)", "[.75,.90)", "[.90,1]"]
EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS = {
    10: [54, 279, 317, 442, 275, 240, 78],
    5: [404, 217, 147, 165, 147, 183, 386],
}
QUANTILES = {"p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
DESCRIPTIVE_MATERIAL_RANGE_MILS = 2.0


def assert_safe_columns(columns, source):
    loaded = set(columns)
    forbidden = sorted(loaded & FORBIDDEN_COLUMNS)
    assert not forbidden, f"{source} loaded forbidden target/outcome/future-only columns: {forbidden}"


def load_allowed_features():
    """Load only train/validation quote fields, using the frozen split ranges."""
    train_start = SPLIT_RANGES["train"][0]
    validation_end = SPLIT_RANGES["validation"][1]
    rows = pd.read_parquet(
        FEATURES_PATH, columns=FEATURE_COLUMNS,
        filters=[(SPLIT_KEY, ">=", train_start), (SPLIT_KEY, "<=", validation_end)],
    )
    assert list(rows.columns) == FEATURE_COLUMNS, "Feature projection changed"
    assert_safe_columns(rows.columns, "market features")
    assert not rows.duplicated(ROW_KEY).any(), f"Market features are not unique on {ROW_KEY}"

    # assign_split() requires all three splits, so apply its imported frozen ranges
    # only after Parquet predicate pushdown has excluded test rows entirely.
    rows["split"] = pd.Series(pd.NA, index=rows.index, dtype="string")
    for split in ALLOWED_SPLITS:
        start, end = SPLIT_RANGES[split]
        rows.loc[rows[SPLIT_KEY].between(start, end, inclusive="both"), "split"] = split
    assert rows["split"].notna().all(), "A loaded feature row is outside train/validation"
    assert rows["split"].isin(ALLOWED_SPLITS).all(), "A forbidden split entered feature analysis"
    assert not rows["split"].eq("test").any(), "Test rows entered feature analysis"

    actual_counts = rows.groupby("split", observed=True).size().to_dict()
    expected_counts = {split: EXPECTED_SPLIT_ROWS[split] for split in ALLOWED_SPLITS}
    assert actual_counts == expected_counts, f"Unexpected all-row split counts: {actual_counts}"
    for split in ALLOWED_SPLITS:
        horizon_counts = rows.loc[rows["split"].eq(split)].groupby("horizon_minutes").size().to_dict()
        expected_horizon_counts = {horizon: EXPECTED_HORIZON_ROWS[split] for horizon in HORIZONS}
        assert horizon_counts == expected_horizon_counts, f"Unexpected {split} horizon counts: {horizon_counts}"
    return rows


def load_allowed_predictions():
    columns = ROW_KEY + ["split"] + PREDICTION_COLUMNS
    rows = pd.read_parquet(PREDICTIONS_PATH, columns=columns, filters=[("split", "in", list(ALLOWED_SPLITS))])
    assert list(rows.columns) == columns, "Prediction projection changed"
    assert_safe_columns(rows.columns, "Stage 0 predictions")
    assert rows["split"].isin(ALLOWED_SPLITS).all(), "A forbidden split entered prediction analysis"
    assert not rows["split"].eq("test").any(), "Test rows entered prediction analysis"
    assert not rows.duplicated(ROW_KEY).any(), f"Stage 0 predictions are not unique on {ROW_KEY}"
    return rows


def identify_common_rows(features, predictions):
    feature_keys = features[ROW_KEY + ["split"]]
    prediction_match = predictions[ROW_KEY + ["split"]].merge(feature_keys, on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert prediction_match["_merge"].eq("both").all(), "A train/validation prediction does not match a feature row"

    mask = common_prediction_mask(predictions, PREDICTION_COLUMNS)
    common_keys = predictions.loc[mask, ROW_KEY + ["split"]].copy()
    actual_counts = common_keys.groupby(["split", "horizon_minutes"]).size().to_dict()
    assert actual_counts == EXPECTED_COMMON_ROWS, f"Unexpected Day 9 common-row counts: {actual_counts}"

    rows = features.merge(common_keys.assign(is_day9_common=True), on=ROW_KEY + ["split"], how="left", validate="one_to_one")
    rows["is_day9_common"] = rows["is_day9_common"].fillna(False).astype(bool)
    assert int(rows["is_day9_common"].sum()) == len(common_keys), "Common-row merge changed the population"
    return rows, common_keys


def verify_platt_carry_forward(predictions, common_keys):
    gap_columns = ROW_KEY[:1] + ["split", "horizon_minutes", "raw_model_probability", "platt_model_probability"]
    gap = pd.read_parquet(MODEL_MARKET_GAP_PATH, columns=gap_columns, filters=[("split", "in", list(ALLOWED_SPLITS))])
    assert_safe_columns(gap.columns, "model-market-gap artifact")
    assert gap["split"].isin(ALLOWED_SPLITS).all() and not gap["split"].eq("test").any(), "Test rows entered the Platt check"
    assert not gap.duplicated(ROW_KEY).any(), f"Model-market-gap rows are not unique on {ROW_KEY}"

    key_check = gap[ROW_KEY + ["split"]].merge(common_keys, on=ROW_KEY + ["split"], how="outer", validate="one_to_one", indicator=True)
    assert key_check["_merge"].eq("both").all(), "Model-market-gap keys differ from the Day 9 common population"

    prediction_columns = ROW_KEY + ["split", "p_5min_ewma_vol"]
    rows = gap.merge(predictions[prediction_columns], on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert rows["_merge"].eq("both").all(), "A model-market-gap row did not match its selected Stage 0 probability"
    raw_difference = np.abs(rows["p_5min_ewma_vol"].to_numpy() - rows["raw_model_probability"].to_numpy())
    assert float(raw_difference.max()) <= 1e-12, "The Day 9 raw selected probability carry-forward changed"

    parameter_columns = ["candidate", "horizon_minutes", "fit_split", "parameter_role", "a", "b"]
    parameters = pd.read_parquet(
        PLATT_PARAMETERS_PATH, columns=parameter_columns,
        filters=[("fit_split", "==", "train"), ("parameter_role", "==", "legitimate_train_fit")],
    )
    assert_safe_columns(parameters.columns, "Platt parameters")
    assert len(parameters) == 2, f"Expected two legitimate train-fitted parameter rows, found {len(parameters)}"
    assert parameters["fit_split"].eq("train").all(), "A non-train Platt parameter entered the check"
    assert parameters["parameter_role"].eq("legitimate_train_fit").all(), "A non-legitimate Platt parameter entered the check"
    assert parameters["candidate"].eq("5min_ewma_vol").all(), "A different sigma candidate entered the Platt check"
    assert not parameters.duplicated("horizon_minutes").any(), "Platt parameters are not unique by horizon"
    assert set(parameters["horizon_minutes"]) == set(HORIZONS), "Platt parameters do not cover both horizons"

    recomputed = np.empty(len(rows), dtype=float)
    for parameter in parameters.itertuples(index=False):
        horizon_mask = rows["horizon_minutes"].eq(parameter.horizon_minutes)
        fit = {"a": parameter.a, "b": parameter.b}
        recomputed[horizon_mask.to_numpy()] = apply_platt(rows.loc[horizon_mask, "p_5min_ewma_vol"], fit)
    differences = np.abs(recomputed - rows["platt_model_probability"].to_numpy())
    maximum_difference = float(differences.max())
    assert maximum_difference <= 1e-12, f"Day 9 Platt reproduction mismatch: {maximum_difference:.17g}"
    return maximum_difference


def integer_mils(values, name):
    scaled = np.asarray(values, dtype=float) * 1000.0
    mils = np.rint(scaled).astype("int64")
    assert np.allclose(scaled, mils, rtol=0.0, atol=1e-9), f"{name} is not representable in integer mils"
    return mils


def assign_price_buckets(rows):
    assert rows["quote_mid"].between(0.0, 1.0, inclusive="both").all(), "quote_mid falls outside [0, 1]"
    buckets = pd.cut(rows["quote_mid"], bins=PRICE_BUCKET_EDGES, labels=PRICE_BUCKET_LABELS, right=False, include_lowest=True).astype("object")
    buckets.loc[rows["quote_mid"].eq(1.0)] = PRICE_BUCKET_LABELS[-1]
    rows["price_bucket"] = pd.Categorical(buckets, categories=PRICE_BUCKET_LABELS, ordered=True)
    assert rows["price_bucket"].notna().all(), "A row was not assigned to a price bucket"
    return rows


def calculate_execution_costs(rows):
    numeric_columns = ["quote_yes_bid", "quote_yes_ask", "quote_mid", "quote_spread", "quote_age_seconds"]
    assert rows[numeric_columns].notna().all().all(), "A required quote field is null"
    assert np.isfinite(rows[numeric_columns].to_numpy(dtype=float)).all(), "A required quote field is non-finite"
    assert rows["hour_utc"].between(0, 23, inclusive="both").all(), "hour_utc falls outside 0 through 23"

    rows["bid_mils"] = integer_mils(rows["quote_yes_bid"], "quote_yes_bid")
    rows["ask_mils"] = integer_mils(rows["quote_yes_ask"], "quote_yes_ask")
    rows["spread_mils"] = integer_mils(rows["quote_spread"], "quote_spread")
    rows["ask_minus_bid_mils"] = np.rint((rows["quote_yes_ask"] - rows["quote_yes_bid"]) * 1000.0).astype("int64")
    assert rows["spread_mils"].gt(0).all(), "Every spread must be positive"
    assert rows["spread_mils"].eq(rows["ask_minus_bid_mils"]).all(), "quote_spread differs from ask - bid in integer mils"
    assert rows["spread_mils"].eq(rows["ask_mils"] - rows["bid_mils"]).all(), "Integer bid/ask spread identity failed"

    bid_prices = [Decimal(int(value)) / 1000 for value in rows["bid_mils"]]
    ask_prices = [Decimal(int(value)) / 1000 for value in rows["ask_mils"]]
    no_prices = [Decimal(1000 - int(value)) / 1000 for value in rows["bid_mils"]]
    assert all(is_on_tick_grid(price) for price in bid_prices), "A YES bid is off the KXBTC15M tapered grid"
    assert all(is_on_tick_grid(price) for price in ask_prices), "A YES ask is off the KXBTC15M tapered grid"
    assert all(is_on_tick_grid(price) for price in no_prices), "A NO executable price is off the KXBTC15M tapered grid"

    yes_results = [
        calculate_single_fill_fee(price, 1, "buy", multiplier=KXBTC15M_FEE_MULTIPLIER,
                                  fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER)
        for price in ask_prices
    ]
    no_results = [
        calculate_single_fill_fee(price, 1, "buy", multiplier=KXBTC15M_FEE_MULTIPLIER,
                                  fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER)
        for price in no_prices
    ]
    yes_executable_checks = (
        result.executable_price == price and result.action == "buy" and result.rounding_mode == DIRECT_MEMBER
        for result, price in zip(yes_results, ask_prices)
    )
    no_executable_checks = (
        result.executable_price == price and result.action == "buy" and result.rounding_mode == DIRECT_MEMBER
        for result, price in zip(no_results, no_prices)
    )
    frozen_fee_checks = (
        result.fee_type == KXBTC15M_FEE_TYPE and result.multiplier == KXBTC15M_FEE_MULTIPLIER
        for result in yes_results + no_results
    )
    assert all(yes_executable_checks), "YES fees did not use executable asks"
    assert all(no_executable_checks), "NO fees did not use 1 - bid"
    assert all(frozen_fee_checks), "A fee used non-frozen KXBTC15M settings"

    rows["fee_yes_dollars"] = np.array([float(result.net_cash_fee) for result in yes_results])
    rows["fee_no_dollars"] = np.array([float(result.net_cash_fee) for result in no_results])
    assert rows[["fee_yes_dollars", "fee_no_dollars"]].ge(0.0).all().all(), "A net cash fee is negative"
    rows["half_spread_probability_units"] = rows["spread_mils"] / 2000.0
    rows["mid_cost_yes_probability_units"] = rows["half_spread_probability_units"] + rows["fee_yes_dollars"]
    rows["mid_cost_no_probability_units"] = rows["half_spread_probability_units"] + rows["fee_no_dollars"]
    rows["bid_subcent"] = rows["bid_mils"].mod(10).ne(0)
    rows["ask_subcent"] = rows["ask_mils"].mod(10).ne(0)
    return assign_price_buckets(rows)


def numeric_statistics(values, prefix, unit_suffix, include_count=True, include_min=True, include_max=True):
    values = np.asarray(values, dtype=float)
    result = {}
    if include_count:
        result[f"{prefix}_count"] = int(len(values))
    names = ["mean", "median"] + list(QUANTILES)
    if include_min:
        names.append("min")
    if include_max:
        names.append("max")
    if len(values) == 0:
        result.update({f"{prefix}_{name}_{unit_suffix}": np.nan for name in names})
        return result

    result[f"{prefix}_mean_{unit_suffix}"] = float(np.mean(values))
    result[f"{prefix}_median_{unit_suffix}"] = float(np.median(values))
    quantiles = np.quantile(values, list(QUANTILES.values()), method="linear")
    for name, value in zip(QUANTILES, quantiles):
        result[f"{prefix}_{name}_{unit_suffix}"] = float(value)
    if include_min:
        result[f"{prefix}_min_{unit_suffix}"] = float(np.min(values))
    if include_max:
        result[f"{prefix}_max_{unit_suffix}"] = float(np.max(values))
    return result


def summarize_group(rows, population, split, horizon, breakdown, breakdown_value):
    record = {
        "population": population,
        "split": split,
        "horizon_minutes": horizon,
        "breakdown": breakdown,
        "breakdown_value": str(breakdown_value),
        "row_count": int(len(rows)),
    }
    record.update(numeric_statistics(rows["spread_mils"], "spread", "mils", include_min=False))
    record["spread_exactly_1_mil_share"] = float(rows["spread_mils"].eq(1).mean()) if len(rows) else np.nan
    record["spread_exactly_10_mils_share"] = float(rows["spread_mils"].eq(10).mean()) if len(rows) else np.nan
    record["spread_above_20_mils_share"] = float(rows["spread_mils"].gt(20).mean()) if len(rows) else np.nan

    if breakdown in {"overall", "price_bucket"}:
        record.update(numeric_statistics(rows["fee_yes_dollars"], "fee_yes", "dollars"))
        record.update(numeric_statistics(rows["fee_no_dollars"], "fee_no", "dollars"))
        record.update(numeric_statistics(rows["mid_cost_yes_probability_units"], "mid_cost_yes", "probability_units", include_count=False))
        record.update(numeric_statistics(rows["mid_cost_no_probability_units"], "mid_cost_no", "probability_units", include_count=False))
        record["bid_subcent_share"] = float(rows["bid_subcent"].mean()) if len(rows) else np.nan
        record["ask_subcent_share"] = float(rows["ask_subcent"].mean()) if len(rows) else np.nan

    if breakdown == "overall":
        record["stale_gt_10s_count"] = int(rows["quote_age_seconds"].gt(10).sum())
        record["stale_gt_60s_count"] = int(rows["quote_age_seconds"].gt(60).sum())
        record["stale_gt_10s_share"] = float(rows["quote_age_seconds"].gt(10).mean())
        record["stale_gt_60s_share"] = float(rows["quote_age_seconds"].gt(60).mean())
    return record


def population_rows(rows, population):
    if population == "all_rows":
        return rows
    if population == "day9_common":
        return rows.loc[rows["is_day9_common"]]
    raise ValueError(f"Unsupported population: {population}")


def build_summary(rows):
    records = []
    for population in POPULATIONS:
        eligible = population_rows(rows, population)
        assert eligible["split"].isin(ALLOWED_SPLITS).all() and not eligible["split"].eq("test").any(), "A test row entered a population"
        for split in ALLOWED_SPLITS:
            for horizon in HORIZONS:
                subset = eligible.loc[eligible["split"].eq(split) & eligible["horizon_minutes"].eq(horizon)]
                expected = EXPECTED_HORIZON_ROWS[split] if population == "all_rows" else EXPECTED_COMMON_ROWS[(split, horizon)]
                assert len(subset) == expected, f"Unexpected {population} {split} T-{horizon} count: {len(subset)}"
                records.append(summarize_group(subset, population, split, horizon, "overall", "all"))

                hour_total = 0
                for hour, hour_rows in subset.groupby("hour_utc", sort=True):
                    records.append(summarize_group(hour_rows, population, split, horizon, "hour_utc", f"{int(hour):02d}"))
                    hour_total += len(hour_rows)
                assert hour_total == len(subset), f"Hour buckets do not cover {population} {split} T-{horizon}"

                price_total = 0
                price_counts = []
                for bucket in PRICE_BUCKET_LABELS:
                    bucket_rows = subset.loc[subset["price_bucket"].eq(bucket)]
                    records.append(summarize_group(bucket_rows, population, split, horizon, "price_bucket", bucket))
                    price_total += len(bucket_rows)
                    price_counts.append(len(bucket_rows))
                assert price_total == len(subset), f"Price buckets do not cover {population} {split} T-{horizon}"
                if population == "day9_common" and split == "validation":
                    frozen_counts = EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS[horizon]
                    assert price_counts == frozen_counts, f"Validation common T-{horizon} price-bucket counts changed: {price_counts}"

    summary = pd.DataFrame(records)
    key = ["population", "split", "horizon_minutes", "breakdown", "breakdown_value"]
    assert not summary.duplicated(key).any(), f"Spread summary key {key} is not unique"
    assert set(summary["population"]) == set(POPULATIONS), "Summary is missing a population"
    assert set(summary["split"]) == set(ALLOWED_SPLITS), "Summary contains an invalid split"
    assert not summary["split"].eq("test").any(), "Test rows entered spread summaries"
    assert set(summary["breakdown"]) == {"overall", "hour_utc", "price_bucket"}, "Summary breakdowns changed"
    count_columns = [column for column in summary.columns if column == "row_count" or column.endswith("_count")]
    summary[count_columns] = summary[count_columns].astype("Int64")
    return summary


def plot_breakdown(summary, breakdown, output_path, x_values, x_labels, title, x_label):
    table = summary.loc[summary["breakdown"].eq(breakdown)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    colors = {"train": "C0", "validation": "C1"}
    for row_index, population in enumerate(POPULATIONS):
        for column_index, horizon in enumerate(HORIZONS):
            axis = axes[row_index, column_index]
            panel = table.loc[table["population"].eq(population) & table["horizon_minutes"].eq(horizon)].set_index("breakdown_value")
            for split in ALLOWED_SPLITS:
                split_rows = panel.loc[panel["split"].eq(split)].reindex(x_values)
                axis.plot(range(len(x_values)), split_rows["spread_mean_mils"], marker="o", markersize=4, color=colors[split], label=f"{split} mean")
                axis.plot(
                    range(len(x_values)), split_rows["spread_median_mils"], marker="x", markersize=4,
                    linestyle="--", color=colors[split], label=f"{split} median",
                )
            population_title = "All rows" if population == "all_rows" else "Day 9 common"
            axis.set_title(f"{population_title} — T-{horizon}")
            axis.set_xticks(range(len(x_values)), x_labels, rotation=35 if breakdown == "price_bucket" else 0)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            if column_index == 0:
                axis.set_ylabel("Spread (mils)")
            if row_index == 1:
                axis.set_xlabel(x_label)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tail_core_comparison(rows):
    records = []
    tail_buckets = {PRICE_BUCKET_LABELS[0], PRICE_BUCKET_LABELS[-1]}
    core_bucket = "p40_60"
    for population in POPULATIONS:
        eligible = population_rows(rows, population)
        for split in ALLOWED_SPLITS:
            for horizon in HORIZONS:
                subset = eligible.loc[eligible["split"].eq(split) & eligible["horizon_minutes"].eq(horizon)]
                tail = subset.loc[subset["price_bucket"].isin(tail_buckets), "spread_mils"]
                core = subset.loc[subset["price_bucket"].eq(core_bucket), "spread_mils"]
                records.append({
                    "population": population,
                    "split": split,
                    "horizon_minutes": horizon,
                    "tail_mean_spread_mils": float(tail.mean()),
                    "core_mean_spread_mils": float(core.mean()),
                })
    return pd.DataFrame(records)


def print_console_summary(rows, summary, maximum_platt_difference):
    overall = summary.loc[summary["breakdown"].eq("overall")].copy()
    print("Day 10 Section 2.2 — train/validation quote-cost summary")
    all_counts = rows.groupby("split", observed=True).size().reindex(ALLOWED_SPLITS)
    print("\nAll-row counts")
    print(all_counts.rename("rows").to_string())

    common_counts = rows.loc[rows["is_day9_common"]].groupby(["split", "horizon_minutes"]).size().rename("rows")
    print("\nDay 9 common-row counts")
    print(common_counts.to_string())

    spread_table = overall[["population", "split", "horizon_minutes", "spread_mean_mils", "spread_median_mils"]]
    print("\nMean/median spread (mils)")
    print(spread_table.to_string(index=False, formatters={"spread_mean_mils": "{:.3f}".format, "spread_median_mils": "{:.3f}".format}))

    tail_core = tail_core_comparison(rows)
    print("\nTail versus core mean spread (mils)")
    print(tail_core.to_string(index=False, formatters={"tail_mean_spread_mils": "{:.3f}".format, "core_mean_spread_mils": "{:.3f}".format}))

    fee_columns = ["population", "split", "horizon_minutes", "fee_yes_median_dollars", "fee_no_median_dollars"]
    print("\nDirect Member one-contract median net cash fee (dollars)")
    print(overall[fee_columns].to_string(index=False, formatters={"fee_yes_median_dollars": "{:.6f}".format, "fee_no_median_dollars": "{:.6f}".format}))

    cost_columns = ["population", "split", "horizon_minutes", "mid_cost_yes_median_probability_units", "mid_cost_no_median_probability_units"]
    print("\nMedian midpoint-scale quoted entry cost (probability units)")
    cost_formatters = {
        "mid_cost_yes_median_probability_units": "{:.6f}".format,
        "mid_cost_no_median_probability_units": "{:.6f}".format,
    }
    print(overall[cost_columns].to_string(index=False, formatters=cost_formatters))

    stale_columns = ["population", "split", "horizon_minutes", "stale_gt_10s_count", "stale_gt_60s_count"]
    print("\nStale quote counts (rows retained)")
    print(overall[stale_columns].to_string(index=False))

    hourly = summary.loc[summary["breakdown"].eq("hour_utc")]
    hourly_ranges = hourly.groupby(["population", "split", "horizon_minutes"])["spread_mean_mils"].agg(lambda values: float(values.max() - values.min()))
    maximum_hourly_range = float(hourly_ranges.max())
    maximum_price_difference = float((tail_core["tail_mean_spread_mils"] - tail_core["core_mean_spread_mils"]).abs().max())
    hour_material = maximum_hourly_range >= DESCRIPTIVE_MATERIAL_RANGE_MILS
    price_material = maximum_price_difference >= DESCRIPTIVE_MATERIAL_RANGE_MILS
    print(
        f"\nHour-of-day materially different: {'YES' if hour_material else 'NO'} "
        f"(maximum within-group mean range {maximum_hourly_range:.3f} mils; "
        f"descriptive threshold {DESCRIPTIVE_MATERIAL_RANGE_MILS:.1f} mils)"
    )
    print(
        f"Price region materially different: {'YES' if price_material else 'NO'} "
        f"(maximum absolute tail/core mean difference {maximum_price_difference:.3f} mils; "
        f"descriptive threshold {DESCRIPTIVE_MATERIAL_RANGE_MILS:.1f} mils)"
    )
    print(f"Platt reproduction: PASS (maximum absolute difference {maximum_platt_difference:.17g})")


def analyze_spreads():
    features = load_allowed_features()
    predictions = load_allowed_predictions()
    rows, common_keys = identify_common_rows(features, predictions)
    maximum_platt_difference = verify_platt_carry_forward(predictions, common_keys)
    rows = calculate_execution_costs(rows)
    summary = build_summary(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUTPUT_PATH, index=False)
    plot_breakdown(
        summary, "price_bucket", PRICE_PLOT_PATH, PRICE_BUCKET_LABELS, PRICE_BUCKET_DISPLAY,
        "KXBTC15M Spread by Midpoint Price Bucket", "Midpoint price bucket",
    )
    hour_values = [f"{hour:02d}" for hour in range(24)]
    plot_breakdown(summary, "hour_utc", HOUR_PLOT_PATH, hour_values, [str(hour) for hour in range(24)], "KXBTC15M Spread by UTC Hour", "UTC hour")

    assert OUTPUT_PATH.exists(), "Spread summary was not saved"
    assert PRICE_PLOT_PATH.exists(), "Price-bucket plot was not saved"
    assert HOUR_PLOT_PATH.exists(), "UTC-hour plot was not saved"
    print_console_summary(rows, summary, maximum_platt_difference)
    print(f"\nSaved {len(summary):,} summary rows to {OUTPUT_PATH}")
    print(f"Saved price-bucket plot to {PRICE_PLOT_PATH}")
    print(f"Saved UTC-hour plot to {HOUR_PLOT_PATH}")
    print("All Section 2.2 assertions passed; no test rows or outcome columns were loaded.")
    return summary


if __name__ == "__main__":
    analyze_spreads()
