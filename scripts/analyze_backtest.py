"""Produce descriptive-only Stage 0 backtest breakdowns from frozen results."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kalshist-matplotlib"))
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_spreads import PRICE_BUCKET_LABELS
from scripts.backtest import FROZEN_TRADE_COUNTS, TIER_LABEL
from scripts.count_threshold_clearance import EDGE_TOLERANCE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATHS = {
    "train": PROJECT_ROOT / "data/backtest/stage0_results_train.parquet",
    "validation": PROJECT_ROOT / "data/backtest/stage0_results_validation.parquet",
}
SUMMARY_PATH = PROJECT_ROOT / "data/backtest/stage0_summary.parquet"
PLOT_PATH = PROJECT_ROOT / "data/backtest/plots/stage0_net_pnl_by_breakdown.png"
DAILY_PATH = PROJECT_ROOT / "data/backtest/stage0_daily.parquet"
CORRELATION_PATH = PROJECT_ROOT / "data/backtest/stage0_correlation.parquet"
DAILY_PLOT_PATH = PROJECT_ROOT / "data/backtest/plots/stage0_daily_net_pnl.png"
CUMULATIVE_PLOT_PATH = PROJECT_ROOT / "data/backtest/plots/stage0_cumulative_net_pnl.png"

ALLOWED_SPLITS = ("train", "validation")
BASIS_SETTINGS = (1.2, 5.0)
SUMMARY_KEY = ["split", "basis_bps", "breakdown", "breakdown_value"]
REALIZED_COLUMNS = ["payoff", "hit", "gross_pnl", "net_pnl", "capital", "surprise"]
GROUPING_COLUMNS = [
    "horizon_minutes", "side", "price_bucket", "abs_z_bucket", "hour_block", "hour_utc",
    "week_start", "close_date", "threshold_role", "market_fair_expected", "net_edge",
]
ACCOUNTING_COLUMNS = ["entry_price", "fee"]

HORIZON_VALUES = [(10, "T-10"), (5, "T-5")]
SIDE_VALUES = [("YES", "YES"), ("NO", "NO")]
ABS_Z_BUCKETS = ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 1.0)", "[1.0, infinity)"]
HOUR_BLOCKS = ["00-05", "06-11", "12-17", "18-23"]
TRAIN_FIRST_PARTIAL_WEEK_START = "2026-05-27"
TRAIN_FIRST_PARTIAL_WEEK_END = "2026-05-31"
TRAIN_EX_FIRST_WEEK_LABEL = "excluding_2026-05-27_through_2026-05-31"
POPULATION_CALENDARS = {
    "train": ("2026-05-27", "2026-07-19", 54),
    "validation": ("2026-07-20", "2026-08-09", 21),
}
PRIMARY_INTERPRETATIONS = {
    "negative": "Stage 0 at the frozen threshold loses money after costs on validation.",
    "contains_zero": "No tier-2 evidence of edge either way; not established.",
    "positive": (
        "A positive tier-2 validation result, not an established edge: one three-week period, "
        "validation already used for the threshold's ECE, optimistic fills."
    ),
}

EXPECTED_VALIDATION_PRIMARY_COUNTS = {
    "horizon": {"T-10": 368, "T-5": 351},
    "side": {"YES": 243, "NO": 476},
    "price_bucket": dict(zip(PRICE_BUCKET_LABELS, [101, 129, 99, 64, 86, 126, 114])),
    "week": dict(zip(["2026-07-20", "2026-07-27", "2026-08-03"], [223, 250, 246])),
    "hour_block": dict(zip(HOUR_BLOCKS, [162, 158, 189, 210])),
    "abs_z_bucket": dict(zip(ABS_Z_BUCKETS, [258, 216, 204, 41])),
}

SUMMARY_COLUMNS = [
    "split", "basis_bps", "threshold_role", "tier_label", "breakdown", "breakdown_value",
    "n_trades", "n_days", "hit_rate", "breakeven_hit_rate", "gross_pnl", "fees", "net_pnl",
    "net_pnl_per_trade", "model_expected_per_trade", "market_fair_per_trade", "mean_surprise",
    "total_capital", "net_pnl_over_capital", "thin",
]
RECONCILIATION_FIELDS = ["gross_pnl", "fees", "net_pnl", "total_capital"]
PARTITION_BREAKDOWNS = ["horizon", "side", "price_bucket", "abs_z_bucket", "hour_block", "hour_utc", "week"]
DAILY_KEY = ["split", "basis_bps", "close_date"]
DAILY_ADDITIVE_COLUMNS = [
    "gross_pnl", "fees", "net_pnl", "model_expected_pnl", "market_fair_expected_pnl", "total_capital",
]
DAILY_COLUMNS = [
    "split", "basis_bps", "threshold_role", "tier_label", "close_date", "n_trades", *DAILY_ADDITIVE_COLUMNS,
]
CORRELATION_KEY = ["split", "basis_bps"]
CORRELATION_COLUMNS = [
    "split", "basis_bps", "threshold_role", "tier_label", "n_trades", "n_population_days",
    "n_days_with_trade", "mean_net_pnl_per_trade", "se_naive", "naive_lower_2se", "naive_upper_2se",
    "se_cluster", "cluster_lower_2se", "cluster_upper_2se", "design_effect", "se_inflation_factor",
    "n_eff", "mean_daily_net_pnl", "sample_sd_daily_net_pnl", "positive_trading_day_share",
    "best_day", "best_day_net_pnl", "worst_day", "worst_day_net_pnl",
    "largest_day_abs_share_of_total_abs_net_pnl", "n_weeks", "n_weeks_same_sign_as_total_net_pnl",
    "largest_week_abs_share_of_total_abs_net_pnl", "total_net_pnl", "primary_interpretation",
]


def close(left, right, n_rows=1) -> bool:
    """Use the repository's absolute tolerance, scaled for aggregate sums."""
    return bool(np.allclose(left, right, rtol=0.0, atol=EDGE_TOLERANCE * max(1, n_rows), equal_nan=True))


def load_frozen_results() -> pd.DataFrame:
    """Load and validate only the two frozen train/validation result files."""
    for path in RESULT_PATHS.values():
        assert path.exists(), f"Missing frozen result file: {path.relative_to(PROJECT_ROOT)}"

    frames = []
    required_columns = set(REALIZED_COLUMNS + GROUPING_COLUMNS + ACCOUNTING_COLUMNS + ["basis_bps", "ticker", "split"])
    for split, path in RESULT_PATHS.items():
        rows = pd.read_parquet(path)
        missing = sorted(required_columns - set(rows.columns))
        assert not missing, f"{split} results are missing required columns: {missing}"
        assert rows["split"].eq(split).all(), f"{split} result file contains another split"
        assert not rows["split"].eq("test").any(), f"{split} result file contains test rows"
        assert not rows.duplicated(["basis_bps", "ticker"]).any(), f"{split} results are not unique by basis/ticker"
        expected_counts = {basis: FROZEN_TRADE_COUNTS[(split, basis)] for basis in BASIS_SETTINGS}
        actual_counts = rows.groupby("basis_bps", observed=True).size().to_dict()
        assert actual_counts == expected_counts, f"{split} frozen row counts changed: {actual_counts}"
        assert rows[REALIZED_COLUMNS + GROUPING_COLUMNS + ACCOUNTING_COLUMNS].notna().all().all(), (
            f"{split} results contain null required values"
        )
        frames.append(rows)

    results = pd.concat(frames, ignore_index=True)
    assert set(results["split"]) == set(ALLOWED_SPLITS), "Combined results contain an unexpected split"
    assert not results["split"].eq("test").any(), "Test rows entered descriptive analysis"
    assert not results.duplicated(["basis_bps", "ticker"]).any(), "Combined results duplicate a basis/ticker key"
    return results


def summarize_group(group: pd.DataFrame) -> dict:
    """Return the pre-registered descriptive metrics for one summary cell."""
    n_trades = len(group)
    total_capital = float(group["capital"].sum())
    return {
        "n_trades": n_trades,
        "n_days": int(group["close_date"].nunique()),
        "hit_rate": float(group["hit"].mean()) if n_trades else np.nan,
        "breakeven_hit_rate": float((group["entry_price"] + group["fee"]).mean()) if n_trades else np.nan,
        "gross_pnl": float(group["gross_pnl"].sum()),
        "fees": float(group["fee"].sum()),
        "net_pnl": float(group["net_pnl"].sum()),
        "net_pnl_per_trade": float(group["net_pnl"].mean()) if n_trades else np.nan,
        "model_expected_per_trade": float(group["net_edge"].mean()) if n_trades else np.nan,
        "market_fair_per_trade": float(group["market_fair_expected"].mean()) if n_trades else np.nan,
        "mean_surprise": float(group["surprise"].mean()) if n_trades else np.nan,
        "total_capital": total_capital,
        "net_pnl_over_capital": float(group["net_pnl"].sum() / total_capital) if total_capital > 0 else np.nan,
        "thin": n_trades < 30,
    }


def summary_record(group, split, basis_bps, threshold_role, breakdown, breakdown_value):
    return {
        "split": split,
        "basis_bps": basis_bps,
        "threshold_role": threshold_role,
        "tier_label": TIER_LABEL,
        "breakdown": breakdown,
        "breakdown_value": str(breakdown_value),
        **summarize_group(group),
    }


def add_partition_records(records, rows, split, basis_bps, threshold_role, breakdown, column, values):
    for frozen_value, display_value in values:
        cell = rows.loc[rows[column].eq(frozen_value)]
        records.append(summary_record(cell, split, basis_bps, threshold_role, breakdown, display_value))


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Build all Section 2.3 breakdowns in frozen, non-performance order."""
    records = []
    for split in ALLOWED_SPLITS:
        for basis_bps in BASIS_SETTINGS:
            rows = results.loc[results["split"].eq(split) & results["basis_bps"].eq(basis_bps)]
            roles = rows["threshold_role"].unique()
            assert len(roles) == 1, f"{split} {basis_bps:g} bps has multiple threshold roles"
            threshold_role = roles[0]
            records.append(summary_record(rows, split, basis_bps, threshold_role, "overall", "all"))
            add_partition_records(records, rows, split, basis_bps, threshold_role, "horizon", "horizon_minutes", HORIZON_VALUES)
            add_partition_records(records, rows, split, basis_bps, threshold_role, "side", "side", SIDE_VALUES)
            add_partition_records(
                records, rows, split, basis_bps, threshold_role, "price_bucket", "price_bucket",
                [(value, value) for value in PRICE_BUCKET_LABELS],
            )
            add_partition_records(
                records, rows, split, basis_bps, threshold_role, "abs_z_bucket", "abs_z_bucket",
                [(value, value) for value in ABS_Z_BUCKETS],
            )
            add_partition_records(
                records, rows, split, basis_bps, threshold_role, "hour_block", "hour_block",
                [(value, value) for value in HOUR_BLOCKS],
            )
            add_partition_records(
                records, rows, split, basis_bps, threshold_role, "hour_utc", "hour_utc",
                [(hour, f"{hour:02d}") for hour in range(24)],
            )
            week_values = sorted(rows["week_start"].unique())
            add_partition_records(
                records, rows, split, basis_bps, threshold_role, "week", "week_start",
                [(value, value) for value in week_values],
            )
            if split == "train":
                sensitivity = rows.loc[
                    ~rows["close_date"].between(TRAIN_FIRST_PARTIAL_WEEK_START, TRAIN_FIRST_PARTIAL_WEEK_END, inclusive="both")
                ]
                assert not sensitivity["close_date"].between(
                    TRAIN_FIRST_PARTIAL_WEEK_START, TRAIN_FIRST_PARTIAL_WEEK_END, inclusive="both"
                ).any(), "First partial train week was not fully excluded"
                records.append(summary_record(
                    sensitivity, split, basis_bps, threshold_role, "train_ex_first_week", TRAIN_EX_FIRST_WEEK_LABEL,
                ))

    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    assert not summary.duplicated(SUMMARY_KEY).any(), f"Summary key {SUMMARY_KEY} is not unique"
    assert set(summary["split"]) == set(ALLOWED_SPLITS) and not summary["split"].eq("test").any(), (
        "Summary contains an unexpected split"
    )
    assert summary["tier_label"].eq(TIER_LABEL).all(), "A summary row has the wrong Tier-2 label"
    assert summary["thin"].eq(summary["n_trades"].lt(30)).all(), "Thin flags disagree with n_trades < 30"
    return summary


def assert_headline_reconciliation(overall: pd.Series, rows: pd.DataFrame) -> None:
    """Reconcile the overall row to the same frozen-result fields used by backtest.headline."""
    assert overall["n_trades"] == len(rows), "Overall trade count differs from the backtest headline"
    headline_values = {
        "gross_pnl": float(rows["gross_pnl"].sum()),
        "fees": float(rows["fee"].sum()),
        "net_pnl": float(rows["net_pnl"].sum()),
        "total_capital": float(rows["capital"].sum()),
    }
    for field, expected in headline_values.items():
        assert close(overall[field], expected, len(rows)), f"Overall {field} differs from the backtest headline"


def assert_reconciliations(summary: pd.DataFrame, results: pd.DataFrame) -> None:
    """Check headline, complete-partition, count, and thin-cell invariants."""
    assert not summary.duplicated(SUMMARY_KEY).any(), f"Summary key {SUMMARY_KEY} is not unique"
    assert summary["thin"].eq(summary["n_trades"].lt(30)).all(), "Thin flags disagree with n_trades < 30"
    for split in ALLOWED_SPLITS:
        for basis_bps in BASIS_SETTINGS:
            source = results.loc[results["split"].eq(split) & results["basis_bps"].eq(basis_bps)]
            subset = summary.loc[summary["split"].eq(split) & summary["basis_bps"].eq(basis_bps)]
            overall_rows = subset.loc[subset["breakdown"].eq("overall")]
            assert len(overall_rows) == 1, f"{split} {basis_bps:g} bps does not have one overall row"
            overall = overall_rows.iloc[0]
            assert_headline_reconciliation(overall, source)
            for breakdown in PARTITION_BREAKDOWNS:
                cells = subset.loc[subset["breakdown"].eq(breakdown)]
                assert len(cells) > 0, f"{split} {basis_bps:g} bps is missing {breakdown} cells"
                assert int(cells["n_trades"].sum()) == int(overall["n_trades"]), (
                    f"{split} {basis_bps:g} bps {breakdown} trade counts do not reconcile"
                )
                for field in RECONCILIATION_FIELDS:
                    assert close(cells[field].sum(), overall[field], len(source)), (
                        f"{split} {basis_bps:g} bps {breakdown} {field} does not reconcile"
                    )

    primary = summary.loc[summary["split"].eq("validation") & summary["basis_bps"].eq(1.2)]
    for breakdown, expected_counts in EXPECTED_VALIDATION_PRIMARY_COUNTS.items():
        actual = primary.loc[primary["breakdown"].eq(breakdown)].set_index("breakdown_value")["n_trades"].to_dict()
        assert actual == expected_counts, f"Validation 1.2 bps {breakdown} counts changed: {actual}"


def write_and_verify_summary(summary: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(SUMMARY_PATH, index=False)
    saved = pd.read_parquet(SUMMARY_PATH)
    assert len(saved) == len(summary), "Summary row count changed after Parquet round trip"
    expected_keys = set(map(tuple, summary[SUMMARY_KEY].itertuples(index=False, name=None)))
    saved_keys = set(map(tuple, saved[SUMMARY_KEY].itertuples(index=False, name=None)))
    assert saved_keys == expected_keys, "Summary keys changed after Parquet round trip"
    assert_reconciliations(saved, results)
    return saved


def population_dates(split: str) -> list[str]:
    start, end, expected_count = POPULATION_CALENDARS[split]
    dates = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    assert len(dates) == expected_count, f"{split} population calendar changed"
    assert "2026-05-26" not in dates, "The excluded 2026-05-26 date entered the population calendar"
    return dates


def build_daily_table(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate frozen trades onto every train/validation population close date."""
    frames = []
    for split in ALLOWED_SPLITS:
        dates = population_dates(split)
        for basis_bps in BASIS_SETTINGS:
            rows = results.loc[results["split"].eq(split) & results["basis_bps"].eq(basis_bps)]
            roles = rows["threshold_role"].unique()
            assert len(roles) == 1, f"{split} {basis_bps:g} bps has multiple threshold roles"
            grouped = rows.groupby("close_date", observed=True, sort=True).agg(
                n_trades=("ticker", "size"),
                gross_pnl=("gross_pnl", "sum"),
                fees=("fee", "sum"),
                net_pnl=("net_pnl", "sum"),
                model_expected_pnl=("net_edge", "sum"),
                market_fair_expected_pnl=("market_fair_expected", "sum"),
                total_capital=("capital", "sum"),
            ).reindex(dates)
            grouped.index.name = "close_date"
            grouped["n_trades"] = grouped["n_trades"].fillna(0).astype("int64")
            grouped[DAILY_ADDITIVE_COLUMNS] = grouped[DAILY_ADDITIVE_COLUMNS].fillna(0.0)
            grouped = grouped.reset_index()
            grouped.insert(0, "tier_label", TIER_LABEL)
            grouped.insert(0, "threshold_role", roles[0])
            grouped.insert(0, "basis_bps", basis_bps)
            grouped.insert(0, "split", split)
            frames.append(grouped[DAILY_COLUMNS])

    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(DAILY_KEY).reset_index(drop=True)
    assert len(daily) == 150, f"Expected 150 daily rows, found {len(daily)}"
    assert not daily.duplicated(DAILY_KEY).any(), f"Daily key {DAILY_KEY} is not unique"
    assert not daily["split"].eq("test").any(), "Test rows entered the daily table"
    zero_trade = daily["n_trades"].eq(0)
    assert daily.loc[zero_trade, DAILY_ADDITIVE_COLUMNS].eq(0.0).all().all(), (
        "A zero-trade population day has a nonzero additive field"
    )
    return daily


def assert_daily_reconciliation(daily: pd.DataFrame, results: pd.DataFrame) -> None:
    assert len(daily) == 150, f"Expected 150 daily rows, found {len(daily)}"
    assert not daily.duplicated(DAILY_KEY).any(), f"Daily key {DAILY_KEY} is not unique"
    assert not daily["split"].eq("test").any(), "Test rows entered daily accounting"
    for split in ALLOWED_SPLITS:
        expected_dates = population_dates(split)
        for basis_bps in BASIS_SETTINGS:
            source = results.loc[results["split"].eq(split) & results["basis_bps"].eq(basis_bps)]
            cells = daily.loc[daily["split"].eq(split) & daily["basis_bps"].eq(basis_bps)]
            assert cells["close_date"].tolist() == expected_dates, f"{split} daily population calendar changed"
            assert len(cells) == POPULATION_CALENDARS[split][2], f"{split} population-day count changed"
            assert int(cells["n_trades"].sum()) == FROZEN_TRADE_COUNTS[(split, basis_bps)], (
                f"{split} {basis_bps:g} bps daily trade count does not reconcile"
            )
            expected_totals = {
                "gross_pnl": source["gross_pnl"].sum(),
                "fees": source["fee"].sum(),
                "net_pnl": source["net_pnl"].sum(),
                "model_expected_pnl": source["net_edge"].sum(),
                "market_fair_expected_pnl": source["market_fair_expected"].sum(),
                "total_capital": source["capital"].sum(),
            }
            for field, expected in expected_totals.items():
                assert close(cells[field].sum(), expected, len(source)), (
                    f"{split} {basis_bps:g} bps daily {field} does not reconcile"
                )


def write_and_verify_daily(daily: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PATH, index=False)
    saved = pd.read_parquet(DAILY_PATH)
    pd.testing.assert_frame_equal(saved, daily, check_exact=True)
    assert_daily_reconciliation(saved, results)
    return saved


def clustered_mean_se(net_pnl, close_date) -> dict:
    """Compute the frozen per-trade naive and traded-day-clustered standard errors."""
    values = pd.Series(net_pnl, dtype="float64").reset_index(drop=True)
    days = pd.Series(close_date, dtype="string").reset_index(drop=True)
    assert len(values) == len(days), "P&L and close-date arrays differ in length"
    assert values.notna().all() and days.notna().all(), "Clustered-SE inputs contain nulls"
    n_trades = len(values)
    if n_trades <= 1:
        raise ValueError("Naive and clustered standard errors require at least two trades")

    mean_pnl = float(values.mean())
    se_naive = float(values.std(ddof=1) / np.sqrt(n_trades))
    clusters = pd.DataFrame({"net_pnl": values, "close_date": days}).groupby(
        "close_date", observed=True, sort=True
    )["net_pnl"].agg(["size", "sum"])
    n_clusters = len(clusters)
    if n_clusters <= 1:
        raise ValueError("The clustered interval requires at least two traded days")
    cluster_scores = clusters["sum"] - mean_pnl * clusters["size"]
    se_cluster = float(np.sqrt(n_clusters / (n_clusters - 1) * np.square(cluster_scores).sum()) / n_trades)

    if se_naive > 0.0:
        se_inflation_factor = float(se_cluster / se_naive)
        design_effect = float(se_inflation_factor ** 2)
        n_eff = float(n_trades / design_effect) if design_effect > 0.0 else np.inf
    elif se_cluster > 0.0:
        se_inflation_factor = np.inf
        design_effect = np.inf
        n_eff = 0.0
    else:
        se_inflation_factor = np.nan
        design_effect = np.nan
        n_eff = np.nan

    return {
        "n_trades": n_trades,
        "n_clusters": n_clusters,
        "mean_net_pnl_per_trade": mean_pnl,
        "se_naive": se_naive,
        "se_cluster": se_cluster,
        "design_effect": design_effect,
        "se_inflation_factor": se_inflation_factor,
        "n_eff": n_eff,
    }


def self_check_clustered_mean_se() -> None:
    values = pd.Series([0.12, -0.31, 0.07, 0.44, -0.18, 0.26], dtype="float64")
    unique_days = pd.Series([f"2026-01-{day:02d}" for day in range(1, len(values) + 1)], dtype="string")
    checked = clustered_mean_se(values, unique_days)
    assert np.isclose(checked["se_cluster"], checked["se_naive"], rtol=0.0, atol=1e-15), (
        "One-trade-per-day clustered SE does not equal naive SE"
    )
    print("PASS: clustered_mean_se synthetic one-trade-per-day check equals naive SE")


def weekly_table(daily: pd.DataFrame) -> pd.DataFrame:
    weekly = daily.copy()
    close_dates = pd.to_datetime(weekly["close_date"], utc=True)
    weekly["week_start"] = (close_dates - pd.to_timedelta(close_dates.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    summary = weekly.groupby(
        ["split", "basis_bps", "threshold_role", "tier_label", "week_start"], observed=True, sort=True, as_index=False
    ).agg(n_population_days=("close_date", "size"), n_trades=("n_trades", "sum"), net_pnl=("net_pnl", "sum"))
    summary["net_pnl_per_trade"] = np.where(summary["n_trades"].gt(0), summary["net_pnl"] / summary["n_trades"], np.nan)
    return summary


def absolute_share(largest_absolute_value, total, n_rows) -> float:
    if close(total, 0.0, n_rows):
        return np.nan
    return float(largest_absolute_value / abs(total))


def primary_interpretation(lower, upper) -> str:
    if upper < 0.0:
        return PRIMARY_INTERPRETATIONS["negative"]
    if lower <= 0.0 <= upper:
        return PRIMARY_INTERPRETATIONS["contains_zero"]
    if lower > 0.0:
        return PRIMARY_INTERPRETATIONS["positive"]
    raise AssertionError("Primary clustered interval did not match a frozen interpretation category")


def build_correlation_table(results: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    records = []
    for split in ALLOWED_SPLITS:
        for basis_bps in BASIS_SETTINGS:
            trades = results.loc[results["split"].eq(split) & results["basis_bps"].eq(basis_bps)]
            days = daily.loc[daily["split"].eq(split) & daily["basis_bps"].eq(basis_bps)]
            weeks = weekly.loc[weekly["split"].eq(split) & weekly["basis_bps"].eq(basis_bps)]
            stats = clustered_mean_se(trades["net_pnl"], trades["close_date"])
            n_trades = stats["n_trades"]
            mean_pnl = stats["mean_net_pnl_per_trade"]
            se_naive = stats["se_naive"]
            se_cluster = stats["se_cluster"]
            naive_lower = mean_pnl - 2.0 * se_naive
            naive_upper = mean_pnl + 2.0 * se_naive
            cluster_lower = mean_pnl - 2.0 * se_cluster
            cluster_upper = mean_pnl + 2.0 * se_cluster
            n_days_with_trade = int(days["n_trades"].gt(0).sum())
            assert stats["n_clusters"] == n_days_with_trade, "Cluster count differs from traded-day count"
            trading_days = days.loc[days["n_trades"].gt(0)]
            best_day = days.loc[days["net_pnl"].idxmax()]
            worst_day = days.loc[days["net_pnl"].idxmin()]
            total_net_pnl = float(days["net_pnl"].sum())
            assert close(weeks["net_pnl"].sum(), total_net_pnl, n_trades), "Weekly net P&L does not reconcile"
            same_sign_weeks = int(np.sign(weeks["net_pnl"]).eq(np.sign(total_net_pnl)).sum())
            interpretation = primary_interpretation(cluster_lower, cluster_upper) if split == "validation" and basis_bps == 1.2 else None
            records.append({
                "split": split,
                "basis_bps": basis_bps,
                "threshold_role": trades["threshold_role"].iloc[0],
                "tier_label": TIER_LABEL,
                "n_trades": n_trades,
                "n_population_days": len(days),
                "n_days_with_trade": n_days_with_trade,
                "mean_net_pnl_per_trade": mean_pnl,
                "se_naive": se_naive,
                "naive_lower_2se": naive_lower,
                "naive_upper_2se": naive_upper,
                "se_cluster": se_cluster,
                "cluster_lower_2se": cluster_lower,
                "cluster_upper_2se": cluster_upper,
                "design_effect": stats["design_effect"],
                "se_inflation_factor": stats["se_inflation_factor"],
                "n_eff": stats["n_eff"],
                "mean_daily_net_pnl": float(days["net_pnl"].mean()),
                "sample_sd_daily_net_pnl": float(days["net_pnl"].std(ddof=1)),
                "positive_trading_day_share": float(trading_days["net_pnl"].gt(0).mean()),
                "best_day": best_day["close_date"],
                "best_day_net_pnl": float(best_day["net_pnl"]),
                "worst_day": worst_day["close_date"],
                "worst_day_net_pnl": float(worst_day["net_pnl"]),
                "largest_day_abs_share_of_total_abs_net_pnl": absolute_share(days["net_pnl"].abs().max(), total_net_pnl, n_trades),
                "n_weeks": len(weeks),
                "n_weeks_same_sign_as_total_net_pnl": same_sign_weeks,
                "largest_week_abs_share_of_total_abs_net_pnl": absolute_share(weeks["net_pnl"].abs().max(), total_net_pnl, n_trades),
                "total_net_pnl": total_net_pnl,
                "primary_interpretation": interpretation,
            })

    correlation = pd.DataFrame(records, columns=CORRELATION_COLUMNS)
    assert not correlation.duplicated(CORRELATION_KEY).any(), f"Correlation key {CORRELATION_KEY} is not unique"
    assert not correlation["split"].eq("test").any(), "Test rows entered correlation statistics"
    assert_correlation_invariants(correlation, results, daily)
    return correlation


def assert_correlation_invariants(correlation: pd.DataFrame, results: pd.DataFrame, daily: pd.DataFrame) -> None:
    assert len(correlation) == len(ALLOWED_SPLITS) * len(BASIS_SETTINGS), "Correlation row count changed"
    assert not correlation.duplicated(CORRELATION_KEY).any(), f"Correlation key {CORRELATION_KEY} is not unique"
    for row in correlation.itertuples(index=False):
        trades = results.loc[results["split"].eq(row.split) & results["basis_bps"].eq(row.basis_bps)]
        days = daily.loc[daily["split"].eq(row.split) & daily["basis_bps"].eq(row.basis_bps)]
        assert row.n_trades == FROZEN_TRADE_COUNTS[(row.split, row.basis_bps)], "Frozen trade count changed"
        assert row.n_population_days == POPULATION_CALENDARS[row.split][2], "Population-day count changed"
        assert row.n_days_with_trade > 1, "Clustered SE requires more than one traded day"
        assert close(row.mean_net_pnl_per_trade, trades["net_pnl"].mean()), "Mean P&L/trade changed"
        assert close((row.naive_lower_2se + row.naive_upper_2se) / 2.0, row.mean_net_pnl_per_trade), (
            "Naive interval is not centered on mean P&L/trade"
        )
        assert close((row.cluster_lower_2se + row.cluster_upper_2se) / 2.0, row.mean_net_pnl_per_trade), (
            "Clustered interval is not centered on mean P&L/trade"
        )
        if np.isfinite(row.design_effect):
            assert close(row.design_effect, row.se_inflation_factor ** 2), "Design effect differs from squared SE inflation"
            if row.design_effect > 0.0:
                assert close(row.n_eff, row.n_trades / row.design_effect), "Effective sample-size identity failed"
        assert close(days["net_pnl"].sum(), trades["net_pnl"].sum(), len(trades)), "Daily net P&L changed"

    primary = correlation.loc[correlation["split"].eq("validation") & correlation["basis_bps"].eq(1.2)]
    assert len(primary) == 1, "Primary validation statistical row is missing"
    primary = primary.iloc[0]
    assert primary["n_trades"] == 719, "Primary validation trade count changed"
    assert primary["n_population_days"] == 21, "Primary validation population-day count changed"
    assert primary["n_days_with_trade"] == 21, "Primary validation traded-day count changed"


def write_and_verify_correlation(correlation: pd.DataFrame, results: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    CORRELATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    correlation.to_parquet(CORRELATION_PATH, index=False)
    saved = pd.read_parquet(CORRELATION_PATH)
    pd.testing.assert_frame_equal(saved, correlation, check_exact=True)
    assert_correlation_invariants(saved, results, daily)
    return saved


def plot_primary_validation(summary: pd.DataFrame) -> None:
    primary = summary.loc[summary["split"].eq("validation") & summary["basis_bps"].eq(1.2)]
    orders = {
        "price_bucket": PRICE_BUCKET_LABELS,
        "abs_z_bucket": ABS_Z_BUCKETS,
        "hour_block": HOUR_BLOCKS,
        "week": ["2026-07-20", "2026-07-27", "2026-08-03"],
    }
    titles = {
        "price_bucket": "Frozen price bucket",
        "abs_z_bucket": "Frozen |z| bucket",
        "hour_block": "UTC decision-hour block",
        "week": "Monday-start week",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, (breakdown, values) in zip(axes.flat, orders.items()):
        cells = primary.loc[primary["breakdown"].eq(breakdown)].set_index("breakdown_value").reindex(values)
        assert cells["n_trades"].notna().all(), f"Primary validation plot is missing a {breakdown} cell"
        positions = np.arange(len(values))
        bars = axis.bar(
            positions, cells["net_pnl_per_trade"], width=0.62, color="C0", alpha=0.72,
            label="Realized net P&L/trade",
        )
        axis.plot(
            positions, cells["market_fair_per_trade"], color="C1", marker="o", linewidth=1.5,
            label="Market-fair/trade",
        )
        thin_positions = positions[cells["thin"].to_numpy(dtype=bool)]
        if len(thin_positions):
            thin_values = cells.loc[cells["thin"], "net_pnl_per_trade"]
            axis.scatter(
                thin_positions, thin_values, marker="x", s=75, linewidths=2, color="black", zorder=4,
                label="Thin (n < 30)",
            )
            for bar, is_thin in zip(bars, cells["thin"]):
                if is_thin:
                    bar.set_hatch("//")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(titles[breakdown])
        axis.set_xticks(positions, values, rotation=30 if breakdown in {"price_bucket", "abs_z_bucket"} else 0)
        axis.set_ylabel("Dollars per one-contract trade")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Stage 0 Primary Validation Diagnostics — 1.2 bps (Tier 2)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert PLOT_PATH.exists(), "Diagnostic breakdown plot was not saved"


def plot_daily_net_pnl(daily: pd.DataFrame) -> None:
    primary = daily.loc[daily["basis_bps"].eq(1.2)]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for axis, split in zip(axes, ALLOWED_SPLITS):
        cells = primary.loc[primary["split"].eq(split)].sort_values("close_date")
        dates = pd.to_datetime(cells["close_date"], utc=True)
        colors = np.where(cells["net_pnl"].ge(0.0), "C0", "C3")
        axis.bar(dates, cells["net_pnl"], width=0.8, color=colors, alpha=0.75)
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_title(f"{split.capitalize()} — 1.2 bps")
        axis.set_ylabel("Daily net P&L ($)")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Stage 0 Primary Daily Net P&L by Frozen close_date (Tier 2)")
    fig.tight_layout()
    DAILY_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DAILY_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert DAILY_PLOT_PATH.exists(), "Daily net-P&L plot was not saved"


def plot_cumulative_net_pnl(results: pd.DataFrame) -> None:
    primary = results.loc[results["basis_bps"].eq(1.2)]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    for axis, split in zip(axes, ALLOWED_SPLITS):
        trades = primary.loc[primary["split"].eq(split)].sort_values(["decision_time", "ticker"], kind="stable")
        axis.plot(trades["decision_time"], trades["net_pnl"].cumsum(), label="Realized net P&L", color="C0")
        axis.plot(trades["decision_time"], trades["net_edge"].cumsum(), label="Model-expected net P&L", color="C2")
        axis.plot(
            trades["decision_time"], trades["market_fair_expected"].cumsum(),
            label="Market-fair expected net P&L", color="C1",
        )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_title(f"{split.capitalize()} — 1.2 bps")
        axis.set_ylabel("Cumulative P&L ($)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Stage 0 Primary Cumulative P&L in Decision-Time Order (Tier 2)")
    fig.tight_layout()
    CUMULATIVE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CUMULATIVE_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert CUMULATIVE_PLOT_PATH.exists(), "Cumulative P&L plot was not saved"


def print_primary_validation_tables(summary: pd.DataFrame) -> None:
    primary = summary.loc[summary["split"].eq("validation") & summary["basis_bps"].eq(1.2)]
    columns = [
        "breakdown_value", "n_trades", "n_days", "hit_rate", "breakeven_hit_rate", "net_pnl",
        "net_pnl_per_trade", "model_expected_per_trade", "market_fair_per_trade", "mean_surprise", "thin",
    ]
    formatters = {
        field: "{:.6f}".format for field in [
            "hit_rate", "breakeven_hit_rate", "net_pnl", "net_pnl_per_trade",
            "model_expected_per_trade", "market_fair_per_trade", "mean_surprise",
        ]
    }
    print(f"\n{TIER_LABEL}")
    print("Primary validation descriptive diagnostics (basis_bps = 1.2)")
    for breakdown in ["horizon", "side", "price_bucket", "abs_z_bucket", "hour_block", "week"]:
        print(f"\n{breakdown}")
        print(primary.loc[primary["breakdown"].eq(breakdown), columns].to_string(index=False, formatters=formatters))
    print("\nNote: T-5 rows are conditional later entries for markets not already traded at T-10; this is not a clean horizon experiment.")


def print_statistical_diagnostics(correlation: pd.DataFrame, weekly: pd.DataFrame) -> None:
    print(f"\n{TIER_LABEL}")
    print("Day-clustered uncertainty (D is traded close_date days)")
    for row in correlation.itertuples(index=False):
        print(
            f"{row.split:10s} {row.basis_bps:>3.1f} bps | N={row.n_trades:4d} | "
            f"population_days={row.n_population_days:2d} | trading_days={row.n_days_with_trade:2d} | "
            f"mean={row.mean_net_pnl_per_trade:.8f} | naive_SE={row.se_naive:.8f} | "
            f"cluster_SE={row.se_cluster:.8f} | "
            f"naive_2SE=[{row.naive_lower_2se:.8f}, {row.naive_upper_2se:.8f}] | "
            f"cluster_2SE=[{row.cluster_lower_2se:.8f}, {row.cluster_upper_2se:.8f}] | "
            f"DE={row.design_effect:.6f} | inflation={row.se_inflation_factor:.6f} | N_eff={row.n_eff:.3f}"
        )

    primary = correlation.loc[correlation["split"].eq("validation") & correlation["basis_bps"].eq(1.2)].iloc[0]
    print("\nPRIMARY VALIDATION 1.2 BPS")
    print(f"mean net P&L/trade: {primary['mean_net_pnl_per_trade']:.8f}")
    print(f"clustered SE: {primary['se_cluster']:.8f}")
    print(f"clustered ±2SE lower: {primary['cluster_lower_2se']:.8f}")
    print(f"clustered ±2SE upper: {primary['cluster_upper_2se']:.8f}")
    print(f"design effect: {primary['design_effect']:.6f}")
    print(f"SE inflation factor: {primary['se_inflation_factor']:.6f}")
    print(f"effective sample size: {primary['n_eff']:.3f}")
    print(f"interpretation: {primary['primary_interpretation']}")

    daily_columns = [
        "split", "basis_bps", "n_population_days", "n_days_with_trade", "mean_daily_net_pnl",
        "sample_sd_daily_net_pnl", "positive_trading_day_share", "best_day", "best_day_net_pnl",
        "worst_day", "worst_day_net_pnl", "largest_day_abs_share_of_total_abs_net_pnl",
    ]
    print("\nDaily concentration diagnostics")
    print(correlation[daily_columns].to_string(index=False, formatters={
        "mean_daily_net_pnl": "{:.6f}".format,
        "sample_sd_daily_net_pnl": "{:.6f}".format,
        "positive_trading_day_share": "{:.6f}".format,
        "best_day_net_pnl": "{:.6f}".format,
        "worst_day_net_pnl": "{:.6f}".format,
        "largest_day_abs_share_of_total_abs_net_pnl": "{:.6f}".format,
    }))

    print("\nWeekly net P&L diagnostics")
    print(weekly[[
        "split", "basis_bps", "week_start", "n_population_days", "n_trades", "net_pnl", "net_pnl_per_trade",
    ]].to_string(
        index=False, formatters={"net_pnl": "{:.6f}".format, "net_pnl_per_trade": "{:.6f}".format},
    ))
    weekly_columns = [
        "split", "basis_bps", "n_weeks", "n_weeks_same_sign_as_total_net_pnl",
        "largest_week_abs_share_of_total_abs_net_pnl",
    ]
    print("\nWeekly concentration diagnostics")
    print(correlation[weekly_columns].to_string(index=False, formatters={
        "largest_week_abs_share_of_total_abs_net_pnl": "{:.6f}".format,
    }))
    print("\nStatistical caveats: primary validation has only 21 day clusters, so its clustered SE may itself be noisy.")
    print("Validation already contributed calibration-error information to the threshold.")
    print("Tier 2 assumes optimistic top-of-book, one-contract fills; the test split remains untouched.")


def main() -> None:
    self_check_clustered_mean_se()
    results = load_frozen_results()
    summary = build_summary(results)
    assert_reconciliations(summary, results)
    saved = write_and_verify_summary(summary, results)
    daily = write_and_verify_daily(build_daily_table(results), results)
    weekly = weekly_table(daily)
    correlation = write_and_verify_correlation(build_correlation_table(results, daily, weekly), results, daily)
    plot_primary_validation(saved)
    plot_daily_net_pnl(daily)
    plot_cumulative_net_pnl(results)
    print_primary_validation_tables(saved)
    print_statistical_diagnostics(correlation, weekly)
    print(f"\nPASS: all complete breakdowns reconcile to their frozen backtest headlines")
    print("PASS: validation 1.2 bps pre-registered cell counts match")
    print("PASS: every thin flag equals n_trades < 30")
    print(f"PASS: {len(saved):,} summary rows and keys round-tripped through {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PASS: {len(daily):,} daily rows round-tripped and reconciled through {DAILY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PASS: {len(correlation):,} correlation rows round-tripped through {CORRELATION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved diagnostic plot to {PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved daily net-P&L plot to {DAILY_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved cumulative P&L plot to {CUMULATIVE_PLOT_PATH.relative_to(PROJECT_ROOT)}")
    print("No test data, source features, predictions, quotes, settlements, or expiration values were loaded.")


if __name__ == "__main__":
    main()
