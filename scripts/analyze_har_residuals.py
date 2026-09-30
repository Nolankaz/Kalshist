"""Describe frozen Day 13 H1 HAR forecast residuals without changing the model."""

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

from scripts.analyze_backtest import HOUR_BLOCKS, clustered_mean_se
from scripts.count_threshold_clearance import assert_safe_columns
from scripts.evaluation_split import SPLIT_RANGES
from scripts.model_har_rv import CONFIG_H1, EXPECTED_HORIZON_SCORE_ROWS, FORECAST_COLUMNS, FORECAST_PATH, assert_forecasts, forecast_fingerprint
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, ROW_KEY
from scripts.walk_forward import FROZEN_FOLDS


TARGET_PATH = PROJECT_ROOT / "data/targets/forward_vol_target.parquet"
MARKET_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
DERIVED_PATH = PROJECT_ROOT / "data/features/derived_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/har_residual_summary.parquet"
REGIME_PLOT_PATH = PROJECT_ROOT / "data/models/plots/har_residuals_by_regime.png"
FORECAST_PLOT_PATH = PROJECT_ROOT / "data/models/plots/har_residuals_vs_forecast.png"
FROZEN_FORECAST_FINGERPRINT = "d2b1ea6776013e3d3ee5582c0bfa89465ce7c87ee07ee0b31aa472eeaa50acc9"
TARGET_COLUMNS = ROW_KEY + ["split", "close_date", "decision_time", "fwd_log_rv"]
MARKET_COLUMNS = ROW_KEY + ["close_date", "decision_time", "5min_ewma_vol"]
DERIVED_COLUMNS = ROW_KEY + ["split", "close_date", "is_common"]
TERCILE_EDGES = {10: (0.6337964396166403, 0.8837207863551418), 5: (0.623421011561805, 0.8606540123954035)}
TERCILE_LABELS = ("low", "middle", "high")
WEEK_LABELS = ("weekday", "weekend")
DECILE_LABELS = tuple(f"D{index}" for index in range(1, 11))
EXPECTED_H1_TARGET_ROWS = {10: 3_473, 5: 3_380}
EXPECTED_FOLD_TARGET_ROWS = {1: {10: 637, 5: 626}, 2: {10: 611, 5: 596}, 3: {10: 581, 5: 570},
                             4: {10: 545, 5: 531}, 5: {10: 581, 5: 571}, 6: {10: 518, 5: 486}}
POPULATIONS = ("all_oof", "train_folds_1_3", "validation_folds_4_6")
COMPARABLE_CUTS = {"hour_block": tuple(HOUR_BLOCKS), "weekend": WEEK_LABELS, "trailing_vol_tercile": TERCILE_LABELS,
                   "forecast_decile": DECILE_LABELS}
KEY = ["population", "horizon_minutes", "cut", "cell"]
OUTPUT_COLUMNS = KEY + ["n", "mean_residual", "naive_se", "clustered_se", "n_clusters", "design_effect", "thin",
                        "significant_clustered", "structure", "train_side_qualifies", "validation_side_qualifies",
                        "cross_boundary_same_sign", "forecast_bin_lower", "forecast_bin_upper", "lag1_autocorrelation",
                        "n_pairs", "n_days_contributing"]


def row_keys(frame):
    assert not frame.duplicated(ROW_KEY).any(), "Duplicate market/horizon row key"
    return set(map(tuple, frame[ROW_KEY].itertuples(index=False, name=None)))


def load_frozen_h1_forecasts():
    """Fingerprint the full artifact before any target is loaded or scored."""
    assert pq.ParquetFile(FORECAST_PATH).metadata.num_rows == 10_346, "Frozen forecast row count changed"
    forecasts = pd.read_parquet(FORECAST_PATH, columns=FORECAST_COLUMNS, filters=[("close_date", "<=", SPLIT_RANGES["validation"][1])])
    assert_forecasts(forecasts)
    assert forecast_fingerprint(forecasts) == FROZEN_FORECAST_FINGERPRINT, "STOP: frozen HAR forecast fingerprint changed"
    assert len(forecasts) == 10_346 and not any(name.startswith("fwd_") for name in forecasts.columns), "Unsafe forecast artifact"
    assert forecasts["split"].isin(("train", "validation")).all() and forecasts["close_date"].le(SPLIT_RANGES["validation"][1]).all(), "Test forecast row loaded"
    h1 = forecasts.loc[forecasts["configuration"].eq(CONFIG_H1)].copy()
    assert len(h1) == 7_012 and h1.groupby("horizon_minutes").size().to_dict() == EXPECTED_HORIZON_SCORE_ROWS[CONFIG_H1], "H1 score population changed"
    assert set(h1["fold"]) == set(range(1, 7)) and not h1.duplicated(ROW_KEY).any(), "H1 folds/keys changed"
    for fold in FROZEN_FOLDS:
        section = h1.loc[h1["fold"].eq(fold.number)]
        assert section["close_date"].between(fold.score_start, fold.score_end).all(), f"Fold {fold.number} score dates changed"
        expected_split = "train" if fold.number <= 3 else "validation"
        assert section["split"].eq(expected_split).all(), f"Fold {fold.number} split changed"
    return h1


def load_context(h1):
    assert_safe_columns(MARKET_COLUMNS, "Day 13 residual structural market projection")
    market = pd.read_parquet(MARKET_PATH, columns=MARKET_COLUMNS, filters=[("close_date", "<=", SPLIT_RANGES["validation"][1])])
    derived = pd.read_parquet(DERIVED_PATH, columns=DERIVED_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    target = pd.read_parquet(TARGET_PATH, columns=TARGET_COLUMNS, filters=[("split", "in", ["train", "validation"])])
    assert list(market.columns) == MARKET_COLUMNS and list(derived.columns) == DERIVED_COLUMNS and list(target.columns) == TARGET_COLUMNS, "Input projection changed"
    assert_safe_columns(market.columns, "Day 13 residual loaded market columns")
    assert len(market) == len(derived) == len(target) == 14_358, "Train/validation structural count changed"
    assert pq.ParquetFile(TARGET_PATH).metadata.num_rows == 14_358, "Target file unexpectedly includes another period"
    for name, frame in (("market", market), ("derived", derived), ("target", target)):
        assert not frame.duplicated(ROW_KEY).any() and frame["close_date"].between(SPLIT_RANGES["train"][0], SPLIT_RANGES["validation"][1]).all(), f"{name} contains duplicate/test keys"
    assert derived["split"].isin(("train", "validation")).all() and target["split"].isin(("train", "validation")).all(), "Test split loaded"
    common = derived.loc[derived["is_common"], ROW_KEY + ["split", "close_date"]].copy()
    assert common.groupby(["split", "horizon_minutes"], observed=True).size().to_dict() == EXPECTED_COMMON_ROWS, "Day 9 common population changed"
    expected_parts = []
    for fold in FROZEN_FOLDS:
        expected_parts.append(common.loc[common["close_date"].between(fold.score_start, fold.score_end)])
    expected = pd.concat(expected_parts, ignore_index=True)
    assert len(expected) == len(h1) and row_keys(expected) == row_keys(h1), "H1 keys do not match frozen common score windows"
    joined = h1.merge(market, on=ROW_KEY + ["close_date"], how="left", validate="one_to_one", indicator=True)
    assert len(joined) == len(h1) and joined["_merge"].eq("both").all(), "H1/market structural join changed"
    joined = joined.drop(columns="_merge")
    joined = joined.merge(target, on=ROW_KEY, how="left", validate="one_to_one", suffixes=("", "_target"), indicator=True)
    assert len(joined) == len(h1) and joined["_merge"].eq("both").all(), "H1/target keys changed"
    assert joined["split"].eq(joined["split_target"]).all() and joined["close_date"].eq(joined["close_date_target"]).all(), "Target split/date disagrees with forecast"
    assert joined["decision_time"].eq(joined["decision_time_target"]).all(), "Target/market decision times disagree"
    joined = joined.drop(columns=["split_target", "close_date_target", "decision_time_target", "_merge"])
    assert isinstance(joined["decision_time"].dtype, pd.DatetimeTZDtype) and str(joined["decision_time"].dt.tz) == "UTC", "Decision time is not UTC-aware"
    assert joined["decision_time"].notna().all() and joined["decision_time"].lt(pd.Timestamp("2026-08-10", tz="UTC")).all(), "Test decision time loaded"
    retained = joined.loc[joined["fwd_log_rv"].notna()].copy()
    assert retained.groupby("horizon_minutes").size().to_dict() == EXPECTED_H1_TARGET_ROWS and not retained.duplicated(ROW_KEY).any(), "H1 target-bearing population changed"
    assert np.isfinite(retained[["fwd_log_rv", "log_rv_har", "5min_ewma_vol"]].to_numpy(dtype=float)).all(), "Non-finite residual input"
    assert retained["5min_ewma_vol"].gt(0).all(), "Trailing volatility must be positive"
    for fold in FROZEN_FOLDS:
        observed = retained.loc[retained["fold"].eq(fold.number)].groupby("horizon_minutes").size().to_dict()
        assert observed == EXPECTED_FOLD_TARGET_ROWS[fold.number], f"Fold {fold.number} target-bearing counts changed: {observed}"
    return retained


def assign_frozen_cuts(rows):
    """Freeze OOF-derived forecast boundaries before computing any residual summary."""
    rows = rows.copy()
    hours = rows["decision_time"].dt.hour
    rows["hour_block"] = pd.cut(hours, bins=[0, 6, 12, 18, 24], labels=HOUR_BLOCKS, right=False).astype("str")
    rows["weekend"] = np.where(rows["decision_time"].dt.dayofweek.ge(5), "weekend", "weekday")
    boundaries = {}
    for horizon in (10, 5):
        mask = rows["horizon_minutes"].eq(horizon)
        subset = rows.loc[mask]
        lower, upper = TERCILE_EDGES[horizon]
        values = subset["5min_ewma_vol"].to_numpy(dtype=float)
        assert 0 < lower < upper and np.isfinite(values).all() and (values > 0).all(), "Frozen tercile/input invalid"
        rows.loc[mask, "trailing_vol_tercile"] = pd.cut(values, bins=[0, lower, upper, np.inf], labels=TERCILE_LABELS, right=False).astype("str").to_numpy()
        qcut_labels, edges = pd.qcut(subset["log_rv_har"], q=10, labels=False, retbins=True, duplicates="raise")
        assert len(edges) == 11 and np.isfinite(edges).all() and np.all(np.diff(edges) > 0), f"T-{horizon} cannot form ten distinct forecast deciles"
        reapplied = pd.cut(subset["log_rv_har"], bins=edges, labels=DECILE_LABELS, right=True, include_lowest=True)
        assert reapplied.notna().all() and np.array_equal(reapplied.cat.codes.to_numpy(), qcut_labels.to_numpy(dtype=int)), "OOF forecast decile boundaries do not reproduce equal-frequency bins"
        rows.loc[mask, "forecast_decile"] = reapplied.astype("str").to_numpy()
        boundaries[horizon] = edges.astype(float)
    for cut, labels in COMPARABLE_CUTS.items():
        assert rows[cut].isin(labels).all(), f"{cut} leaves a target-bearing row unassigned"
        assert rows.groupby("horizon_minutes")[cut].count().to_dict() == EXPECTED_H1_TARGET_ROWS, f"{cut} does not cover H1"
    return rows, boundaries


def select_populations(rows):
    parts = {"all_oof": rows, "train_folds_1_3": rows.loc[rows["fold"].between(1, 3)],
             "validation_folds_4_6": rows.loc[rows["fold"].between(4, 6)]}
    assert parts["train_folds_1_3"]["split"].eq("train").all() and parts["validation_folds_4_6"]["split"].eq("validation").all(), "Fold-side split changed"
    assert len(parts["train_folds_1_3"]) + len(parts["validation_folds_4_6"]) == len(rows), "Fold-side populations do not partition H1"
    for population, part in parts.items():
        assert not part.duplicated(ROW_KEY).any() and part["fwd_log_rv"].notna().all(), f"{population}: invalid residual population"
    return parts


def base_record(population, horizon, cut, cell, n):
    return {"population": population, "horizon_minutes": horizon, "cut": cut, "cell": str(cell), "n": n,
            "mean_residual": np.nan, "naive_se": np.nan, "clustered_se": np.nan, "n_clusters": 0,
            "design_effect": np.nan, "thin": n < 30, "significant_clustered": pd.NA, "structure": pd.NA,
            "train_side_qualifies": pd.NA, "validation_side_qualifies": pd.NA, "cross_boundary_same_sign": pd.NA,
            "forecast_bin_lower": np.nan, "forecast_bin_upper": np.nan, "lag1_autocorrelation": np.nan,
            "n_pairs": pd.NA, "n_days_contributing": pd.NA}


def mean_cell(population, horizon, cut, cell, rows, boundaries):
    record = base_record(population, horizon, cut, cell, len(rows))
    if cut == "forecast_decile":
        index = DECILE_LABELS.index(cell)
        record["forecast_bin_lower"] = float(boundaries[horizon][index])
        record["forecast_bin_upper"] = float(boundaries[horizon][index + 1])
    if rows.empty:
        return record
    values = rows["residual"].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "Non-finite residual in a cell"
    record["mean_residual"] = float(values.mean())
    record["n_clusters"] = int(rows["close_date"].nunique())
    if len(rows) > 1:
        record["naive_se"] = float(np.std(values, ddof=1) / np.sqrt(len(rows)))
    if len(rows) > 1 and record["n_clusters"] > 1:
        checked = clustered_mean_se(values, rows["close_date"])
        assert checked["n_trades"] == len(rows) and checked["n_clusters"] == record["n_clusters"], "Cluster helper cell mismatch"
        assert abs(checked["mean_net_pnl_per_trade"] - record["mean_residual"]) <= 1e-12, "Cluster helper mean changed"
        record["naive_se"] = checked["se_naive"]
        record["clustered_se"] = checked["se_cluster"]
        record["design_effect"] = checked["design_effect"]
        record["significant_clustered"] = bool(abs(record["mean_residual"]) > 2 * record["clustered_se"])
    return record


def lag1_cell(population, horizon, rows):
    record = base_record(population, horizon, "lag1_autocorrelation", "within_close_date", len(rows))
    lagged, current = [], []
    for _, day in rows.groupby("close_date", observed=True, sort=True):
        ordered = day.sort_values(["decision_time", "ticker"], kind="stable")
        assert ordered["horizon_minutes"].eq(horizon).all() and ordered["close_date"].nunique() == 1, "Lag pair crosses horizon/date"
        values = ordered["residual"].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        lagged.extend(values[:-1])
        current.extend(values[1:])
    n_pairs = len(lagged)
    n_days = int(rows.groupby("close_date", observed=True).size().ge(2).sum())
    assert n_pairs == sum(max(len(day) - 1, 0) for _, day in rows.groupby("close_date", observed=True)), "Lag pairs crossed a date boundary"
    record["n_clusters"] = int(rows["close_date"].nunique())
    record["n_pairs"] = n_pairs
    record["n_days_contributing"] = n_days
    if n_pairs >= 2 and np.std(lagged) > 0 and np.std(current) > 0:
        record["lag1_autocorrelation"] = float(np.corrcoef(lagged, current)[0, 1])
    assert pd.isna(record["lag1_autocorrelation"]) or -1 <= record["lag1_autocorrelation"] <= 1, "Invalid lag-1 correlation"
    return record


def build_summary(rows, boundaries):
    parts = select_populations(rows)
    records = []
    for population in POPULATIONS:
        for horizon in (10, 5):
            parent = parts[population].loc[parts[population]["horizon_minutes"].eq(horizon)]
            assert len(parent) > 0, f"{population} T-{horizon} is empty"
            records.append(mean_cell(population, horizon, "overall", "overall", parent, boundaries))
            for cut, labels in COMPARABLE_CUTS.items():
                total = 0
                for cell in labels:
                    subset = parent.loc[parent[cut].eq(cell)]
                    total += len(subset)
                    records.append(mean_cell(population, horizon, cut, cell, subset, boundaries))
                assert total == len(parent), f"{population} T-{horizon} {cut} cells do not partition the parent"
            records.append(lag1_cell(population, horizon, parent))
    for horizon in (10, 5):
        parent = parts["all_oof"].loc[parts["all_oof"]["horizon_minutes"].eq(horizon)]
        total = 0
        for fold in FROZEN_FOLDS:
            subset = parent.loc[parent["fold"].eq(fold.number)]
            assert len(subset) == EXPECTED_FOLD_TARGET_ROWS[fold.number][horizon], f"Fold {fold.number} T-{horizon} count changed"
            total += len(subset)
            records.append(mean_cell("all_oof", horizon, "fold", str(fold.number), subset, boundaries))
        assert total == len(parent), "Fold cells do not partition all OOF"

    summary = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    lookup = summary.set_index(KEY)
    for horizon in (10, 5):
        for cut, cells in COMPARABLE_CUTS.items():
            for cell in cells:
                train = lookup.loc[("train_folds_1_3", horizon, cut, cell)]
                validation = lookup.loc[("validation_folds_4_6", horizon, cut, cell)]
                train_qualifies = bool(train["n"] >= 30 and train["significant_clustered"] is not pd.NA and train["significant_clustered"])
                validation_qualifies = bool(validation["n"] >= 30 and validation["significant_clustered"] is not pd.NA and validation["significant_clustered"])
                same_sign = bool(pd.notna(train["mean_residual"]) and pd.notna(validation["mean_residual"])
                                 and train["mean_residual"] * validation["mean_residual"] > 0)
                structure = train_qualifies and validation_qualifies and same_sign
                mask = summary["horizon_minutes"].eq(horizon) & summary["cut"].eq(cut) & summary["cell"].eq(cell)
                assert mask.sum() == 3, "Comparable train/validation/all-OOF cell missing"
                summary.loc[mask, "train_side_qualifies"] = train_qualifies
                summary.loc[mask, "validation_side_qualifies"] = validation_qualifies
                summary.loc[mask, "cross_boundary_same_sign"] = same_sign
                summary.loc[mask, "structure"] = structure
    for column in ("significant_clustered", "structure", "train_side_qualifies", "validation_side_qualifies", "cross_boundary_same_sign"):
        summary[column] = summary[column].astype("boolean")
    for column in ("n_pairs", "n_days_contributing"):
        summary[column] = summary[column].astype("Int64")
    return summary.sort_values(KEY).reset_index(drop=True)


def assert_summary(summary, boundaries):
    assert list(summary.columns) == OUTPUT_COLUMNS and len(summary) == 138 and not summary.duplicated(KEY).any(), "Residual summary schema/key/count changed"
    assert set(summary["population"]) == set(POPULATIONS) and set(summary["horizon_minutes"]) == {10, 5}, "Residual populations changed"
    assert summary["n"].ge(0).all() and summary["thin"].eq(summary["n"].lt(30)).all(), "Thin/count invariant failed"
    for population in POPULATIONS:
        for horizon in (10, 5):
            parent = summary.loc[summary["population"].eq(population) & summary["horizon_minutes"].eq(horizon)]
            overall = parent.loc[parent["cut"].eq("overall")]
            assert len(overall) == 1 and int(overall["n"].iloc[0]) > 0, "Overall residual row missing"
            for cut, labels in COMPARABLE_CUTS.items():
                cells = parent.loc[parent["cut"].eq(cut)]
                assert set(cells["cell"]) == set(labels) and int(cells["n"].sum()) == int(overall["n"].iloc[0]), f"{population} T-{horizon} {cut} partition failed"
                if cut == "forecast_decile":
                    ordered = cells.set_index("cell").loc[list(DECILE_LABELS)]
                    assert np.array_equal(ordered["forecast_bin_lower"].to_numpy(dtype=float), boundaries[horizon][:-1]), "Forecast lower boundaries changed"
                    assert np.array_equal(ordered["forecast_bin_upper"].to_numpy(dtype=float), boundaries[horizon][1:]), "Forecast upper boundaries changed"
            lag = parent.loc[parent["cut"].eq("lag1_autocorrelation")]
            assert len(lag) == 1 and int(lag["n"].iloc[0]) == int(overall["n"].iloc[0]), "Lag population changed"
            assert lag[["mean_residual", "naive_se", "clustered_se", "design_effect"]].isna().all().all(), "Lag correlation disguised as mean residual"
    for horizon in (10, 5):
        fold = summary.loc[summary["population"].eq("all_oof") & summary["horizon_minutes"].eq(horizon) & summary["cut"].eq("fold")]
        assert set(fold["cell"]) == {str(i) for i in range(1, 7)} and int(fold["n"].sum()) == EXPECTED_H1_TARGET_ROWS[horizon], "Fold summary partition changed"
    comparable = summary["cut"].isin(COMPARABLE_CUTS)
    assert summary.loc[comparable, "structure"].notna().all() and summary.loc[~comparable, "structure"].isna().all(), "Formal structure scope changed"
    assert summary.loc[summary["structure"].fillna(False), "thin"].eq(False).all(), "Thin cell marked as structure"
    assert summary.loc[summary["cut"].eq("forecast_decile"), ["forecast_bin_lower", "forecast_bin_upper"]].notna().all().all(), "Forecast boundaries missing"
    assert summary.loc[summary["cut"].ne("forecast_decile"), ["forecast_bin_lower", "forecast_bin_upper"]].isna().all().all(), "Forecast boundary on unrelated cut"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(summary, boundaries):
    assert_summary(summary, boundaries)
    prior_hash = None
    if OUTPUT_PATH.exists():
        prior = pd.read_parquet(OUTPUT_PATH)
        assert_summary(prior, boundaries)
        pd.testing.assert_frame_equal(prior, summary, check_exact=True)
        prior_hash = sha256(OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, prefix=".har_residual_summary_", suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        summary.to_parquet(temporary, index=False)
        saved = pd.read_parquet(temporary)
        assert_summary(saved, boundaries)
        pd.testing.assert_frame_equal(saved, summary, check_exact=True)
        if prior_hash is not None:
            assert sha256(temporary) == prior_hash, "Deterministic residual Parquet hash changed"
        os.replace(temporary, OUTPUT_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return prior_hash is not None


def atomic_plot(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}_", suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, dpi=160, bbox_inches="tight")
        assert temporary.stat().st_size > 0, "Residual plot is empty"
        if path.exists():
            assert sha256(temporary) == sha256(path), f"Deterministic plot hash changed: {path.name}"
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def plot_regimes(summary):
    cut_specs = (("hour_block", HOUR_BLOCKS, "UTC decision hour"), ("weekend", WEEK_LABELS, "UTC weekday/weekend"),
                 ("trailing_vol_tercile", TERCILE_LABELS, "Frozen EWMA-5 tercile"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    for row_index, horizon in enumerate((10, 5)):
        for col_index, (cut, labels, title) in enumerate(cut_specs):
            axis = axes[row_index, col_index]
            for population, offset, color, label in (("train_folds_1_3", -0.1, "#1665a5", "Train folds 1–3"),
                                                      ("validation_folds_4_6", 0.1, "#e3821c", "Validation folds 4–6")):
                cells = summary.loc[summary["population"].eq(population) & summary["horizon_minutes"].eq(horizon) & summary["cut"].eq(cut)].set_index("cell").loc[list(labels)]
                assert cells["clustered_se"].notna().all(), f"Cannot plot missing clustered SE: {population}/T-{horizon}/{cut}"
                x = np.arange(len(labels), dtype=float) + offset
                axis.errorbar(x, cells["mean_residual"].to_numpy(dtype=float), yerr=2 * cells["clustered_se"].to_numpy(dtype=float),
                              marker="o", capsize=3, linewidth=1.4, color=color, label=label)
            axis.axhline(0, color="black", linestyle="--", linewidth=1)
            axis.set_xticks(range(len(labels)), labels)
            axis.set_title(f"T-{horizon} · {title}")
            axis.grid(alpha=0.2)
            if col_index == 0:
                axis.set_ylabel("Mean residual ± 2 day-clustered SE")
    axes[0, 2].legend(loc="best", fontsize=8)
    fig.suptitle("Frozen HAR residuals by regime · target − forecast")
    fig.tight_layout()
    atomic_plot(fig, REGIME_PLOT_PATH)


def plot_forecast_deciles(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, horizon in zip(axes, (10, 5)):
        for population, color, label in (("train_folds_1_3", "#1665a5", "Train folds 1–3"),
                                         ("validation_folds_4_6", "#e3821c", "Validation folds 4–6")):
            cells = summary.loc[summary["population"].eq(population) & summary["horizon_minutes"].eq(horizon) & summary["cut"].eq("forecast_decile")].set_index("cell").loc[list(DECILE_LABELS)]
            assert cells["clustered_se"].notna().all(), f"Cannot plot missing decile clustered SE: {population}/T-{horizon}"
            assert np.all(np.diff(cells["forecast_bin_lower"].to_numpy(dtype=float)) > 0), "Forecast deciles are not ordered"
            axis.errorbar(range(1, 11), cells["mean_residual"].to_numpy(dtype=float), yerr=2 * cells["clustered_se"].to_numpy(dtype=float),
                          marker="o", capsize=3, linewidth=1.4, color=color, label=label)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"T-{horizon}", xlabel="HAR log-volatility forecast decile", xticks=range(1, 11))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean residual ± 2 day-clustered SE")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Frozen HAR residuals versus forecast · target − forecast")
    fig.tight_layout()
    atomic_plot(fig, FORECAST_PLOT_PATH)


def main():
    h1 = load_frozen_h1_forecasts()
    print(f"Frozen forecast fingerprint: {FROZEN_FORECAST_FINGERPRINT} (PASS before target load)")
    rows = load_context(h1)
    rows, boundaries = assign_frozen_cuts(rows)
    rows["residual"] = rows["fwd_log_rv"] - rows["log_rv_har"]
    assert np.isfinite(rows["residual"].to_numpy(dtype=float)).all(), "Non-finite HAR residual"
    summary = build_summary(rows, boundaries)
    deterministic_prior = write_summary(summary, boundaries)
    plot_regimes(summary)
    plot_forecast_deciles(summary)
    print("H1 target-bearing rows: T-10=3,473; T-5=3,380")
    print("Frozen terciles, OOF forecast deciles, fold/side populations, partition sums, clustered structure rule, lag-day boundaries, and round trip: PASS")
    print("Forecast decile edges: " + "; ".join(f"T-{horizon}={boundaries[horizon].tolist()}" for horizon in (10, 5)))
    flagged = summary.loc[summary["structure"].fillna(False) & summary["population"].eq("all_oof"), ["horizon_minutes", "cut", "cell"]]
    print(f"Formal cross-boundary structure cells: {flagged.to_dict('records')}")
    print(f"Summary rows: {len(summary)}; deterministic prior comparison: {'PASS' if deterministic_prior else 'first write'}")
    print(f"Written: {OUTPUT_PATH}; {REGIME_PLOT_PATH}; {FORECAST_PLOT_PATH}")


if __name__ == "__main__":
    main()
