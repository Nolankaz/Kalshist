"""Rebuild Day 10 candidates and freeze outcome-free Day 11 Stage 0 trades."""

from decimal import Decimal
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.count_threshold_clearance import (
    ALLOWED_SPLITS,
    BASIS_SETTINGS,
    EDGE_TOLERANCE,
    FEATURES_PATH,
    HORIZONS,
    OUTPUT_PATH as DAY10_SUMMARY_PATH,
    PLATT_TOLERANCE,
    PREDICTIONS_PATH,
    PROBABILITY_VERSIONS,
    ROW_KEY,
    WATERFALL_COUNT_COLUMNS,
    MARKET_COUNT_COLUMNS,
    MARKET_SUMMARY_HORIZON,
    SELECTED_CANDIDATE,
    apply_stale_rule_and_buckets,
    assert_safe_columns,
    build_probability_rows,
    build_summary,
    calculate_execution_fees,
    join_quotes,
    join_thresholds,
    load_common_prediction_rows,
    load_legitimate_platt_parameters,
    load_quote_rows,
    load_thresholds,
    reconstruct_platt_probability,
    validate_waterfalls,
    verify_platt_carry_forward,
)
from scripts.evaluation_split import SPLIT_KEY, SPLIT_RANGES
from scripts.analyze_spreads import PRICE_BUCKET_LABELS, integer_mils
from scripts.fees import DIRECT_MEMBER, KXBTC15M_FEE_MULTIPLIER, KXBTC15M_FEE_TYPE, calculate_single_fill_fee, is_on_tick_grid
from scripts.score_stage0 import EXPECTED_COMMON_ROWS


FEATURE_EXTRA_COLUMNS = ROW_KEY + ["decision_time", "hour_utc"]
PREDICTION_EXTRA_COLUMNS = ROW_KEY + ["z_5min_ewma_vol"]
SUMMARY_KEY = ["split", "horizon_minutes", "basis_bps", "probability_version", "breakdown", "breakdown_value"]
SUMMARY_COUNT_COLUMNS = WATERFALL_COUNT_COLUMNS + MARKET_COUNT_COLUMNS
CANDIDATE_KEY = ROW_KEY + ["split", "probability_version", "basis_bps"]
TRADE_KEY = ["basis_bps", "ticker"]
TRADE_PATH = PROJECT_ROOT / "data/backtest/stage0_trade_decisions.parquet"
TRADE_COLUMNS = [
    "basis_bps", "threshold_role", "ticker", "split", "close_date", "week_start",
    "horizon_minutes", "decision_time", "hour_utc", "hour_block", "side",
    "quote_yes_bid", "quote_yes_ask", "quote_mid", "half_spread", "entry_price",
    "entry_price_mils", "model_probability", "p_side", "mid_side", "z_5min_ewma_vol",
    "abs_z_bucket", "price_bucket", "gross_edge", "net_edge", "required_net_edge",
    "best_net_edge", "signal_eligible", "threshold_clear", "model_fee", "trade_fee",
    "rounding_adjustment", "rebate", "fee", "market_fair_expected",
    "other_horizon_cleared", "other_horizon_side", "probability_version",
    "sigma_candidate", "fee_rounding_mode", "order_size",
]


def add_outcome_free_columns(rows):
    """Attach Day 11 decision metadata while rows remain unique by market/horizon."""
    assert not rows.duplicated(ROW_KEY).any(), "Day 10 quote rows are not unique by market/horizon"
    assert_safe_columns(FEATURE_EXTRA_COLUMNS, "Day 11 feature projection")
    assert_safe_columns(PREDICTION_EXTRA_COLUMNS, "Day 11 prediction projection")

    start = SPLIT_RANGES["train"][0]
    end = SPLIT_RANGES["validation"][1]
    features = pd.read_parquet(
        FEATURES_PATH, columns=FEATURE_EXTRA_COLUMNS,
        filters=[(SPLIT_KEY, ">=", start), (SPLIT_KEY, "<=", end)],
    )
    predictions = pd.read_parquet(
        PREDICTIONS_PATH, columns=PREDICTION_EXTRA_COLUMNS,
        filters=[("split", "in", list(ALLOWED_SPLITS))],
    )
    assert list(features.columns) == FEATURE_EXTRA_COLUMNS, "Day 11 feature projection changed"
    assert list(predictions.columns) == PREDICTION_EXTRA_COLUMNS, "Day 11 prediction projection changed"
    assert_safe_columns(features.columns, "Day 11 features")
    assert_safe_columns(predictions.columns, "Day 11 predictions")

    for name, extra in (("features", features), ("predictions", predictions)):
        assert not extra.duplicated(ROW_KEY).any(), f"Day 11 {name} keys are not unique"
        before = len(rows)
        rows = rows.merge(extra, on=ROW_KEY, how="left", validate="one_to_one", indicator=True)
        assert len(rows) == before, f"Day 11 {name} merge changed candidate-row count"
        assert rows["_merge"].eq("both").all(), f"A common row lacks Day 11 {name}"
        rows = rows.drop(columns="_merge")
        assert not rows.duplicated(ROW_KEY).any(), f"Day 11 {name} merge changed key uniqueness"

    assert rows[["decision_time", "hour_utc", "z_5min_ewma_vol"]].notna().all().all(), "Day 11 metadata is missing"
    assert rows["hour_utc"].between(0, 23).all(), "Day 11 UTC decision hour is invalid"
    assert_safe_columns(rows.columns, "Day 11 enriched quote rows")
    return rows


def assert_day10_reproduction(rows):
    """Compare every Day 10 summary count with the saved, outcome-free artifact."""
    assert_safe_columns(rows.columns, "Day 11 candidate rows")
    assert not rows.duplicated(CANDIDATE_KEY).any(), "Candidate rows are not unique by full key"
    assert set(rows["split"]) == set(ALLOWED_SPLITS), "Candidate splits changed"
    assert not rows["split"].eq("test").any(), "Test rows entered Day 11 candidates"
    assert rows["close_date"].between(SPLIT_RANGES["train"][0], "2026-08-09", inclusive="both").all(), (
        "A candidate row falls outside the frozen train/validation dates"
    )
    assert set(rows["probability_version"]) == set(PROBABILITY_VERSIONS), "Day 10 probability versions changed"
    assert set(rows["basis_bps"]) == set(BASIS_SETTINGS), "Day 10 basis settings changed"
    assert set(rows["horizon_minutes"]) == set(HORIZONS), "Day 10 candidate horizons changed"

    common_counts = rows.loc[rows["probability_version"].eq("train_fitted_platt") & rows["basis_bps"].eq(BASIS_SETTINGS[0])]
    actual_common = common_counts.groupby(["split", "horizon_minutes"], observed=True).size().to_dict()
    assert actual_common == EXPECTED_COMMON_ROWS, f"Day 9 common-row population changed: {actual_common}"

    rebuilt = build_summary(rows)
    validate_waterfalls(rebuilt)
    columns = SUMMARY_KEY + SUMMARY_COUNT_COLUMNS
    assert_safe_columns(columns, "Day 10 saved summary projection")
    saved = pd.read_parquet(DAY10_SUMMARY_PATH, columns=columns)
    assert list(saved.columns) == columns, "Saved Day 10 summary projection changed"
    assert_safe_columns(saved.columns, "Day 10 saved summary")
    assert set(saved["split"]) == set(ALLOWED_SPLITS), "Saved Day 10 summary has unexpected splits"
    assert not saved.duplicated(SUMMARY_KEY).any(), "Saved Day 10 summary key is not unique"
    assert len(saved) == len(rebuilt), f"Day 10 summary row count differs: saved={len(saved)}, rebuilt={len(rebuilt)}"

    expected = saved.set_index(SUMMARY_KEY)[SUMMARY_COUNT_COLUMNS].sort_index()
    actual = rebuilt.set_index(SUMMARY_KEY)[SUMMARY_COUNT_COLUMNS].sort_index()
    assert actual.index.equals(expected.index), "Rebuilt Day 10 summary groups differ from the saved artifact"
    for column in SUMMARY_COUNT_COLUMNS:
        matches = (actual[column].eq(expected[column]) | (actual[column].isna() & expected[column].isna())).fillna(False)
        if not matches.all():
            key = matches.index[~matches][0]
            raise AssertionError(
                f"Day 10 {column} mismatch at {key}: saved={expected.loc[key, column]!r}, "
                f"rebuilt={actual.loc[key, column]!r}"
            )

    print(f"PASS: all {len(saved)} saved Day 10 summary groups and count fields reproduced exactly")
    platt_overall = rebuilt.loc[
        rebuilt["probability_version"].eq("train_fitted_platt") & rebuilt["breakdown"].eq("overall"),
        ["split", "horizon_minutes", "basis_bps", "n_common", "n_stale_excluded", "n_threshold_clear"],
    ]
    print(platt_overall.to_string(index=False))


def build_candidate_rows() -> pd.DataFrame:
    """Follow Day 10's call order, pass reproduction, then retain Platt rows."""
    predictions = load_common_prediction_rows()
    parameters = load_legitimate_platt_parameters()
    predictions = reconstruct_platt_probability(predictions, parameters)
    maximum_platt_difference = verify_platt_carry_forward(predictions)

    features = load_quote_rows()
    rows = join_quotes(predictions, features)
    rows = add_outcome_free_columns(rows)
    rows = calculate_execution_fees(rows)
    rows = apply_stale_rule_and_buckets(rows)
    edge_rows = build_probability_rows(rows)
    thresholds = load_thresholds()
    candidate_rows = join_thresholds(edge_rows, thresholds)

    assert_day10_reproduction(candidate_rows)
    assert maximum_platt_difference <= PLATT_TOLERANCE, "Day 9 train-fitted Platt carry-forward tolerance failed"
    print(f"PASS: train-fitted Platt carry-forward maximum absolute difference = {maximum_platt_difference:.17g}")

    platt_rows = candidate_rows.loc[candidate_rows["probability_version"].eq("train_fitted_platt")].copy()
    assert len(platt_rows) * len(PROBABILITY_VERSIONS) == len(candidate_rows), "Dropping raw candidates changed Platt coverage"
    assert not platt_rows.duplicated(CANDIDATE_KEY).any(), "Platt candidate keys are not unique"
    assert platt_rows["probability_version"].eq("train_fitted_platt").all(), "Raw candidates remain"
    assert_safe_columns(platt_rows.columns, "final outcome-free Day 11 candidates")
    print(f"PASS: {len(platt_rows):,} Platt candidate rows retained; outcome-free, train/validation only; no trades selected")
    return platt_rows


def assert_safe_trade_columns(columns):
    assert_safe_columns(columns, "Day 11 trade decisions")
    forbidden = [
        column for column in columns if column.lower() == "y" or any(
            token in column.lower() for token in
            ("settlement", "expiration", "fwd_", "target_", "outcome", "pnl", "payoff", "hit", "return", "profit", "surprise")
        )
    ]
    assert not forbidden, f"Outcome/future/result columns entered trade decisions: {forbidden}"


def select_first_clearing_trade(rows: pd.DataFrame) -> pd.DataFrame:
    """Choose the earliest clearing decision per basis and ticker, then add decision fields."""
    assert_safe_trade_columns(rows.columns)
    assert rows["probability_version"].eq("train_fitted_platt").all(), "Only train-fitted Platt candidates may be selected"
    assert not rows.duplicated(CANDIDATE_KEY).any(), "Candidate key is not unique"
    clears = rows["signal_eligible"] & rows["best_net_edge"].ge(rows["required_net_edge"])
    assert clears.equals(rows["threshold_clear"]), "Day 10 threshold-clearance rule changed"
    clearing = rows.loc[clears].copy()
    assert clearing.groupby(TRADE_KEY, observed=True).size().le(2).all(), "A market has more than two clearing horizons"
    assert not clearing.duplicated(TRADE_KEY + ["horizon_minutes"]).any(), "A market/horizon clears twice"

    # There are no observed timestamp ties. The horizon key makes ordering deterministic if one appears.
    selected = clearing.sort_values(TRADE_KEY + ["decision_time", "horizon_minutes"], ascending=[True, True, True, False])
    selected = selected.drop_duplicates(TRADE_KEY, keep="first").copy()
    other = clearing[TRADE_KEY + ["horizon_minutes", "best_side"]].copy()
    other["horizon_minutes"] = other["horizon_minutes"].map({10: 5, 5: 10})
    other = other.rename(columns={"best_side": "other_horizon_side"})
    selected = selected.merge(other, on=TRADE_KEY + ["horizon_minutes"], how="left", validate="one_to_one")
    selected["other_horizon_cleared"] = selected["other_horizon_side"].notna()
    selected["other_horizon_side"] = selected["other_horizon_side"].astype("string")
    selected = selected.rename(columns={"best_side": "side"})

    bid_mils = integer_mils(selected["quote_yes_bid"], "selected quote_yes_bid")
    ask_mils = integer_mils(selected["quote_yes_ask"], "selected quote_yes_ask")
    yes = selected["side"].eq("YES").to_numpy()
    assert selected["side"].isin(["YES", "NO"]).all(), "Selected side is invalid"
    selected["entry_price_mils"] = np.where(yes, ask_mils, 1000 - bid_mils)
    selected["entry_price"] = selected["entry_price_mils"] / 1000.0
    selected["p_side"] = np.where(yes, selected["model_probability"], 1.0 - selected["model_probability"])
    selected["mid_side"] = np.where(yes, selected["quote_mid"], 1.0 - selected["quote_mid"])
    selected["gross_edge"] = selected["best_gross_edge"]
    selected["net_edge"] = selected["best_net_edge"]
    selected["fee"] = np.where(yes, selected["fee_yes_probability_units"], selected["fee_no_probability_units"])
    selected["half_spread"] = (selected["quote_yes_ask"] - selected["quote_yes_bid"]) / 2.0
    selected["market_fair_expected"] = selected["mid_side"] - selected["entry_price"] - selected["fee"]

    fee_results = [
        calculate_single_fill_fee(
            Decimal(int(mils)) / Decimal(1000), 1, "buy", multiplier=KXBTC15M_FEE_MULTIPLIER,
            fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER,
        )
        for mils in selected["entry_price_mils"]
    ]
    selected["model_fee"] = [float(result.model_fee) for result in fee_results]
    selected["trade_fee"] = [float(result.trade_fee) for result in fee_results]
    selected["rounding_adjustment"] = [float(result.rounding_adjustment) for result in fee_results]
    selected["rebate"] = [float(result.rebate) for result in fee_results]
    selected["fee_rounding_mode"] = [result.rounding_mode for result in fee_results]
    assert np.array_equal(selected["fee"].to_numpy(), np.array([float(result.net_cash_fee) for result in fee_results])), (
        "Recomputed one-fill fees differ from the carried Day 10 side fees"
    )

    close_dates = pd.to_datetime(selected["close_date"], utc=True)
    selected["week_start"] = (close_dates - pd.to_timedelta(close_dates.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    selected["hour_block"] = pd.cut(
        selected["hour_utc"], bins=[-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"],
    ).astype("string")
    selected["abs_z_bucket"] = pd.cut(
        selected["z_5min_ewma_vol"].abs(), bins=[0.0, 0.25, 0.5, 1.0, np.inf],
        labels=["[0, 0.25)", "[0.25, 0.5)", "[0.5, 1.0)", "[1.0, infinity)"], right=False,
    ).astype("string")
    selected["sigma_candidate"] = SELECTED_CANDIDATE
    assert selected[["week_start", "hour_block", "abs_z_bucket"]].notna().all().all(), "A descriptive bucket is missing"
    trades = selected[TRADE_COLUMNS].sort_values(TRADE_KEY).reset_index(drop=True)
    assert_safe_trade_columns(trades.columns)
    return trades


def load_saved_market_summary():
    columns = SUMMARY_KEY + MARKET_COUNT_COLUMNS
    assert_safe_trade_columns(columns)
    saved = pd.read_parquet(
        DAY10_SUMMARY_PATH, columns=columns,
        filters=[("breakdown", "==", "market_summary"), ("probability_version", "==", "train_fitted_platt")],
    )
    assert len(saved) == len(ALLOWED_SPLITS) * len(BASIS_SETTINGS), "Saved Platt market-summary row count changed"
    assert saved["horizon_minutes"].eq(MARKET_SUMMARY_HORIZON).all(), "Market-summary horizon sentinel changed"
    assert saved["breakdown_value"].eq("cross_horizon").all(), "Market-summary row label changed"
    assert not saved.duplicated(["split", "basis_bps"]).any(), "Saved market-summary keys are not unique"
    return saved.set_index(["split", "basis_bps"])


def assert_trade_invariants(trades, candidates, saved_market):
    assert list(trades.columns) == TRADE_COLUMNS, "Trade-list columns changed"
    assert_safe_trade_columns(trades.columns)
    assert not trades.duplicated(TRADE_KEY).any(), "More than one trade was selected per basis/market"
    assert trades["probability_version"].eq("train_fitted_platt").all(), "A non-Platt trade was selected"
    assert trades["sigma_candidate"].eq(SELECTED_CANDIDATE).all(), "Selected sigma changed"
    assert trades.loc[trades["basis_bps"].eq(BASIS_SETTINGS[0]), "threshold_role"].eq("primary").all(), (
        "Primary threshold role changed"
    )
    assert trades.loc[trades["basis_bps"].eq(BASIS_SETTINGS[1]), "threshold_role"].eq("conservative_sensitivity").all(), (
        "Sensitivity threshold role changed"
    )
    assert trades["signal_eligible"].all() and trades["threshold_clear"].all(), "A non-clearing trade was selected"
    assert trades["net_edge"].ge(trades["required_net_edge"]).all(), "A selected trade is below its threshold"
    assert trades["required_net_edge"].gt(0).all(), "A selected threshold is nonpositive"
    assert set(trades["split"]) == set(ALLOWED_SPLITS) and not trades["split"].eq("test").any(), "Trade splits changed"
    assert trades["close_date"].le("2026-08-09").all(), "A trade has a post-validation close date"
    for split in ALLOWED_SPLITS:
        start, end = SPLIT_RANGES[split]
        subset = trades.loc[trades["split"].eq(split)]
        assert subset["close_date"].between(start, end, inclusive="both").all(), f"{split} trade dates changed"

    common = candidates.loc[candidates["basis_bps"].eq(BASIS_SETTINGS[0]), TRADE_KEY + ["horizon_minutes", "decision_time"]]
    paired = common.pivot(index="ticker", columns="horizon_minutes", values="decision_time").dropna()
    assert paired[10].lt(paired[5]).all(), "T-10 is not strictly earlier than T-5 for a common market"
    clearing = candidates.loc[candidates["threshold_clear"]].sort_values(
        TRADE_KEY + ["decision_time", "horizon_minutes"], ascending=[True, True, True, False],
    )
    earliest = clearing.drop_duplicates(TRADE_KEY, keep="first")[TRADE_KEY + ["horizon_minutes", "decision_time", "best_side"]]
    selected = trades[TRADE_KEY + ["horizon_minutes", "decision_time", "side"]].rename(columns={"side": "best_side"})
    pd.testing.assert_frame_equal(
        selected.sort_values(TRADE_KEY).reset_index(drop=True), earliest.sort_values(TRADE_KEY).reset_index(drop=True),
        check_dtype=False, check_exact=True,
    )
    assert trades.loc[trades["other_horizon_cleared"], "horizon_minutes"].eq(10).all(), (
        "A dual-horizon market did not select T-10"
    )
    assert trades["other_horizon_cleared"].eq(trades["other_horizon_side"].notna()).all(), "Suppression metadata disagrees"
    other = clearing[TRADE_KEY + ["horizon_minutes", "best_side"]].copy()
    other["horizon_minutes"] = other["horizon_minutes"].map({10: 5, 5: 10})
    other = other.rename(columns={"best_side": "expected_other_side"})
    checked_other = trades.merge(other, on=TRADE_KEY + ["horizon_minutes"], how="left", validate="one_to_one")
    assert (checked_other["other_horizon_side"].eq(checked_other["expected_other_side"]) |
            (checked_other["other_horizon_side"].isna() & checked_other["expected_other_side"].isna())).all(), (
        "The recorded other-horizon side differs from the suppressed clearing row"
    )

    for (split, basis_bps), expected in saved_market.iterrows():
        group = trades.loc[trades["split"].eq(split) & trades["basis_bps"].eq(basis_bps)]
        dual = group.loc[group["other_horizon_cleared"]]
        actual = {
            "n_unique_markets_threshold_clear": len(group),
            "n_dual_horizon_markets_threshold_clear": len(dual),
            "n_dual_horizon_same_side": int(dual["side"].eq(dual["other_horizon_side"]).sum()),
            "n_dual_horizon_opposite_side": int(dual["side"].ne(dual["other_horizon_side"]).sum()),
            "n_dual_horizon_yes_yes": int((dual["side"].eq("YES") & dual["other_horizon_side"].eq("YES")).sum()),
            "n_dual_horizon_no_no": int((dual["side"].eq("NO") & dual["other_horizon_side"].eq("NO")).sum()),
        }
        for field, value in actual.items():
            assert value == expected[field], f"{split} {basis_bps:g} bps {field}: saved={expected[field]}, selected={value}"

    close = lambda left, right: np.allclose(left, right, rtol=0.0, atol=EDGE_TOLERANCE)
    yes = trades["side"].eq("YES")
    assert trades["side"].isin(["YES", "NO"]).all(), "Trade side is invalid"
    assert close(trades["p_side"], np.where(yes, trades["model_probability"], 1.0 - trades["model_probability"])), (
        "Selected-side model probability changed"
    )
    assert close(trades["mid_side"], np.where(yes, trades["quote_mid"], 1.0 - trades["quote_mid"])), (
        "Selected-side midpoint probability changed"
    )
    assert np.array_equal(
        trades["entry_price_mils"].to_numpy(),
        np.where(yes, integer_mils(trades["quote_yes_ask"], "trade ask"), 1000 - integer_mils(trades["quote_yes_bid"], "trade bid")),
    ), "Selected-side executable price changed"
    assert close(trades["entry_price"], trades["entry_price_mils"] / 1000.0), "Entry mils and prices disagree"
    assert close(trades["net_edge"], trades["best_net_edge"]), "Selected edge differs from Day 10 best edge"
    assert close(trades["gross_edge"], trades["p_side"] - trades["entry_price"]), "Gross executable edge identity failed"
    assert close(trades["net_edge"], trades["gross_edge"] - trades["fee"]), "Net executable edge identity failed"
    assert close(trades["market_fair_expected"], trades["mid_side"] - trades["entry_price"] - trades["fee"]), (
        "Market-fair reference identity failed"
    )
    assert close(trades["market_fair_expected"], -(trades["half_spread"] + trades["fee"])), (
        "Market-fair midpoint-cost identity failed"
    )
    assert close(trades["fee"], trades["trade_fee"] + trades["rounding_adjustment"] - trades["rebate"]), (
        "Net cash fee components do not reconcile"
    )
    assert np.array_equal(integer_mils(trades["entry_price"], "trade entry_price"), trades["entry_price_mils"].to_numpy()), (
        "Entry prices changed on the mil grid"
    )
    assert all(is_on_tick_grid(Decimal(int(mils)) / Decimal(1000)) for mils in trades["entry_price_mils"]), (
        "A trade entry price is off the executable tick grid"
    )
    assert trades["fee_rounding_mode"].eq(DIRECT_MEMBER).all() and trades["order_size"].eq(1).all(), (
        "Fee route or order size changed"
    )
    assert trades["price_bucket"].isin(PRICE_BUCKET_LABELS).all(), "A trade has an invalid price bucket"
    assert trades["hour_utc"].eq(trades["decision_time"].dt.hour).all(), "UTC decision hour disagrees with decision time"
    close_dates = pd.to_datetime(trades["close_date"], utc=True)
    expected_week = (close_dates - pd.to_timedelta(close_dates.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
    assert trades["week_start"].eq(expected_week).all(), "Monday-start week changed"
    expected_hour_block = pd.cut(
        trades["hour_utc"], bins=[-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"],
    ).astype("string")
    expected_abs_z_bucket = pd.cut(
        trades["z_5min_ewma_vol"].abs(), bins=[0.0, 0.25, 0.5, 1.0, np.inf],
        labels=["[0, 0.25)", "[0.25, 0.5)", "[0.5, 1.0)", "[1.0, infinity)"], right=False,
    ).astype("string")
    assert trades["hour_block"].eq(expected_hour_block).all(), "UTC hour block changed"
    assert trades["abs_z_bucket"].eq(expected_abs_z_bucket).all(), "Pre-registered absolute z bucket changed"
    recomputed_fees = [
        calculate_single_fill_fee(
            Decimal(int(mils)) / Decimal(1000), 1, "buy", multiplier=KXBTC15M_FEE_MULTIPLIER,
            fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER,
        )
        for mils in trades["entry_price_mils"]
    ]
    for column, field in (("model_fee", "model_fee"), ("trade_fee", "trade_fee"),
                          ("rounding_adjustment", "rounding_adjustment"), ("rebate", "rebate"), ("fee", "net_cash_fee")):
        assert np.array_equal(trades[column].to_numpy(), np.array([float(getattr(result, field)) for result in recomputed_fees])), (
            f"Fresh one-fill {column} recomputation differs from the carried fee detail"
        )


def decision_fingerprint(trades: pd.DataFrame) -> str:
    """Hash canonical decision fields, independent of row order and Parquet encoding."""
    assert not trades.duplicated(TRADE_KEY).any(), "Cannot fingerprint duplicate trade decisions"
    fields = trades[TRADE_KEY + ["horizon_minutes", "side", "entry_price_mils"]].sort_values(TRADE_KEY)
    lines = []
    for row in fields.itertuples(index=False):
        assert "|" not in row.ticker and "\n" not in row.ticker, "Ticker cannot be encoded canonically"
        lines.append(f"{float(row.basis_bps):.1f}|{row.ticker}|{int(row.horizon_minutes)}|{row.side}|{int(row.entry_price_mils)}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def check_fingerprint_invariants(trades):
    fingerprint = decision_fingerprint(trades)
    assert decision_fingerprint(trades.sample(frac=1, random_state=11)) == fingerprint, "Fingerprint depends on row order"
    changed = trades.copy()
    first = changed.index[0]
    changed.loc[first, "side"] = "NO" if changed.loc[first, "side"] == "YES" else "YES"
    assert decision_fingerprint(changed) != fingerprint, "Fingerprint missed a changed trade decision"
    return fingerprint


def save_and_verify_trades(trades, candidates, saved_market):
    assert_trade_invariants(trades, candidates, saved_market)
    fingerprint = check_fingerprint_invariants(trades)
    TRADE_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(TRADE_PATH, index=False)
    saved = pd.read_parquet(TRADE_PATH)
    assert_safe_trade_columns(saved.columns)
    assert len(saved) == len(trades), "Trade-list row count changed after Parquet round trip"
    assert set(map(tuple, saved[TRADE_KEY].itertuples(index=False, name=None))) == set(
        map(tuple, trades[TRADE_KEY].itertuples(index=False, name=None))
    ), "Trade-list keys changed after Parquet round trip"
    assert decision_fingerprint(saved) == fingerprint, "Decision fingerprint changed after Parquet round trip"
    assert_trade_invariants(saved, candidates, saved_market)
    print("PASS: Parquet round-trip fingerprint and trade invariants match")
    return saved, fingerprint


def print_trade_summary(trades, fingerprint):
    for split in ALLOWED_SPLITS:
        for basis_bps in BASIS_SETTINGS:
            group = trades.loc[trades["split"].eq(split) & trades["basis_bps"].eq(basis_bps)]
            horizons = group["horizon_minutes"].value_counts()
            sides = group["side"].value_counts()
            print(
                f"{split} {basis_bps:.1f} bps: trades={len(group)}, T-10={horizons.get(10, 0)}, "
                f"T-5={horizons.get(5, 0)}, YES={sides.get('YES', 0)}, NO={sides.get('NO', 0)}, "
                f"later horizon suppressed={int(group['other_horizon_cleared'].sum())}"
            )
    print(f"SHA-256 decision fingerprint: {fingerprint}")
    print(f"Saved: {TRADE_PATH.relative_to(PROJECT_ROOT)}")
    print("Outcome-free train/validation decisions only; no outcome or test rows loaded")


if __name__ == "__main__":
    candidate_rows = build_candidate_rows()
    trade_rows = select_first_clearing_trade(candidate_rows)
    saved_market_rows = load_saved_market_summary()
    saved_trades, decision_hash = save_and_verify_trades(trade_rows, candidate_rows, saved_market_rows)
    print_trade_summary(saved_trades, decision_hash)
