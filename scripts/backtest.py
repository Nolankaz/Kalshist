"""Settle frozen Stage 0 trades with one-contract Tier 2 accounting."""

import argparse
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.analyze_spreads import PREDICTION_COLUMNS
from scripts.build_stage0_trades import (
    TRADE_COLUMNS, TRADE_KEY, TRADE_PATH, assert_safe_trade_columns, decision_fingerprint,
)
from scripts.count_threshold_clearance import EDGE_TOLERANCE, FEATURES_PATH, PREDICTIONS_PATH
from scripts.evaluation_split import EXPECTED_SPLIT_ROWS, SPLIT_KEY, SPLIT_RANGES
from scripts.fees import DIRECT_MEMBER, KXBTC15M_FEE_MULTIPLIER, KXBTC15M_FEE_TYPE, calculate_single_fill_fee
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, common_prediction_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SPLITS = ("train", "validation")
FROZEN_FINGERPRINT = "f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96"
FROZEN_TRADE_COUNTS = {("train", 1.2): 1_881, ("validation", 1.2): 719, ("train", 5.0): 277, ("validation", 5.0): 100}
SETTLEMENT_COLUMNS = ["ticker", "horizon_minutes", SPLIT_KEY, "settlement_value", "y"]
RESULT_COLUMNS = TRADE_COLUMNS + ["settlement_value", "payoff", "hit", "gross_pnl", "net_pnl", "capital", "surprise"]
TIER_LABEL = "Tier 2 — quote-aware, top-of-book, size-unaware. One contract, taker entry, held to settlement."


def require_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unsupported split {split!r}; choose train or validation")


def close(left, right) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=EDGE_TOLERANCE))


def fresh_buy_fee(entry_price_mils):
    """A separate Direct Member one-contract order for each supplied entry."""
    return calculate_single_fill_fee(
        Decimal(int(entry_price_mils)) / Decimal(1000), 1, "buy",
        multiplier=KXBTC15M_FEE_MULTIPLIER, fee_type=KXBTC15M_FEE_TYPE, rounding_mode=DIRECT_MEMBER,
    )


def load_trade_decisions(split: str) -> pd.DataFrame:
    require_split(split)
    disk_columns = pq.read_schema(TRADE_PATH).names
    assert_safe_trade_columns(disk_columns)
    assert disk_columns == TRADE_COLUMNS, "Frozen trade-decision schema changed"
    all_trades = pd.read_parquet(TRADE_PATH, columns=TRADE_COLUMNS)
    assert_safe_trade_columns(all_trades.columns)
    assert not all_trades.duplicated(TRADE_KEY).any(), "Frozen decisions contain duplicate basis/market keys"
    assert set(all_trades["split"]) == set(ALLOWED_SPLITS), "Frozen decisions contain an unexpected split"
    assert not all_trades["split"].eq("test").any(), "Test decisions entered the backtest"
    actual_fingerprint = decision_fingerprint(all_trades)
    assert actual_fingerprint == FROZEN_FINGERPRINT, (
        f"Frozen trade-decision fingerprint changed: expected {FROZEN_FINGERPRINT}, got {actual_fingerprint}"
    )
    counts = all_trades.groupby(["split", "basis_bps"], observed=True).size().to_dict()
    assert counts == FROZEN_TRADE_COUNTS, f"Frozen trade counts changed: {counts}"
    print(f"PASS: frozen decision fingerprint {actual_fingerprint} and all four trade counts verified")

    trades = all_trades.loc[all_trades["split"].eq(split)].copy()
    assert trades["split"].eq(split).all() and not trades["split"].eq("test").any(), "Requested trade split changed"
    assert not trades.duplicated(TRADE_KEY).any(), "Requested split has duplicate trade keys"
    assert_safe_trade_columns(trades.columns)
    return trades


def load_settlements(split: str) -> pd.DataFrame:
    require_split(split)
    start, end = SPLIT_RANGES[split]
    rows = pd.read_parquet(
        FEATURES_PATH, columns=SETTLEMENT_COLUMNS,
        filters=[(SPLIT_KEY, ">=", start), (SPLIT_KEY, "<=", end)],
    )
    assert list(rows.columns) == SETTLEMENT_COLUMNS, "Settlement projection changed"
    assert len(rows) == EXPECTED_SPLIT_ROWS[split], f"{split} settlement row count changed"
    assert rows[SPLIT_KEY].between(start, end, inclusive="both").all(), "A settlement row is outside the requested split"
    assert rows[SPLIT_KEY].le(SPLIT_RANGES["validation"][1]).all(), "Test settlement rows were loaded"
    assert not rows.duplicated(["ticker", "horizon_minutes"]).any(), "Settlement keys are not unique"
    assert rows[["settlement_value", "y"]].notna().all().all(), "A settlement outcome is missing"
    assert rows["settlement_value"].isin([0, 1]).all() and rows["y"].isin([0, 1]).all(), "Settlement values must be binary"
    assert rows["y"].eq(rows["settlement_value"]).all(), "Stored label disagrees with Kalshi settlement"
    print(f"PASS: {split}-only settlement projection and Parquet close_date filter ({start} through {end})")
    return rows.drop(columns="y")


def add_accounting_fields(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply binary payoff and cash identities to an already matched decision table."""
    results = rows.copy()
    assert results["side"].isin(["YES", "NO"]).all(), "A trade has an unsupported side"
    results["payoff"] = np.where(results["side"].eq("YES"), results["settlement_value"], 1 - results["settlement_value"])
    results["gross_pnl"] = results["payoff"] - results["entry_price"]
    results["net_pnl"] = results["gross_pnl"] - results["fee"]
    results["hit"] = results["payoff"].eq(1)
    results["capital"] = results["entry_price"] + results["fee"]
    results["surprise"] = results["payoff"] - results["p_side"]
    return results


def self_check_accounting() -> None:
    cases = [
        ("YES settles", "YES", 400, 1, Decimal("0.0168"), Decimal("0.5832")),
        ("NO settles after YES buy", "YES", 400, 0, Decimal("0.0168"), Decimal("-0.4168")),
        ("NO settles after NO buy", "NO", 970, 0, Decimal("0.0021"), Decimal("0.0279")),
        ("YES settles after NO buy", "NO", 240, 1, Decimal("0.0128"), Decimal("-0.2528")),
    ]
    records = []
    for name, side, price_mils, settlement_value, expected_fee, expected_net in cases:
        fee_result = fresh_buy_fee(price_mils)
        assert fee_result.net_cash_fee == expected_fee, f"{name}: fee differs from the worked example"
        records.append({
            "side": side, "entry_price": price_mils / 1000.0, "fee": float(fee_result.net_cash_fee),
            "settlement_value": settlement_value, "p_side": 0.5, "name": name,
        })
    checked = add_accounting_fields(pd.DataFrame(records))
    for row, case in zip(checked.itertuples(index=False), cases):
        assert close(row.net_pnl, float(case[5])), f"{row.name}: net cash result differs from the worked example"
    print("PASS: self_check_accounting — four YES/NO payoff and Direct Member fee examples")


def settle_trades(trades: pd.DataFrame, settlements: pd.DataFrame) -> pd.DataFrame:
    """Join one requested split's official settlements to passed frozen decisions."""
    assert_safe_trade_columns(trades.columns)
    assert not trades.duplicated(TRADE_KEY).any(), "Passed decisions contain duplicate basis/market keys"
    assert not settlements.duplicated(["ticker", "horizon_minutes"]).any(), "Passed settlements are not unique"
    assert set(trades["split"]) <= set(ALLOWED_SPLITS) and trades["split"].nunique() == 1, "Passed trades must have one allowed split"
    parts = []
    for basis_bps, group in trades.groupby("basis_bps", observed=True, sort=True):
        assert not group.duplicated(["ticker", "horizon_minutes"]).any(), f"{basis_bps:g} bps trades have duplicate settlement keys"
        merged = group.merge(
            settlements, on=["ticker", "horizon_minutes"], how="left", validate="one_to_one",
            suffixes=("", "_settlement"), indicator=True,
        )
        assert len(merged) == len(group), f"{basis_bps:g} bps settlement join changed trade count"
        assert merged["_merge"].eq("both").all(), f"{basis_bps:g} bps trade lacks a settlement row"
        assert merged["close_date"].eq(merged["close_date_settlement"]).all(), "Trade and settlement close dates disagree"
        parts.append(merged.drop(columns=["_merge", "close_date_settlement"]))
    joined = pd.concat(parts, ignore_index=True).sort_values(TRADE_KEY).reset_index(drop=True)
    assert len(joined) == len(trades), "Settlement join added or dropped a trade"
    assert not joined.duplicated(TRADE_KEY).any(), "Settlement join duplicated a basis/market key"
    print(f"PASS: one-to-one settlement join for {len(joined):,} frozen trades")
    return add_accounting_fields(joined)[RESULT_COLUMNS]


def assert_accounting_invariants(results: pd.DataFrame, trades: pd.DataFrame, split: str) -> None:
    require_split(split)
    assert list(results.columns) == RESULT_COLUMNS, "Result schema changed"
    assert len(results) == len(trades), "Result row count differs from frozen trades"
    assert not results.duplicated(TRADE_KEY).any(), "Results contain duplicate basis/market keys"
    assert results["split"].eq(split).all() and not results["split"].eq("test").any(), "Result split changed"
    expected_counts = {basis: count for (name, basis), count in FROZEN_TRADE_COUNTS.items() if name == split}
    assert results.groupby("basis_bps", observed=True).size().to_dict() == expected_counts, "Frozen split trade counts changed"
    expected_decisions = trades[TRADE_COLUMNS].sort_values(TRADE_KEY).reset_index(drop=True)
    actual_decisions = results[TRADE_COLUMNS].sort_values(TRADE_KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_decisions, expected_decisions, check_dtype=False, check_exact=True)

    assert results["settlement_value"].isin([0, 1]).all() and results["payoff"].isin([0, 1]).all(), "A binary payoff is invalid"
    assert results["hit"].eq(results["payoff"].eq(1)).all(), "Hit flag disagrees with side payoff"
    assert close(results["net_pnl"], results["net_edge"] + results["surprise"]), "Net P&L decomposition failed"
    assert close(results["net_pnl"], results["gross_pnl"] - results["fee"]), "Gross-to-net accounting failed"
    assert close(results["capital"], results["entry_price"] + results["fee"]), "Capital identity failed"
    expected_cash = np.where(results["hit"], 1 - results["capital"], -results["capital"])
    assert close(results["net_pnl"], expected_cash), "Binary win/loss cash identity failed"

    fee_results = [fresh_buy_fee(mils) for mils in results["entry_price_mils"]]
    assert np.array_equal(results["fee"].to_numpy(), np.array([float(value.net_cash_fee) for value in fee_results])), (
        "Recomputed Direct Member net cash fee differs from the frozen entry fee"
    )
    assert close(results["capital"], np.array([-float(value.posted_cash_change) for value in fee_results])), (
        "Capital differs from the fee engine's posted cash debit"
    )
    print(f"PASS: accounting identities and fresh posted-cash checks for {len(results):,} {split} trades")


def population_day_count(split: str) -> int:
    """Count represented close dates in this split's Day 9 common population."""
    require_split(split)
    columns = ["ticker", "horizon_minutes", "close_date", "split"] + PREDICTION_COLUMNS
    assert_safe_trade_columns(columns)
    start, end = SPLIT_RANGES[split]
    rows = pd.read_parquet(PREDICTIONS_PATH, columns=columns, filters=[("split", "==", split)])
    assert rows["split"].eq(split).all() and rows["close_date"].between(start, end, inclusive="both").all(), (
        "Common-population rows fall outside the requested split"
    )
    common = rows.loc[common_prediction_mask(rows, PREDICTION_COLUMNS)]
    counts = common.groupby("horizon_minutes", observed=True).size().to_dict()
    expected = {horizon: count for (name, horizon), count in EXPECTED_COMMON_ROWS.items() if name == split}
    assert counts == expected, f"Day 9 {split} common-row counts changed: {counts}"
    return int(common["close_date"].nunique())


def headline(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize the pre-registered Stage 0 headline, one row per basis."""
    assert results["split"].nunique() == 1, "Headline requires one split"
    split = str(results["split"].iloc[0])
    population_days = population_day_count(split)
    records = []
    for basis_bps, group in results.groupby("basis_bps", observed=True, sort=True):
        n_trades = len(group)
        total_capital = float(group["capital"].sum())
        assert n_trades > 0 and total_capital > 0, "Headline denominator is nonpositive"
        records.append({
            "split": split,
            "basis_bps": basis_bps,
            "threshold_role": group["threshold_role"].iloc[0],
            "tier_label": TIER_LABEL,
            "n_trades": n_trades,
            "n_distinct_markets": group["ticker"].nunique(),
            "n_days_with_trade": group["close_date"].nunique(),
            "n_population_days": population_days,
            "n_t10": int(group["horizon_minutes"].eq(10).sum()),
            "n_t5": int(group["horizon_minutes"].eq(5).sum()),
            "n_yes": int(group["side"].eq("YES").sum()),
            "n_no": int(group["side"].eq("NO").sum()),
            "hit_rate": float(group["hit"].mean()),
            "mean_entry_price": float(group["entry_price"].mean()),
            "breakeven_hit_rate": float(group["capital"].mean()),
            "gross_pnl": float(group["gross_pnl"].sum()),
            "fees": float(group["fee"].sum()),
            "fee_rounding_component": float((group["rounding_adjustment"] - group["rebate"]).sum()),
            "net_pnl": float(group["net_pnl"].sum()),
            "net_pnl_per_trade": float(group["net_pnl"].mean()),
            "mean_net_edge": float(group["net_edge"].mean()),
            "model_expected_total_pnl": float(group["net_edge"].sum()),
            "market_fair_expected_total_pnl": float(group["market_fair_expected"].sum()),
            "mean_surprise": float(group["surprise"].mean()),
            "total_capital": total_capital,
            "net_pnl_over_capital": float(group["net_pnl"].sum() / total_capital),
        })
    summary = pd.DataFrame(records)
    assert np.isclose(
        summary["gross_pnl"], summary["net_pnl"] + summary["fees"],
        rtol=0.0, atol=EDGE_TOLERANCE * summary["n_trades"],
    ).all(), "Headline fee reconciliation failed"
    assert (summary["n_trades"] == summary["n_t10"] + summary["n_t5"]).all(), "Headline horizon counts do not reconcile"
    assert (summary["n_trades"] == summary["n_yes"] + summary["n_no"]).all(), "Headline side counts do not reconcile"
    return summary


def write_results(results: pd.DataFrame, trades: pd.DataFrame, split: str) -> Path:
    require_split(split)
    path = PROJECT_ROOT / f"data/backtest/stage0_results_{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(path, index=False)
    saved = pd.read_parquet(path)
    assert len(saved) == len(results), "Result row count changed after Parquet round trip"
    assert set(map(tuple, saved[TRADE_KEY].itertuples(index=False, name=None))) == set(
        map(tuple, results[TRADE_KEY].itertuples(index=False, name=None))
    ), "Result keys changed after Parquet round trip"
    assert_accounting_invariants(saved, trades, split)
    print(f"PASS: {split} results Parquet round trip ({path.relative_to(PROJECT_ROOT)})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    args = parser.parse_args()
    self_check_accounting()
    trades = load_trade_decisions(args.split)
    settlements = load_settlements(args.split)
    results = settle_trades(trades, settlements)
    assert_accounting_invariants(results, trades, args.split)
    summary = headline(results)
    write_results(results, trades, args.split)
    print(f"\n{TIER_LABEL}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
