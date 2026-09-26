"""Prove the Day 11 selector, then apply it to frozen Day 12 validation probabilities."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_stage0_trades import (
    TRADE_COLUMNS, TRADE_KEY, assert_safe_trade_columns, build_candidate_rows,
    decision_fingerprint, select_first_clearing_trade,
)
from scripts.compare_models import FROZEN_FINGERPRINT, load_frozen_predictions
from scripts.count_threshold_clearance import (
    CALIBRATED_PROBABILITY_COLUMN, EDGE_TOLERANCE, PRIMARY_BASIS_BPS,
    SELECTED_PROBABILITY_COLUMN, build_probability_rows, load_thresholds,
)
from scripts.evaluation_split import SPLIT_RANGES
from scripts.model_logistic import CONFIG_M1, CONFIG_M2
from scripts.score_stage0 import ROW_KEY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data/backtest/model_trade_decisions.parquet"
FROZEN_STAGE0_FINGERPRINT = "f0f3fbb27ac36c8fdfeca436828fbd84f16ff614c2bc27016aaa6a1dfff08e96"
FROZEN_STAGE0_COUNTS = {("train", 1.2): 1_881, ("validation", 1.2): 719, ("train", 5.0): 277, ("validation", 5.0): 100}
MODEL_COLUMNS = TRADE_COLUMNS
MODEL_TRADE_KEY = ["probability_version", "threshold_role", "basis_bps", "ticker"]
EXECUTION_COLUMNS = ROW_KEY + [
    "split", "close_date", "decision_time", "hour_utc", "z_5min_ewma_vol",
    "quote_yes_bid", "quote_yes_ask", "quote_mid", "quote_age_seconds",
    "fee_yes_probability_units", "fee_no_probability_units", "signal_eligible",
    "price_bucket", SELECTED_PROBABILITY_COLUMN, CALIBRATED_PROBABILITY_COLUMN,
]


def prove_stage0_selector():
    """Rebuild the original outcome-free candidates and prove the exact decisions."""
    candidates = build_candidate_rows()
    trades = select_first_clearing_trade(candidates)
    fingerprint = decision_fingerprint(trades)
    counts = trades.groupby(["split", "basis_bps"], observed=True).size().to_dict()
    assert fingerprint == FROZEN_STAGE0_FINGERPRINT, f"STOP: Stage 0 selector fingerprint {fingerprint} differs"
    assert counts == FROZEN_STAGE0_COUNTS, f"STOP: Stage 0 selector counts differ: {counts}"
    assert_safe_trade_columns(trades.columns)
    return candidates, fingerprint, counts


def execution_template(candidates):
    rows = candidates.loc[candidates["basis_bps"].eq(PRIMARY_BASIS_BPS), EXECUTION_COLUMNS].copy()
    assert len(rows) == 12_883 and not rows.duplicated(ROW_KEY).any(), "Execution template population changed"
    assert rows["split"].isin(("train", "validation")).all(), "Test entered execution template"
    return rows


def generic_select(template, probabilities, thresholds, probability_version, threshold_role):
    """Use Day 10 edge calculations and Day 11 chronological selection for any probability source."""
    assert_safe_trade_columns(template.columns)
    required = ROW_KEY + ["split", "close_date", "probability"]
    assert set(required) <= set(probabilities.columns), "Probability input lacks a required field"
    probabilities = probabilities[required].copy()
    assert_safe_trade_columns(probabilities.columns)
    assert not probabilities.duplicated(ROW_KEY).any(), "Duplicate input probability key"
    assert probabilities["split"].eq("validation").all(), "Only validation model probabilities may trade"
    assert probabilities["close_date"].between(*SPLIT_RANGES["validation"]).all(), "A model trade input is outside validation"
    values = probabilities["probability"].to_numpy(dtype=float)
    assert np.isfinite(values).all() and ((values > 0) & (values < 1)).all(), "Invalid model probabilities"
    rows = probabilities[required].merge(template, on=ROW_KEY + ["split", "close_date"], how="left", validate="one_to_one", indicator=True)
    assert len(rows) == len(probabilities) and rows["_merge"].eq("both").all(), "Execution template did not match model rows exactly"
    rows = rows.drop(columns="_merge")
    rows[CALIBRATED_PROBABILITY_COLUMN] = rows["probability"]
    edge_rows = build_probability_rows(rows.drop(columns="probability"))
    edge_rows = edge_rows.loc[edge_rows["probability_version"].eq("train_fitted_platt")].copy()
    assert len(edge_rows) == len(probabilities), "Edge construction changed model population"
    keys = ["horizon_minutes", "price_bucket"]
    assert not thresholds.duplicated(keys).any() and len(thresholds) == 14, "Expected one threshold per horizon and price bucket"
    selected_thresholds = thresholds.copy()
    selected_thresholds["threshold_role"] = threshold_role
    candidate = edge_rows.merge(selected_thresholds, on=keys, how="left", validate="many_to_one", indicator=True)
    assert len(candidate) == len(edge_rows) and candidate["_merge"].eq("both").all(), "Threshold join incomplete"
    candidate = candidate.drop(columns="_merge")
    candidate["threshold_clear"] = candidate["signal_eligible"] & candidate["best_net_edge"].ge(candidate["required_net_edge"])
    assert candidate["required_net_edge"].ge(0).all(), "Invalid required net edge"
    assert candidate.loc[candidate["threshold_clear"], "best_net_edge"].gt(0).all(), "A clearing edge is nonpositive"
    # The frozen selector requires its historical internal version label. Restore the
    # caller's provenance only after its side, fee, and earliest-decision logic runs.
    trades = select_first_clearing_trade(candidate)
    trades["probability_version"] = probability_version
    trades["threshold_role"] = threshold_role
    assert list(trades.columns) == MODEL_COLUMNS and not trades.duplicated(MODEL_TRADE_KEY).any(), "Model trade schema/key changed"
    assert trades["split"].eq("validation").all() and trades["close_date"].between(*SPLIT_RANGES["validation"]).all(), "Invalid model trade dates"
    assert trades["signal_eligible"].all() and trades["threshold_clear"].all(), "Ineligible model trade"
    assert trades["net_edge"].ge(trades["required_net_edge"]).all(), "Model trade missed threshold"
    assert np.isfinite(trades[["model_probability", "entry_price", "fee", "net_edge", "required_net_edge"]].to_numpy()).all(), "Non-finite trade field"
    assert trades["fee"].ge(0).all() and trades["entry_price"].between(0, 1).all(), "Invalid execution fee or price"
    assert np.allclose(trades["net_edge"], trades["gross_edge"] - trades["fee"], rtol=0, atol=EDGE_TOLERANCE), "Executable edge identity failed"
    clearing = candidate.loc[candidate["threshold_clear"]].sort_values(TRADE_KEY + ["decision_time", "horizon_minutes"], ascending=[True, True, True, False])
    earliest = clearing.drop_duplicates(TRADE_KEY)[TRADE_KEY + ["horizon_minutes", "decision_time"]].reset_index(drop=True)
    actual = trades.sort_values(TRADE_KEY)[TRADE_KEY + ["horizon_minutes", "decision_time"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, earliest, check_exact=True)
    assert_safe_trade_columns(trades.columns)
    return trades


def model_decision_fingerprint(trades):
    assert not trades.duplicated(MODEL_TRADE_KEY).any(), "Duplicate model decision key"
    fields = trades[MODEL_TRADE_KEY + ["horizon_minutes", "side", "entry_price_mils"]].sort_values(MODEL_TRADE_KEY)
    lines = [
        f"{row.probability_version}|{row.threshold_role}|{float(row.basis_bps):.1f}|{row.ticker}|{int(row.horizon_minutes)}|{row.side}|{int(row.entry_price_mils)}"
        for row in fields.itertuples(index=False)
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def write_decisions(trades):
    assert list(trades.columns) == MODEL_COLUMNS and not trades.duplicated(MODEL_TRADE_KEY).any(), "Model decision output invalid"
    assert trades["split"].eq("validation").all() and trades["basis_bps"].eq(PRIMARY_BASIS_BPS).all(), "Model decisions contain another split or basis"
    fingerprint = model_decision_fingerprint(trades)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(OUTPUT_PATH, index=False)
    saved = pd.read_parquet(OUTPUT_PATH)
    pd.testing.assert_frame_equal(saved, trades, check_exact=True)
    assert model_decision_fingerprint(saved) == fingerprint, "Model fingerprint changed on round trip"
    return fingerprint


def main():
    predictions = load_frozen_predictions()
    assert FROZEN_FINGERPRINT == "78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512"
    candidates, stage0_fingerprint, counts = prove_stage0_selector()
    template = execution_template(candidates)
    thresholds = load_thresholds()
    thresholds = thresholds.loc[thresholds["basis_bps"].eq(PRIMARY_BASIS_BPS)].copy()
    frames = []
    for configuration in (CONFIG_M2, CONFIG_M1):
        selected = predictions.loc[predictions["configuration"].eq(configuration) & predictions["split"].eq("validation")].copy()
        assert len(selected) == 3_334 and selected.groupby("horizon_minutes").size().to_dict() == {10: 1_685, 5: 1_649}, "Validation score population changed"
        frames.append(generic_select(template, selected, thresholds, configuration, "stage0_fixed_bar"))
    trades = pd.concat(frames, ignore_index=True)[MODEL_COLUMNS]
    fingerprint = write_decisions(trades)
    print(f"Day 12 frozen prediction fingerprint: {FROZEN_FINGERPRINT} PASS")
    print(f"Day 11 Stage 0 selector fingerprint: {stage0_fingerprint} PASS")
    for (split, basis_bps), count in counts.items():
        print(f"Stage 0 selector {basis_bps:.1f} bps {split}: {count:,} trades PASS")
    for configuration in (CONFIG_M2, CONFIG_M1):
        count = len(trades.loc[trades["probability_version"].eq(configuration)])
        print(f"{configuration} validation Stage 0 fixed bar, 1.2 bps: {count:,} outcome-free decisions")
    print(f"Model trade-decision fingerprint: {fingerprint}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return trades


if __name__ == "__main__":
    main()
