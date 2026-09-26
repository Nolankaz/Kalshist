"""Settle frozen Day 12 model trades, then build the separate model-own bar."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kalshist-matplotlib"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_backtest import clustered_mean_se
from scripts.analyze_calibration import platt_feature, stable_sigmoid
from scripts.analyze_spreads import EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS, PRICE_BUCKET_LABELS, assign_price_buckets
from scripts.backtest import (
    RESULT_COLUMNS, SETTLEMENT_COLUMNS, TIER_LABEL, headline, load_settlements,
    self_check_accounting, settle_trades,
)
from scripts.build_derived_features import OUTPUT_PATH as DERIVED_PATH
from scripts.build_model_trades import (
    MODEL_COLUMNS, MODEL_TRADE_KEY, OUTPUT_PATH as DECISION_PATH,
    execution_template, generic_select, model_decision_fingerprint,
    prove_stage0_selector, write_decisions,
)
from scripts.build_stage0_trades import TRADE_COLUMNS, TRADE_KEY, assert_safe_trade_columns
from scripts.compare_models import FROZEN_FINGERPRINT, OUTPUT_PATH as COMPARISON_PATH, load_frozen_predictions
from scripts.count_threshold_clearance import PRIMARY_BASIS_BPS, load_thresholds
from scripts.evaluation_split import SPLIT_RANGES
from scripts.model_logistic import (
    BASE_FEATURE_NAMES, COEFFICIENT_OUTPUT_PATH, CONFIG_M1, CONFIG_M2,
    FEATURE_NAMES, build_design,
)
from scripts.score_stage0 import ROW_KEY
from scripts.stage0_baseline import stage0_probability


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "data/backtest/model_results_validation.parquet"
STAGE0_PATH = PROJECT_ROOT / "data/backtest/stage0_results_validation.parquet"
STAGE0_SUMMARY_PATH = PROJECT_ROOT / "data/backtest/stage0_summary.parquet"
THRESHOLD_PATH = PROJECT_ROOT / "data/models/model_edge_threshold.parquet"
PLOT_PATH = PROJECT_ROOT / "data/backtest/plots/model_vs_stage0_daily_net_pnl.png"
MARKET_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
VALIDATION_DATES = pd.date_range(*SPLIT_RANGES["validation"], freq="D").strftime("%Y-%m-%d").tolist()
RESULT_KEY = MODEL_TRADE_KEY
THRESHOLD_COLUMNS = [
    "horizon_minutes", "basis_bps", "threshold_role", "price_bucket", "basis_term",
    "model_error_term", "required_net_edge", "basis_validation_row_count",
    "basis_statistic", "model_error_statistic", "probability_version",
    "fee_model_status", "order_size", "bucket_merge_fired",
]


def validate_accounting(results, trades):
    assert len(results) == len(trades) and not results.duplicated(RESULT_KEY).any(), "Result keys or count changed"
    assert results["split"].eq("validation").all() and results["close_date"].isin(VALIDATION_DATES).all(), "Non-validation result"
    assert results["settlement_value"].isin((0, 1)).all(), "Invalid settlement"
    assert np.allclose(results["net_pnl"], results["gross_pnl"] - results["fee"], rtol=0, atol=1e-12), "Gross-to-net identity failed"
    assert np.allclose(results["net_pnl"], results["net_edge"] + results["surprise"], rtol=0, atol=1e-12), "Net-edge/surprise identity failed"
    assert np.allclose(results["capital"], results["entry_price"] + results["fee"], rtol=0, atol=1e-12), "Capital identity failed"
    assert_safe_trade_columns(trades.columns)


def settle_book(trades, settlements):
    assert list(trades.columns) == MODEL_COLUMNS and not trades.duplicated(TRADE_KEY).any(), "A book has duplicate market decisions"
    result = settle_trades(trades, settlements)
    assert list(result.columns) == RESULT_COLUMNS, "Day 11 result schema changed"
    validate_accounting(result, trades)
    return result


def book_stats(rows):
    assert len(rows) > 1 and rows["split"].eq("validation").all(), "Headline population changed"
    record = headline(rows).iloc[0].to_dict()
    cluster = clustered_mean_se(rows["net_pnl"], rows["close_date"])
    record["clustered_se"] = cluster["se_cluster"]
    record["cluster_lower_2se"] = record["net_pnl_per_trade"] - 2 * record["clustered_se"]
    record["cluster_upper_2se"] = record["net_pnl_per_trade"] + 2 * record["clustered_se"]
    assert record["n_population_days"] == 21 and cluster["n_clusters"] == record["n_days_with_trade"], "Population/traded day count changed"
    return record


def frozen_stage0_book():
    rows = pd.read_parquet(STAGE0_PATH, columns=RESULT_COLUMNS, filters=[("basis_bps", "==", PRIMARY_BASIS_BPS)])
    assert len(rows) == 719 and rows["split"].eq("validation").all(), "Frozen Stage 0 validation count changed"
    assert not rows.duplicated(TRADE_KEY).any() and rows["basis_bps"].eq(PRIMARY_BASIS_BPS).all(), "Frozen Stage 0 book key changed"
    assert np.isclose(rows["net_pnl"].sum(), -12.7339, atol=1e-12, rtol=0), "Frozen Stage 0 P&L changed"
    assert np.isclose(rows["net_pnl"].mean(), -12.7339 / 719, atol=1e-12, rtol=0), "Frozen Stage 0 mean changed"
    summary = pd.read_parquet(STAGE0_SUMMARY_PATH, filters=[
        ("split", "==", "validation"), ("basis_bps", "==", PRIMARY_BASIS_BPS), ("breakdown", "==", "overall")
    ])
    assert len(summary) == 1 and int(summary.iloc[0]["n_trades"]) == 719, "Frozen Stage 0 summary changed"
    actuals = {
        "net_pnl": rows["net_pnl"].sum(),
        "net_pnl_per_trade": rows["net_pnl"].mean(),
        "mean_surprise": rows["surprise"].mean(),
        "gross_pnl": rows["gross_pnl"].sum(),
        "fees": rows["fee"].sum(),
    }
    for field, actual in actuals.items():
        assert np.isclose(actual, summary.iloc[0][field], atol=1e-12, rtol=0), f"Frozen Stage 0 {field} disagrees with saved summary"
    return rows


def paired_daily(model, stage0):
    calendar = pd.DataFrame({"close_date": VALIDATION_DATES})
    assert len(calendar) == 21, "Validation calendar changed"
    for name, rows in (("model", model), ("stage0", stage0)):
        daily = rows.groupby("close_date", observed=True)["net_pnl"].sum().rename(name)
        calendar = calendar.join(daily, on="close_date")
    calendar[["model", "stage0"]] = calendar[["model", "stage0"]].fillna(0.0)
    calendar["difference"] = calendar["model"] - calendar["stage0"]
    assert len(calendar) == 21 and calendar["close_date"].is_unique, "Paired day population changed"
    assert np.isclose(calendar["model"].sum(), model["net_pnl"].sum(), atol=1e-12, rtol=0), "Model daily P&L mismatch"
    assert np.isclose(calendar["stage0"].sum(), stage0["net_pnl"].sum(), atol=1e-12, rtol=0), "Stage 0 daily P&L mismatch"
    mean = float(calendar["difference"].mean())
    sd = float(calendar["difference"].std(ddof=1))
    se = sd / np.sqrt(21)
    return calendar, {"n_days": 21, "mean": mean, "sample_sd": sd, "se": se, "two_se": 2 * se, "lower": mean - 2 * se, "upper": mean + 2 * se}


def plot_daily(daily):
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(daily["close_date"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(dates, daily["stage0"], marker="o", label="Stage 0 fixed bar", color="C0")
    axes[0].plot(dates, daily["model"], marker="o", label="M2 fixed bar", color="C1")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Daily net P&L ($)")
    axes[0].legend()
    axes[1].bar(dates, daily["difference"], color="C2", width=0.7)
    axes[1].axhline(daily["difference"].mean(), color="C3", linestyle="--", label="Mean daily difference")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("M2 − Stage 0 ($)")
    axes[1].legend()
    fig.suptitle("Tier-2 validation · fixed Stage 0 bar · basis 1.2 bps")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)


def load_model_ece():
    columns = ["configuration", "comparison_window", "horizon_minutes", "breakdown", "breakdown_value", "metric", "model_value", "n"]
    rows = pd.read_parquet(COMPARISON_PATH, columns=columns, filters=[
        ("configuration", "==", CONFIG_M2), ("comparison_window", "==", "validation_primary"),
        ("breakdown", "==", "overall"), ("metric", "==", "ece"),
    ])
    assert len(rows) == 2 and set(rows["horizon_minutes"]) == {10, 5}, "§3.1 M2 ECE rows changed"
    assert rows.set_index("horizon_minutes")["n"].to_dict() == {10: 1_685, 5: 1_649}, "§3.1 ECE counts changed"
    assert np.isclose(rows.set_index("horizon_minutes").loc[10, "model_value"], 0.037055016, atol=5e-10, rtol=0)
    assert np.isclose(rows.set_index("horizon_minutes").loc[5, "model_value"], 0.018318881, atol=5e-10, rtol=0)
    return rows.set_index("horizon_minutes")["model_value"].to_dict()


def model_basis_rows(predictions):
    """Perturb only the Stage 0 path, then apply the stored M2 scaler and coefficients."""
    chosen = predictions.loc[predictions["configuration"].eq(CONFIG_M2), ROW_KEY + ["split", "close_date", "probability"]].copy()
    assert len(chosen) == 3_334 and chosen["split"].eq("validation").all(), "M2 validation population changed"
    derived_columns = ROW_KEY + ["split", "close_date", "is_common"] + BASE_FEATURE_NAMES
    derived = pd.read_parquet(DERIVED_PATH, columns=derived_columns, filters=[("split", "==", "validation")])
    derived = derived.loc[derived["is_common"]].drop(columns="is_common")
    assert len(derived) == 3_334 and not derived.duplicated(ROW_KEY).any(), "Derived common validation keys changed"
    market_columns = ROW_KEY + ["close_date", "log_moneyness", "T_years", "5min_ewma_vol", "quote_mid"]
    market = pd.read_parquet(MARKET_PATH, columns=market_columns, filters=[
        ("close_date", ">=", SPLIT_RANGES["validation"][0]), ("close_date", "<=", SPLIT_RANGES["validation"][1]),
    ])
    assert not market.duplicated(ROW_KEY).any() and market["close_date"].isin(VALIDATION_DATES).all(), "Market basis inputs are invalid"
    rows = chosen.merge(derived, on=ROW_KEY + ["split", "close_date"], how="left", validate="one_to_one", indicator=True)
    assert len(rows) == 3_334 and rows["_merge"].eq("both").all(), "M2/derived basis join failed"
    rows = rows.drop(columns="_merge").merge(market, on=ROW_KEY + ["close_date"], how="left", validate="one_to_one", indicator=True)
    assert len(rows) == 3_334 and rows["_merge"].eq("both").all(), "M2/market basis join failed"
    rows = rows.drop(columns="_merge")
    assert np.isfinite(rows[["log_moneyness", "T_years", "5min_ewma_vol", "quote_mid"]].to_numpy(dtype=float)).all(), "Nonfinite basis inputs"
    rows = assign_price_buckets(rows)
    coefficient_columns = ["configuration", "fold", "horizon_minutes", "feature", "coefficient_standardized", "fit_mean", "fit_sd", "intercept_standardized"]
    coefficients = pd.read_parquet(COEFFICIENT_OUTPUT_PATH, columns=coefficient_columns, filters=[("configuration", "==", CONFIG_M2)])
    assert len(coefficients) == 18 and set(coefficients["horizon_minutes"]) == {10, 5}, "Frozen M2 coefficient artifact changed"
    records = []
    maximum_reconstruction_error = 0.0
    maximum_spot_independent_shift = 0.0
    for horizon in (10, 5):
        group = rows.loc[rows["horizon_minutes"].eq(horizon)].copy()
        fit = coefficients.loc[coefficients["horizon_minutes"].eq(horizon)]
        assert fit["feature"].tolist() == FEATURE_NAMES and fit["fold"].nunique() == 1, "Frozen M2 feature order changed"
        assert fit["intercept_standardized"].nunique() == 1 and fit["fit_sd"].gt(0).all(), "Frozen M2 scaler/intercept changed"
        x, names = build_design(group)
        assert names == FEATURE_NAMES, "Amended feature order changed"
        means = fit["fit_mean"].to_numpy(dtype=float)
        sds = fit["fit_sd"].to_numpy(dtype=float)
        beta = fit["coefficient_standardized"].to_numpy(dtype=float)
        intercept = float(fit["intercept_standardized"].iloc[0])
        base = stable_sigmoid(intercept + ((x - means) / sds) @ beta)
        error = float(np.max(np.abs(base - group["probability"].to_numpy(dtype=float))))
        maximum_reconstruction_error = max(maximum_reconstruction_error, error)
        assert error <= 1e-12, f"STOP: M2 unperturbed probability reconstruction T-{horizon} differs by {error:.3e}"
        log_m = group["log_moneyness"].to_numpy(dtype=float)
        sigma = group["5min_ewma_vol"].to_numpy(dtype=float)
        t_years = group["T_years"].to_numpy(dtype=float)
        raw_base, _ = stage0_probability(log_m, sigma, t_years)
        logit_base = platt_feature(raw_base)
        assert np.max(np.abs(logit_base - x[:, 0])) <= 1e-12, "Frozen Stage 0 logit path did not reconstruct"
        shifts = []
        for sign in (1, -1):
            # Day 9/10 convention is log1p(± basis_bps / 10_000), not a symmetric log offset.
            raw_shifted, _ = stage0_probability(log_m + np.log1p(sign * PRIMARY_BASIS_BPS / 10_000), sigma, t_years)
            shifted_logit = platt_feature(raw_shifted)
            changed = x.copy()
            changed[:, 0] = shifted_logit
            changed[:, 7] = shifted_logit * changed[:, 2]
            changed[:, 8] = shifted_logit * changed[:, 1]
            independent_shift = float(np.max(np.abs(changed[:, 1:7] - x[:, 1:7])))
            maximum_spot_independent_shift = max(maximum_spot_independent_shift, independent_shift)
            assert independent_shift == 0.0, "STOP: A spot-independent raw feature moved"
            assert np.array_equal(changed[:, 1:7], x[:, 1:7]), "Spot-independent feature changed"
            perturbed = stable_sigmoid(intercept + ((changed - means) / sds) @ beta)
            assert np.isfinite(perturbed).all() and ((perturbed > 0) & (perturbed < 1)).all(), "Invalid perturbed model probability"
            shifts.append(np.abs(perturbed - base))
        group["max_abs_shift"] = np.maximum(*shifts)
        assert group["max_abs_shift"].ge(0).all() and np.isfinite(group["max_abs_shift"]).all(), "Invalid basis shift"
        records.append(group[ROW_KEY + ["price_bucket", "max_abs_shift"]])
    basis = pd.concat(records, ignore_index=True)
    assert len(basis) == 3_334 and not basis.duplicated(ROW_KEY).any(), "Basis sensitivity population changed"
    return basis, maximum_reconstruction_error, maximum_spot_independent_shift


def derive_model_threshold(basis, ece):
    records = []
    for horizon in (10, 5):
        group = basis.loc[basis["horizon_minutes"].eq(horizon)]
        counts = group["price_bucket"].value_counts(sort=False).reindex(PRICE_BUCKET_LABELS, fill_value=0)
        assert counts.tolist() == EXPECTED_VALIDATION_COMMON_BUCKET_COUNTS[horizon], "Frozen price bucket counts changed"
        # Day 10's preregistered sparse-cell rule has no merge on this population.
        assert counts.ge(50).all(), f"Sparse bucket requires the frozen toward-0.50 merge: {counts.to_dict()}"
        for bucket in PRICE_BUCKET_LABELS:
            cell = group.loc[group["price_bucket"].eq(bucket), "max_abs_shift"]
            term = float(cell.median())
            records.append({
                "horizon_minutes": horizon, "basis_bps": PRIMARY_BASIS_BPS,
                "threshold_role": "model_own_bar", "price_bucket": bucket,
                "basis_term": term, "model_error_term": float(ece[horizon]),
                "required_net_edge": term + float(ece[horizon]), "basis_validation_row_count": len(cell),
                "basis_statistic": "median_max_abs_shift", "model_error_statistic": "validation_10_decile_ece",
                "probability_version": CONFIG_M2, "fee_model_status": "verified", "order_size": 1,
                "bucket_merge_fired": False,
            })
    threshold = pd.DataFrame(records)[THRESHOLD_COLUMNS]
    threshold["price_bucket"] = threshold["price_bucket"].astype("string")
    assert len(threshold) == 14 and not threshold.duplicated(["horizon_minutes", "basis_bps", "price_bucket"]).any(), "Model threshold key changed"
    assert np.array_equal(threshold["required_net_edge"], threshold["basis_term"] + threshold["model_error_term"]), "Additive threshold changed"
    assert threshold.groupby("horizon_minutes")["basis_validation_row_count"].sum().to_dict() == {10: 1_685, 5: 1_649}, "Threshold counts changed"
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    threshold.to_parquet(THRESHOLD_PATH, index=False)
    saved = pd.read_parquet(THRESHOLD_PATH)
    pd.testing.assert_frame_equal(saved, threshold, check_exact=True)
    return threshold


def print_book(label, stats):
    print(f"{label} — {TIER_LABEL} basis {stats['basis_bps']:.1f} bps")
    print(f"  trades={stats['n_trades']:,}; traded days={stats['n_days_with_trade']}/{stats['n_population_days']}; hit rate={stats['hit_rate']:.9f}; break-even hit rate={stats['breakeven_hit_rate']:.9f}")
    print(f"  gross P&L=${stats['gross_pnl']:+.6f}")
    print(f"  fees=${stats['fees']:.6f}")
    print(f"  net P&L=${stats['net_pnl']:+.6f}; net P&L/trade=${stats['net_pnl_per_trade']:+.9f}")
    print(f"  mean net edge={stats['mean_net_edge']:+.9f}; market-fair expected P&L=${stats['market_fair_expected_total_pnl']:+.6f}; mean surprise={stats['mean_surprise']:+.9f}")
    print(f"  traded-day clustered SE={stats['clustered_se']:.9f}; ±2SE interval=[{stats['cluster_lower_2se']:+.9f}, {stats['cluster_upper_2se']:+.9f}]")


def main():
    predictions = load_frozen_predictions()
    assert FROZEN_FINGERPRINT == "78a0e3cda08734a778d7a3c49fa7cbe5156f71a0f7d1bc93df404903a1a36512"
    self_check_accounting()
    saved_decisions = pd.read_parquet(DECISION_PATH)
    assert list(saved_decisions.columns) == MODEL_COLUMNS and not saved_decisions.duplicated(MODEL_TRADE_KEY).any(), "Model decisions changed"
    assert set(saved_decisions["threshold_role"]) <= {"stage0_fixed_bar", "model_own_bar"}, "Unknown model threshold role"
    decisions = saved_decisions.loc[saved_decisions["threshold_role"].eq("stage0_fixed_bar")].copy()
    assert set(decisions["probability_version"]) == {CONFIG_M2, CONFIG_M1} and decisions["threshold_role"].eq("stage0_fixed_bar").all(), "Fixed-bar provenance changed"
    fixed_fingerprint = model_decision_fingerprint(decisions)
    assert fixed_fingerprint == "72a50a6494e75f2de221756e22aba66053f9e7a597b2f809ec606a0274c828a2", "Frozen fixed-bar model decisions changed"
    settlements = load_settlements("validation")
    assert list(settlements.columns) == [column for column in SETTLEMENT_COLUMNS if column != "y"], "Settlement projection changed"
    fixed_results = []
    for configuration in (CONFIG_M2, CONFIG_M1):
        trades = decisions.loc[decisions["probability_version"].eq(configuration), MODEL_COLUMNS].copy()
        assert trades["basis_bps"].eq(PRIMARY_BASIS_BPS).all() and trades["threshold_role"].eq("stage0_fixed_bar").all(), "Primary bar changed"
        fixed_results.append(settle_book(trades, settlements))
    m2_result, m1_result = fixed_results
    m2_stats, m1_stats = book_stats(m2_result), book_stats(m1_result)
    stage0 = frozen_stage0_book()
    stage0_stats = book_stats(stage0)
    daily, paired = paired_daily(m2_result, stage0)
    plot_daily(daily)

    # The primary fixed-bar book is fully settled and compared before the
    # validation ECE or model-own threshold is read below.
    ece = load_model_ece()
    basis, reconstruction_error, independent_shift = model_basis_rows(predictions)
    threshold = derive_model_threshold(basis, ece)
    candidates, stage0_fingerprint, stage0_counts = prove_stage0_selector()
    assert stage0_counts[("validation", PRIMARY_BASIS_BPS)] == 719, "Stage 0 proof changed"
    template = execution_template(candidates)
    own_probabilities = predictions.loc[predictions["configuration"].eq(CONFIG_M2)]
    selector_threshold = threshold.rename(columns={"probability_version": "threshold_probability_version"})
    own_trades = generic_select(template, own_probabilities, selector_threshold, CONFIG_M2, "model_own_bar")
    all_decisions = pd.concat([decisions, own_trades], ignore_index=True)[MODEL_COLUMNS]
    all_fingerprint = write_decisions(all_decisions)
    own_result = settle_book(own_trades, settlements)
    own_stats = book_stats(own_result)
    results = pd.concat([m2_result, m1_result, own_result], ignore_index=True)[RESULT_COLUMNS]
    assert not results.duplicated(RESULT_KEY).any() and results["split"].eq("validation").all(), "Combined model result key changed"
    assert len(results) == len(all_decisions), "Decision/result counts differ"
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(RESULT_PATH, index=False)
    saved = pd.read_parquet(RESULT_PATH)
    pd.testing.assert_frame_equal(saved, results, check_exact=True)
    validate_accounting(saved, all_decisions)
    assert model_decision_fingerprint(pd.read_parquet(DECISION_PATH)) == all_fingerprint, "Decision round-trip fingerprint changed"
    assert load_frozen_predictions().shape == predictions.shape, "Frozen predictions changed"

    print("INTEGRITY")
    print(f"  frozen Day 12 prediction fingerprint {FROZEN_FINGERPRINT}: PASS")
    print(f"  frozen Stage 0 selector fingerprint {stage0_fingerprint}: PASS")
    print(f"  Stage 0 counts {stage0_counts}: PASS")
    print("  accounting self-check, one-to-one settlement join, frozen Stage 0 validation P&L: PASS")
    print(f"  model trade-decision fingerprint {all_fingerprint} (fixed-bar-only {fixed_fingerprint})")
    print("PRIMARY FIXED-BAR COMPARISON")
    print_book("M2 stage0_fixed_bar", m2_stats)
    print_book("Frozen Stage 0 stage0_fixed_bar", stage0_stats)
    print(f"PAIRED 21-DAY M2 MINUS STAGE 0: mean=${paired['mean']:+.9f}; sample SD={paired['sample_sd']:.9f}; SE={paired['se']:.9f}; 2SE={paired['two_se']:.9f}; interval=[{paired['lower']:+.9f}, {paired['upper']:+.9f}]")
    print("MODEL-OWN THRESHOLD INPUTS — secondary, 1.2 bps")
    for row in threshold.itertuples(index=False):
        print(f"  T-{row.horizon_minutes} {row.price_bucket}: n={row.basis_validation_row_count}, ECE={row.model_error_term:.9f}, basis={row.basis_term:.9f}, required={row.required_net_edge:.9f}, sparse_merge={row.bucket_merge_fired}")
    print(f"  unperturbed M2 reconstruction max error={reconstruction_error:.3e}; spot-independent raw feature max shift={independent_shift:.1f}")
    print_book("M2 model_own_bar (secondary)", own_stats)
    print_book("M1 stage0_fixed_bar (context)", m1_stats)
    print("Round trips: decisions, model threshold, validation results PASS")
    print(f"Outputs: {DECISION_PATH.relative_to(PROJECT_ROOT)}; {RESULT_PATH.relative_to(PROJECT_ROOT)}; {THRESHOLD_PATH.relative_to(PROJECT_ROOT)}; {PLOT_PATH.relative_to(PROJECT_ROOT)}")
    return results, threshold, paired


if __name__ == "__main__":
    main()
