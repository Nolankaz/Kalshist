from itertools import product
from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = PROJECT_ROOT / "data/models/stage0_scores.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/stage0_sigma_selection.parquet"
HORIZONS = (10, 5)
EXPECTED_COMMON_ROWS = {10: 1_685, 5: 1_649}
COVERAGE_FLOOR = 0.80
WINDOW_ORDER = {"5min": 0, "15min": 1, "1hr": 2, "4hr": 3, "24hr": 4}
MATRIX_TOLERANCE = 1e-15


def load_validation_inputs():
    scores = pd.read_parquet(SCORES_PATH, filters=[("split", "==", "validation"), ("population", "==", "common")])
    assert len(scores) == 20, f"Expected 20 validation/common score rows, found {len(scores)}"
    assert scores["split"].eq("validation").all(), "Sigma selection scores must be validation-only"
    assert scores["population"].eq("common").all(), "Candidate-population scores cannot enter sigma selection"
    assert set(scores["horizon_minutes"]) == set(HORIZONS), f"Unexpected score horizons: {set(scores['horizon_minutes'])}"

    predictions = pd.read_parquet(PREDICTIONS_PATH, filters=[("split", "==", "validation")])
    prediction_columns = [column for column in predictions.columns if column.startswith("p_")]
    candidates = [column.removeprefix("p_") for column in prediction_columns]
    assert len(prediction_columns) == 10, f"Expected 10 p_* columns, found {len(prediction_columns)}: {prediction_columns}"
    assert len(candidates) == len(set(candidates)), "Stage 0 candidates must be unique"
    assert predictions["split"].eq("validation").all(), "Non-validation rows entered sigma selection"
    assert not predictions["split"].eq("test").any(), "Test rows entered sigma selection"
    assert not predictions.duplicated(["ticker", "horizon_minutes"]).any(), "Validation prediction keys must be unique"
    assert set(scores["candidate"]) == set(candidates), "Score and prediction candidates differ"
    assert not scores.duplicated(["candidate", "horizon_minutes"]).any(), "Validation/common score keys must be unique"
    for horizon, expected_rows in EXPECTED_COMMON_ROWS.items():
        score_counts = set(scores.loc[scores["horizon_minutes"].eq(horizon), "n"])
        assert score_counts == {expected_rows}, f"T-{horizon} score rows do not use the expected common population: {score_counts}"
    return scores, predictions, prediction_columns, candidates


def build_common_rows(predictions, prediction_columns):
    common_rows = {}
    for horizon in HORIZONS:
        horizon_rows = predictions.loc[predictions["horizon_minutes"].eq(horizon)]
        rows = horizon_rows.loc[horizon_rows[prediction_columns].notna().all(axis=1)].copy()
        assert len(rows) == EXPECTED_COMMON_ROWS[horizon], (
            f"Expected {EXPECTED_COMMON_ROWS[horizon]:,} validation T-{horizon} common rows, found {len(rows):,}"
        )
        assert rows["split"].eq("validation").all(), f"T-{horizon} common rows must be validation-only"
        assert not rows["split"].eq("test").any(), f"Test rows entered the T-{horizon} common population"
        assert rows["y"].notna().all(), f"T-{horizon} common rows contain null y values"
        assert rows["y"].isin([0, 1]).all(), f"T-{horizon} common rows contain invalid y values"
        assert rows[prediction_columns].notna().all().all(), f"T-{horizon} common rows contain null predictions"
        common_rows[horizon] = rows
    return common_rows


def build_brier_summary(scores, candidates):
    indexed = scores.set_index(["candidate", "horizon_minutes"])
    rows = []
    for candidate in candidates:
        brier_t10 = float(indexed.loc[(candidate, 10), "brier"])
        brier_t5 = float(indexed.loc[(candidate, 5), "brier"])
        coverage_t10 = float(indexed.loc[(candidate, 10), "coverage"])
        coverage_t5 = float(indexed.loc[(candidate, 5), "coverage"])
        joint_brier = (EXPECTED_COMMON_ROWS[10] * brier_t10 + EXPECTED_COMMON_ROWS[5] * brier_t5) / sum(EXPECTED_COMMON_ROWS.values())
        rows.append({
            "candidate": candidate,
            "brier_t10": brier_t10,
            "brier_t5": brier_t5,
            "joint_brier": joint_brier,
            "coverage_t10": coverage_t10,
            "coverage_t5": coverage_t5,
            "eligible": coverage_t10 >= COVERAGE_FLOOR and coverage_t5 >= COVERAGE_FLOOR,
        })
    summary = pd.DataFrame(rows)
    assert len(summary) == 10, f"Expected 10 candidate summaries, found {len(summary)}"
    return summary


def paired_statistics(difference):
    n = len(difference)
    mean_difference = float(np.mean(difference))
    standard_error = float(np.std(difference, ddof=1) / math.sqrt(n))
    two_se = 2.0 * standard_error
    return n, mean_difference, standard_error, two_se


def build_paired_comparisons(common_rows, prediction_columns, candidates, brier_summary):
    errors = {}
    score_lookup = brier_summary.set_index("candidate")
    for horizon, rows in common_rows.items():
        targets = rows["y"].to_numpy(dtype=float)
        for column, candidate in zip(prediction_columns, candidates):
            errors[(horizon, candidate)] = (rows[column].to_numpy(dtype=float) - targets) ** 2
            expected_brier = score_lookup.loc[candidate, f"brier_t{horizon}"]
            assert abs(errors[(horizon, candidate)].mean() - expected_brier) <= MATRIX_TOLERANCE, (
                f"Reconstructed T-{horizon} Brier differs from stage0_scores for {candidate}"
            )

    comparisons = []
    for horizon in HORIZONS:
        for candidate_a, candidate_b in product(candidates, repeat=2):
            difference = errors[(horizon, candidate_a)] - errors[(horizon, candidate_b)]
            n, mean_difference, standard_error, two_se = paired_statistics(difference)
            comparisons.append({
                "record_type": "paired_comparison",
                "horizon": f"T-{horizon}",
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "n": n,
                "mean_difference": mean_difference,
                "standard_error": standard_error,
                "two_se": two_se,
                "meaningful_worse": mean_difference > two_se,
                "tied_under_rule": not mean_difference > two_se,
            })

    for candidate_a, candidate_b in product(candidates, repeat=2):
        difference = np.concatenate([
            errors[(10, candidate_a)] - errors[(10, candidate_b)],
            errors[(5, candidate_a)] - errors[(5, candidate_b)],
        ])
        n, mean_difference, standard_error, two_se = paired_statistics(difference)
        comparisons.append({
            "record_type": "paired_comparison",
            "horizon": "joint",
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "n": n,
            "mean_difference": mean_difference,
            "standard_error": standard_error,
            "two_se": two_se,
            "meaningful_worse": mean_difference > two_se,
            "tied_under_rule": not mean_difference > two_se,
        })

    comparisons = pd.DataFrame(comparisons)
    assert len(comparisons) == 300, f"Expected 300 ordered paired comparisons, found {len(comparisons)}"
    validate_pair_matrices(comparisons, candidates)
    return comparisons


def validate_pair_matrices(comparisons, candidates):
    for horizon in ("T-10", "T-5", "joint"):
        rows = comparisons.loc[comparisons["horizon"].eq(horizon)]
        mean_matrix = rows.pivot(index="candidate_a", columns="candidate_b", values="mean_difference").reindex(index=candidates, columns=candidates)
        se_matrix = rows.pivot(index="candidate_a", columns="candidate_b", values="standard_error").reindex(index=candidates, columns=candidates)
        assert mean_matrix.shape == (10, 10), f"{horizon} paired mean-difference matrix must be 10x10"
        assert np.allclose(np.diag(mean_matrix), 0.0, rtol=0.0, atol=MATRIX_TOLERANCE), f"{horizon} mean-difference diagonal is not zero"
        assert np.allclose(mean_matrix, -mean_matrix.T, rtol=0.0, atol=MATRIX_TOLERANCE), f"{horizon} mean-difference matrix is not antisymmetric"
        assert np.allclose(np.diag(se_matrix), 0.0, rtol=0.0, atol=MATRIX_TOLERANCE), f"{horizon} standard-error diagonal is not zero"
        assert np.allclose(se_matrix, se_matrix.T, rtol=0.0, atol=MATRIX_TOLERANCE), f"{horizon} standard-error matrix is not symmetric"


def comparison_row(comparisons, horizon, candidate_a, candidate_b):
    mask = comparisons["horizon"].eq(horizon) & comparisons["candidate_a"].eq(candidate_a) & comparisons["candidate_b"].eq(candidate_b)
    rows = comparisons.loc[mask]
    assert len(rows) == 1, f"Expected one {horizon} comparison for {candidate_a} versus {candidate_b}, found {len(rows)}"
    return rows.iloc[0]


def decision_meaningfully_worse(comparisons, candidate_a, candidate_b):
    joint_worse = bool(comparison_row(comparisons, "joint", candidate_a, candidate_b)["meaningful_worse"])
    horizon_worse = [bool(comparison_row(comparisons, f"T-{horizon}", candidate_a, candidate_b)["meaningful_worse"]) for horizon in HORIZONS]
    return joint_worse and any(horizon_worse)


def apply_frozen_rule(summary, comparisons):
    assert summary["eligible"].notna().all(), "Coverage eligibility must be evaluated for every candidate"
    eligible = summary.loc[summary["eligible"]]
    assert not eligible.empty, "No candidate passes the frozen 80% coverage floor"
    numerically_best = summary.loc[summary["joint_brier"].idxmin(), "candidate"]

    summary = summary.copy()
    summary["meaningfully_worse_than_best"] = summary["candidate"].map(
        lambda candidate: decision_meaningfully_worse(comparisons, candidate, numerically_best)
    )
    summary["tie_set_member"] = summary["eligible"] & ~summary["meaningfully_worse_than_best"]
    assert summary.loc[summary["candidate"].eq(numerically_best), "tie_set_member"].item(), "Numerically best candidate must be in the tie set"

    tie_set = summary.loc[summary["tie_set_member"], "candidate"].tolist()
    surviving_windows = {candidate.split("_", 1)[0] for candidate in tie_set}
    assert surviving_windows <= set(WINDOW_ORDER), f"Unrecognized volatility windows: {surviving_windows - set(WINDOW_ORDER)}"
    selected_window = min(surviving_windows, key=WINDOW_ORDER.get)
    window_candidates = [candidate for candidate in tie_set if candidate.startswith(f"{selected_window}_")]
    simple_candidate = f"{selected_window}_vol"
    ewma_candidate = f"{selected_window}_ewma_vol"

    if simple_candidate in window_candidates and ewma_candidate in window_candidates:
        simple_meaningfully_worse = decision_meaningfully_worse(comparisons, simple_candidate, ewma_candidate)
        final_candidate = ewma_candidate if simple_meaningfully_worse else simple_candidate
        simplicity_decision = "EWMA selected because simple is meaningfully worse" if simple_meaningfully_worse else "simple selected because it is not meaningfully worse than EWMA"
    elif simple_candidate in window_candidates:
        final_candidate = simple_candidate
        simplicity_decision = "simple is the only tied candidate in the selected window"
    elif ewma_candidate in window_candidates:
        final_candidate = ewma_candidate
        simplicity_decision = "EWMA is the only tied candidate in the selected window"
    else:
        raise AssertionError(f"No recognized estimator survives in selected window {selected_window}")

    summary["final_selected"] = summary["candidate"].eq(final_candidate)
    assert summary["final_selected"].sum() == 1, "Exactly one sigma candidate must be selected"
    assert final_candidate in tie_set, "Final sigma candidate must come from the tie set"
    return summary, numerically_best, tie_set, selected_window, simplicity_decision, final_candidate


def build_output(summary, comparisons, numerically_best, selected_window, final_candidate):
    summary_rows = []
    for row in summary.itertuples(index=False):
        summary_rows.append({
            "record_type": "candidate_summary",
            "horizon": "joint",
            "candidate_a": row.candidate,
            "candidate_b": None,
            "candidate_joint_brier": row.joint_brier,
            "brier_t10": row.brier_t10,
            "brier_t5": row.brier_t5,
            "coverage_t10": row.coverage_t10,
            "coverage_t5": row.coverage_t5,
            "eligible": row.eligible,
            "meaningfully_worse_than_best": row.meaningfully_worse_than_best,
            "tie_set_member": row.tie_set_member,
            "final_selected": row.final_selected,
        })

    selection_row = {
        "record_type": "selection",
        "horizon": "joint",
        "candidate_a": final_candidate,
        "candidate_b": numerically_best,
        "coverage_floor": COVERAGE_FLOOR,
        "selected_window": selected_window,
        "numerically_best": numerically_best,
        "final_selected_sigma": final_candidate,
        "selection_rule": "joint Brier, paired 2-SE, shortest window, simple unless meaningfully worse",
    }
    output = pd.concat([comparisons, pd.DataFrame(summary_rows), pd.DataFrame([selection_row])], ignore_index=True, sort=False)
    assert len(output) == 311, f"Expected 311 selection output rows, found {len(output)}"
    assert output.loc[output["record_type"].eq("selection"), "final_selected_sigma"].notna().sum() == 1
    return output


def print_brier_summary(summary):
    print("Validation common-row Stage 0 Brier summary")
    print(summary[["candidate", "brier_t10", "brier_t5", "joint_brier", "coverage_t10", "coverage_t5"]].to_string(
        index=False,
        formatters={
            "brier_t10": "{:.6f}".format,
            "brier_t5": "{:.6f}".format,
            "joint_brier": "{:.6f}".format,
            "coverage_t10": "{:.2%}".format,
            "coverage_t5": "{:.2%}".format,
        },
    ))


def print_horizon_support(comparisons, final_candidate, numerically_best):
    contenders = comparisons.loc[
        comparisons["candidate_b"].eq(numerically_best)
        & comparisons["candidate_a"].ne(numerically_best)
        & comparisons["horizon"].isin(["T-10", "T-5", "joint"])
    ]
    flags = contenders.pivot(index="candidate_a", columns="horizon", values="meaningful_worse")
    both_horizons = flags.index[flags["T-10"] & flags["T-5"]].tolist()
    t5_and_joint = flags.index[~flags["T-10"] & flags["T-5"] & flags["joint"]].tolist()
    pooled_only = flags.index[~flags["T-10"] & ~flags["T-5"] & flags["joint"]].tolist()
    print("\nPer-horizon paired-comparison support")
    print(f"Meaningfully worse than {numerically_best} at both T-10 and T-5: {both_horizons}")
    print(f"Meaningfully worse at T-5 and jointly, but tied at T-10: {t5_and_joint}")
    print(f"Meaningfully worse only after pooling: {pooled_only}")
    print(f"Final selection {final_candidate} is supported without a pooled-only override.")


def select_sigma():
    assert HORIZONS == (10, 5), "Frozen selection horizons changed"
    assert COVERAGE_FLOOR == 0.80, "Frozen coverage floor changed"
    assert WINDOW_ORDER == {"5min": 0, "15min": 1, "1hr": 2, "4hr": 3, "24hr": 4}, "Frozen window tie-break order changed"
    scores, predictions, prediction_columns, candidates = load_validation_inputs()
    common_rows = build_common_rows(predictions, prediction_columns)
    summary = build_brier_summary(scores, candidates)
    comparisons = build_paired_comparisons(common_rows, prediction_columns, candidates, summary)
    summary, numerically_best, tie_set, selected_window, simplicity_decision, final_candidate = apply_frozen_rule(summary, comparisons)

    print_brier_summary(summary)
    meaningfully_worse = summary.loc[summary["meaningfully_worse_than_best"], "candidate"].tolist()
    print(f"\nNumerically best joint-Brier candidate: {numerically_best}")
    print(f"Candidates meaningfully worse than the best: {meaningfully_worse}")
    print(f"Tie set: {tie_set}")
    print(f"Tie-break 1 — shortest surviving window: {selected_window}")
    print(f"Tie-break 2 — {simplicity_decision}")
    print(f"Final selected sigma candidate: {final_candidate}")
    print_horizon_support(comparisons, final_candidate, numerically_best)

    output = build_output(summary, comparisons, numerically_best, selected_window, final_candidate)
    assert not predictions["split"].eq("test").any(), "Test rows entered sigma-selection inputs"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(output):,} selection records to {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    select_sigma()
