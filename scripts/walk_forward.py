"""Frozen Day 8 walk-forward schedule and model-agnostic OOF prediction harness."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.analyze_calibration import PREDICTION_COLUMN, fit_platt, platt_feature, stable_sigmoid
from scripts.build_derived_features import OUTPUT_PATH as DERIVED_PATH
from scripts.evaluation_split import EXPECTED_SPLIT_ROWS, SPLIT_KEY, SPLIT_RANGES
from scripts.score_stage0 import EXPECTED_COMMON_ROWS, EXPECTED_SCORING_ROWS, HORIZONS, PREDICTIONS_PATH, ROW_KEY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data/models/walk_forward_schedule.parquet"
LAST_ALLOWED_DATE = SPLIT_RANGES["validation"][1]
EXPECTED_FULL_FIT = (6_414, 7_740, 9_066, 10_392, 11_718, 13_032)
EXPECTED_FULL_SCORE = (1_326, 1_326, 1_326, 1_326, 1_314, 1_326)
EXPECTED_COMMON_FIT = (5_871, 7_155, 8_384, 9_549, 10_654, 11_833)
EXPECTED_COMMON_SCORE = (1_284, 1_229, 1_165, 1_105, 1_179, 1_050)
EXPECTED_COMMON_SCORE_HORIZON = ((644, 640), (616, 613), (586, 579), (556, 549), (594, 585), (535, 515))
EXPECTED_OOF_HORIZON = {10: 3_531, 5: 3_481}
SCHEDULE_COLUMNS = ["fold", "fit_start", "fit_end", "score_start", "score_end", "horizon_minutes", "fit_rows", "score_rows"]
OOF_COLUMNS = ROW_KEY + [SPLIT_KEY, "fold", "probability"]
PLATT_PARAMETER_TOLERANCE = 1e-6
PLATT_PROBABILITY_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Fold:
    number: int
    fit_start: str
    fit_end: str
    score_start: str
    score_end: str


FROZEN_FOLDS = (
    Fold(1, "2026-05-26", "2026-06-28", "2026-06-29", "2026-07-05"),
    Fold(2, "2026-05-26", "2026-07-05", "2026-07-06", "2026-07-12"),
    Fold(3, "2026-05-26", "2026-07-12", "2026-07-13", "2026-07-19"),
    Fold(4, "2026-05-26", "2026-07-19", "2026-07-20", "2026-07-26"),
    Fold(5, "2026-05-26", "2026-07-26", "2026-07-27", "2026-08-02"),
    Fold(6, "2026-05-26", "2026-08-02", "2026-08-03", "2026-08-09"),
)


def validate_folds(folds):
    """Validate chronology and the shared expanding-window shape."""
    assert folds and all(isinstance(fold, Fold) for fold in folds), "Expected one or more Fold objects"
    assert [fold.number for fold in folds] == list(range(1, len(folds) + 1)), "Fold numbers must be consecutive"
    first_start = folds[0].fit_start
    previous_score_end = None
    for fold in folds:
        fit_start, fit_end, score_start, score_end = (date.fromisoformat(value) for value in (
            fold.fit_start, fold.fit_end, fold.score_start, fold.score_end
        ))
        assert fit_start.isoformat() == first_start and fit_start <= fit_end, "Invalid expanding fit window"
        assert fit_end + timedelta(days=1) == score_start and score_start <= score_end, "Fit and score dates are not adjacent"
        assert score_end.isoformat() <= LAST_ALLOWED_DATE, "A fold reaches beyond validation"
        if previous_score_end is not None:
            assert fit_end == previous_score_end, "Fit window must absorb every preceding score window"
        previous_score_end = score_end


def single_fold_schedule(fit_range, score_range):
    """Use the same harness for a single train-fit, validation-score fold."""
    fold = Fold(1, *fit_range, *score_range)
    validate_folds((fold,))
    return (fold,)


def validate_rows(rows):
    required = ROW_KEY + [SPLIT_KEY]
    assert set(required).issubset(rows.columns), f"Missing required columns: {set(required) - set(rows.columns)}"
    assert len(rows) and not rows[required].isna().any().any(), "Missing row key or close_date"
    assert not rows.duplicated(ROW_KEY).any(), f"Duplicate row key {ROW_KEY}"
    assert rows[SPLIT_KEY].map(lambda value: isinstance(value, str)).all(), "close_date must contain stored strings"
    assert rows[SPLIT_KEY].between(SPLIT_RANGES["train"][0], LAST_ALLOWED_DATE).all(), "A test or out-of-range row entered"
    assert set(rows["horizon_minutes"]) == set(HORIZONS), "Unexpected horizon"
    if "split" in rows:
        assert rows["split"].isin(("train", "validation")).all(), "A test or unsupported split entered"
        for split in ("train", "validation"):
            start, end = SPLIT_RANGES[split]
            assert rows.loc[rows["split"].eq(split), SPLIT_KEY].between(start, end).all(), f"{split} label disagrees with close_date"


def fold_slices(rows, fold):
    fit = rows.loc[rows[SPLIT_KEY].between(fold.fit_start, fold.fit_end)].copy()
    score = rows.loc[rows[SPLIT_KEY].between(fold.score_start, fold.score_end)].copy()
    assert len(fit) and len(score), f"Fold {fold.number} has an empty fit or score window"
    assert fit[SPLIT_KEY].between(fold.fit_start, fold.fit_end).all(), "Fit row outside declared window"
    assert score[SPLIT_KEY].between(fold.score_start, fold.score_end).all(), "Score row outside declared window"
    assert fit[SPLIT_KEY].max() < score[SPLIT_KEY].min(), f"Fold {fold.number} violates temporal ordering"
    assert not set(fit["ticker"]) & set(score["ticker"]), f"Fold {fold.number} shares a ticker"
    assert not set(map(tuple, fit[ROW_KEY].itertuples(index=False, name=None))) & set(map(tuple, score[ROW_KEY].itertuples(index=False, name=None))), f"Fold {fold.number} shares a row key"
    return fit, score


def assert_frozen_schedule(all_rows):
    """Regress the Day 8 counts on every train and validation row."""
    validate_rows(all_rows)
    validate_folds(FROZEN_FOLDS)
    assert len(all_rows) == sum(EXPECTED_SPLIT_ROWS[name] for name in ("train", "validation")), "Full population count changed"
    counts = []
    for fold, expected_fit, expected_score in zip(FROZEN_FOLDS, EXPECTED_FULL_FIT, EXPECTED_FULL_SCORE):
        fit, score = fold_slices(all_rows, fold)
        assert len(fit) == expected_fit and len(score) == expected_score, f"Fold {fold.number} full counts changed: fit={len(fit)}, score={len(score)}"
        counts.append((len(fit), len(score)))
    assert sum(score for _, score in counts) == 7_944, "Frozen full score total changed"
    return counts


def common_fold_counts(rows):
    """Return and verify pooled and per-horizon counts for all frozen folds."""
    validate_rows(rows)
    validate_folds(FROZEN_FOLDS)
    assert len(rows) == sum(EXPECTED_COMMON_ROWS.values()), "Full common population changed"
    records = []
    for index, fold in enumerate(FROZEN_FOLDS):
        fit, score = fold_slices(rows, fold)
        assert len(fit) == EXPECTED_COMMON_FIT[index], f"Fold {fold.number} common fit count changed: {len(fit)}"
        assert len(score) == EXPECTED_COMMON_SCORE[index], f"Fold {fold.number} common score count changed: {len(score)}"
        for horizon, expected_score in zip(HORIZONS, EXPECTED_COMMON_SCORE_HORIZON[index]):
            fit_count = int(fit["horizon_minutes"].eq(horizon).sum())
            score_count = int(score["horizon_minutes"].eq(horizon).sum())
            assert score_count == expected_score, f"Fold {fold.number} T-{horizon} common score count changed: {score_count}"
            records.append((fold.number, fold.fit_start, fold.fit_end, fold.score_start, fold.score_end, horizon, fit_count, score_count))
    schedule = pd.DataFrame(records, columns=SCHEDULE_COLUMNS)
    assert schedule["score_rows"].sum() == 7_012, "Common OOF score total changed"
    assert schedule.groupby("horizon_minutes")["score_rows"].sum().to_dict() == EXPECTED_OOF_HORIZON, "Common horizon totals changed"
    return schedule


def run_walk_forward(rows, fit_predict, folds=FROZEN_FOLDS, group_by="horizon_minutes"):
    """Call fit_predict(fit_rows, score_rows, fold=..., horizon=...) per fold/horizon.

    The callback returns (probabilities in score-row order, optional fit metadata).
    No target or model feature is selected by this harness.
    """
    folds = tuple(folds)
    validate_folds(folds)
    validate_rows(rows)
    assert group_by == "horizon_minutes", "Only the frozen horizon grouping is supported"
    if folds == FROZEN_FOLDS:
        common_fold_counts(rows)
    predictions, fit_records = [], []
    scored_keys = set()
    expected_score_keys = set()
    for fold in folds:
        fit, score = fold_slices(rows, fold)
        expected_score_keys.update(map(tuple, score[ROW_KEY].itertuples(index=False, name=None)))
        for horizon in HORIZONS:
            fit_horizon = fit.loc[fit[group_by].eq(horizon)].copy()
            score_horizon = score.loc[score[group_by].eq(horizon)].copy()
            assert len(fit_horizon) and len(score_horizon), f"Fold {fold.number} T-{horizon} has empty fit or score rows"
            assert fit_horizon[SPLIT_KEY].max() < score_horizon[SPLIT_KEY].min(), "Horizon fit dates do not precede score dates"
            assert not set(fit_horizon["ticker"]) & set(score_horizon["ticker"]), "Horizon fit and score tickers overlap"
            before_fit = fit_horizon[ROW_KEY + [SPLIT_KEY]].copy()
            before_score = score_horizon[ROW_KEY + [SPLIT_KEY]].copy()
            probabilities, metadata = fit_predict(fit_horizon, score_horizon, fold=fold, horizon=horizon)
            pd.testing.assert_frame_equal(fit_horizon[ROW_KEY + [SPLIT_KEY]], before_fit)
            pd.testing.assert_frame_equal(score_horizon[ROW_KEY + [SPLIT_KEY]], before_score)
            probabilities = np.asarray(probabilities, dtype=float)
            assert probabilities.ndim == 1 and len(probabilities) == len(score_horizon), "Callback prediction length changed"
            assert np.isfinite(probabilities).all() and ((probabilities > 0) & (probabilities < 1)).all(), "Invalid prediction probability"
            scored = score_horizon[ROW_KEY + [SPLIT_KEY]].copy()
            scored["fold"] = fold.number
            scored["probability"] = probabilities
            keys = set(map(tuple, scored[ROW_KEY].itertuples(index=False, name=None)))
            assert len(keys) == len(scored) and not keys & scored_keys, "A score key repeats within or across folds"
            scored_keys.update(keys)
            predictions.append(scored[OOF_COLUMNS])
            fit_records.append({"fold": fold.number, "horizon_minutes": horizon, "fit_rows": len(fit_horizon), "score_rows": len(score_horizon), "metadata": metadata})
    oof = pd.concat(predictions, ignore_index=True)
    assert len(oof) == len(expected_score_keys) == len(scored_keys), "OOF rows do not cover the exact score windows"
    assert scored_keys == expected_score_keys and not oof.duplicated(ROW_KEY).any(), "OOF keys differ from the declared score population"
    if folds == FROZEN_FOLDS:
        assert len(oof) == 7_012 and oof.groupby("horizon_minutes").size().to_dict() == EXPECTED_OOF_HORIZON, "Frozen common OOF counts changed"
        assert oof[SPLIT_KEY].min() == FROZEN_FOLDS[0].score_start and oof[SPLIT_KEY].max() == FROZEN_FOLDS[-1].score_end, "Frozen OOF date endpoints changed"
        expected_dates = {day.isoformat() for fold in FROZEN_FOLDS for day in (
            date.fromisoformat(fold.score_start) + timedelta(days=offset)
            for offset in range((date.fromisoformat(fold.score_end) - date.fromisoformat(fold.score_start)).days + 1)
        )}
        assert set(oof[SPLIT_KEY]) == expected_dates, "Frozen OOF date coverage changed"
        for fold in FROZEN_FOLDS:
            assert oof.loc[oof["fold"].eq(fold.number), SPLIT_KEY].between(fold.score_start, fold.score_end).all(), "OOF date outside its fold"
    return oof, fit_records


def write_schedule(schedule):
    assert list(schedule.columns) == SCHEDULE_COLUMNS and len(schedule) == 12, "Schedule schema or row count changed"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(OUTPUT_PATH, index=False)
    saved = pd.read_parquet(OUTPUT_PATH)
    pd.testing.assert_frame_equal(saved, schedule, check_exact=True)
    assert list(saved.columns) == SCHEDULE_COLUMNS and len(saved) == 12, "Saved schedule schema or row count changed"
    assert saved["fold"].tolist() == [fold for fold in range(1, 7) for _ in HORIZONS], "Saved fold ordering changed"
    assert saved["horizon_minutes"].tolist() == list(HORIZONS) * 6, "Saved horizon ordering changed"
    assert saved["score_rows"].sum() == 7_012 and saved.groupby("horizon_minutes")["score_rows"].sum().to_dict() == EXPECTED_OOF_HORIZON, "Saved score counts changed"
    for index, fold in enumerate(FROZEN_FOLDS):
        part = saved.iloc[index * len(HORIZONS):(index + 1) * len(HORIZONS)]
        assert part["fit_rows"].sum() == EXPECTED_COMMON_FIT[index] and part["score_rows"].sum() == EXPECTED_COMMON_SCORE[index], "Saved pooled counts changed"
        assert part["score_rows"].tolist() == list(EXPECTED_COMMON_SCORE_HORIZON[index]), "Saved horizon counts changed"
        for column in ("fit_start", "fit_end", "score_start", "score_end"):
            assert part[column].eq(getattr(fold, column)).all(), f"Saved {column} changed"
    assert saved[["fit_start", "fit_end", "score_start", "score_end"]].stack().between(SPLIT_RANGES["train"][0], LAST_ALLOWED_DATE).all(), "Saved date outside frozen range"
    return saved


def self_check_platt_equivalence():
    """Prove the harness using one-feature sklearn and independent Platt fits."""
    feature_columns = ROW_KEY + ["split", SPLIT_KEY, "is_common", "stage0_logit"]
    features = pd.read_parquet(DERIVED_PATH, columns=feature_columns, filters=[("split", "in", ["train", "validation"])])
    assert features["split"].isin(("train", "validation")).all() and len(features) == EXPECTED_SCORING_ROWS, "Derived proof population changed"
    common = features.loc[features["is_common"], ROW_KEY + ["split", SPLIT_KEY, "stage0_logit"]].copy()
    common_fold_counts(common)
    target_columns = ROW_KEY + ["split", "y", PREDICTION_COLUMN]
    targets = pd.read_parquet(PREDICTIONS_PATH, columns=target_columns, filters=[("split", "in", ["train", "validation"])])
    assert list(targets.columns) == target_columns and len(targets) == EXPECTED_SCORING_ROWS, "Proof target projection changed"
    assert targets["split"].isin(("train", "validation")).all() and not targets.duplicated(ROW_KEY).any(), "Unsafe or duplicate proof target rows"
    joined = common.merge(targets, on=ROW_KEY + ["split"], how="left", validate="one_to_one", indicator=True)
    assert len(joined) == len(common) == sum(EXPECTED_COMMON_ROWS.values()) and joined["_merge"].eq("both").all(), "Proof targets do not match common rows"
    assert joined["y"].isin((0, 1)).all() and joined[PREDICTION_COLUMN].notna().all(), "Invalid proof target or Stage 0 input"
    assert np.allclose(joined["stage0_logit"], platt_feature(joined[PREDICTION_COLUMN]), rtol=0, atol=1e-12), "Derived Stage 0 logit changed"
    target_lookup = targets.set_index(ROW_KEY)
    errors = []

    def fit_predict(fit_rows, score_rows, *, fold, horizon):
        fit_targets = target_lookup.loc[pd.MultiIndex.from_frame(fit_rows[ROW_KEY])]
        fit_y = fit_targets["y"].to_numpy(dtype=int)
        assert set(np.unique(fit_y)) == {0, 1}, "Proof fit rows need both target classes"
        fit_x = fit_rows["stage0_logit"].to_numpy(dtype=float).reshape(-1, 1)
        score_x = score_rows["stage0_logit"].to_numpy(dtype=float).reshape(-1, 1)
        model = LogisticRegression(fit_intercept=True, penalty="l2", C=1e12, solver="newton-cholesky", max_iter=10_000, tol=1e-12)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning, module="sklearn.linear_model._logistic")
            model.fit(fit_x, fit_y)
        platt = fit_platt(fit_targets[PREDICTION_COLUMN].to_numpy(dtype=float), fit_y)
        sklearn_p = model.predict_proba(score_x)[:, 1]
        platt_p = stable_sigmoid(platt["a"] + platt["b"] * score_x[:, 0])
        error = (abs(float(model.intercept_[0]) - platt["a"]), abs(float(model.coef_[0, 0]) - platt["b"]), float(np.max(np.abs(sklearn_p - platt_p))))
        assert error[0] <= PLATT_PARAMETER_TOLERANCE and error[1] <= PLATT_PARAMETER_TOLERANCE and error[2] <= PLATT_PROBABILITY_TOLERANCE, f"Fold {fold.number} T-{horizon} Platt disagreement: {error}"
        errors.append(error)
        return sklearn_p, {"platt_intercept_error": error[0], "platt_slope_error": error[1], "platt_probability_error": error[2]}

    oof, fit_records = run_walk_forward(common, fit_predict)
    assert len(fit_records) == len(FROZEN_FOLDS) * len(HORIZONS), "Proof omitted a fold/horizon"
    return tuple(max(error[index] for error in errors) for index in range(3)), oof


def main():
    columns = ROW_KEY + ["split", SPLIT_KEY, "is_common"]
    all_rows = pd.read_parquet(DERIVED_PATH, columns=columns, filters=[("split", "in", ["train", "validation"])])
    assert all_rows["split"].isin(("train", "validation")).all(), "A test row entered the schedule"
    full_counts = assert_frozen_schedule(all_rows)
    common = all_rows.loc[all_rows["is_common"]].copy()
    schedule = common_fold_counts(common)
    assert single_fold_schedule(SPLIT_RANGES["train"], SPLIT_RANGES["validation"]) == (Fold(1, "2026-05-26", "2026-07-19", "2026-07-20", "2026-08-09"),)
    errors, oof = self_check_platt_equivalence()
    write_schedule(schedule)
    print("Frozen full-population schedule (fold: fit, score): " + "; ".join(f"{i}: {fit:,}, {score:,}" for i, (fit, score) in enumerate(full_counts, 1)))
    print("Common fold counts (fold: fit, score, T-10, T-5): " + "; ".join(
        f"{i}: {EXPECTED_COMMON_FIT[i-1]:,}, {EXPECTED_COMMON_SCORE[i-1]:,}, {EXPECTED_COMMON_SCORE_HORIZON[i-1][0]:,}, {EXPECTED_COMMON_SCORE_HORIZON[i-1][1]:,}" for i in range(1, 7)
    ))
    print(f"Common OOF coverage: {len(oof):,}; T-10: {EXPECTED_OOF_HORIZON[10]:,}; T-5: {EXPECTED_OOF_HORIZON[5]:,}")
    print("Temporal ordering assertions: PASS")
    print("Fit/score disjointness: PASS")
    print("Schedule Parquet round trip: PASS")
    print(f"Platt equivalence max intercept error: {errors[0]:.12g}")
    print(f"Platt equivalence max slope error: {errors[1]:.12g}")
    print(f"Platt equivalence max probability error: {errors[2]:.12g}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
