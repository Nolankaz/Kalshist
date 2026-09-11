from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_market_features import FEATURE_COLUMNS
from scripts.evaluation_split import assign_split


INPUT_PATH = PROJECT_ROOT / "data/features/market_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
EXPECTED_ROWS = 17_182
ROW_KEY = ["ticker", "horizon_minutes"]
BASE_OUTPUT_COLUMNS = [
    "ticker",
    "horizon_minutes",
    "close_date",
    "split",
    "log_moneyness",
    "T_years",
    "y",
]

_erf = np.vectorize(math.erf, otypes=[float])


def norm_cdf(z):
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def stage0_probability(log_moneyness, sigma_annualized, T_years):
    """P(YES) under zero-drift lognormal prices. NaN in -> NaN out."""
    sigma_sqrt_t = sigma_annualized * np.sqrt(T_years)
    z = log_moneyness / sigma_sqrt_t
    return norm_cdf(z), z


def build_stage0_predictions():
    features = pd.read_parquet(INPUT_PATH)
    sigma_candidates = [column for column in FEATURE_COLUMNS if column.endswith(("_vol", "_ewma_vol"))]

    assert len(sigma_candidates) == 10, f"Expected 10 sigma candidates, found {len(sigma_candidates)}: {sigma_candidates}"
    assert len(sigma_candidates) == len(set(sigma_candidates)), "Sigma candidates must be unique"

    required_columns = BASE_OUTPUT_COLUMNS.copy()
    required_columns.remove("split")
    required_columns.extend(sigma_candidates)
    missing_columns = [column for column in required_columns if column not in features.columns]
    assert not missing_columns, f"Missing required market-feature columns: {missing_columns}"

    assert len(features) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} market-feature rows, found {len(features):,}"
    assert not features.duplicated(ROW_KEY).any(), f"Market-feature row key {ROW_KEY} must be unique"
    assert features["T_years"].notna().all(), "T_years must not contain nulls"
    assert features["T_years"].gt(0).all(), "Every T_years value must be positive"

    predictions = features[[column for column in BASE_OUTPUT_COLUMNS if column != "split"]].copy()
    predictions.insert(BASE_OUTPUT_COLUMNS.index("split"), "split", assign_split(features))

    log_moneyness = features["log_moneyness"].to_numpy(dtype=float)
    T_years = features["T_years"].to_numpy(dtype=float)
    for candidate in sigma_candidates:
        sigma = features[candidate].to_numpy(dtype=float)
        non_null_sigma = sigma[~np.isnan(sigma)]
        assert np.all(non_null_sigma > 0), f"Every non-null {candidate} value must be positive"

        p, z = stage0_probability(log_moneyness, sigma, T_years)
        expected_nan = np.isnan(log_moneyness) | np.isnan(sigma) | np.isnan(T_years)
        assert len(p) == len(features) and len(z) == len(features), f"{candidate} predictions must cover every source row"
        assert np.array_equal(np.isnan(z), expected_nan), f"z_{candidate} did not preserve the input NaN mask"
        assert np.array_equal(np.isnan(p), expected_nan), f"p_{candidate} did not preserve the input NaN mask"
        predictions[f"p_{candidate}"] = p
        predictions[f"z_{candidate}"] = z

    prediction_columns = BASE_OUTPUT_COLUMNS + [f"p_{candidate}" for candidate in sigma_candidates]
    prediction_columns += [f"z_{candidate}" for candidate in sigma_candidates]
    predictions = predictions[prediction_columns]

    assert len(predictions) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} prediction rows, found {len(predictions):,}"
    assert len(predictions) == len(features), "Prediction generation dropped or duplicated rows"
    assert not predictions.duplicated(ROW_KEY).any(), f"Prediction row key {ROW_KEY} must be unique"

    source_keys = features[ROW_KEY].sort_values(ROW_KEY).reset_index(drop=True)
    prediction_keys = predictions[ROW_KEY].sort_values(ROW_KEY).reset_index(drop=True)
    assert prediction_keys.equals(source_keys), "Prediction row keys do not match market-feature row keys after sorting"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(predictions):,} rows to {OUTPUT_PATH}")
    print(f"Sigma candidates ({len(sigma_candidates)}): {sigma_candidates}")
    print(f"Output columns: {predictions.columns.tolist()}")
    return predictions


if __name__ == "__main__":
    build_stage0_predictions()
