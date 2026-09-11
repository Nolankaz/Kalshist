from itertools import combinations

import pandas as pd


SPLIT_RANGES = {
    "train": ("2026-05-26", "2026-07-19"),
    "validation": ("2026-07-20", "2026-08-09"),
    "test": ("2026-08-10", "2026-08-24"),
}
EXPECTED_SPLIT_ROWS = {"train": 10_392, "validation": 3_966, "test": 2_824}
EXPECTED_SPLIT_MARKETS = {"train": 5_196, "validation": 1_983, "test": 1_412}
SPLIT_KEY = "close_date"


def split_masks(features) -> dict[str, pd.Series]:
    assert SPLIT_KEY in features.columns, f"Missing split column: {SPLIT_KEY}"
    assert "ticker" in features.columns, "Missing market identifier column: ticker"
    assert features[SPLIT_KEY].notna().all(), f"{SPLIT_KEY} must not contain nulls"
    assert features[SPLIT_KEY].map(lambda value: isinstance(value, str)).all(), f"{SPLIT_KEY} must contain stored strings"

    close_key = features[SPLIT_KEY]
    masks = {
        split_name: close_key.between(start_date, end_date, inclusive="both")
        for split_name, (start_date, end_date) in SPLIT_RANGES.items()
    }

    for left_name, right_name in combinations(masks, 2):
        assert not (masks[left_name] & masks[right_name]).any(), f"{left_name} and {right_name} masks overlap"

    assignments_per_row = sum(mask.astype("int64") for mask in masks.values())
    assert assignments_per_row.eq(1).all(), "Every row must be assigned to exactly one split"

    assigned_rows = sum(int(mask.sum()) for mask in masks.values())
    assert assigned_rows == len(features), f"Assigned {assigned_rows:,} rows but received {len(features):,}"

    actual_rows = {split_name: int(mask.sum()) for split_name, mask in masks.items()}
    assert actual_rows == EXPECTED_SPLIT_ROWS, f"Unexpected split row counts: {actual_rows}"

    actual_markets = {split_name: int(features.loc[mask, "ticker"].nunique()) for split_name, mask in masks.items()}
    assert actual_markets == EXPECTED_SPLIT_MARKETS, f"Unexpected split market counts: {actual_markets}"
    return masks


def assign_split(features) -> pd.Series:
    masks = split_masks(features)
    labels = pd.Series(pd.NA, index=features.index, dtype="string", name="split")

    for split_name, mask in masks.items():
        labels.loc[mask] = split_name

    assert labels.notna().all(), "Split labels must not contain nulls"
    return labels.astype(pd.CategoricalDtype(categories=list(SPLIT_RANGES)))
