from pathlib import Path

import pandas as pd

from scripts.realized_vol import realized_vol_features


START_TIME = pd.Timestamp("2026-06-16 00:00:00", tz="UTC")
END_TIME = pd.Timestamp("2026-08-24 23:59:59", tz="UTC")
N_TIMESTAMPS = 300


def build_realized_vol_sample():
    timestamps = pd.date_range(
        start=START_TIME,
        end=END_TIME,
        periods=N_TIMESTAMPS,
    )

    results = []

    for i, t in enumerate(timestamps, start=1):
        features = realized_vol_features(t)

        row = {
            "timestamp": t,
            **features,
        }

        results.append(row)

        print(f"Processed {i}/{len(timestamps)}: {t}")

    results_df = pd.DataFrame(results)

    n_obs_columns = [column for column in results_df.columns if column.endswith("_n_obs")]

    print("\nObservation count summary:")
    print(results_df[n_obs_columns].describe())

    EXPECTED_OBS = {
        "5min": 300,
        "15min": 900,
        "1hr": 3600,
        "4hr": 14400,
        "24hr": 86400,
    }

    print("\nCoverage summary:")

    for window, expected in EXPECTED_OBS.items():
        coverage = results_df[f"{window}_n_obs"] / expected

        print(
            f"{window}: "
            f"min={coverage.min():.1%}, "
            f"median={coverage.median():.1%}, "
            f"mean={coverage.mean():.1%}"
        )

    vol_columns = [
        column
        for column in results_df.columns
        if column.endswith("_vol")
    ]

    print("\nVolatility NaN counts:")
    print(results_df[vol_columns].isna().sum())

    complete_rows = results_df[vol_columns].notna().all(axis=1).sum()

    print(f"\nRows with all vol features valid: {complete_rows}/{len(results_df)}")

    print("\nRejected windows due to coverage:")

    for window in EXPECTED_OBS:
        rejected = results_df[f"{window}_vol"].isna().sum()
        print(f"{window}: {rejected}/{len(results_df)}")

    output_path = Path("data/realized_vol/realized_vol_sample.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_parquet(output_path, index=False)

    print(f"\nSaved {len(results_df)} rows to {output_path}")


if __name__ == "__main__":
    build_realized_vol_sample()