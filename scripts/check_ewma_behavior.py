import pandas as pd


INPUT_PATH = "data/realized_vol/realized_vol_sample.parquet"

df = pd.read_parquet(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp")


WINDOWS = [
    "5min",
    "15min",
    "1hr",
    "4hr",
    "24hr",
]


for window in WINDOWS:
    simple_col = f"{window}_vol"
    ewma_col = f"{window}_ewma_vol"

    valid = df[[simple_col, ewma_col, "timestamp"]].dropna().copy()

    valid["difference"] = valid[ewma_col] - valid[simple_col]
    valid["abs_difference"] = valid["difference"].abs()

    print(f"\n{window}")
    print("Largest EWMA vs simple differences:")

    largest = valid.nlargest(5, "abs_difference")

    print(
        largest[
            [
                "timestamp",
                simple_col,
                ewma_col,
                "difference",
            ]
        ].to_string(index=False)
    )

    print("\nClosest EWMA vs simple values:")

    closest = valid.nsmallest(5, "abs_difference")

    print(
        closest[
            [
                "timestamp",
                simple_col,
                ewma_col,
                "difference",
            ]
        ].to_string(index=False)
    )