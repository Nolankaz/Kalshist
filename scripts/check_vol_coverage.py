import pandas as pd


EXPECTED_OBS = {
    "5min": 300,
    "15min": 900,
    "1hr": 3600,
    "4hr": 14400,
    "24hr": 86400,
}

df = pd.read_parquet("data/realized_vol/realized_vol_sample.parquet")

for window, expected in EXPECTED_OBS.items():
    coverage = df[f"{window}_n_obs"] / expected

    print(f"\n{window}")
    print(f"Min:    {coverage.min():.1%}")
    print(f"25th:   {coverage.quantile(0.25):.1%}")
    print(f"Median: {coverage.median():.1%}")
    print(f"Mean:   {coverage.mean():.1%}")
    print(f"75th:   {coverage.quantile(0.75):.1%}")
    print(f"Max:    {coverage.max():.1%}")
    print(f"Below 80%: {(coverage < 0.80).sum()}/{len(df)}")
    print(f"Below 90%: {(coverage < 0.90).sum()}/{len(df)}")
