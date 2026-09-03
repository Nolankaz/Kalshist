from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("data/realized_vol/realized_vol_sample.parquet")
PLOT_DIR = Path("data/realized_vol/plots")

VOL_COLUMNS = [
    "5min_vol",
    "5min_ewma_vol",
    "15min_vol",
    "15min_ewma_vol",
    "1hr_vol",
    "1hr_ewma_vol",
    "4hr_vol",
    "4hr_ewma_vol",
    "24hr_vol",
    "24hr_ewma_vol",
]

SIMPLE_VOL_COLUMNS = [
    "5min_vol",
    "15min_vol",
    "1hr_vol",
    "4hr_vol",
    "24hr_vol",
]

EWMA_VOL_COLUMNS = [
    "5min_ewma_vol",
    "15min_ewma_vol",
    "1hr_ewma_vol",
    "4hr_ewma_vol",
    "24hr_ewma_vol",
]


def analyze_realized_vol():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    print("\nSummary statistics:")
    print(df[VOL_COLUMNS].describe())

    percent_stats = df[VOL_COLUMNS].describe()

    percent_rows = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    percent_stats.loc[percent_rows] = percent_stats.loc[percent_rows] * 100

    print("\nSummary statistics in annualized percent:")
    print(percent_stats)

    negative_counts = (df[VOL_COLUMNS] < 0).sum()

    print("\nNegative value counts:")
    print(negative_counts)

    print("\nNaN counts:")
    print(df[VOL_COLUMNS].isna().sum())

    for column in VOL_COLUMNS:
        values = df[column].dropna() * 100

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30)
        plt.title(f"{column} Distribution")
        plt.xlabel("Annualized volatility (%)")
        plt.ylabel("Frequency")
        plt.tight_layout()

        output_path = PLOT_DIR / f"{column}_histogram.png"
        plt.savefig(output_path)
        plt.close()

    for column in VOL_COLUMNS:
        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df[column] * 100)
        plt.title(f"{column} Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Annualized volatility (%)")
        plt.tight_layout()

        output_path = PLOT_DIR / f"{column}_timeseries.png"
        plt.savefig(output_path)
        plt.close()

    plt.figure(figsize=(12, 6))

    for column in SIMPLE_VOL_COLUMNS:
        plt.plot(df["timestamp"], df[column] * 100, label=column)

    plt.title("Simple Realized Volatility by Window")
    plt.xlabel("Timestamp")
    plt.ylabel("Annualized volatility (%)")
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / "simple_vol_windows_timeseries.png"
    plt.savefig(output_path)
    plt.close()

    print("\nSimple realized vol ranges:")

    for column in SIMPLE_VOL_COLUMNS:
        values = df[column].dropna() * 100

        print(
            f"{column}: "
            f"min={values.min():.2f}%, "
            f"median={values.median():.2f}%, "
            f"max={values.max():.2f}%"
        )

    print("\nEWMA realized vol ranges:")

    for column in EWMA_VOL_COLUMNS:
        values = df[column].dropna() * 100

        print(
            f"{column}: "
            f"min={values.min():.2f}%, "
            f"median={values.median():.2f}%, "
            f"max={values.max():.2f}%"
        )

    print(f"\nPlots saved to {PLOT_DIR}")


if __name__ == "__main__":
    analyze_realized_vol()