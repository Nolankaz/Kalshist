from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = (
    "data/basis_risk/samples/"
    "basis_risk_sample.parquet"
)

PLOTS_FOLDER = Path("data/basis_risk/plots")


def main():
    df = pd.read_parquet(DATA_PATH)

    if df.empty:
        print("Basis-risk sample is empty.")
        return

    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)

    df["abs_diff_bps"] = (df["diff_bps"].abs())

    print(
        f"Loaded {len(df)} "
        f"basis-risk rows"
    )

    # -------------------------
    # 3.1 Overall distribution
    # -------------------------

    mean_bps = df["diff_bps"].mean()
    median_bps = df["diff_bps"].median()
    std_bps = df["diff_bps"].std()

    min_bps = df["diff_bps"].min()
    max_bps = df["diff_bps"].max()

    print("\nBasis-risk summary:")
    print(
        f"Mean diff_bps:   "
        f"{mean_bps:.3f}"
    )
    print(
        f"Median diff_bps: "
        f"{median_bps:.3f}"
    )
    print(
        f"Std diff_bps:    "
        f"{std_bps:.3f}"
    )
    print(
        f"Min diff_bps:    "
        f"{min_bps:.3f}"
    )
    print(
        f"Max diff_bps:    "
        f"{max_bps:.3f}"
    )

    positive_count = ( df["diff_bps"] > 0).sum()

    negative_count = (df["diff_bps"] < 0).sum()

    zero_count = (df["diff_bps"] == 0).sum()

    print("\nDirection:")
    print(f"Positive: {positive_count}")
    print(f"Negative: {negative_count}")
    print(f"Zero:     {zero_count}")

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["diff_bps"],
        bins=20,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero error"
    )

    plt.axvline(
        mean_bps,
        linestyle="--",
        label=(
            f"Mean = "
            f"{mean_bps:.2f} bps"
        )
    )

    plt.axvline(
        median_bps,
        linestyle=":",
        label=(
            f"Median = "
            f"{median_bps:.2f} bps"
        )
    )

    plt.xlabel(
        "Proxy - Kalshi "
        "expiration value (bps)"
    )
    plt.ylabel("Number of markets")
    plt.title("Basis-risk distribution")
    plt.legend()

    plt.tight_layout()

    histogram_path = (
        PLOTS_FOLDER
        / "basis_risk_histogram.png"
    )

    plt.savefig(
        histogram_path,
        dpi=150
    )

    print(
        f"\nSaved histogram to "
        f"{histogram_path}"
    )

    plt.show()

    # -------------------------
    # 3.2 Time drift
    # -------------------------

    df["time_period"] = pd.cut(
        df["close_time"].astype("int64"),
        bins=3,
        labels=[
            "Early",
            "Middle",
            "Late"
        ]
    )

    time_summary = (
        df.groupby("time_period", observed=True)["diff_bps"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
    )

    print("\nBasis risk by time period:")
    print(time_summary)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["close_time"],
        df["diff_bps"],
        alpha=0.7
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Settlement time")
    plt.ylabel("Basis error (bps)")

    plt.title("Basis risk over time")

    plt.tight_layout()

    time_plot_path = (
        PLOTS_FOLDER
        / "basis_risk_over_time.png"
    )

    plt.savefig(
        time_plot_path,
        dpi=150
    )

    print(
        f"\nSaved time plot to "
        f"{time_plot_path}"
    )

    plt.show()

    # -------------------------
    # 3.2 Exchange completeness
    # -------------------------

    exchange_summary = (
        df.groupby("num_exchanges")["diff_bps"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
    )

    print(
        "\nBasis risk by number "
        "of exchanges:"
    )
    print(exchange_summary)

    exchange_abs_summary = (
        df.groupby("num_exchanges")["abs_diff_bps"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "max"
        ])
    )

    print(
        "\nAbsolute basis risk by "
        "number of exchanges:"
    )
    print(exchange_abs_summary)

    combination_summary = (
        df.groupby("exchanges_used")["abs_diff_bps"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "max"
        ])
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        "\nAbsolute basis risk by "
        "exchange combination:"
    )
    print(combination_summary)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["num_exchanges"],
        df["abs_diff_bps"],
        alpha=0.7
    )

    plt.xlabel(
        "Number of exchanges "
        "contributing"
    )

    plt.ylabel("Absolute basis error (bps)")

    plt.title(
        "Basis risk vs "
        "exchange completeness"
    )

    plt.xticks(sorted(df["num_exchanges"].unique()))

    plt.tight_layout()

    completeness_plot_path = (
        PLOTS_FOLDER
        / "basis_risk_by_completeness.png"
    )

    plt.savefig(completeness_plot_path, dpi=150)

    print(
        f"\nSaved completeness plot to "
        f"{completeness_plot_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()