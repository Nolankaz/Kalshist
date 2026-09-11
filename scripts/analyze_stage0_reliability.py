from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/models/stage0_predictions.parquet"
PLOT_PATH = PROJECT_ROOT / "data/models/plots/stage0_5min_vol_reliability_train.png"
PREDICTION_COLUMN = "p_5min_vol"
HORIZONS = (10, 5)
EXPECTED_TRAIN_ROWS = 10_392


def build_reliability_table(train, horizon):
    rows = train.loc[train["horizon_minutes"].eq(horizon)].dropna(subset=[PREDICTION_COLUMN]).copy()
    assert rows["split"].eq("train").all(), f"T-{horizon} reliability rows must be train-only"
    assert rows["y"].isin([0, 1]).all(), f"T-{horizon} targets must contain only zero or one"

    rows["probability_interval"] = pd.qcut(rows[PREDICTION_COLUMN], 10)
    table = rows.groupby("probability_interval", observed=True).agg(
        n=("y", "size"),
        mean_predicted_probability=(PREDICTION_COLUMN, "mean"),
        observed_yes_frequency=("y", "mean"),
    ).reset_index()
    table.insert(0, "decile", np.arange(1, len(table) + 1))

    assert len(table) == 10, f"T-{horizon} must produce exactly 10 populated deciles"
    assert table["n"].sum() == len(rows), f"T-{horizon} deciles must contain every non-null prediction exactly once"
    assert table["mean_predicted_probability"].is_monotonic_increasing, f"T-{horizon} decile means must be monotonic"
    return table


def analyze_stage0_reliability():
    columns = ["horizon_minutes", "split", "y", PREDICTION_COLUMN]
    train = pd.read_parquet(INPUT_PATH, columns=columns, filters=[("split", "==", "train")])
    assert len(train) == EXPECTED_TRAIN_ROWS, f"Expected {EXPECTED_TRAIN_ROWS:,} train rows, found {len(train):,}"
    assert train["split"].eq("train").all(), "Reliability analysis must not contain validation or test rows"

    tables = {}
    for horizon in HORIZONS:
        table = build_reliability_table(train, horizon)
        tables[horizon] = table
        print(f"\n5min_vol train reliability: T-{horizon}")
        print(table.to_string(index=False, formatters={
            "mean_predicted_probability": "{:.6f}".format,
            "observed_yes_frequency": "{:.6f}".format,
        }))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, horizon in zip(axes, HORIZONS):
        table = tables[horizon]
        axis.plot(table["mean_predicted_probability"], table["observed_yes_frequency"], marker="o", label="Stage 0 deciles")
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect reliability")
        axis.set_title(f"T-{horizon}")
        axis.set_xlabel("Mean predicted probability")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0].set_ylabel("Observed YES frequency")
    fig.suptitle("Stage 0 Reliability — 5min_vol (Train Only)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved reliability figure to {PLOT_PATH}")
    return tables


if __name__ == "__main__":
    analyze_stage0_reliability()
