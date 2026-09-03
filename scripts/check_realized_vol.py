import numpy as np
import pandas as pd

from storage import load_range
from scripts.realized_vol import realized_vol


EXCHANGES = [
    "bullish",
    "kraken",
    "crypto_com",
]

MAX_FFILL_SECONDS = 10


t = pd.Timestamp("2026-07-01 12:00:00", tz="UTC")

start_time = t - pd.Timedelta(minutes=15)
load_start_time = start_time - pd.Timedelta(seconds=1)

dataframes = []

start_day = load_start_time.strftime("%Y-%m-%d")
end_day = t.strftime("%Y-%m-%d")

for exchange in EXCHANGES:
    df = load_range(f"btc_prices_1s/{exchange}", start_day, end_day)

    if df.empty:
        continue

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df[
        (df["timestamp"] >= load_start_time)
        & (df["timestamp"] < t)
    ].copy()

    if df.empty:
        continue

    df["exchange"] = exchange

    dataframes.append(df)

combined = pd.concat(dataframes, ignore_index=True)

combined["price_volume"] = combined["price"] * combined["volume"]

proxy = combined.groupby("timestamp").agg(
    total_price_volume=("price_volume", "sum"),
    total_volume=("volume", "sum"),
)

proxy = proxy[proxy["total_volume"] > 0].copy()

proxy["price"] = proxy["total_price_volume"] / proxy["total_volume"]

proxy = proxy.sort_index()

grid_start = load_start_time.ceil("s")
grid_end = (t - pd.Timedelta(nanoseconds=1)).floor("s")

full_index = pd.date_range(start=grid_start, end=grid_end, freq="1s", tz="UTC")

proxy = proxy.reindex(full_index)

proxy["price"] = proxy["price"].ffill(limit=MAX_FFILL_SECONDS)

log_returns = np.log(proxy["price"] / proxy["price"].shift(1))

log_returns = log_returns.dropna()

log_returns = log_returns[
    (log_returns.index >= start_time)
    & (log_returns.index < t)
]

manual_std = log_returns.std(ddof=1)

seconds_per_year = 365 * 24 * 60 * 60

manual_vol = manual_std * np.sqrt(seconds_per_year)

function_vol = realized_vol(t, "15min")

print(f"Number of proxy prices: {len(proxy)}")
print(f"Number of valid 1-second returns after short-gap fill: {len(log_returns)}")

print(f"\nManual vol:   {manual_vol:.10f}")
print(f"Function vol: {function_vol:.10f}")
print(f"Difference:   {abs(manual_vol - function_vol):.10f}")
