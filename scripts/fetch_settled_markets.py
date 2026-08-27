import httpx
import time
import pandas as pd
from datetime import datetime, timezone
from storage import save_daily

LIVE_URL = "https://external-api.kalshi.com/trade-api/v2/markets"
HISTORICAL_URL = ("https://external-api.kalshi.com/trade-api/v2/historical/markets")

start_date = datetime(2026, 5, 26, tzinfo=timezone.utc)
end_date = datetime(2026, 8, 24, 23, 59, 59, tzinfo=timezone.utc)

def fetch_all_pages(url, params):
    markets = []
    cursor = None

    while True:
        request_params = params.copy()

        if cursor:
            request_params["cursor"] = cursor

        resp = httpx.get(
            url,
            params=request_params,
            timeout=30.0
        )

        resp.raise_for_status()

        data = resp.json()
        page_markets = data["markets"]

        markets.extend(page_markets)

        print(
            f"Fetched {len(page_markets)} markets "
            f"from page; total = {len(markets)}"
        )

        cursor = data.get("cursor")

        if not cursor:
            break

        time.sleep(0.25)

    return markets

live_params = {
    "limit": 1000,
    "status": "settled",
    "series_ticker": "KXBTC15M",
    "min_settled_ts": int(start_date.timestamp()),
    "max_settled_ts": int(end_date.timestamp()),
}

print("Fetching recent/live settled markets...")

live_markets = fetch_all_pages(LIVE_URL, live_params)

historical_params = {
    "limit": 1000,
    "series_ticker": "KXBTC15M",
}

print("Fetching archived historical markets...")

historical_markets = fetch_all_pages(HISTORICAL_URL, historical_params)

all_markets = live_markets + historical_markets

unique_markets = {}

for market in all_markets:
    unique_markets[market["ticker"]] = market

all_markets = list(unique_markets.values())

rows = []

for m in all_markets:
    row = {
        "ticker": m["ticker"],
        "strike": m.get("floor_strike"),
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "settlement_result": m.get("result"),
        "expiration_value": m.get("expiration_value"),
        "settlement_value": m.get("settlement_value_dollars"),
        "volume": m.get("volume_fp"),
        "settlement_ts": m.get("settlement_ts"),
    }

    rows.append(row)


df = pd.DataFrame(rows)

df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
df["settlement_ts"] = pd.to_datetime(df["settlement_ts"], utc=True)

df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["settlement_value"] = pd.to_numeric(df["settlement_value"], errors="coerce")
df["expiration_value"] = pd.to_numeric(df["expiration_value"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df[(df["settlement_ts"] >= start_date) & (df["settlement_ts"] <= end_date)]
df = df.sort_values("settlement_ts").reset_index(drop=True)
df["date"] = (df["settlement_ts"].dt.strftime("%Y-%m-%d"))

for date, daily_df in df.groupby("date"):
    daily_df = daily_df.drop(
        columns=["date"]
    )

    save_daily(daily_df, "kalshi_markets", date)

    print(f"Saved {len(daily_df)} markets " f"for {date}")

print()
print("Backfill complete.")
print(f"Total markets: {len(df)}")
print(f"First settlement: {df['settlement_ts'].min()}")
print(f"Last settlement: {df['settlement_ts'].max()}")