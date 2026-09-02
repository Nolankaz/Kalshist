import pandas as pd
from pathlib import Path
from storage import load_range


EXCHANGES = [
    "bullish",
    "kraken",
    "crypto_com",
]


def compute_vwap_proxy(close_time):
    close_time = pd.Timestamp(close_time)

    if close_time.tzinfo is None:
        close_time = close_time.tz_localize("UTC")
    else:
        close_time = close_time.tz_convert("UTC")

    start_time = close_time - pd.Timedelta(seconds=60)

    total_price_volume = 0.0
    total_volume = 0.0

    exchanges_used = []
    exchange_details = {}

    for exchange in EXCHANGES:
        day = close_time.strftime("%Y-%m-%d")

        df = load_range(f"btc_prices_1s/{exchange}", day, day)

        if df.empty:
            exchange_details[exchange] = {"rows": 0, "volume": 0.0}
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        window = df[
            (df["timestamp"] > start_time)
            & (df["timestamp"] <= close_time)
        ].copy()

        if window.empty:
            exchange_details[exchange] = {
                "rows": 0,
                "volume": 0.0
            }
            continue

        price_volume = (
            window["price"]
            * window["volume"]
        ).sum()

        volume = window["volume"].sum()

        if volume <= 0:
            exchange_details[exchange] = {
                "rows": len(window),
                "volume": 0.0
            }
            continue

        total_price_volume += price_volume
        total_volume += volume

        exchanges_used.append(exchange)

        exchange_details[exchange] = {
            "rows": len(window),
            "volume": volume
        }

    if total_volume == 0:
        return None

    vwap_proxy = (total_price_volume / total_volume)

    return {
        "vwap_proxy": vwap_proxy,
        "num_exchanges": len(exchanges_used),
        "exchanges_used": exchanges_used,
        "exchange_details": exchange_details,
    }

def build_basis_risk_sample():
    kalshi = load_range(
        "kalshi_markets",
        "2026-05-27",
        "2026-08-25"
    )

    if kalshi.empty:
        print("No Kalshi market data found.")
        return

    settled = kalshi[
        kalshi["settlement_result"].notna()
        & kalshi["expiration_value"].notna()
    ].copy()

    sample_size = min(100, len(settled))

    sample = settled.sample(n=sample_size, random_state=42).copy()

    sample = sample.sort_values("close_time")

    results = []

    for _, market in sample.iterrows():
        ticker = market["ticker"]
        close_time = market["close_time"]
        expiration_value = market["expiration_value"]

        proxy_result = compute_vwap_proxy(close_time)

        if proxy_result is None:
            print(
                f"No price data for {ticker} "
                f"at {close_time}"
            )
            continue

        vwap_proxy = proxy_result["vwap_proxy"]

        diff_dollars = (vwap_proxy - expiration_value)

        diff_bps = (diff_dollars / expiration_value * 10_000)

        results.append({
            "ticker": ticker,
            "close_time": close_time,
            "expiration_value": expiration_value,
            "vwap_proxy": vwap_proxy,
            "diff_dollars": diff_dollars,
            "diff_bps": diff_bps,
            "num_exchanges": proxy_result["num_exchanges"],
            "exchanges_used": ",".join(proxy_result["exchanges_used"]),
        })

        print(
            f"{ticker}: "
            f"{diff_dollars:.2f} dollars, "
            f"{diff_bps:.2f} bps"
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No basis-risk results were created.")
        return
    
    Path("data/basis_risk/samples").mkdir(parents=True, exist_ok=True)

    output_path = "data/basis_risk/samples/basis_risk_sample.parquet"

    results_df.to_parquet(output_path, index=False)

    print(
        f"\nSaved {len(results_df)} results "
        f"to {output_path}"
    )

    print("\nFirst 10 rows:")
    print(results_df.head(10))


if __name__ == "__main__":
    build_basis_risk_sample()