import numpy as np
import pandas as pd

from storage import load_range

SECONDS_PER_YEAR = 365 * 24 * 60 * 60
MAX_FFILL_SECONDS = 10
MIN_COVERAGE = 0.80

WINDOWS = {
    "5min": pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "1hr": pd.Timedelta(hours=1),
    "4hr": pd.Timedelta(hours=4),
    "24hr": pd.Timedelta(hours=24),
}

EWMA_HALFLIVES = {
    "5min": pd.Timedelta(minutes=2.5),
    "15min": pd.Timedelta(minutes=7.5),
    "1hr": pd.Timedelta(minutes=30),
    "4hr": pd.Timedelta(hours=2),
    "24hr": pd.Timedelta(hours=12),
}

EXPECTED_OBS = {
    "5min": 300,
    "15min": 900,
    "1hr": 3600,
    "4hr": 14400,
    "24hr": 86400,
}

EXCHANGES = [
    "bullish",
    "kraken",
    "crypto_com",
]

def load_exchange_prices(start_time, end_time):
    dataframes = []

    start_day = start_time.strftime("%Y-%m-%d")
    end_day = end_time.strftime("%Y-%m-%d")

    for exchange in EXCHANGES:
        df = load_range(f"btc_prices_1s/{exchange}", start_day, end_day)

        if df.empty:
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df = df[
            (df["timestamp"] >= start_time)
            & (df["timestamp"] < end_time)
        ].copy()

        if df.empty:
            continue

        dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)

def build_vwap_proxy(df):
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    df["price_volume"] = df["price"] * df["volume"]

    proxy = df.groupby("timestamp").agg(
        total_price_volume=("price_volume", "sum"),
        total_volume=("volume", "sum"),
    )

    proxy = proxy[proxy["total_volume"] > 0].copy()

    proxy["price"] = proxy["total_price_volume"] / proxy["total_volume"]

    return proxy[["price"]].sort_index()

def make_1s_proxy(proxy, start_time, end_time):
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    grid_start = start_time.ceil("s")
    grid_end = (end_time - pd.Timedelta(nanoseconds=1)).floor("s")

    if grid_end < grid_start:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=["price"])

    full_index = pd.date_range(
        start=grid_start,
        end=grid_end,
        freq="1s",
        tz="UTC",
    )

    proxy = proxy.reindex(full_index)

    proxy["price"] = proxy["price"].ffill(limit=MAX_FFILL_SECONDS)

    return proxy

def get_window_log_returns(t, window):
    if window not in WINDOWS:
        raise ValueError(f"Unsupported window: {window}")

    t = pd.Timestamp(t)

    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    start_time = t - WINDOWS[window]
    load_start_time = start_time - pd.Timedelta(seconds=1)

    df = load_exchange_prices(load_start_time, t)

    if df.empty:
        return pd.Series(dtype=float)

    proxy = build_vwap_proxy(df)

    if len(proxy) < 2:
        return pd.Series(dtype=float)

    proxy = make_1s_proxy(proxy, load_start_time, t)

    log_returns = np.log(proxy["price"] / proxy["price"].shift(1)).dropna()

    log_returns = log_returns[
        (log_returns.index >= start_time)
        & (log_returns.index < t)
    ]

    return log_returns

def get_all_window_log_returns(t):
    t = pd.Timestamp(t)

    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    max_window = max(WINDOWS.values())

    start_time = t - max_window
    load_start_time = start_time - pd.Timedelta(seconds=1)

    df = load_exchange_prices(load_start_time, t)

    if df.empty:
        return {
            window: pd.Series(dtype=float)
            for window in WINDOWS
        }

    proxy = build_vwap_proxy(df)

    if len(proxy) < 2:
        return {
            window: pd.Series(dtype=float)
            for window in WINDOWS
        }

    proxy = make_1s_proxy(proxy, load_start_time, t)

    log_returns = np.log(proxy["price"] / proxy["price"].shift(1)).dropna()

    window_returns = {}

    for window, duration in WINDOWS.items():
        window_start = t - duration

        returns = log_returns[
            (log_returns.index >= window_start)
            & (log_returns.index < t)
        ]

        window_returns[window] = returns

    return window_returns

def realized_vol(t, window):
    log_returns = get_window_log_returns(t, window)

    coverage = len(log_returns) / EXPECTED_OBS[window]

    if coverage < MIN_COVERAGE:
        return np.nan

    vol = log_returns.std(ddof=1)

    annualized_vol = vol * np.sqrt(SECONDS_PER_YEAR)

    return annualized_vol

def ewma_realized_vol(t, window):
    log_returns = get_window_log_returns(t, window)

    coverage = len(log_returns) / EXPECTED_OBS[window]

    if coverage < MIN_COVERAGE:
        return np.nan

    squared_returns = log_returns ** 2

    ewma_variance = squared_returns.ewm(
        halflife=EWMA_HALFLIVES[window],
        times=squared_returns.index,
    ).mean().iloc[-1]

    ewma_vol = np.sqrt(ewma_variance)

    annualized_vol = ewma_vol * np.sqrt(SECONDS_PER_YEAR)

    return annualized_vol

def realized_vol_features(t):
    window_returns = get_all_window_log_returns(t)

    features = {}

    for window, log_returns in window_returns.items():
        n_obs = len(log_returns)
        coverage = n_obs / EXPECTED_OBS[window]

        if coverage < MIN_COVERAGE:
            simple_vol = np.nan
            ewma_vol = np.nan
        else:
            simple_vol = log_returns.std(ddof=1) * np.sqrt(SECONDS_PER_YEAR)

            squared_returns = log_returns ** 2

            ewma_variance = squared_returns.ewm(
                halflife=EWMA_HALFLIVES[window],
                times=squared_returns.index,
            ).mean().iloc[-1]

            ewma_vol = np.sqrt(ewma_variance) * np.sqrt(SECONDS_PER_YEAR)

        features[f"{window}_vol"] = simple_vol
        features[f"{window}_ewma_vol"] = ewma_vol
        features[f"{window}_n_obs"] = n_obs
        features[f"{window}_coverage"] = coverage

    return features
