# Kalshist

Kalshist is a Python research and data pipeline for Kalshi's `KXBTC15M` Bitcoin prediction markets. It builds a local, timestamp-aligned dataset of settled markets, historical Kalshi quotes, live top-of-book observations, and BTC spot-market data for later probability modeling and trading-strategy evaluation.

The project deliberately separates two questions:

- **Probability-model evaluation:** Are predicted YES probabilities calibrated and informative relative to settled outcomes?
- **Trading-strategy evaluation:** Could those predictions have been traded after accounting for available quotes, fees, size, and execution constraints?

Data collection and validation are complete through Day 5. Modeling and strategy backtesting have not started, so the repository makes no claim of predictive edge or profitability.

## Data Pipeline

The current pipeline collects and validates:

- Settled `KXBTC15M` market metadata, including market windows, strikes, BTC expiration values, and YES/NO outcomes.
- Historical one-minute Kalshi candlesticks containing bid, ask, and trade-price OHLC values. These are partitioned by market close date.
- Live Kalshi snapshots sampled roughly every 10 seconds, including top-of-book bid/ask prices and actual bid/ask sizes.
- One-second BTC trade aggregates from Bullish, Kraken, and Crypto.com, with VWAP price, volume, and trade count.
- Leakage-safe realized-volatility features over 5-minute through 24-hour windows, using data strictly before each feature timestamp.
- Basis-risk samples used to compare exchange-derived BTC reference prices around Kalshi settlement times.

All generated datasets are stored locally as daily Parquet files under `data/`. Raw and derived data are intentionally ignored by Git and must be generated locally.

See [schema.md](schema.md) for exact columns and UTC timestamp semantics, [exchange_notes.md](exchange_notes.md) for exchange selection, [feature_notes.md](feature_notes.md) for realized-volatility methodology, and [quote_notes.md](quote_notes.md) for Kalshi quote behavior and coverage.

## Day 5 Validation

Historical quote validation covers 8,597 settled markets across 91 close dates:

- 8,591 markets have at least one candle; 8,586 have all 15 expected one-minute candles.
- 128,820 quote rows were collected: 48,066 from the historical endpoint and 80,754 from the series endpoint.
- There were zero bid/ask close-ordering violations, zero non-null price values outside `[0, 1]`, and zero entirely missing close dates.
- Six markets have no returned candles, and five have partial coverage. These gaps were reproduced directly against Kalshi's API and were not filled or synthesized.

Historical candlesticks do not contain bid size, ask size, or full order-book depth. They support quote-aware analysis but cannot establish whether an arbitrary quantity could have filled. The live recorder captures top-of-book sizes for forward testing, but it does not capture full depth.

## Evaluation Tiers

Every future P&L result is intended to carry one of these labels:

- **Tier 1: probability-only.** No market prices are used.
- **Tier 2: quote-aware/top-of-book.** Historical quotes are used, but historical size is unavailable.
- **Tier 3: depth-aware/execution-aware.** Available size and execution constraints are modeled.

An unlabeled P&L figure is not considered a complete project result.

## Repository Structure

```text
.
|-- storage.py                 # Shared daily Parquet save/load helpers
|-- scripts/                   # Backfills, feature builders, checks, and analyses
|-- data/                      # Generated local datasets and plots (mostly ignored)
|-- schema.md                  # Dataset schemas and timestamp conventions
|-- exchange_notes.md          # BTC exchange-selection evidence
|-- feature_notes.md           # Feature definitions and leakage controls
`-- quote_notes.md             # Kalshi endpoint, coverage, and execution notes
```

## Setup

Create and activate a virtual environment from the repository root, then install the direct dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The data-fetching scripts use public exchange and Kalshi HTTP endpoints. They write into `data/` and may take substantial time to complete. Run scripts from the repository root so local imports resolve consistently.

## Representative Commands

```bash
# Fetch settled Kalshi markets and exchange trade data
python3 -m scripts.fetch_settled_markets
python3 -m scripts.backfill_bullish
python3 -m scripts.backfill_kraken
python3 -m scripts.backfill_crypto_com

# Backfill and validate historical Kalshi quotes
python3 -m scripts.backfill_kalshi_quotes
python3 -m scripts.check_kalshi_quotes
python3 -m scripts.inspect_kalshi_market

# Build and validate volatility features
python3 -m scripts.build_realized_vol_sample
python3 -m scripts.check_realized_vol
python3 -m scripts.check_vol_coverage

# Run the live top-of-book recorder; Ctrl+C flushes buffered observations
python3 -m scripts.record_kalshi_quotes
```

Analysis entry points include `python3 -m scripts.analyze_realized_vol`, `python3 -m scripts.basis_risk`, and `python3 -m scripts.analyze_basis_risk`.

## Current Limitations

- No probability model, backtest, or trading system has been implemented yet.
- Historical Kalshi quotes are top-of-book candles without historical size or full depth.
- A small number of historical markets have reproducible upstream candle-coverage gaps.
- Exchange-derived BTC settlement proxies are not the official Kalshi settlement index.
- Live quote snapshots contain top-of-book size, but not a complete order book.
