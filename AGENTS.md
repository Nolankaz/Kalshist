# Repository Guidelines

## Project Structure & Module Organization

This is a Python research/data pipeline for Kalshi KXBTC15M markets and BTC exchange pricing. Shared daily Parquet helpers live in `storage.py`. Backfills, feature builders, checks, and analysis scripts live in `scripts/`. Project notes and schemas live at the root: `schema.md`, `exchange_notes.md`, and `feature_notes.md`.

Generated local data is under `data/`: Kalshi market files in `data/kalshi_markets/`, per-exchange BTC 1-second buckets in `data/btc_prices_1s/<exchange>/`, realized-vol samples in `data/realized_vol/`, and basis-risk samples/plots in `data/basis_risk/`. Do not dump large Parquet contents unless needed.

## Build, Test, and Development Commands

Use the existing virtual environment when available:

```bash
source .venv/bin/activate
```

Run scripts directly from the repository root so imports such as `from storage import load_range` resolve correctly:

```bash
python scripts/fetch_settled_markets.py
python scripts/backfill_kraken.py
python scripts/check_realized_vol.py
python scripts/analyze_basis_risk.py
```

Backfill scripts fetch and persist source data. `build_*` scripts create derived samples. `check_*` scripts perform validation checks, and `analyze_*` scripts produce research summaries or plots.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, clear snake_case names, and module-level constants in `UPPER_SNAKE_CASE`. Keep normal-length expressions and function calls on one line; split only when readability improves.

Prefer `pathlib.Path` for filesystem paths and pandas/numpy APIs for tabular and numerical work. Keep timestamps timezone-aware in UTC, matching existing `pd.to_datetime(..., utc=True)` usage.

## Quantitative Pipeline Rules

Understand existing code before changing it, and make narrow edits only. Preserve daily storage conventions from `save_daily()` and `load_range()`. Feature values at timestamp `t` must use only data strictly before `t` unless a task explicitly defines otherwise. Do not invent statistical choices silently; document assumptions such as windows, annualization constants, fill limits, or sampling rules.

Before changing validated quantitative code, inspect and preserve the independent checks. Realized volatility currently has independent recomputation in `scripts/check_realized_vol.py` and coverage checks in `scripts/check_vol_coverage.py`. Basis-risk validation has sample creation in `scripts/basis_risk.py` and distribution/completeness analysis in `scripts/analyze_basis_risk.py`.

## Testing Guidelines

There is no formal test framework configured yet. For current work, validate behavior with focused check scripts such as:

```bash
python scripts/check_realized_vol.py
python scripts/check_vol_coverage.py
```

When adding reusable logic, prefer importable helpers and add a small `scripts/check_<feature>.py` validation script if no test harness is introduced. Include expected row counts, date ranges, coverage, or numerical comparisons in output where practical.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries, for example `Add and validate BTC exchange backfill pipelines` and `Complete Day 3 basis risk validation`. Follow that style: describe the completed change in one line.

Pull requests should include the purpose of the change, commands run, affected data ranges or exchanges, and any generated artifacts. Link related issues or notes when relevant. For data pipeline changes, mention whether scripts perform network fetches and whether local `data/` outputs were regenerated.

## Security & Configuration Tips

Keep secrets and API material out of git. `.env`, `*.pem`, generated Parquet files, and major `data/` subdirectories are ignored. Do not commit local credentials or bulky derived datasets unless the repository policy changes.
