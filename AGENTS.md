# Repository Guidelines

## Project Structure & Module Organization

This is a Python research/data pipeline for Kalshi KXBTC15M markets and BTC exchange pricing. Shared daily Parquet helpers live in `storage.py`. Backfills, feature builders, checks, and analysis scripts live in `scripts/`. Start with `README.md` for the project overview and `ARCHITECTURE.md` for stage and technical-note navigation.

Generated local data is under `data/`: Kalshi market files in `data/kalshi_markets/`, per-exchange BTC 1-second buckets in `data/btc_prices_1s/<exchange>/`, realized-vol samples in `data/realized_vol/`, and basis-risk samples/plots in `data/basis_risk/`. Do not dump large Parquet contents unless needed.

## Build, Test, and Development Commands

Use the existing virtual environment when available:

```bash
source .venv/bin/activate
```

Run scripts as modules from the repository root so imports such as `from storage import load_range` resolve correctly:

```bash
python -m scripts.fetch_settled_markets
python -m scripts.backfill_kraken
python -m scripts.check_realized_vol
python -m scripts.analyze_basis_risk
```

Backfill scripts fetch and persist source data. `build_*` scripts create derived samples. `check_*` scripts perform validation checks, and `analyze_*` scripts produce research summaries or plots.

## Code formatting preferences

- Keep normal-length function calls and expressions on one line.
- Only split code across multiple lines when the line is genuinely long or readability clearly improves.
- Do not vertically expand a function call just because it has one argument.
- Prefer compact, readable formatting over unnecessary line breaks.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, clear snake_case names, and module-level constants in `UPPER_SNAKE_CASE`.

Prefer `pathlib.Path` for filesystem paths and pandas/numpy APIs for tabular and numerical work. Keep timestamps timezone-aware in UTC, matching existing `pd.to_datetime(..., utc=True)` usage.

## Quantitative Pipeline Rules

Understand existing code before changing it, and make narrow edits only. Preserve daily storage conventions from `save_daily()` and `load_range()`. Feature values at timestamp `t` must use only data strictly before `t` unless a task explicitly defines otherwise. Do not invent statistical choices silently; document assumptions such as windows, annualization constants, fill limits, or sampling rules.

Before changing validated quantitative code, inspect and preserve the independent checks; see the integrity controls in `ARCHITECTURE.md` for navigation. Realized volatility currently has independent recomputation in `scripts/check_realized_vol.py` and coverage checks in `scripts/check_vol_coverage.py`. Basis-risk validation has sample creation in `scripts/basis_risk.py` and distribution/completeness analysis in `scripts/analyze_basis_risk.py`.

## Testing Guidelines

There is no formal test framework configured yet. For current work, validate behavior with focused check scripts such as:

```bash
python -m scripts.check_realized_vol
python -m scripts.check_vol_coverage
```

When adding reusable logic, prefer importable helpers and add a small `scripts/check_<feature>.py` validation script if no test harness is introduced. Include expected row counts, date ranges, coverage, or numerical comparisons in output where practical.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries, for example `Add and validate BTC exchange backfill pipelines`. Follow that style: describe the completed change in one line.

Pull requests should include the purpose of the change, commands run, affected data ranges or exchanges, and any generated artifacts. Link related issues or notes when relevant. For data pipeline changes, mention whether scripts perform network fetches and whether local `data/` outputs were regenerated.

## Security & Configuration Tips

Keep secrets and API material out of git. `.env`, `*.pem`, generated Parquet files, and major `data/` subdirectories are ignored. Do not commit local credentials or bulky derived datasets unless the repository policy changes.
