# CLAUDE.md

Project context and conventions for Claude Code.

## Project overview

`fred-econ` is an installable Python package that fetches GDP, unemployment rate,
and CPI data from the FRED API, cleans it, and renders an interactive Plotly Dash
dashboard with recession shading and correlation analysis.

## Package structure

```
fred_econ/
  __init__.py    public API re-exports, __version__
  config.py      SERIES map, RECESSIONS list, DateRange dataclass
  fetcher.py     fetch_raw()  — wraps fredapi.Fred.get_series
  cleaner.py     clean_and_align()  — resample / ffill / round
  storage.py     save() / load()  — CSV I/O + derived columns
  dashboard.py   create_app(df)  — returns configured Dash app
  cli.py         Click CLI  — fred-econ fetch / serve
tests/
  conftest.py         shared pytest fixtures
  test_cleaner.py
  test_fetcher.py
  test_storage.py
  test_cli.py
pyproject.toml
.env                  FRED_API_KEY goes here (git-ignored)
```

## Environment setup

```bash
pip install -e ".[dev]"   # installs package + pytest + pytest-cov
```

Requires a `.env` file with:
```
FRED_API_KEY=<your_key>
```
Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html

## Common commands

```bash
# Fetch 10 years of data (default)
fred-econ fetch

# Fetch last 5 years to a custom path
fred-econ fetch --years 5 --output data/econ_5y.csv

# Launch the Plotly Dash dashboard (default port 8050)
fred-econ serve

# Serve from a specific CSV on a different port
fred-econ serve --csv data/econ_5y.csv --port 8080

# Enable verbose (DEBUG) logging on any command
fred-econ --verbose fetch
```

## Running tests

```bash
pytest                        # all 53 tests, verbose (configured in pyproject.toml)
pytest tests/test_cleaner.py  # single module
pytest --cov                  # with coverage report
```

Tests are fully mocked — no FRED API key required.

## Code conventions

- **Python 3.11+** with `from __future__ import annotations` in every module.
- **Type hints** on all public functions and methods.
- **Docstrings** on every module, class, and function (parameters / returns / raises).
- **Logging** via `logging.getLogger(__name__)` — never `print()` in library code.
- **No top-level side effects** — modules are safe to import without triggering I/O.
- Constants live in `fred_econ/config.py`; do not scatter magic values elsewhere.
- `fredapi` is only imported inside function bodies (lazy import) to keep startup fast
  and make mocking straightforward (`patch("fredapi.Fred", ...)`).
- `dashboard.py` uses a factory (`create_app(df)`) so the Dash app can be tested
  without a running server.

## Key data flow

```
FRED API
  └─ fetch_raw()          → dict[str, pd.Series]  (raw, mixed frequency)
       └─ clean_and_align() → pd.DataFrame         (monthly MS, ffilled, rounded)
            └─ save()       → CSV on disk
                 └─ load()  → pd.DataFrame + gdp_growth_yoy + cpi_yoy
                      └─ create_app(df) → Dash app
```

## Git workflow

- `main` — stable baseline (original standalone scripts)
- Feature branches named `feat/<topic>`, e.g. `feat/package-refactor`
- PRs target `main`; remote is https://github.com/quentinb305/ant-ds
