"""
Command-line interface for fred_econ.

Usage
-----
::

    fred-econ [--verbose] fetch [--output PATH] [--years N]
    fred-econ [--verbose] serve [--csv PATH] [--port N] [--debug]

Examples
--------
Fetch the last 10 years of data and save to the default CSV::

    fred-econ fetch

Fetch only the last 5 years and save to a custom path::

    fred-econ fetch --years 5 --output data/econ_5y.csv

Launch the dashboard on port 8080::

    fred-econ serve --port 8080

Enable verbose logging for any command::

    fred-econ --verbose fetch
"""

from __future__ import annotations

import logging
import os

import click
from dotenv import load_dotenv

from fred_econ.config import DEFAULT_LOOKBACK_YEARS, DEFAULT_OUTPUT, DateRange, SERIES
from fred_econ.cleaner import clean_and_align
from fred_econ.fetcher import fetch_raw
from fred_econ.storage import load, save

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
    )


def _get_api_key() -> str:
    """Load and validate the FRED API key from the environment / .env file.

    Raises
    ------
    click.ClickException
        If the key is missing or still set to the placeholder value.
    """
    load_dotenv()
    key = os.getenv("FRED_API_KEY", "")
    if not key or key == "your_api_key_here":
        raise click.ClickException(
            "FRED_API_KEY is not set. Add it to your .env file and retry."
        )
    return key


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """FRED economic data pipeline and dashboard.

    Use the sub-commands below to fetch data or launch the dashboard.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# fetch sub-command
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--output", "-o",
    default=DEFAULT_OUTPUT,
    show_default=True,
    metavar="PATH",
    help="Destination CSV file.",
)
@click.option(
    "--years", "-y",
    default=DEFAULT_LOOKBACK_YEARS,
    show_default=True,
    metavar="N",
    type=click.IntRange(1, 50),
    help="Number of years to look back.",
)
@click.pass_context
def fetch(ctx: click.Context, output: str, years: int) -> None:
    """Fetch GDP, unemployment, and CPI from FRED and save to CSV."""
    from fredapi import Fred  # imported here to keep startup fast

    api_key = _get_api_key()
    date_range = DateRange.last_n_years(years)

    logger.info(
        "Fetching %d years of data (%s to %s)",
        years,
        date_range.start,
        date_range.end,
    )

    fred = Fred(api_key=api_key)
    raw = fetch_raw(
        fred,
        SERIES,
        start_date=date_range.start.isoformat(),
        end_date=date_range.end.isoformat(),
    )
    df = clean_and_align(raw)
    save(df, output)

    click.echo(f"Saved {len(df)} rows to '{output}'")
    click.echo(f"Period : {df.index.min().date()} to {df.index.max().date()}")
    click.echo(f"Columns: {list(df.columns)}")
    click.echo("\nMissing values:")
    for col, n in df.isna().sum().items():
        click.echo(f"  {col}: {n}")


# ---------------------------------------------------------------------------
# serve sub-command
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--csv", "csv_path",
    default=DEFAULT_OUTPUT,
    show_default=True,
    metavar="PATH",
    help="CSV data file to visualise.",
)
@click.option(
    "--port", "-p",
    default=8050,
    show_default=True,
    metavar="PORT",
    type=click.IntRange(1024, 65535),
    help="Port for the Dash server.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Run Dash in debug / hot-reload mode.",
)
@click.pass_context
def serve(ctx: click.Context, csv_path: str, port: int, debug: bool) -> None:
    """Launch the interactive Plotly Dash dashboard."""
    from fred_econ.dashboard import create_app

    df = load(csv_path)
    app = create_app(df)

    click.echo(f"Dashboard running at http://127.0.0.1:{port}")
    app.run(debug=debug, port=port)
