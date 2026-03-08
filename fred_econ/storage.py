"""
CSV persistence helpers.

:func:`save` writes a cleaned DataFrame to disk.
:func:`load` reads it back and appends the derived columns
``gdp_growth_yoy`` and ``cpi_yoy`` that are needed by the dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save(df: pd.DataFrame, path: str | Path) -> None:
    """Persist *df* to a CSV file at *path*.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    df:
        DataFrame to persist.  Must have a :class:`DatetimeIndex`
        named ``"date"``.
    path:
        Destination file path (created or overwritten).
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest)
    logger.info("Saved %d rows to '%s'", len(df), dest)


def load(path: str | Path) -> pd.DataFrame:
    """Load a CSV previously written by :func:`save`.

    In addition to the stored columns, two derived columns are appended:

    * ``gdp_growth_yoy`` – GDP year-over-year growth rate (%).
    * ``cpi_yoy``        – CPI year-over-year change rate (%) i.e. inflation.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a :class:`DatetimeIndex` named ``"date"``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Data file not found: {src}")

    df = pd.read_csv(src, index_col="date", parse_dates=True)
    df["gdp_growth_yoy"] = df["gdp"].pct_change(12) * 100
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100
    logger.info("Loaded %d rows from '%s'", len(df), src)
    return df
