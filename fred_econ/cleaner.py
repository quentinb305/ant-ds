"""
Data cleaning and frequency alignment.

GDP is reported quarterly; :func:`clean_and_align` resamples every series
to a common monthly (MS) index and forward-fills intra-quarter months so
all columns share a single :class:`DatetimeIndex`.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Rounding precision (decimal places) applied to each column.
_PRECISION: dict[str, int] = {
    "gdp": 3,
    "unemployment_rate": 2,
    "cpi": 3,
}


def clean_and_align(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """Align all series to a common monthly (MS) :class:`DatetimeIndex`.

    Steps applied in order:

    1. Convert each series index to :class:`pd.DatetimeIndex`.
    2. Resample to month-start (``"MS"``), taking the mean within each bin.
    3. Forward-fill NaNs introduced by the quarterly → monthly expansion.
    4. Concatenate all series into a single :class:`pd.DataFrame`.
    5. Drop rows where *every* column is NaN.
    6. Round each column to its defined precision.

    Parameters
    ----------
    raw:
        Dict of raw ``pd.Series`` as returned by :func:`~fred_econ.fetcher.fetch_raw`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a :class:`DatetimeIndex` named ``"date"``
        and columns ``gdp``, ``unemployment_rate``, ``cpi``.

    Notes
    -----
    All-NaN rows typically appear at the leading or trailing edge of the
    window when a FRED series has not yet been published for those dates.
    """
    logger.debug("Aligning %d series to monthly (MS) frequency", len(raw))
    monthly: list[pd.Series] = []

    for col, series in raw.items():
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        s.name = col
        resampled = s.resample("MS").mean().ffill()
        logger.debug("  %s: %d raw obs → %d monthly", col, len(s), len(resampled))
        monthly.append(resampled)

    df = pd.concat(monthly, axis=1)

    before = len(df)
    df.dropna(how="all", inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d all-NaN row(s)", dropped)

    for col, decimals in _PRECISION.items():
        if col in df.columns:
            df[col] = df[col].round(decimals)

    df.index.name = "date"
    logger.info("Cleaned DataFrame: %d rows x %d columns", *df.shape)
    return df
