"""
FRED API data fetching.

This module wraps ``fredapi.Fred.get_series`` so the rest of the package
never imports ``fredapi`` directly, making it easy to mock in tests.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from fredapi import Fred

logger = logging.getLogger(__name__)


def fetch_raw(
    fred: "Fred",
    series_map: dict[str, str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    """Fetch FRED series and return them as a dict of raw :class:`pd.Series`.

    Parameters
    ----------
    fred:
        Authenticated ``fredapi.Fred`` client instance.
    series_map:
        Mapping of column name → FRED series ID,
        e.g. ``{"gdp": "GDP", "unemployment_rate": "UNRATE"}``.
    start_date:
        Observation start in ISO-8601 format (``"YYYY-MM-DD"``).
    end_date:
        Observation end in ISO-8601 format (``"YYYY-MM-DD"``).

    Returns
    -------
    dict[str, pd.Series]
        One entry per key in *series_map*, each with a :class:`DatetimeIndex`.
    """
    raw: dict[str, pd.Series] = {}
    for col, series_id in series_map.items():
        logger.info("Fetching %s (%s) …", series_id, col)
        raw[col] = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
        logger.debug("  %s: %d observations", series_id, len(raw[col]))
    return raw
