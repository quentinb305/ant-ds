"""
Central configuration for the fred_econ package.

All constants, FRED series mappings, and recession date ranges live here
so the rest of the package has a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

#: FRED series IDs mapped to friendly DataFrame column names.
SERIES: Final[dict[str, str]] = {
    "gdp": "GDP",
    "unemployment_rate": "UNRATE",
    "cpi": "CPIAUCSL",
}

#: NBER recession periods as (ISO start, ISO end) string pairs.
RECESSIONS: Final[list[tuple[str, str]]] = [
    ("2020-02-01", "2020-04-30"),  # COVID-19
]

#: Default output CSV filename.
DEFAULT_OUTPUT: Final[str] = "fred_economic_data.csv"

#: Default look-back window in years.
DEFAULT_LOOKBACK_YEARS: Final[int] = 10


@dataclass(frozen=True)
class DateRange:
    """Immutable start/end date pair used for FRED API requests."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(
                f"start ({self.start}) must be strictly before end ({self.end})"
            )

    @classmethod
    def last_n_years(cls, years: int = DEFAULT_LOOKBACK_YEARS) -> "DateRange":
        """Return a :class:`DateRange` covering the last *years* years to today.

        Parameters
        ----------
        years:
            Number of calendar years to look back (default 10).
        """
        end = date.today()
        start = end - timedelta(days=365 * years)
        return cls(start=start, end=end)
