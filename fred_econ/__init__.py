"""
fred_econ
=========
FRED economic data pipeline and interactive dashboard.

Public API
----------
fetch_raw        -- pull raw series from the FRED API
clean_and_align  -- resample, forward-fill, and round all series
save             -- persist a DataFrame to CSV
load             -- load a CSV and add derived columns
create_app       -- build the Plotly Dash application
"""

from fred_econ.cleaner import clean_and_align
from fred_econ.fetcher import fetch_raw
from fred_econ.storage import load, save

__all__ = ["clean_and_align", "fetch_raw", "load", "save"]
__version__ = "0.1.0"
