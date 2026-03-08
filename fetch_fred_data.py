"""
Fetches GDP, Unemployment Rate, and CPI data from the FRED API
for the last 10 years, cleans it, and saves as a pandas DataFrame.

FRED series used:
  - GDP      : Gross Domestic Product (quarterly, billions of USD)
  - UNRATE   : Unemployment Rate (monthly, %)
  - CPIAUCSL : CPI for All Urban Consumers (monthly, index 1982-84=100)

Requirements:
  pip install fredapi pandas python-dotenv
"""

import os
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

SERIES = {
    "gdp": "GDP",
    "unemployment_rate": "UNRATE",
    "cpi": "CPIAUCSL",
}

OUTPUT_FILE = "fred_economic_data.csv"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fetch_raw(
    fred: Fred,
    series_map: dict[str, str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    """Fetch each FRED series and return a dict of raw pd.Series."""
    raw: dict[str, pd.Series] = {}
    for col, series_id in series_map.items():
        print(f"Fetching {series_id} ...")
        raw[col] = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
    return raw


def clean_and_align(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Align all series to a common monthly (MS) index.

    GDP is quarterly; forward-filling fills the two intra-quarter months
    so every series has the same date index.  Rows where every column is
    NaN are dropped (can appear at the tail before GDP is released).
    """
    monthly_frames = []
    for col, series in raw.items():
        series = series.copy()
        series.index = pd.to_datetime(series.index)
        series.name = col
        resampled = series.resample("MS").mean().ffill()
        monthly_frames.append(resampled)

    df = pd.concat(monthly_frames, axis=1)
    df.dropna(how="all", inplace=True)

    df["gdp"] = df["gdp"].round(3)
    df["unemployment_rate"] = df["unemployment_rate"].round(2)
    df["cpi"] = df["cpi"].round(3)

    df.index.name = "date"
    return df


def save(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
    print(f"Saved to '{path}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Set FRED_API_KEY in your .env file before running.")

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * 10)

    fred = Fred(api_key=api_key)
    raw = fetch_raw(fred, SERIES, start_date.isoformat(), end_date.isoformat())
    df = clean_and_align(raw)

    print("\n--- Data preview ---")
    print(df.head())
    print(f"\nShape : {df.shape}")
    print(f"Period: {df.index.min().date()} to {df.index.max().date()}")
    print("\nMissing values per column:")
    print(df.isna().sum())

    save(df, OUTPUT_FILE)
    return df


if __name__ == "__main__":
    main()
