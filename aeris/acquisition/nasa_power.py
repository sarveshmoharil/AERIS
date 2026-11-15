
"""
AERIS v1.0
NASA POWER Climate Data Acquisition
Region: Nashik, Maharashtra, India

Downloads daily agroclimatological data.
Preserves raw JSON and produces a cleaned CSV.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "aeris" / "data" / "raw" / "nasa_power"
PROCESSED_DIR = PROJECT_ROOT / "aeris" / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Nashik city representative coordinate.
# This is an initial pilot point, NOT a district-wide average.
LATITUDE = 19.9975
LONGITUDE = 73.7898

START_DATE = "20010101"

# Leave a 7-day buffer for data publication latency.
END_DATE = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

PARAMETERS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN"
]

BASE_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AERIS.NASA.POWER")


# ============================================================
# DATA ACQUISITION
# ============================================================

def download_data():

    logger.info("Starting NASA POWER acquisition")
    logger.info("Region: Nashik, Maharashtra")
    logger.info(f"Coordinates: {LATITUDE}, {LONGITUDE}")
    logger.info(f"Period: {START_DATE} - {END_DATE}")

    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON"
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    if "properties" not in data:
        raise ValueError("Unexpected NASA POWER response structure")

    return data


# ============================================================
# RAW DATA STORAGE
# ============================================================

def save_raw_data(data):

    filename = f"nashik_power_{START_DATE}_{END_DATE}.json"
    filepath = RAW_DIR / filename

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    logger.info(f"Raw JSON saved: {filepath}")

    return filepath


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(data):

    logger.info("Beginning data cleaning")

    parameters = data["properties"]["parameter"]

    frames = []

    for parameter_name, values in parameters.items():

        series = pd.Series(values, name=parameter_name)
        frames.append(series)

    df = pd.concat(frames, axis=1)

    df.index.name = "date"
    df = df.reset_index()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%d",
        errors="coerce"
    )

    # NASA POWER uses -999 for unavailable measurements.
    df.replace(-999, float("nan"), inplace=True)

    # Remove invalid dates.
    df.dropna(subset=["date"], inplace=True)

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove duplicate dates.
    df.drop_duplicates(subset=["date"], inplace=True)

    # Ensure numeric columns are numeric.
    for column in PARAMETERS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    logger.info(f"Records after cleaning: {len(df)}")

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_data(df):

    logger.info("Running data validation")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected")

    logger.info("Missing values:")
    logger.info("\n" + str(df.isna().sum()))

    logger.info("Dataset statistics:")
    logger.info("\n" + str(df.describe()))

    return True


# ============================================================
# PROCESSED DATA STORAGE
# ============================================================

def save_processed_data(df):

    filename = "nashik_nasa_power_daily.csv"
    filepath = PROCESSED_DIR / filename

    df.to_csv(filepath, index=False)

    logger.info(f"Processed CSV saved: {filepath}")

    return filepath


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    logger.info("=" * 55)
    logger.info("AERIS | NASA POWER DATA PIPELINE")
    logger.info("=" * 55)

    try:

        raw_data = download_data()

        save_raw_data(raw_data)

        cleaned_data = clean_data(raw_data)

        validate_data(cleaned_data)

        save_processed_data(cleaned_data)

        logger.info("=" * 55)
        logger.info("NASA POWER PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total records: {len(cleaned_data)}")
        logger.info("=" * 55)

        print("\nFIRST 10 RECORDS:")
        print(cleaned_data.head(10).to_string(index=False))

    except Exception:

        logger.exception("NASA POWER PIPELINE FAILED")
        raise


if __name__ == "__main__":
    main()
