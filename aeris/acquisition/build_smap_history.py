
"""
AERIS | Historical SMAP Manifest Builder

Dataset: NASA SMAP L4 Global 3-hourly 9 km
Collection: SPL4SMGP, Version 008

Creates:
    data/metadata/smap_all_granules.csv
    data/metadata/smap_daily_manifest.csv
    data/metadata/smap_daily_manifest.json

No science data is downloaded by this script.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import earthaccess
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "aeris" / "data" / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

SHORT_NAME = "SPL4SMGP"
VERSION = "008"

START_DATE = "2015-04-01"

# Allow seven days for NASA processing latency.
END_DATE = (
    datetime.now(timezone.utc).date() - timedelta(days=7)
).isoformat()

TARGET_HOUR_UTC = 12

LOG_FILE = PROJECT_ROOT / "logs" / "smap_history.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AERIS-SMAP")


# --------------------------------------------------
# METADATA HELPERS
# --------------------------------------------------

def get_umm(granule):
    """Extract UMM metadata from an earthaccess granule."""

    data = dict(granule)

    umm = data.get("umm", {})

    if not isinstance(umm, dict):
        umm = {}

    return data, umm


def get_granule_name(data, umm):
    return (
        umm.get("GranuleUR")
        or data.get("producer_granule_id")
        or data.get("title")
        or "UNKNOWN"
    )


def get_beginning_time(umm):
    try:
        return (
            umm["TemporalExtent"]
            ["RangeDateTime"]
            ["BeginningDateTime"]
        )
    except (KeyError, TypeError):
        return None


def get_processing_version(name):
    match = re.search(r"_Vv(\d+)_", name)

    if match:
        return int(match.group(1))

    return 0


def get_size_mb(data, granule):
    size = data.get("size")

    if size is None:
        try:
            size = granule.size
        except Exception:
            size = 0

    try:
        return round(float(size) / (1024 * 1024), 3)
    except (TypeError, ValueError):
        return 0.0


def get_utc_datetime(value):
    if not value:
        return None

    try:
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        return timestamp

    except Exception:
        return None


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def authenticate():
    logger.info("Authenticating with NASA Earthdata...")

    auth = earthaccess.login(
        strategy="interactive",
        persist=True
    )

    if not auth.authenticated:
        raise RuntimeError("NASA Earthdata authentication failed.")

    logger.info("Earthdata authentication successful.")


# --------------------------------------------------
# HISTORICAL SEARCH
# --------------------------------------------------

def search_historical_granules():

    start = datetime.fromisoformat(START_DATE).date()
    end = datetime.fromisoformat(END_DATE).date()

    all_records = []

    for year in range(start.year, end.year + 1):

        year_start = max(
            start,
            datetime(year, 1, 1).date()
        )

        year_end = min(
            end,
            datetime(year, 12, 31).date()
        )

        temporal_start = f"{year_start.isoformat()}T00:00:00"
        temporal_end = f"{year_end.isoformat()}T23:59:59"

        logger.info(
            "Searching %s: %s to %s",
            year,
            temporal_start,
            temporal_end
        )

        granules = earthaccess.search_data(
            short_name=SHORT_NAME,
            version=VERSION,
            temporal=(
                temporal_start,
                temporal_end
            ),
            count=-1
        )

        logger.info(
            "%s: discovered %s granules",
            year,
            len(granules)
        )

        for granule in granules:

            data, umm = get_umm(granule)

            name = get_granule_name(data, umm)

            beginning = get_utc_datetime(
                get_beginning_time(umm)
            )

            if beginning is None:
                logger.warning(
                    "Skipping granule with missing timestamp: %s",
                    name
                )
                continue

            record = {
                "granule": name,
                "beginning_utc": beginning.isoformat(),
                "date_utc": beginning.date().isoformat(),
                "hour_utc": beginning.hour,
                "processing_version": get_processing_version(name),
                "size_mb": get_size_mb(data, granule)
            }

            all_records.append(record)

    if not all_records:
        raise RuntimeError(
            "No granules discovered. Check authentication and search dates."
        )

    df = pd.DataFrame(all_records)

    df = df.drop_duplicates(
        subset=["granule"],
        keep="first"
    )

    df["beginning_utc"] = pd.to_datetime(
        df["beginning_utc"],
        utc=True
    )

    df = df.sort_values(
        ["beginning_utc", "processing_version"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# DAILY OBSERVATION SELECTION
# --------------------------------------------------

def build_daily_manifest(df):

    df = df.copy()

    df["hour_distance"] = (
        df["hour_utc"] - TARGET_HOUR_UTC
    ).abs()

    # Prefer the observation closest to 12 UTC.
    # If equally close, prefer the earlier observation.
    # If multiple revisions exist, prefer the newest.
    df = df.sort_values(
        [
            "date_utc",
            "hour_distance",
            "hour_utc",
            "processing_version"
        ],
        ascending=[True, True, True, False]
    )

    daily = df.drop_duplicates(
        subset=["date_utc"],
        keep="first"
    ).copy()

    daily = daily.sort_values(
        "beginning_utc"
    ).reset_index(drop=True)

    daily["selected_hour_distance"] = daily[
        "hour_distance"
    ]

    daily["selected_time_utc"] = daily[
        "beginning_utc"
    ].astype(str)

    return daily


# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------

def save_manifests(all_df, daily_df):

    all_csv = METADATA_DIR / "smap_all_granules.csv"

    daily_csv = METADATA_DIR / "smap_daily_manifest.csv"

    daily_json = METADATA_DIR / "smap_daily_manifest.json"

    all_output = all_df.copy()
    all_output["beginning_utc"] = (
        all_output["beginning_utc"].astype(str)
    )

    daily_output = daily_df.copy()
    daily_output["beginning_utc"] = (
        daily_output["beginning_utc"].astype(str)
    )

    all_output.to_csv(
        all_csv,
        index=False
    )

    daily_output.to_csv(
        daily_csv,
        index=False
    )

    metadata = {
        "project": "AERIS",
        "dataset": SHORT_NAME,
        "collection_version": VERSION,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "selection_strategy": (
            "One observation per UTC day, closest to 12:00 UTC; "
            "newest processing revision preferred."
        ),
        "total_granules": int(len(all_output)),
        "daily_observations": int(len(daily_output)),
        "records": daily_output.to_dict(orient="records")
    }

    with open(
        daily_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            default=str
        )

    logger.info("Saved all-granule manifest: %s", all_csv)
    logger.info("Saved daily manifest: %s", daily_csv)
    logger.info("Saved JSON manifest: %s", daily_json)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 55)
    print("AERIS | HISTORICAL SMAP MANIFEST BUILDER")
    print("=" * 55)

    print(f"Dataset: {SHORT_NAME}")
    print(f"Collection version: {VERSION}")
    print(f"Start: {START_DATE}")
    print(f"End: {END_DATE}")
    print(f"Target observation: {TARGET_HOUR_UTC:02d}:00 UTC")

    authenticate()

    logger.info("Starting historical archive search.")

    all_df = search_historical_granules()

    daily_df = build_daily_manifest(all_df)

    save_manifests(
        all_df,
        daily_df
    )

    print("\n" + "=" * 55)
    print("HISTORICAL MANIFEST SUMMARY")
    print("=" * 55)

    print(f"All discovered granules: {len(all_df):,}")
    print(f"Selected daily records: {len(daily_df):,}")

    print(
        "Date coverage:",
        daily_df["date_utc"].min(),
        "to",
        daily_df["date_utc"].max()
    )

    print(
        "Total archive size represented:",
        f"{all_df['size_mb'].sum() / 1024:.2f} GB"
    )

    print("\nSelected observation hours:")
    print(
        daily_df["hour_utc"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nLatest selected records:")
    print(
        daily_df[
            [
                "date_utc",
                "hour_utc",
                "processing_version",
                "granule"
            ]
        ].tail(10).to_string(index=False)
    )

    print("\nMANIFEST BUILD COMPLETED.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")

    except Exception:
        logger.exception("Manifest construction failed.")
        raise
