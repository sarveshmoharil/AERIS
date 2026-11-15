
"""
AERIS | SMAP Nashik Downloader
Corrected validation + parallel acquisition.
"""

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ==================================================
# CONFIG
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "aeris/data/metadata/smap_daily_manifest.csv"
OUTPUT = ROOT / "aeris/data/raw/nasa_smap/historical"
PLAN = ROOT / "aeris/data/metadata/smap_download_plan.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)
PLAN.parent.mkdir(parents=True, exist_ok=True)

COLLECTION = "C3480440870-NSIDC_CPRD"

BASE_URL = (
    f"https://harmony.earthdata.nasa.gov/{COLLECTION}/"
    "ogc-api-coverages/1.0.0/"
    "collections/parameter_vars/coverage/rangeset"
)

VARIABLES = {
    "rootzone": "Geophysical_Data/sm_rootzone"
}

BBOX = {
    "south": 19.5,
    "north": 20.5,
    "west": 73.5,
    "east": 74.1
}

WORKERS = 4
RECENT_DAILY_DAYS = 30

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "smap_download.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

log = logging.getLogger("AERIS-SMAP")


# ==================================================
# SELECT OBSERVATIONS
# ==================================================

def select_observations():

    df = pd.read_csv(MANIFEST)

    df["date_utc"] = pd.to_datetime(
        df["date_utc"],
        errors="coerce"
    )

    df = df.dropna(subset=["date_utc"])

    df = df[
        df["date_utc"] >= pd.Timestamp("2015-04-01")
    ].copy()

    df["month"] = df["date_utc"].dt.to_period("M")

    recent = (
        df.sort_values("date_utc")
        .drop_duplicates("date_utc")
        .tail(RECENT_DAILY_DAYS)
    )

    recent_dates = set(recent["date_utc"])

    df["midmonth_distance"] = (
        df["date_utc"].dt.day - 15
    ).abs()

    monthly = (
        df.sort_values(
            [
                "month",
                "midmonth_distance",
                "hour_utc",
                "processing_version"
            ],
            ascending=[True, True, True, False]
        )
        .drop_duplicates("month")
    )

    monthly = monthly[
        ~monthly["date_utc"].isin(recent_dates)
    ]

    selected = pd.concat(
        [monthly, recent],
        ignore_index=True
    )

    selected = (
        selected.drop_duplicates("date_utc")
        .sort_values("date_utc")
        .reset_index(drop=True)
    )

    return selected


# ==================================================
# VALIDATE RASTER
# ==================================================

def validate_raster(path):

    try:
        with rasterio.open(path) as src:

            if src.count != 1:
                return False, f"Expected 1 band, got {src.count}"

            if src.width < 1 or src.height < 1:
                return False, "Invalid raster dimensions"

            tags = src.tags()

            long_name = tags.get(
                "long_name", ""
            ).lower()

            units = tags.get(
                "units", ""
            ).lower().replace(" ", "")

            # Actual SMAP metadata wording:
            # "Root zone soil moisture (0-100 cm)"
            if "root zone soil moisture" not in long_name:
                return False, f"Unexpected long_name: {long_name}"

            if units not in ("m3m-3", "m^3m^-3"):
                return False, f"Unexpected units: {units}"

            if src.crs is None:
                return False, "Missing CRS"

            data = src.read(1, masked=True)
            values = data.compressed()

            if values.size == 0:
                return False, "No valid pixels"

            if not np.isfinite(values).all():
                return False, "Non-finite values"

            return True, (
                f"Valid {src.width}x{src.height}, "
                f"{values.size} pixels, "
                f"mean={values.mean():.4f} m3/m3"
            )

    except Exception as exc:
        return False, str(exc)


# ==================================================
# DOWNLOAD ONE FILE
# ==================================================

def download_one(row, variable_key, variable_path):

    date = row["date_utc"].strftime("%Y%m%d")
    hour = int(row["hour_utc"])

    filename = (
        f"smap_{variable_key}_{date}_{hour:02d}00Z.tif"
    )

    destination = OUTPUT / filename
    temporary = OUTPUT / (filename + ".part")

    if destination.exists():

        valid, reason = validate_raster(destination)

        if valid:
            return "skipped", filename, reason

        log.warning(
            "Invalid existing file %s: %s",
            filename,
            reason
        )

        destination.unlink()

    granule = row["granule"]

    command = [
        "curl.exe",
        "-L",
        "-n",
        "-b",
        "-j",
        "--fail",
        "--silent",
        "--show-error",
        "--retry", "3",
        "--retry-all-errors",
        "--connect-timeout", "30",
        "--max-time", "180",
        "-G",
        BASE_URL,
        "--data-urlencode",
        f"granuleName={granule}",
        "--data-urlencode",
        f"subset=lat({BBOX['south']}:{BBOX['north']})",
        "--data-urlencode",
        f"subset=lon({BBOX['west']}:{BBOX['east']})",
        "--data-urlencode",
        f"variable={variable_path}",
        "--data-urlencode",
        "format=image/tiff",
        "-o",
        str(temporary)
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=220
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        if not temporary.exists():
            raise RuntimeError("No output file created")

        valid, reason = validate_raster(temporary)

        if not valid:
            raise RuntimeError(
                f"Raster validation failed: {reason}"
            )

        temporary.replace(destination)

        return "downloaded", filename, reason

    except Exception as exc:

        if temporary.exists():
            temporary.unlink()

        return "failed", filename, str(exc)


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 55)
    print("AERIS | SMAP NASHIK ACQUISITION")
    print("=" * 55)

    selected = select_observations()

    selected.to_csv(PLAN, index=False)

    total = len(selected) * len(VARIABLES)

    print(f"\nObservations: {len(selected)}")
    print(f"Files planned: {total}")
    print(f"Parallel workers: {WORKERS}")
    print(f"Output: {OUTPUT}")
    print("\nStarting acquisition...\n")

    summary = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 0
    }

    futures = {}

    try:

        with ThreadPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            for _, row in selected.iterrows():

                for key, variable in VARIABLES.items():

                    future = executor.submit(
                        download_one,
                        row,
                        key,
                        variable
                    )

                    futures[future] = row["date_utc"]

            completed = 0

            for future in as_completed(futures):

                completed += 1

                try:
                    status, filename, detail = future.result()

                except Exception as exc:
                    status = "failed"
                    filename = "unknown"
                    detail = str(exc)

                summary[status] += 1

                if status == "failed":
                    log.error(
                        "[%s/%s] FAILED %s: %s",
                        completed, total, filename, detail
                    )

                else:
                    log.info(
                        "[%s/%s] %s %s | %s",
                        completed, total,
                        status.upper(), filename, detail
                    )

    except KeyboardInterrupt:

        print("\nInterrupted. Completed files are preserved.")
        raise

    report = {
        "dataset": "NASA SMAP SPL4SMGP",
        "collection_version": "008",
        "variable": "sm_rootzone",
        "units": "m3 m-3",
        "bbox": BBOX,
        "observations": len(selected),
        "workers": WORKERS,
        "results": summary
    }

    (OUTPUT / "acquisition_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print("ACQUISITION SUMMARY")
    print("=" * 55)

    for key, value in summary.items():
        print(f"{key.capitalize():15}: {value}")

    print("\nAcquisition completed.")


if __name__ == "__main__":
    main()
