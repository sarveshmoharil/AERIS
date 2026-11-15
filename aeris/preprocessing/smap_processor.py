
"""
AERIS | SMAP Raster Preprocessor

Converts Nashik SMAP GeoTIFFs into:
1. Pixel-level time series
2. Spatial summary time series
3. Processing quality report
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from pyproj import Transformer
from rasterio.transform import xy


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "aeris/data/raw/nasa_smap/historical"
OUTPUT_DIR = ROOT / "aeris/data/processed/nasa_smap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "smap_preprocessing.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

log = logging.getLogger("AERIS-SMAP-PREPROCESSOR")


# ==================================================
# CONFIGURATION
# ==================================================

EXPECTED_PREFIX = "smap_rootzone_"

PIXEL_OUTPUT = OUTPUT_DIR / "smap_pixel_timeseries.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "smap_spatial_summary.csv"
REPORT_OUTPUT = OUTPUT_DIR / "smap_processing_report.json"


# ==================================================
# FILE METADATA
# ==================================================

def extract_date(filename):

    match = re.search(
        r"(\d{8})_(\d{4})Z",
        filename
    )

    if not match:
        raise ValueError(
            f"Cannot extract date from {filename}"
        )

    return pd.to_datetime(
        match.group(1),
        format="%Y%m%d"
    )


# ==================================================
# PROCESS ONE RASTER
# ==================================================

def process_raster(path, transformer):

    observation_date = extract_date(path.name)

    with rasterio.open(path) as src:

        if src.crs is None:
            raise ValueError("Raster has no CRS")

        if src.count != 1:
            raise ValueError("Expected exactly one band")

        tags = src.tags()

        if "root zone soil moisture" not in (
            tags.get("long_name", "").lower()
        ):
            raise ValueError(
                "Unexpected measurement variable"
            )

        if tags.get("units") != "m3 m-3":
            raise ValueError("Unexpected measurement units")

        raster = src.read(1, masked=True)

        values = raster.filled(np.nan)

        rows, cols = np.where(
            np.isfinite(values)
        )

        if len(rows) == 0:
            raise ValueError("No valid raster pixels")

        x_coords, y_coords = xy(
            src.transform,
            rows,
            cols,
            offset="center"
        )

        lon, lat = transformer.transform(
            x_coords,
            y_coords
        )

        moisture = values[rows, cols]

        pixel_df = pd.DataFrame({
            "date": observation_date.date().isoformat(),
            "pixel_row": rows,
            "pixel_col": cols,
            "latitude": lat,
            "longitude": lon,
            "soil_moisture_m3_m3": moisture
        })

        summary = {
            "date": observation_date.date().isoformat(),
            "valid_pixels": len(moisture),
            "total_pixels": src.width * src.height,
            "valid_fraction": (
                len(moisture) / (src.width * src.height)
            ),
            "mean": float(np.mean(moisture)),
            "median": float(np.median(moisture)),
            "minimum": float(np.min(moisture)),
            "maximum": float(np.max(moisture)),
            "std_dev": float(np.std(moisture)),
            "crs": str(src.crs),
            "resolution_x_m": abs(src.transform.a),
            "resolution_y_m": abs(src.transform.e)
        }

        return pixel_df, summary


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 55)
    print("AERIS | SMAP PREPROCESSING")
    print("=" * 55)

    files = sorted(
        INPUT_DIR.glob(f"{EXPECTED_PREFIX}*.tif")
    )

    if not files:
        raise FileNotFoundError(
            f"No SMAP rasters found in {INPUT_DIR}"
        )

    log.info("Found %s SMAP rasters", len(files))

    pixel_frames = []
    summaries = []
    failures = []

    total = len(files)

    for index, path in enumerate(files, start=1):

        try:

            with rasterio.open(path) as src:
                source_crs = src.crs

            transformer = Transformer.from_crs(
                source_crs,
                "EPSG:4326",
                always_xy=True
            )

            pixels, summary = process_raster(
                path,
                transformer
            )

            pixel_frames.append(pixels)
            summaries.append(summary)

            log.info(
                "[%s/%s] Processed %s | %s valid pixels",
                index,
                total,
                path.name,
                summary["valid_pixels"]
            )

        except Exception as exc:

            failures.append({
                "file": path.name,
                "error": str(exc)
            })

            log.exception("Failed: %s", path.name)

    if not pixel_frames:
        raise RuntimeError("No rasters were successfully processed")

    pixel_df = pd.concat(
        pixel_frames,
        ignore_index=True
    )

    summary_df = pd.DataFrame(summaries)

    pixel_df = pixel_df.sort_values(
        ["date", "pixel_row", "pixel_col"]
    ).reset_index(drop=True)

    summary_df = summary_df.sort_values(
        "date"
    ).reset_index(drop=True)

    pixel_df.to_csv(
        PIXEL_OUTPUT,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    report = {
        "dataset": "NASA SMAP SPL4SMGP",
        "variable": "Root zone soil moisture",
        "units": "m3 m-3",
        "input_rasters": total,
        "processed_rasters": len(summaries),
        "failed_rasters": len(failures),
        "pixel_observations": len(pixel_df),
        "date_start": summary_df["date"].min(),
        "date_end": summary_df["date"].max(),
        "failures": failures,
        "outputs": {
            "pixel_timeseries": str(PIXEL_OUTPUT),
            "spatial_summary": str(SUMMARY_OUTPUT)
        }
    }

    REPORT_OUTPUT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print("PREPROCESSING SUMMARY")
    print("=" * 55)

    print("Input rasters:", total)
    print("Processed:", len(summaries))
    print("Failed:", len(failures))
    print("Pixel observations:", len(pixel_df))

    print("\nDate coverage:")
    print(summary_df["date"].min(), "to", summary_df["date"].max())

    print("\nSpatial summary:")
    print(
        summary_df[
            ["mean", "median", "minimum", "maximum"]
        ].describe().round(4)
    )

    print("\nSaved:")
    print(PIXEL_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
