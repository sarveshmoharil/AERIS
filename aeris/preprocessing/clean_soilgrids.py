
"""
AERIS | SoilGrids Zero-Value Cleaner
Version 1.0

Creates cleaned copies of SoilGrids rasters.
Zero-valued pixels are converted to NoData.
Original rasters remain untouched.
"""

from pathlib import Path
import json

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "aeris" / "data" / "raw" / "soilgrids"
OUTPUT_DIR = ROOT / "aeris" / "data" / "processed" / "soilgrids_clean"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODATA_VALUE = -9999.0


def clean_raster(source):
    destination = OUTPUT_DIR / source.name

    with rasterio.open(source) as src:
        profile = src.profile.copy()
        data = src.read()

        zero_count = int(np.count_nonzero(data == 0))
        valid_count = int(np.count_nonzero(data != 0))

        data = data.astype(np.float32)
        data[data == 0] = NODATA_VALUE

        profile.update(
            dtype="float32",
            nodata=NODATA_VALUE,
            compress="lzw"
        )

        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(data)

    return {
        "file": source.name,
        "zero_pixels_converted": zero_count,
        "remaining_valid_pixels": valid_count,
        "output": str(destination)
    }


def main():
    print("=" * 60)
    print("AERIS | SOILGRIDS CLEANER")
    print("=" * 60)

    files = sorted(INPUT_DIR.glob("*.tif"))

    if not files:
        raise FileNotFoundError(f"No rasters found: {INPUT_DIR}")

    results = []

    for index, source in enumerate(files, start=1):
        result = clean_raster(source)
        results.append(result)

        print(
            f"[{index:02}/{len(files):02}] {source.name} | "
            f"Zero pixels converted: "
            f"{result['zero_pixels_converted']:,}"
        )

    report = {
        "version": "1.0",
        "input_rasters": len(files),
        "cleaned_rasters": len(results),
        "nodata_value": NODATA_VALUE,
        "zero_values_converted": sum(
            item["zero_pixels_converted"] for item in results
        ),
        "results": results
    }

    report_path = OUTPUT_DIR / "cleaning_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== CLEANING COMPLETE ==========")
    print(f"Rasters cleaned: {len(results)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print("Original rasters were preserved.")


if __name__ == "__main__":
    main()
