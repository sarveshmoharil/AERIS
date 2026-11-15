
"""
AERIS | SoilGrids Raster Validation
Version 1.0

Read-only inspection of all SoilGrids GeoTIFF files.
Checks NoData metadata, zero values, valid pixels,
statistics, data types and spatial alignment.
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "aeris" / "data" / "raw" / "soilgrids"

OUTPUT_DIR = ROOT / "aeris" / "data" / "processed" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "soilgrids_validation.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def inspect_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        values = np.asarray(data.compressed(), dtype=np.float64)

        total_pixels = src.width * src.height
        masked_pixels = total_pixels - values.size

        finite = values[np.isfinite(values)]

        zero_count = int(np.sum(finite == 0))
        negative_count = int(np.sum(finite < 0))
        positive_count = int(np.sum(finite > 0))

        if finite.size:
            stats = {
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p10": float(np.percentile(finite, 10)),
                "p90": float(np.percentile(finite, 90))
            }
        else:
            stats = {
                "minimum": None,
                "maximum": None,
                "mean": None,
                "median": None,
                "p10": None,
                "p90": None
            }

        return {
            "file": path.name,
            "property": path.stem,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "nodata_metadata": (
                float(src.nodata)
                if src.nodata is not None else None
            ),
            "total_pixels": int(total_pixels),
            "masked_pixels": int(masked_pixels),
            "valid_pixels": int(values.size),
            "zero_pixels": zero_count,
            "negative_pixels": negative_count,
            "positive_pixels": positive_count,
            "zero_percentage_of_valid": round(
                zero_count / values.size * 100, 3
            ) if values.size else None,
            "transform": list(src.transform)[:6],
            **stats
        }


def main():
    print("=" * 65)
    print("AERIS | SOILGRIDS RASTER VALIDATION")
    print("=" * 65)

    files = sorted(INPUT_DIR.glob("*.tif"))

    if not files:
        raise FileNotFoundError(
            f"No GeoTIFF files found in {INPUT_DIR}"
        )

    print(f"Found {len(files)} raster files.\n")

    results = []

    for index, path in enumerate(files, start=1):
        try:
            result = inspect_raster(path)
            results.append(result)

            print(
                f"[{index:02}/{len(files):02}] {path.name}"
            )
            print(
                f"  Valid: {result['valid_pixels']:,} / "
                f"{result['total_pixels']:,}"
            )
            print(
                f"  Masked: {result['masked_pixels']:,} | "
                f"Zeros: {result['zero_pixels']:,} | "
                f"Negative: {result['negative_pixels']:,}"
            )
            print(
                f"  Range: {result['minimum']} to "
                f"{result['maximum']} | "
                f"Median: {result['median']}"
            )
            print(
                f"  NoData metadata: {result['nodata_metadata']}"
            )
            print()

        except Exception as exc:
            logging.exception("Failed to inspect %s", path)
            print(f"[ERROR] {path.name}: {exc}")

    if not results:
        raise RuntimeError("No rasters could be inspected.")

    df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "soilgrids_raster_validation.csv"
    json_path = OUTPUT_DIR / "soilgrids_raster_validation.json"

    df.to_csv(csv_path, index=False)

    report = {
        "version": "1.0",
        "read_only": True,
        "input_directory": str(INPUT_DIR),
        "raster_count": len(results),
        "successful_inspections": len(results),
        "failed_inspections": len(files) - len(results),
        "rasters": results
    }

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Rasters inspected: {len(results)}")
    print(
        f"Rasters with zero values: "
        f"{int((df['zero_pixels'] > 0).sum())}"
    )
    print(
        f"Rasters with masked pixels: "
        f"{int((df['masked_pixels'] > 0).sum())}"
    )
    print(
        f"Rasters without NoData metadata: "
        f"{int(df['nodata_metadata'].isna().sum())}"
    )

    print("\nSaved:")
    print(csv_path)
    print(json_path)

    print("\nNo source rasters were modified.")


if __name__ == "__main__":
    main()
