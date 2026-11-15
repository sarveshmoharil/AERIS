
from pathlib import Path
from datetime import datetime, timezone
import gzip
import json
import requests
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "aeris" / "data" / "raw" / "nasa_srtm"
OUT = ROOT / "aeris" / "data" / "processed" / "nasa_srtm"
META = ROOT / "aeris" / "data" / "metadata"

RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

# Nashik-area bounding box
WEST, SOUTH, EAST, NORTH = 73.45, 19.40, 74.15, 20.56

# Four 1-degree SRTM tiles cover this bounding box
TILES = [
    "N18E073", "N18E074",
    "N19E073", "N19E074",
    "N20E073", "N20E074",
    "N21E073", "N21E074"
]

BASE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"


def download_tile(tile):
    folder = RAW / tile[:3]
    folder.mkdir(parents=True, exist_ok=True)

    hgt_path = folder / f"{tile}.hgt"
    gz_path = folder / f"{tile}.hgt.gz"

    if hgt_path.exists() and hgt_path.stat().st_size > 1_000_000:
        print(f"[SKIP] {tile}")
        return hgt_path

    url = f"{BASE_URL}/{tile[:3]}/{tile}.hgt.gz"
    print(f"[DOWNLOAD] {tile}")

    response = requests.get(url, timeout=180)
    response.raise_for_status()

    gz_path.write_bytes(response.content)

    with gzip.open(gz_path, "rb") as source:
        with open(hgt_path, "wb") as destination:
            destination.write(source.read())

    gz_path.unlink(missing_ok=True)

    print(f"[OK] {tile}: {hgt_path.stat().st_size / 1024 / 1024:.1f} MB")
    return hgt_path


def main():
    print("AERIS | SRTM terrain acquisition")

    tile_paths = [download_tile(tile) for tile in TILES]

    # Mosaic source tiles
    datasets = [rasterio.open(path) for path in tile_paths]

    mosaic, mosaic_transform = merge(datasets)

    profile = datasets[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        transform=mosaic_transform,
        compress="deflate",
        nodata=-32768
    )

    for dataset in datasets:
        dataset.close()

    source_path = RAW / "srtm_mosaic_full.tif"

    with rasterio.open(source_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)

    print("[OK] Source mosaic created")

    # Reproject to UTM Zone 43N for terrain calculations in metres
    dst_crs = "EPSG:32643"

    left, bottom, right, top = transform_bounds(
        "EPSG:4326", dst_crs,
        WEST, SOUTH, EAST, NORTH
    )

    resolution = 30.0
    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))

    dst_transform = from_origin(left, top, resolution, resolution)

    elevation = np.full((height, width), -32768, dtype=np.float32)

    with rasterio.open(source_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=elevation,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=-32768,
            resampling=Resampling.bilinear
        )

    valid = elevation != -32768

    if not np.any(valid):
        raise RuntimeError("No valid elevation pixels found")

    # Fill NoData temporarily for stable terrain derivatives
    filled = elevation.copy()
    filled[~valid] = np.nan

    # Fill edge NoData with nearest valid values
    from scipy.ndimage import distance_transform_edt

    invalid = ~np.isfinite(filled)
    if invalid.any():
        indices = distance_transform_edt(
            invalid, return_distances=False, return_indices=True
        )
        filled[invalid] = filled[tuple(indices[:, invalid])]

    # Terrain derivatives, using metre-based pixel spacing
    dy, dx = np.gradient(filled, resolution, resolution)

    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Aspect: 0 = north, 90 = east, 180 = south, 270 = west
    aspect = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": -9999.0,
        "compress": "deflate"
    }

    outputs = {
        "elevation_m": elevation,
        "slope_degrees": slope,
        "aspect_degrees": aspect
    }

    for name, array in outputs.items():
        array = array.copy()
        array[~valid] = -9999.0

        path = OUT / f"{name}.tif"

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)

        print(f"[SAVED] {path}")

    report = {
        "source": "SRTM-derived elevation tiles",
        "source_resolution": "1 arc-second (~30 m)",
        "processing_crs": dst_crs,
        "bbox": [WEST, SOUTH, EAST, NORTH],
        "tiles": TILES,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "valid_pixels": int(valid.sum()),
        "elevation_min_m": float(np.min(elevation[valid])),
        "elevation_max_m": float(np.max(elevation[valid])),
        "elevation_mean_m": float(np.mean(elevation[valid])),
        "slope_mean_degrees": float(np.mean(slope[valid])),
        "slope_max_degrees": float(np.max(slope[valid])),
        "outputs": list(outputs.keys())
    }

    report_path = META / "srtm_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== SRTM REPORT ==========")
    print(f"Elevation: {report['elevation_min_m']:.1f}–"
          f"{report['elevation_max_m']:.1f} m")
    print(f"Mean elevation: {report['elevation_mean_m']:.1f} m")
    print(f"Mean slope: {report['slope_mean_degrees']:.2f}°")
    print(f"Valid pixels: {report['valid_pixels']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
