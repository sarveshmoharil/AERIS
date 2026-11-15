"""
AERIS District-wide Spatial Grid & Raster Zonal Statistics
Creates a 4x4 grid clipped to Nashik district and calculates raster statistics
directly from valid raster pixels. Does not invent values outside raster coverage.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from pyproj import CRS, Transformer
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union, transform as shp_transform

ROOT = Path(__file__).resolve().parents[1]  # .../aeris
DATA = ROOT / "data" / "processed"
BOUNDARY_FILE = DATA / "spatial" / "nashik_district_boundary.geojson"
OUT_DIR = DATA / "district_grid"
GRID_ROWS = 4
GRID_COLS = 4
GRID_CRS = CRS.from_epsg(32643)

SOIL_DIR = DATA / "soilgrids_clean"
SRTM_DIR = DATA / "nasa_srtm"

def read_district():
    with open(BOUNDARY_FILE, encoding="utf-8") as f:
        gj = json.load(f)
    geoms = [shape(ft["geometry"]) for ft in gj.get("features", []) if ft.get("geometry")]
    if not geoms:
        raise RuntimeError("No district geometry found in boundary GeoJSON.")
    # GeoJSON is expected in WGS84 per the previous diagnostic.
    return unary_union(geoms)

def transform_geom(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(transformer.transform, geom)

def raster_stats_for_polygon(raster_path, polygon_utm):
    """Mean/median/std/min/max and approximate valid-area coverage for one polygon."""
    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise RuntimeError(f"Raster has no CRS: {raster_path}")
        poly_native = transform_geom(polygon_utm, GRID_CRS, ds.crs)

        # Crop to polygon and mask pixels outside the polygon.
        try:
            data, out_transform = mask(
                ds, [mapping(poly_native)], crop=True, filled=False,
                all_touched=False
            )
        except ValueError:
            return {
                "mean": np.nan, "median": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan, "valid_pixels": 0,
                "coverage_pct": 0.0
            }

        band = data[0]
        arr = np.asarray(band.compressed(), dtype="float64")
        arr = arr[np.isfinite(arr)]
        if ds.nodata is not None:
            arr = arr[arr != ds.nodata]

        # Denominator estimates the number of raster pixels covering the full
        # clipped management cell, including portions beyond raster extent.
        pixel_area = abs(ds.transform.a * ds.transform.e)
        expected_pixels = poly_native.area / pixel_area if pixel_area else 0
        valid_pixels = int(arr.size)
        coverage = min(100.0, 100.0 * valid_pixels / expected_pixels) if expected_pixels > 0 else 0.0

        if valid_pixels == 0:
            return {
                "mean": np.nan, "median": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan, "valid_pixels": 0,
                "coverage_pct": round(coverage, 2)
            }

        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "valid_pixels": valid_pixels,
            "coverage_pct": round(coverage, 2)
        }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    district_wgs84 = read_district()
    district_utm = transform_geom(district_wgs84, "EPSG:4326", GRID_CRS)

    minx, miny, maxx, maxy = district_utm.bounds
    dx = (maxx - minx) / GRID_COLS
    dy = (maxy - miny) / GRID_ROWS

    # SoilGrids 18 rasters + SRTM 3 rasters.
    raster_paths = sorted(SOIL_DIR.glob("*.tif")) + [
        SRTM_DIR / "elevation_m.tif",
        SRTM_DIR / "slope_degrees.tif",
        SRTM_DIR / "aspect_degrees.tif",
    ]
    raster_paths = [p for p in raster_paths if p.exists()]
    if not raster_paths:
        raise RuntimeError("No input rasters found.")

    features = []
    rows = []
    print(f"District area: {district_utm.area / 1e6:.1f} km²")
    print(f"Raster layers found: {len(raster_paths)}")
    print(f"Building {GRID_ROWS} x {GRID_COLS} district grid...")

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell_box = box(
                minx + c * dx, miny + r * dy,
                minx + (c + 1) * dx, miny + (r + 1) * dy
            )
            cell_geom = cell_box.intersection(district_utm)
            if cell_geom.is_empty or cell_geom.area < 1000:
                continue

            cell_id = f"D{r+1:02d}{c+1:02d}"
            row = {
                "grid_cell": cell_id,
                "area_km2": round(cell_geom.area / 1e6, 3),
                "centroid_longitude": transform_geom(
                    cell_geom.centroid, GRID_CRS, "EPSG:4326"
                ).x,
                "centroid_latitude": transform_geom(
                    cell_geom.centroid, GRID_CRS, "EPSG:4326"
                ).y,
            }
            props = dict(row)
            per_layer_coverages = []

            for raster_path in raster_paths:
                layer = raster_path.stem
                try:
                    stats = raster_stats_for_polygon(raster_path, cell_geom)
                except Exception as exc:
                    print(f"WARNING {cell_id} {layer}: {exc}")
                    stats = {
                        "mean": np.nan, "median": np.nan, "std": np.nan,
                        "min": np.nan, "max": np.nan, "valid_pixels": 0,
                        "coverage_pct": 0.0
                    }

                for stat_name, value in stats.items():
                    row[f"{layer}_{stat_name}"] = value
                props[f"{layer}_mean"] = (
                    stats["mean"] if np.isfinite(stats["mean"]) else None
                )
                props[f"{layer}_coverage_pct"] = stats["coverage_pct"]
                per_layer_coverages.append(stats["coverage_pct"])

            # Evidence status is based on both a minimum raster coverage threshold
            # and actual availability of soil + terrain data.
            soil_cov = [
                row.get(f"{p.stem}_coverage_pct", 0)
                for p in raster_paths if p.parent == SOIL_DIR
            ]
            terrain_cov = [
                row.get(f"{p.stem}_coverage_pct", 0)
                for p in raster_paths if p.parent == SRTM_DIR
            ]
            soil_mean_cov = float(np.mean(soil_cov)) if soil_cov else 0.0
            terrain_mean_cov = float(np.mean(terrain_cov)) if terrain_cov else 0.0
            row["soil_mean_coverage_pct"] = round(soil_mean_cov, 2)
            row["terrain_mean_coverage_pct"] = round(terrain_mean_cov, 2)

            if soil_mean_cov >= 80 and terrain_mean_cov >= 80:
                status = "ADEQUATE_RASTER_EVIDENCE"
            elif soil_mean_cov > 0 and terrain_mean_cov > 0:
                status = "PARTIAL_RASTER_EVIDENCE"
            else:
                status = "INSUFFICIENT_RASTER_EVIDENCE"

            row["evidence_status"] = status
            props["soil_mean_coverage_pct"] = row["soil_mean_coverage_pct"]
            props["terrain_mean_coverage_pct"] = row["terrain_mean_coverage_pct"]
            props["evidence_status"] = status

            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": mapping(transform_geom(cell_geom, GRID_CRS, "EPSG:4326"))
            })
            rows.append(row)
            print(
                f"{cell_id}: {row['area_km2']} km² | "
                f"soil {soil_mean_cov:.1f}% | terrain {terrain_mean_cov:.1f}% | {status}"
            )

    csv_path = OUT_DIR / "nashik_district_grid_environmental_profiles.csv"
    geojson_path = OUT_DIR / "nashik_district_grid.geojson"
    report_path = OUT_DIR / "district_grid_report.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    report = {
        "grid": f"{GRID_ROWS}x{GRID_COLS}",
        "grid_cells_created": len(rows),
        "district_area_km2": round(district_utm.area / 1e6, 2),
        "raster_layers_processed": [p.name for p in raster_paths],
        "outputs": {
            "profiles_csv": str(csv_path),
            "grid_geojson": str(geojson_path)
        },
        "evidence_status_counts": pd.Series(
            [r["evidence_status"] for r in rows]
        ).value_counts().to_dict(),
        "limitations": [
            "Grid cells are research management units, not official administrative or soil zones.",
            "Raster coverage is estimated from valid pixel area within each clipped cell.",
            "SoilGrids and SRTM coverage is incomplete across Nashik district.",
            "No crop suitability or rotation recommendation is generated by this script.",
            "Raster source scale factors and units must be checked before interpreting absolute values."
        ]
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== COMPLETE ==========")
    print(json.dumps(report, indent=2))
    print("\nNext: inspect coverage and units before connecting this grid to crop scoring.")

if __name__ == "__main__":
    main()
