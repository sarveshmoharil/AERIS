
import json
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import shape, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

boundary_file = DATA / "spatial" / "nashik_district_boundary.geojson"

with open(boundary_file, encoding="utf-8") as f:
    geo = json.load(f)

district = unary_union([
    shape(feature["geometry"])
    for feature in geo["features"]
    if feature.get("geometry")
])

print("\n========== NASHIK DISTRICT ==========")
print("Geometry:", district.geom_type)
print("Area (degrees squared):", district.area)
print("Bounds:", district.bounds)

rasters = [
    DATA / "soilgrids_clean" / "phh2o_0-5cm.tif",
    DATA / "soilgrids_clean" / "cec_0-5cm.tif",
    DATA / "nasa_srtm" / "elevation_m.tif",
    DATA / "nasa_srtm" / "slope_degrees.tif",
    DATA / "nasa_srtm" / "aspect_degrees.tif",
]

print("\n========== RASTER COVERAGE ==========")

for path in rasters:
    print(f"\n{path.name}")

    with rasterio.open(path) as ds:
        print("CRS:", ds.crs)
        print("Dimensions:", ds.width, "x", ds.height)
        print("Native bounds:", ds.bounds)

        try:
            bounds_wgs84 = transform_bounds(
                ds.crs, "EPSG:4326", *ds.bounds,
                densify_pts=41
            )
            raster_footprint = box(*bounds_wgs84)

            intersection = district.intersection(raster_footprint)

            coverage = (
                intersection.area / district.area * 100
                if district.area else 0
            )

            print("WGS84 bounds:", bounds_wgs84)
            print("Approx district coverage:", round(coverage, 2), "%")
        except Exception as e:
            print("CRS transformation failed:", repr(e))

print("\nDiagnostic complete.")
