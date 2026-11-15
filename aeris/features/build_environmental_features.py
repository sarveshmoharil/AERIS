
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coords

ROOT = Path(__file__).resolve().parents[2]

SMAP_CSV = ROOT / "aeris/data/processed/nasa_smap/smap_pixel_timeseries.csv"
POWER_CSV = ROOT / "aeris/data/processed/nashik_nasa_power_daily.csv"

# Use cleaned SoilGrids rasters
SOIL_DIR = ROOT / "aeris/data/processed/soilgrids_clean"
TERRAIN_DIR = ROOT / "aeris/data/processed/nasa_srtm"

OUTPUT_DIR = ROOT / "aeris/data/processed/environmental"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "environmental_features.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def locate_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def sample_raster(path, lon, lat):
    with rasterio.open(path) as src:
        xs, ys = transform_coords(
            "EPSG:4326",
            src.crs,
            lon.tolist(),
            lat.tolist()
        )

        values = np.array([
            value[0] for value in src.sample(zip(xs, ys))
        ], dtype=float)

        # Ignore NoData, zero and invalid values
        if src.nodata is not None:
            values[values == src.nodata] = np.nan

        values[~np.isfinite(values)] = np.nan
        values[values == 0] = np.nan
        values[values < 0] = np.nan

    return values


def build_smap_locations():
    df = pd.read_csv(SMAP_CSV)

    lat_col = locate_column(df, ["latitude", "lat"])
    lon_col = locate_column(df, ["longitude", "lon"])

    value_col = locate_column(
        df,
        [
            "soil_moisture_m3_m3",
            "sm_rootzone",
            "soil_moisture",
            "value",
            "moisture"
        ]
    )

    for col in [lat_col, lon_col, value_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[lat_col, lon_col, value_col])

    locations = (
        df.groupby([lat_col, lon_col])[value_col]
        .agg(
            smap_mean="mean",
            smap_median="median",
            smap_min="min",
            smap_max="max",
            smap_std="std",
            smap_observations="count"
        )
        .reset_index()
        .rename(
            columns={
                lat_col: "latitude",
                lon_col: "longitude"
            }
        )
    )

    locations["smap_std"] = locations["smap_std"].fillna(0)

    return locations


def add_soil_features(features):
    properties = [
        "phh2o",
        "soc",
        "clay",
        "sand",
        "nitrogen",
        "cec"
    ]

    depths = ["0-5cm", "5-15cm", "15-30cm"]

    for prop in properties:
        for depth in depths:
            name = f"{prop}_{depth}"
            path = SOIL_DIR / f"{name}.tif"

            if not path.exists():
                logging.warning("Missing soil raster: %s", path)
                features[name] = np.nan
                continue

            features[name] = sample_raster(
                path,
                features["longitude"],
                features["latitude"]
            )

    return features


def add_terrain_features(features):
    rasters = {
        "elevation_m": TERRAIN_DIR / "elevation_m.tif",
        "slope_degrees": TERRAIN_DIR / "slope_degrees.tif",
        "aspect_degrees": TERRAIN_DIR / "aspect_degrees.tif"
    }

    for name, path in rasters.items():
        if not path.exists():
            raise FileNotFoundError(path)

        features[name] = sample_raster(
            path,
            features["longitude"],
            features["latitude"]
        )

    radians = np.radians(features["aspect_degrees"])

    features["aspect_sin"] = np.sin(radians)
    features["aspect_cos"] = np.cos(radians)

    return features


def add_climate_features(features):
    climate = pd.read_csv(POWER_CSV)

    date_col = locate_column(climate, ["date", "DATE"])

    climate[date_col] = pd.to_datetime(
        climate[date_col].astype(str),
        errors="coerce"
    )

    climate = climate.dropna(subset=[date_col]).copy()
    climate["month"] = climate[date_col].dt.month

    seasons = {
        "monsoon": [6, 7, 8, 9],
        "post_monsoon": [10, 11],
        "winter": [12, 1, 2],
        "summer": [3, 4, 5]
    }

    numeric_columns = [
        col for col in climate.columns
        if col not in [date_col, "month"]
        and pd.api.types.is_numeric_dtype(climate[col])
    ]

    for col in numeric_columns:
        climate[col] = pd.to_numeric(
            climate[col],
            errors="coerce"
        )

    climate_summary = {}

    for season, months in seasons.items():
        subset = climate[climate["month"].isin(months)]

        for col in numeric_columns:
            climate_summary[f"{col}_{season}_mean"] = (
                subset[col].mean()
            )

            if col == "PRECTOTCORR":
                climate_summary[
                    f"{col}_{season}_total_mean"
                ] = (
                    subset.groupby(
                        subset[date_col].dt.year
                    )[col].sum().mean()
                )

    for col in numeric_columns:
        climate_summary[f"{col}_annual_mean"] = (
            climate[col].mean()
        )

        if col == "PRECTOTCORR":
            climate_summary["annual_rainfall_mean_mm"] = (
                climate.groupby(
                    climate[date_col].dt.year
                )[col].sum().mean()
            )

    for name, value in climate_summary.items():
        features[name] = value

    return features, climate_summary


def main():
    print("=" * 60)
    print("AERIS | ENVIRONMENTAL FEATURE INTEGRATION")
    print("=" * 60)

    features = build_smap_locations()
    print(f"[OK] Spatial locations: {len(features)}")

    features = add_soil_features(features)
    print("[OK] Cleaned SoilGrids sampled")

    features = add_terrain_features(features)
    print("[OK] Terrain sampled")

    features, climate_summary = add_climate_features(features)
    print("[OK] Climate summaries attached")

    missing = features.isna().sum().to_dict()

    output_csv = (
        OUTPUT_DIR / "nashik_environmental_features.csv"
    )

    features.to_csv(output_csv, index=False)

    report = {
        "region": "Nashik-area bounding box",
        "spatial_reference": "SMAP pixel centres",
        "soil_source": "Cleaned SoilGrids rasters",
        "zero_values_treated_as_missing": True,
        "spatial_locations": len(features),
        "feature_count": len(features.columns),
        "missing_values": {
            key: int(value)
            for key, value in missing.items()
            if value > 0
        },
        "climate_summary": {
            key: float(value) if pd.notna(value) else None
            for key, value in climate_summary.items()
        },
        "output": str(output_csv)
    }

    report_path = (
        OUTPUT_DIR / "environmental_feature_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== FEATURE REPORT ==========")
    print(f"Locations: {len(features)}")
    print(f"Features: {len(features.columns)}")
    print(f"Columns containing missing values: {len(report['missing_values'])}")

    print(f"\nSaved: {output_csv}")
    print(f"Report: {report_path}")

    if report["missing_values"]:
        print("\nMissing-value counts:")

        for key, count in report["missing_values"].items():
            print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
