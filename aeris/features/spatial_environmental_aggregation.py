
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed" / "environmental"
    / "nashik_environmental_features.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "spatial"

PROFILE_FILE = OUTPUT_DIR / "grid_cell_environmental_profiles.csv"
ASSIGNMENT_FILE = OUTPUT_DIR / "sample_grid_assignments.csv"
GEOJSON_FILE = OUTPUT_DIR / "nashik_environmental_grid.geojson"
REPORT_FILE = OUTPUT_DIR / "spatial_aggregation_report.json"

# Existing study-area bounds derived from the 105 sample locations.
LAT_MIN = 19.462916674982402
LAT_MAX = 20.513201077058827
LON_MIN = 73.52178793713323
LON_MAX = 74.08195394046376

GRID_COLUMNS = 2
GRID_ROWS = 4

# UTM zone 43N, suitable for this study area.
TO_PROJECTED = Transformer.from_crs(
    "EPSG:4326", "EPSG:32643", always_xy=True
)

TO_GEOGRAPHIC = Transformer.from_crs(
    "EPSG:32643", "EPSG:4326", always_xy=True
)


def assign_grid(x, y, xmin, ymin, cell_size):
    col = int(np.floor((x - xmin) / cell_size))
    row = int(np.floor((y - ymin) / cell_size))

    col = min(max(col, 0), GRID_COLUMNS - 1)
    row = min(max(row, 0), GRID_ROWS - 1)

    return row, col, f"G{row + 1}0{col + 1}"


def make_cell_polygon(xmin, ymin, cell_size, row, col):
    x0 = xmin + col * cell_size
    x1 = min(x0 + cell_size, xmin + GRID_COLUMNS * cell_size)

    y0 = ymin + row * cell_size
    y1 = min(y0 + cell_size, ymin + GRID_ROWS * cell_size)

    # Clip outer edges to the study-area bounding rectangle.
    return [
        (max(x0, xmin), max(y0, ymin)),
        (min(x1, X_MAX), max(y0, ymin)),
        (min(x1, X_MAX), min(y1, Y_MAX)),
        (max(x0, xmin), min(y1, Y_MAX)),
        (max(x0, xmin), max(y0, ymin))
    ]


def to_lonlat_polygon(projected_coordinates):
    coordinates = []

    for x, y in projected_coordinates:
        lon, lat = TO_GEOGRAPHIC.transform(x, y)
        coordinates.append([float(lon), float(lat)])

    return coordinates


def main():
    global X_MAX, Y_MAX

    print("=" * 65)
    print("AERIS | SEAE-01 SPATIAL ENVIRONMENTAL AGGREGATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required = {"latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError("Input must contain latitude and longitude.")

    # Project sample coordinates into metres.
    x, y = TO_PROJECTED.transform(
        df["longitude"].to_numpy(),
        df["latitude"].to_numpy()
    )

    df["_x"] = x
    df["_y"] = y

    xmin, ymin = float(np.min(y) * 0 + np.min(x)), float(np.min(y))
    xmax, ymax = float(np.max(x)), float(np.max(y))

    X_MAX = xmax
    Y_MAX = ymax

    width = xmax - xmin
    height = ymax - ymin

    cell_size = max(
        width / GRID_COLUMNS,
        height / GRID_ROWS
    )

    print(f"Input samples: {len(df)}")
    print(f"Grid: {GRID_COLUMNS} x {GRID_ROWS}")
    print(f"Square cell size: {cell_size / 1000:.2f} km")

    # Assign every sample to one cell.
    assignments = []

    for index, row in df.iterrows():
        grid_row, grid_col, cell_id = assign_grid(
            row["_x"], row["_y"], xmin, ymin, cell_size
        )

        assignments.append({
            "source_index": int(index),
            "location_id": (
                str(row["location_id"])
                if "location_id" in df.columns
                else f"SAMPLE_{index + 1:03d}"
            ),
            "grid_cell": cell_id,
            "grid_row": grid_row + 1,
            "grid_column": grid_col + 1,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"])
        })

    assignment_df = pd.DataFrame(assignments)
    assignment_df.to_csv(ASSIGNMENT_FILE, index=False)

    df["grid_cell"] = assignment_df["grid_cell"].values

    # Select numeric environmental variables, excluding coordinates.
    excluded = {
        "latitude", "longitude", "_x", "_y",
        "smap_observations"
    }

    feature_columns = [
        column for column in df.columns
        if column not in excluded
        and column != "grid_cell"
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    # Create eight cell profiles, including empty cells.
    profile_rows = []
    cell_counts = df.groupby("grid_cell").size().to_dict()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLUMNS):
            cell_id = f"G{row + 1}0{col + 1}"
            subset = df[df["grid_cell"] == cell_id]

            profile = {
                "grid_cell": cell_id,
                "grid_row": row + 1,
                "grid_column": col + 1,
                "sample_count": len(subset)
            }

            for feature in feature_columns:
                values = pd.to_numeric(
                    subset[feature], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).dropna()

                valid_count = len(values)

                profile[f"{feature}_mean"] = (
                    float(values.mean()) if valid_count else np.nan
                )
                profile[f"{feature}_median"] = (
                    float(values.median()) if valid_count else np.nan
                )
                profile[f"{feature}_min"] = (
                    float(values.min()) if valid_count else np.nan
                )
                profile[f"{feature}_max"] = (
                    float(values.max()) if valid_count else np.nan
                )
                profile[f"{feature}_std"] = (
                    float(values.std()) if valid_count > 1 else np.nan
                )
                profile[f"{feature}_valid_count"] = valid_count
                profile[f"{feature}_coverage_pct"] = (
                    round(valid_count / len(subset) * 100, 2)
                    if len(subset) else 0.0
                )

            profile_rows.append(profile)

    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(PROFILE_FILE, index=False)

    # Create GeoJSON polygons.
    features = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLUMNS):
            cell_id = f"G{row + 1}0{col + 1}"

            polygon_xy = make_cell_polygon(
                xmin, ymin, cell_size, row, col
            )

            polygon_lonlat = to_lonlat_polygon(polygon_xy)

            subset = df[df["grid_cell"] == cell_id]

            properties = {
                "grid_cell": cell_id,
                "grid_row": row + 1,
                "grid_column": col + 1,
                "sample_count": int(len(subset)),
                "cell_size_km": round(cell_size / 1000, 3)
            }

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_lonlat]
                },
                "properties": properties
            })

    geojson = {
        "type": "FeatureCollection",
        "name": "AERIS_Nashik_Environmental_Grid",
        "features": features
    }

    with open(GEOJSON_FILE, "w", encoding="utf-8") as file:
        json.dump(geojson, file, indent=2)

    report = {
        "engine": "SEAE-01",
        "study_area": "Nashik sampled study area",
        "source_samples": len(df),
        "grid_rows": GRID_ROWS,
        "grid_columns": GRID_COLUMNS,
        "total_cells": GRID_ROWS * GRID_COLUMNS,
        "cell_size_km": round(cell_size / 1000, 3),
        "environmental_variables": len(feature_columns),
        "profile_statistics": [
            "mean", "median", "minimum", "maximum",
            "standard_deviation", "valid_count", "coverage_percentage"
        ],
        "cell_sample_counts": {
            cell: int(cell_counts.get(cell, 0))
            for cell in [
                f"G{r + 1}0{c + 1}"
                for r in range(GRID_ROWS)
                for c in range(GRID_COLUMNS)
            ]
        },
        "outputs": {
            "profiles": str(PROFILE_FILE),
            "assignments": str(ASSIGNMENT_FILE),
            "geojson": str(GEOJSON_FILE)
        },
        "limitations": [
            "Cell profiles aggregate the 105 sampled points, not every raster pixel.",
            "Climate variables originate from one Nashik city point.",
            "SMAP has substantially coarser native resolution than SoilGrids.",
            "The grid is a research management grid, not an official district boundary."
        ]
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n========== CELL DISTRIBUTION ==========")
    print(profiles[["grid_cell", "sample_count"]].to_string(index=False))

    print("\nEnvironmental variables:", len(feature_columns))
    print("Cells created:", len(features))

    print("\nSaved:")
    print(PROFILE_FILE)
    print(ASSIGNMENT_FILE)
    print(GEOJSON_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
