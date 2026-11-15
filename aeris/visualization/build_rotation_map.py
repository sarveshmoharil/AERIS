
"""
AERIS | Rotation Map Data Builder
Version 1.0

Converts rotation recommendations into map-ready GeoJSON
and a compact CSV for the AERIS interface.
"""

from pathlib import Path
import json
import logging

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "aeris" / "data" / "processed" /
    "rotations" / "rotation_recommendations.csv"
)

OUTPUT_DIR = (
    ROOT / "aeris" / "data" / "processed" / "map_data"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "rotation_map_builder.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main():
    print("=" * 60)
    print("AERIS | ROTATION MAP DATA BUILDER")
    print("=" * 60)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "location_id",
        "latitude",
        "longitude",
        "rank",
        "rotation",
        "rotation_score",
        "average_suitability",
        "rotation_compatibility"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["latitude", "longitude", "rank", "rotation_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["latitude", "longitude", "rank", "rotation_score"]
    )

    # Select the highest-ranked recommendation for each location.
    top = (
        df.sort_values(
            ["location_id", "rank", "rotation_score"],
            ascending=[True, True, False]
        )
        .drop_duplicates(subset=["location_id"], keep="first")
        .copy()
    )

    features = []

    for _, row in top.iterrows():
        properties = {
            "location_id": str(row["location_id"]),
            "rank": int(row["rank"]),
            "rotation": str(row["rotation"]),
            "rotation_score": round(float(row["rotation_score"]), 2),
            "average_suitability": round(
                float(row["average_suitability"]), 2
            ),
            "rotation_compatibility": round(
                float(row["rotation_compatibility"]), 2
            )
        }

        # Include individual crop names if available.
        if "crop_names" in row and pd.notna(row["crop_names"]):
            properties["crop_names"] = str(row["crop_names"])

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(row["longitude"]),
                    float(row["latitude"])
                ]
            },
            "properties": properties
        })

    geojson = {
        "type": "FeatureCollection",
        "name": "AERIS_Nashik_Rotation_Recommendations",
        "description": (
            "Top-ranked annual crop rotation per sampled "
            "SMAP grid location in the Nashik-area bounding box."
        ),
        "features": features
    }

    geojson_path = OUTPUT_DIR / "nashik_rotation_map.geojson"
    csv_path = OUTPUT_DIR / "nashik_top_rotations.csv"
    report_path = OUTPUT_DIR / "rotation_map_report.json"

    geojson_path.write_text(
        json.dumps(geojson, indent=2),
        encoding="utf-8"
    )

    top.to_csv(csv_path, index=False)

    report = {
        "builder_version": "1.0",
        "region": "Nashik-area bounding box",
        "input_recommendations": int(len(df)),
        "mapped_locations": int(len(features)),
        "geometry_type": "Point",
        "coordinate_order": "longitude, latitude",
        "outputs": {
            "geojson": str(geojson_path),
            "csv": str(csv_path)
        },
        "score_range": {
            "minimum": round(float(top["rotation_score"].min()), 2),
            "maximum": round(float(top["rotation_score"].max()), 2),
            "mean": round(float(top["rotation_score"].mean()), 2)
        }
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"\nInput recommendations: {len(df)}")
    print(f"Unique locations mapped: {len(features)}")

    print("\nTOP ROTATION DISTRIBUTION")
    print(
        top["rotation"].value_counts().head(10).to_string()
    )

    print("\nScore statistics")
    print(top["rotation_score"].describe().round(2).to_string())

    print("\nSaved:")
    print(geojson_path)
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
