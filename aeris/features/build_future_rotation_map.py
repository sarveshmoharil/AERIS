
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FUTURE_DIR = ROOT / "data" / "processed" / "future_rotations"

INPUT_FILE = FUTURE_DIR / "future_rotation_recommendations.csv"

OUTPUT_GEOJSON = FUTURE_DIR / "future_rotation_map.geojson"
OUTPUT_CSV = FUTURE_DIR / "future_rotation_map.csv"
OUTPUT_REPORT = FUTURE_DIR / "future_rotation_map_report.json"


def main():
    print("=" * 65)
    print("AERIS | FUTURE ROTATION SPATIAL MAP BUILDER")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "location_id",
        "latitude",
        "longitude",
        "rotation",
        "future_rotation_score",
        "baseline_comparable_score",
        "score_change",
        "rank"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only the highest-ranked future recommendation per location.
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = (
        df.sort_values(["location_id", "rank"])
        .drop_duplicates("location_id")
        .copy()
    )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])

    df["recommendation_changed"] = (
        df["rotation"].astype(str).str.strip()
        != df.get("baseline_rotation", df["rotation"]).astype(str).str.strip()
    ) if "baseline_rotation" in df.columns else False

    features = []

    for _, row in df.iterrows():
        properties = {
            "location_id": str(row["location_id"]),
            "target_year": int(row["target_year"]),
            "rotation": str(row["rotation"]),
            "crop_ids": str(row["crop_ids"]),
            "crop_names": str(row["crop_names"]),
            "future_score": float(row["future_rotation_score"]),
            "baseline_score": float(row["baseline_comparable_score"]),
            "score_change": float(row["score_change"]),
            "rank": int(row["rank"]),
            "recommendation_changed": bool(
                row["recommendation_changed"]
            ),
            "explanation": str(row["explanation"])
        }

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
        "name": "AERIS_2027_Future_Rotation_Map",
        "features": features
    }

    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    df.to_csv(OUTPUT_CSV, index=False)

    report = {
        "engine": "AERIS Future Rotation Spatial Map Builder",
        "target_year": 2027,
        "locations_processed": len(df),
        "geojson_features": len(features),
        "mean_future_score": round(
            float(df["future_rotation_score"].mean()), 3
        ),
        "mean_baseline_score": round(
            float(df["baseline_comparable_score"].mean()), 3
        ),
        "mean_score_change": round(
            float(df["score_change"].mean()), 3
        ),
        "positive_score_changes": int(
            (df["score_change"] > 0).sum()
        ),
        "negative_score_changes": int(
            (df["score_change"] < 0).sum()
        ),
        "geographic_note": (
            "Sampled Nashik-area coordinates, not an official "
            "district boundary."
        )
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== MAP BUILD REPORT ==========")
    print(f"Locations processed: {len(df)}")
    print(f"GeoJSON points created: {len(features)}")
    print(f"Mean future score: {report['mean_future_score']}")
    print(f"Mean baseline score: {report['mean_baseline_score']}")
    print(f"Mean score change: {report['mean_score_change']}")

    print("\nSaved:")
    print(OUTPUT_GEOJSON)
    print(OUTPUT_CSV)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
