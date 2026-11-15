
"""
AERIS | Rotation Analytics Engine
Version 1.0

Analyses:
    - Crop recommendation frequency
    - Rotation distribution
    - Spatial score variation
    - Environmental suitability factors
    - Rotation consistency
    - Best and weakest sampled locations

Outputs:
    analytics_summary.json
    crop_frequency.csv
    rotation_frequency.csv
    location_analytics.csv
    factor_analytics.csv
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "aeris" / "data" / "processed"

ROTATION_FILE = DATA / "rotations" / "rotation_recommendations.csv"
TOP_FILE = DATA / "map_data" / "nashik_top_rotations.csv"
SUITABILITY_FILE = DATA / "crop_suitability" / "crop_suitability_all.csv"
ENVIRONMENT_FILE = DATA / "environmental" / "nashik_environmental_features.csv"

OUTPUT_DIR = DATA / "analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "rotation_analytics.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")

    return pd.read_csv(path)


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def analyse_rotations(rotations):
    """Analyse how frequently complete rotations are recommended."""

    frequency = (
        rotations["rotation"]
        .value_counts()
        .rename_axis("rotation")
        .reset_index(name="location_count")
    )

    frequency["percentage"] = (
        frequency["location_count"] / len(rotations) * 100
    ).round(2)

    return frequency


def analyse_crops(top_rotations):
    """Count crops appearing in the top recommendation at each location."""

    crop_counts = {}

    for rotation in top_rotations["rotation"].dropna():
        for crop in str(rotation).split(" -> "):
            crop = crop.strip()
            crop_counts[crop] = crop_counts.get(crop, 0) + 1

    result = pd.DataFrame([
        {
            "crop": crop,
            "location_count": count,
            "percentage": round(
                count / len(top_rotations) * 100, 2
            )
        }
        for crop, count in crop_counts.items()
    ])

    return result.sort_values(
        "location_count",
        ascending=False
    ).reset_index(drop=True)


def analyse_locations(top_rotations):
    """Summarise rotation scores for each sampled location."""

    df = top_rotations.copy()

    for col in [
        "rotation_score",
        "average_suitability",
        "rotation_compatibility"
    ]:
        df[col] = numeric(df[col])

    result = df[
        [
            "location_id",
            "latitude",
            "longitude",
            "rotation",
            "rotation_score",
            "average_suitability",
            "rotation_compatibility"
        ]
    ].copy()

    result["score_band"] = pd.cut(
        result["rotation_score"],
        bins=[-np.inf, 60, 70, 80, 90, np.inf],
        labels=[
            "Very low",
            "Low",
            "Moderate",
            "High",
            "Very high"
        ]
    ).astype(str)

    return result.sort_values(
        "rotation_score",
        ascending=False
    ).reset_index(drop=True)


def analyse_factors(suitability):
    """
    Analyse average factor scores and identify factors
    most frequently receiving low scores.
    """

    if "factor_scores_json" not in suitability.columns:
        raise ValueError(
            "factor_scores_json missing from suitability data."
        )

    records = []

    for _, row in suitability.iterrows():
        try:
            factors = json.loads(row["factor_scores_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        for factor, value in factors.items():
            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(value):
                records.append({
                    "factor": factor,
                    "score": value
                })

    if not records:
        return pd.DataFrame(
            columns=[
                "factor",
                "mean_score",
                "minimum_score",
                "low_score_count",
                "observations"
            ]
        )

    factor_df = pd.DataFrame(records)

    summary = (
        factor_df.groupby("factor")["score"]
        .agg(
            mean_score="mean",
            minimum_score="min",
            observations="count",
            low_score_count=lambda x: (x < 45).sum()
        )
        .reset_index()
    )

    summary["mean_score"] = summary["mean_score"].round(2)
    summary["minimum_score"] = summary["minimum_score"].round(2)

    summary["low_score_percentage"] = (
        summary["low_score_count"]
        / summary["observations"] * 100
    ).round(2)

    return summary.sort_values(
        "low_score_percentage",
        ascending=False
    ).reset_index(drop=True)


def analyse_suitability(suitability):
    """Summarise crop suitability by crop and season."""

    df = suitability.copy()
    df["suitability_score"] = numeric(df["suitability_score"])

    result = (
        df.groupby(["crop_name", "season"], dropna=False)
        .agg(
            average_score=("suitability_score", "mean"),
            maximum_score=("suitability_score", "max"),
            minimum_score=("suitability_score", "min"),
            locations_evaluated=("location_id", "nunique")
        )
        .reset_index()
    )

    for col in ["average_score", "maximum_score", "minimum_score"]:
        result[col] = result[col].round(2)

    return result.sort_values(
        ["season", "average_score"],
        ascending=[True, False]
    ).reset_index(drop=True)


def score_statistics(location_data):
    scores = numeric(location_data["rotation_score"]).dropna()

    if scores.empty:
        return {}

    return {
        "mean": round(float(scores.mean()), 2),
        "median": round(float(scores.median()), 2),
        "minimum": round(float(scores.min()), 2),
        "maximum": round(float(scores.max()), 2),
        "standard_deviation": round(float(scores.std()), 2)
    }


def main():
    print("=" * 60)
    print("AERIS | ROTATION ANALYTICS ENGINE")
    print("=" * 60)

    rotations = load_csv(ROTATION_FILE)
    top_rotations = load_csv(TOP_FILE)
    suitability = load_csv(SUITABILITY_FILE)
    environment = load_csv(ENVIRONMENT_FILE)

    print(f"Rotation records: {len(rotations)}")
    print(f"Mapped locations: {len(top_rotations)}")
    print(f"Suitability records: {len(suitability)}")
    print(f"Environmental locations: {len(environment)}")

    print("\nAnalysing rotation distribution...")
    rotation_frequency = analyse_rotations(top_rotations)

    print("Analysing crop frequency...")
    crop_frequency = analyse_crops(top_rotations)

    print("Analysing spatial score variation...")
    location_data = analyse_locations(top_rotations)

    print("Analysing environmental factors...")
    factor_data = analyse_factors(suitability)

    print("Analysing crop suitability...")
    suitability_data = analyse_suitability(suitability)

    # Persist detailed tables.
    save_csv(rotation_frequency, "rotation_frequency.csv")
    save_csv(crop_frequency, "crop_frequency.csv")
    save_csv(location_data, "location_analytics.csv")
    save_csv(factor_data, "factor_analytics.csv")
    save_csv(suitability_data, "crop_suitability_analytics.csv")

    # Score bands.
    score_distribution = {
        str(k): int(v)
        for k, v in location_data["score_band"].value_counts().items()
    }

    # Top and bottom sampled locations.
    best_locations = location_data.head(5).to_dict("records")
    weakest_locations = (
        location_data.sort_values("rotation_score")
        .head(5)
        .to_dict("records")
    )

    # Environmental data overview.
    climate_columns = [
        col for col in environment.columns
        if col.startswith("T2M_")
        or col.startswith("PRECTOTCORR_")
    ]

    environmental_summary = {}

    for col in climate_columns:
        values = numeric(environment[col]).dropna()

        if not values.empty:
            environmental_summary[col] = round(
                float(values.mean()), 3
            )

    summary = {
        "engine": "AERIS Rotation Analytics",
        "version": "1.0",
        "region": "Nashik-area bounding box",
        "overview": {
            "locations": int(len(top_rotations)),
            "rotation_records": int(len(rotations)),
            "suitability_evaluations": int(len(suitability)),
            "environmental_features": int(len(environment.columns))
        },
        "rotation_score_statistics": score_statistics(location_data),
        "score_distribution": score_distribution,
        "top_rotations": rotation_frequency.head(10).to_dict("records"),
        "crop_frequency": crop_frequency.to_dict("records"),
        "best_locations": best_locations,
        "weakest_locations": weakest_locations,
        "environmental_factor_analysis": factor_data.to_dict("records"),
        "crop_suitability": suitability_data.to_dict("records"),
        "climate_overview": environmental_summary,
        "interpretation_notes": [
            "Scores represent environmental screening, not predicted yields.",
            "The sampled points are from a Nashik-area bounding box, not a district-clipped grid.",
            "Climate attributes are shared across locations because NASA POWER was sampled at one Nashik city point.",
            "Low factor scores indicate model limitations or mismatches, not proven agronomic constraints.",
            "Crop frequency counts appearances in the top rotation, not actual cultivated area."
        ]
    }

    output_path = OUTPUT_DIR / "analytics_summary.json"

    output_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8"
    )

    print("\n========== ANALYTICS REPORT ==========")

    print("\nRotation score statistics:")
    for key, value in summary["rotation_score_statistics"].items():
        print(f"  {key}: {value}")

    print("\nMost frequent rotations:")
    print(rotation_frequency.head(5).to_string(index=False))

    print("\nMost frequently recommended crops:")
    print(crop_frequency.head(8).to_string(index=False))

    print("\nEnvironmental factors with most low scores:")
    if not factor_data.empty:
        print(factor_data.head(5).to_string(index=False))
    else:
        print("No factor score data available.")

    print("\nBest sampled locations:")
    print(
        location_data.head(5)[
            ["location_id", "rotation", "rotation_score"]
        ].to_string(index=False)
    )

    print("\nSaved analytics to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
