
"""
AERIS | Crop Rotation Engine
Version 1.0

Builds Kharif -> Rabi -> Summer annual crop rotations
using crop suitability results and agronomic rotation rules.

Outputs:
    rotation_recommendations.csv
    rotation_engine_report.json
"""

from pathlib import Path
from itertools import product
import sys
import json
import logging

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "aeris" / "agronomy"))

from crop_knowledge_base import CROPS, get_rotation_penalty, validate_database


INPUT = (
    ROOT / "aeris" / "data" / "processed" /
    "crop_suitability" / "crop_suitability_ranked.csv"
)

OUTPUT_DIR = (
    ROOT / "aeris" / "data" / "processed" / "rotations"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "rotation_engine.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

SEASONS = ["kharif", "rabi", "summer"]

TOP_CANDIDATES_PER_SEASON = 8
MAX_RECOMMENDATIONS_PER_LOCATION = 10

# Suitability and rotation compatibility contribution.
SUITABILITY_WEIGHT = 0.75
ROTATION_WEIGHT = 0.25


def crop_is_annual(crop):
    """Exclude perennial orchard crops from annual rotations."""
    if crop.get("crop_group") == "fruit":
        return False

    if "perennial" in crop.get("season", []):
        return False

    return True


def get_candidates(location_df, season):
    """Return the strongest eligible annual crops for a season."""

    candidates = location_df[
        (location_df["season"] == season)
        & (location_df["classification"] != "out_of_season")
        & (location_df["classification"] != "insufficient_evidence")
    ].copy()

    candidates = candidates.dropna(subset=["suitability_score"])

    candidates = candidates[
        candidates["crop_id"].isin(CROPS.keys())
    ]

    candidates = candidates[
        candidates["crop_id"].apply(
            lambda crop_id: crop_is_annual(CROPS[crop_id])
        )
    ]

    candidates = candidates.sort_values(
        "suitability_score",
        ascending=False
    )

    return candidates.head(TOP_CANDIDATES_PER_SEASON).to_dict("records")


def evaluate_sequence(sequence):
    """
    Evaluate a three-crop sequence.

    Sequence order:
        Kharif -> Rabi -> Summer
    """

    crop_ids = [item["crop_id"] for item in sequence]
    crop_names = [item["crop_name"] for item in sequence]

    suitability_scores = [
        float(item["suitability_score"]) for item in sequence
    ]

    average_suitability = sum(suitability_scores) / len(
        suitability_scores
    )

    total_penalty = 0.0
    transition_reasons = []
    valid = True

    for first, second in zip(crop_ids[:-1], crop_ids[1:]):
        result = get_rotation_penalty(first, second)

        penalty = float(result.get("penalty", 0))
        total_penalty += penalty

        transition_reasons.extend(result.get("reasons", []))

        if not result.get("valid", True):
            valid = False

    if not valid:
        return None

    transition_count = len(crop_ids) - 1

    average_penalty = (
        total_penalty / transition_count
        if transition_count else 0
    )

    # Rotation compatibility is reduced by penalties.
    # Negative penalties represent benefits, such as legume rotation.
    compatibility_score = max(
        0.0,
        min(100.0, 100.0 - average_penalty * 2)
    )

    combined_score = (
        average_suitability * SUITABILITY_WEIGHT
        + compatibility_score * ROTATION_WEIGHT
    )

    # Build crop-specific explanations.
    explanations = []

    for item in sequence:
        explanations.append(
            f"{item['crop_name']} has a suitability score of "
            f"{float(item['suitability_score']):.1f} in "
            f"{item['season']}."
        )

    explanations.extend(transition_reasons)

    if not transition_reasons:
        explanations.append(
            "No basic sequence penalty was triggered."
        )

    return {
        "rotation": " -> ".join(crop_names),
        "crop_ids": crop_ids,
        "crop_names": crop_names,
        "seasonal_suitability": suitability_scores,
        "average_suitability": round(average_suitability, 2),
        "total_rotation_penalty": round(total_penalty, 2),
        "rotation_compatibility": round(compatibility_score, 2),
        "rotation_score": round(combined_score, 2),
        "explanation": " | ".join(explanations)
    }


def main():
    print("=" * 60)
    print("AERIS | CROP ROTATION ENGINE")
    print("=" * 60)

    validation = validate_database()

    if not validation["valid"]:
        raise RuntimeError(
            f"Crop database invalid: {validation['errors']}"
        )

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Suitability results not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = [
        "location_id",
        "latitude",
        "longitude",
        "crop_id",
        "crop_name",
        "season",
        "suitability_score",
        "classification"
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing suitability columns: {missing}"
        )

    df["suitability_score"] = pd.to_numeric(
        df["suitability_score"],
        errors="coerce"
    )

    all_recommendations = []

    location_groups = df.groupby("location_id")

    print(f"Locations: {len(location_groups)}")
    print(f"Candidate limit per season: {TOP_CANDIDATES_PER_SEASON}")
    print("Rotation: Kharif -> Rabi -> Summer\n")

    total_valid_sequences = 0
    locations_without_rotation = 0

    for location_id, location_df in location_groups:
        first_row = location_df.iloc[0]

        seasonal_candidates = {
            season: get_candidates(location_df, season)
            for season in SEASONS
        }

        if any(not seasonal_candidates[s] for s in SEASONS):
            locations_without_rotation += 1
            logging.warning(
                "%s has no eligible candidate in at least one season",
                location_id
            )
            continue

        combinations = product(
            seasonal_candidates["kharif"],
            seasonal_candidates["rabi"],
            seasonal_candidates["summer"]
        )

        location_recommendations = []

        for sequence in combinations:
            result = evaluate_sequence(sequence)

            if result is None:
                continue

            total_valid_sequences += 1

            recommendation = {
                "location_id": location_id,
                "latitude": float(first_row["latitude"]),
                "longitude": float(first_row["longitude"]),
                **result
            }

            location_recommendations.append(recommendation)

        location_recommendations.sort(
            key=lambda item: item["rotation_score"],
            reverse=True
        )

        for rank, recommendation in enumerate(
            location_recommendations[
                :MAX_RECOMMENDATIONS_PER_LOCATION
            ],
            start=1
        ):
            recommendation["rank"] = rank
            all_recommendations.append(recommendation)

    output_path = OUTPUT_DIR / "rotation_recommendations.csv"

    output_df = pd.DataFrame(all_recommendations)

    if not output_df.empty:
        output_df.to_csv(output_path, index=False)

    report = {
        "engine_version": "1.0",
        "region": "Nashik-area bounding box",
        "locations_processed": int(len(location_groups)),
        "locations_without_complete_rotation": locations_without_rotation,
        "valid_sequences_evaluated": total_valid_sequences,
        "recommendations_saved": len(all_recommendations),
        "seasons": SEASONS,
        "top_candidates_per_season": TOP_CANDIDATES_PER_SEASON,
        "maximum_recommendations_per_location":
            MAX_RECOMMENDATIONS_PER_LOCATION,
        "weights": {
            "suitability": SUITABILITY_WEIGHT,
            "rotation_compatibility": ROTATION_WEIGHT
        },
        "outputs": {
            "recommendations": str(output_path)
        },
        "score_definition": (
            "75% average environmental suitability + "
            "25% rotation compatibility"
        ),
        "limitations": [
            "Scores are screening recommendations, not yield predictions.",
            "Water availability and irrigation infrastructure are not yet modelled.",
            "Crop economics, market demand and input costs are not included.",
            "Rotation rules are based on the current heuristic knowledge base."
        ]
    }

    report_path = OUTPUT_DIR / "rotation_engine_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== ROTATION REPORT ==========")
    print(f"Locations processed: {len(location_groups)}")
    print(f"Valid sequences evaluated: {total_valid_sequences}")
    print(f"Locations without complete rotation: {locations_without_rotation}")
    print(f"Recommendations saved: {len(all_recommendations)}")

    if not output_df.empty:
        print("\nTOP 5 ROTATIONS")

        top = output_df.sort_values(
            "rotation_score",
            ascending=False
        ).head(5)

        for _, item in top.iterrows():
            print(
                f"\n{item['rotation']}"
                f"\n  Score: {item['rotation_score']:.1f}/100"
                f"\n  Suitability: {item['average_suitability']:.1f}"
                f"\n  Compatibility: {item['rotation_compatibility']:.1f}"
                f"\n  Location: {item['location_id']}"
            )

    print("\nSaved:")
    print(output_path)
    print(report_path)


if __name__ == "__main__":
    main()

