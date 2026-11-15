
"""
AERIS | Future Crop Rotation Engine
Version 1.0

Uses RIE-01 seasonal rainfall outlook to adjust existing crop
suitability scores and rank future annual crop rotations.

This is a heuristic scenario model, not a yield prediction.
"""

from pathlib import Path
from itertools import product
import sys
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "aeris" / "agronomy"))

from crop_knowledge_base import CROPS, get_rotation_penalty, validate_database

SUITABILITY_FILE = (
    ROOT / "aeris" / "data" / "processed" /
    "crop_suitability" / "crop_suitability_ranked.csv"
)

FORECAST_FILE = (
    ROOT / "aeris" / "data" / "processed" /
    "rainfall_prediction" / "seasonal_rainfall_forecast.csv"
)

DRY_SPELL_FILE = (
    ROOT / "aeris" / "data" / "processed" /
    "rainfall_prediction" / "rainfall_prediction_report.json"
)

BASELINE_ROTATION_FILE = (
    ROOT / "aeris" / "data" / "processed" /
    "rotations" / "rotation_recommendations.csv"
)

OUTPUT_DIR = (
    ROOT / "aeris" / "data" / "processed" /
    "future_rotations"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["kharif", "rabi", "summer"]

TOP_CANDIDATES_PER_SEASON = 8
MAX_RECOMMENDATIONS_PER_LOCATION = 10

SUITABILITY_WEIGHT = 0.75
ROTATION_WEIGHT = 0.25

# Maximum absolute rainfall-driven score adjustment.
MAX_RAINFALL_ADJUSTMENT = 6.0

# Water-demand index: low demand to very high demand.
WATER_DEMAND = {
    "low": 1,
    "low_medium": 2,
    "medium": 3,
    "medium_high": 4,
    "high": 5,
    "very_high": 6,
}


def load_inputs():
    required_files = [
        SUITABILITY_FILE,
        FORECAST_FILE,
        DRY_SPELL_FILE,
        BASELINE_ROTATION_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    suitability = pd.read_csv(SUITABILITY_FILE)
    forecasts = pd.read_csv(FORECAST_FILE)

    with open(DRY_SPELL_FILE, "r", encoding="utf-8") as file:
        rainfall_report = json.load(file)

    baseline = pd.read_csv(BASELINE_ROTATION_FILE)

    required_columns = [
        "location_id", "latitude", "longitude",
        "crop_id", "crop_name", "season",
        "suitability_score", "classification",
    ]

    missing = [
        col for col in required_columns
        if col not in suitability.columns
    ]

    if missing:
        raise ValueError(f"Missing suitability columns: {missing}")

    return suitability, forecasts, rainfall_report, baseline


def build_seasonal_rainfall_context(forecasts):
    """
    Map RIE-01 seasonal forecasts to crop seasons.

    Rabi uses the combined post-monsoon and winter outlook.
    """

    forecasts = forecasts.set_index("season")

    required = ["kharif", "post_monsoon", "winter", "summer"]

    missing = [s for s in required if s not in forecasts.index]

    if missing:
        raise ValueError(f"Missing forecast seasons: {missing}")

    context = {}

    for season in ["kharif", "summer"]:
        row = forecasts.loc[season]

        historical = float(row["historical_mean_mm"])
        predicted = float(row["forecast_mm"])

        context[season] = {
            "predicted_mm": predicted,
            "historical_mean_mm": historical,
            "rainfall_ratio": (
                predicted / historical if historical > 0 else 1.0
            ),
        }

    # Rabi covers post-monsoon and winter rainfall.
    post = forecasts.loc["post_monsoon"]
    winter = forecasts.loc["winter"]

    predicted = (
        float(post["forecast_mm"])
        + float(winter["forecast_mm"])
    )

    historical = (
        float(post["historical_mean_mm"])
        + float(winter["historical_mean_mm"])
    )

    context["rabi"] = {
        "predicted_mm": predicted,
        "historical_mean_mm": historical,
        "rainfall_ratio": (
            predicted / historical if historical > 0 else 1.0
        ),
    }

    return context


def crop_is_annual(crop):
    return (
        crop.get("crop_group") != "fruit"
        and "perennial" not in crop.get("season", [])
    )


def calculate_rainfall_adjustment(crop_id, season, rainfall_context):
    """
    Estimate a modest score adjustment from seasonal rainfall anomaly.

    Higher water-demand crops benefit relatively more from wetter
    than normal conditions and are penalised more under drier conditions.

    This is a heuristic screening adjustment, not crop water modelling.
    """

    crop = CROPS[crop_id]

    demand = WATER_DEMAND.get(
        crop.get("water_demand", "medium"), 3
    )

    rainfall_ratio = rainfall_context[season]["rainfall_ratio"]

    anomaly = rainfall_ratio - 1.0

    # Centre the adjustment around medium water demand.
    adjustment = anomaly * (demand - 3) * 6.0

    adjustment = float(np.clip(
        adjustment,
        -MAX_RAINFALL_ADJUSTMENT,
        MAX_RAINFALL_ADJUSTMENT,
    ))

    explanation = (
        f"Rainfall outlook for {season}: "
        f"{rainfall_context[season]['predicted_mm']:.1f} mm "
        f"versus historical seasonal mean "
        f"{rainfall_context[season]['historical_mean_mm']:.1f} mm. "
        f"Water-demand adjustment: {adjustment:+.1f} points."
    )

    return adjustment, explanation


def apply_rainfall_adjustments(df, rainfall_context, dry_spell_probability):
    result = df.copy()

    result["suitability_score"] = pd.to_numeric(
        result["suitability_score"],
        errors="coerce",
    )

    result["baseline_suitability"] = result["suitability_score"]
    result["rainfall_adjustment"] = 0.0
    result["future_suitability_score"] = np.nan
    result["rainfall_explanation"] = ""

    for index, row in result.iterrows():
        crop_id = row["crop_id"]
        season = row["season"]

        if crop_id not in CROPS or season not in rainfall_context:
            continue

        base = row["baseline_suitability"]

        if pd.isna(base):
            continue

        adjustment, explanation = calculate_rainfall_adjustment(
            crop_id, season, rainfall_context
        )

        # Historical dry-spell risk adds a small Kharif caution
        # for crops with higher water demand.
        if season == "kharif":
            demand = WATER_DEMAND.get(
                CROPS[crop_id].get("water_demand", "medium"), 3
            )

            dry_spell_adjustment = (
                -2.0 * dry_spell_probability * (demand / 6.0)
            )

            adjustment += dry_spell_adjustment

            explanation += (
                f" Historical 7-day dry-spell risk adjustment: "
                f"{dry_spell_adjustment:+.1f} points."
            )

        adjustment = float(np.clip(
            adjustment,
            -MAX_RAINFALL_ADJUSTMENT,
            MAX_RAINFALL_ADJUSTMENT,
        ))

        future_score = float(np.clip(base + adjustment, 0, 100))

        result.at[index, "rainfall_adjustment"] = round(adjustment, 2)
        result.at[index, "future_suitability_score"] = round(future_score, 2)
        result.at[index, "rainfall_explanation"] = explanation

    return result


def get_candidates(location_df, season):
    candidates = location_df[
        (location_df["season"] == season)
        & (~location_df["classification"].isin([
            "out_of_season",
            "insufficient_evidence",
        ]))
    ].copy()

    candidates = candidates.dropna(
        subset=["future_suitability_score"]
    )

    candidates = candidates[
        candidates["crop_id"].isin(CROPS.keys())
    ]

    candidates = candidates[
        candidates["crop_id"].apply(
            lambda crop_id: crop_is_annual(CROPS[crop_id])
        )
    ]

    candidates = candidates.sort_values(
        "future_suitability_score",
        ascending=False,
    )

    return candidates.head(
        TOP_CANDIDATES_PER_SEASON
    ).to_dict("records")


def evaluate_sequence(sequence):
    crop_ids = [item["crop_id"] for item in sequence]
    crop_names = [item["crop_name"] for item in sequence]

    scores = [
        float(item["future_suitability_score"])
        for item in sequence
    ]

    baseline_scores = [
        float(item["baseline_suitability"])
        for item in sequence
    ]

    average_suitability = float(np.mean(scores))
    baseline_average = float(np.mean(baseline_scores))

    total_penalty = 0.0
    transition_reasons = []

    for first, second in zip(crop_ids[:-1], crop_ids[1:]):
        result = get_rotation_penalty(first, second)

        if not result.get("valid", True):
            return None

        total_penalty += float(result.get("penalty", 0) or 0)
        transition_reasons.extend(result.get("reasons", []))

    transition_count = len(crop_ids) - 1

    average_penalty = (
        total_penalty / transition_count
        if transition_count else 0
    )

    compatibility = float(np.clip(
        100.0 - average_penalty * 2,
        0,
        100,
    ))

    future_score = (
        average_suitability * SUITABILITY_WEIGHT
        + compatibility * ROTATION_WEIGHT
    )

    baseline_score = (
        baseline_average * SUITABILITY_WEIGHT
        + compatibility * ROTATION_WEIGHT
    )

    explanations = []

    for item in sequence:
        explanations.append(
            f"{item['crop_name']} ({item['season']}): "
            f"baseline {float(item['baseline_suitability']):.1f}, "
            f"future {float(item['future_suitability_score']):.1f}. "
            f"{item['rainfall_explanation']}"
        )

    explanations.extend(transition_reasons)

    return {
        "rotation": " -> ".join(crop_names),
        "crop_ids": crop_ids,
        "crop_names": crop_names,
        "seasonal_baseline_suitability": baseline_scores,
        "seasonal_future_suitability": scores,
        "average_baseline_suitability": round(baseline_average, 2),
        "average_future_suitability": round(average_suitability, 2),
        "total_rotation_penalty": round(total_penalty, 2),
        "rotation_compatibility": round(compatibility, 2),
        "baseline_comparable_score": round(baseline_score, 2),
        "future_rotation_score": round(future_score, 2),
        "score_change": round(future_score - baseline_score, 2),
        "explanation": " | ".join(explanations),
    }


def main():
    print("=" * 65)
    print("AERIS | FUTURE CROP ROTATION ENGINE")
    print("=" * 65)

    validation = validate_database()

    if not validation["valid"]:
        raise RuntimeError(validation["errors"])

    suitability, forecasts, rainfall_report, baseline = load_inputs()

    target_year = int(forecasts["target_year"].iloc[0])

    rainfall_context = build_seasonal_rainfall_context(forecasts)

    dry_spell_probability = float(
        rainfall_report["dry_spell_analysis"][
            "probability_7_day_dry_spell"
        ]
    )

    print(f"Target year: {target_year}")
    print(f"Locations in suitability data: {suitability['location_id'].nunique()}")
    print(f"Historical 7-day dry-spell frequency: {dry_spell_probability:.1%}")

    adjusted = apply_rainfall_adjustments(
        suitability,
        rainfall_context,
        dry_spell_probability,
    )

    adjusted.to_csv(
        OUTPUT_DIR / "future_crop_suitability.csv",
        index=False,
    )

    recommendations = []

    location_groups = adjusted.groupby("location_id")

    total_sequences = 0
    incomplete_locations = 0

    for location_id, location_df in location_groups:
        first = location_df.iloc[0]

        candidates = {
            season: get_candidates(location_df, season)
            for season in SEASONS
        }

        if any(not candidates[season] for season in SEASONS):
            incomplete_locations += 1
            continue

        combinations = product(
            candidates["kharif"],
            candidates["rabi"],
            candidates["summer"],
        )

        location_results = []

        for sequence in combinations:
            result = evaluate_sequence(sequence)

            if result is None:
                continue

            total_sequences += 1

            location_results.append({
                "target_year": target_year,
                "location_id": location_id,
                "latitude": float(first["latitude"]),
                "longitude": float(first["longitude"]),
                **result,
            })

        location_results.sort(
            key=lambda item: item["future_rotation_score"],
            reverse=True,
        )

        for rank, item in enumerate(
            location_results[:MAX_RECOMMENDATIONS_PER_LOCATION],
            start=1,
        ):
            item["rank"] = rank
            recommendations.append(item)

    output_df = pd.DataFrame(recommendations)

    if output_df.empty:
        raise RuntimeError("No complete future rotations were generated.")

    output_path = OUTPUT_DIR / "future_rotation_recommendations.csv"
    output_df.to_csv(output_path, index=False)

    # Compare best future recommendation with best existing baseline.
    baseline_best = (
        baseline.sort_values("rotation_score", ascending=False)
        .groupby("location_id")
        .first()
        .reset_index()
    )

    future_best = (
        output_df[output_df["rank"] == 1]
        .copy()
    )

    comparison = baseline_best[
        ["location_id", "rotation", "rotation_score"]
    ].rename(columns={
        "rotation": "baseline_rotation",
        "rotation_score": "baseline_rotation_score",
    })

    comparison = comparison.merge(
        future_best[[
            "location_id",
            "rotation",
            "future_rotation_score",
            "score_change",
        ]].rename(columns={
            "rotation": "future_rotation",
        }),
        on="location_id",
        how="outer",
    )

    comparison["rotation_changed"] = (
        comparison["baseline_rotation"]
        != comparison["future_rotation"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "future_rotation_comparison.csv",
        index=False,
    )

    report = {
        "engine": "AERIS Future Crop Rotation Engine",
        "version": "1.0",
        "target_year": target_year,
        "locations_processed": int(len(location_groups)),
        "locations_without_complete_rotation": incomplete_locations,
        "valid_sequences_evaluated": total_sequences,
        "recommendations_saved": int(len(output_df)),
        "rainfall_context": rainfall_context,
        "historical_7_day_dry_spell_frequency": dry_spell_probability,
        "scoring": {
            "suitability_weight": SUITABILITY_WEIGHT,
            "rotation_compatibility_weight": ROTATION_WEIGHT,
            "maximum_rainfall_adjustment_points": MAX_RAINFALL_ADJUSTMENT,
        },
        "comparison": {
            "locations_compared": int(len(comparison)),
            "recommendation_changed_count": int(
                comparison["rotation_changed"].fillna(False).sum()
            ),
        },
        "limitations": [
            "Rainfall adjustment is a heuristic based on crop water-demand categories.",
            "Seasonal rainfall totals do not describe rainfall distribution or irrigation availability.",
            "The same Nashik climate forecast is shared across all locations.",
            "This is not a crop yield or profit prediction.",
            "Rotation compatibility uses the existing heuristic knowledge base.",
        ],
        "outputs": {
            "future_suitability": "future_crop_suitability.csv",
            "future_rotations": "future_rotation_recommendations.csv",
            "comparison": "future_rotation_comparison.csv",
        },
    }

    report_path = OUTPUT_DIR / "future_rotation_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n========== FUTURE ROTATION REPORT ==========")
    print(f"Locations processed: {len(location_groups)}")
    print(f"Valid sequences evaluated: {total_sequences}")
    print(f"Locations without complete rotation: {incomplete_locations}")
    print(f"Recommendations saved: {len(output_df)}")
    print(
        "Locations with changed top recommendation: "
        f"{report['comparison']['recommendation_changed_count']}"
    )

    print("\nTOP 5 FUTURE ROTATIONS")

    for _, item in output_df.head(5).iterrows():
        print(
            f"\n{item['rotation']}"
            f"\n  Score: {item['future_rotation_score']:.2f}/100"
            f"\n  Baseline comparable score: {item['baseline_comparable_score']:.2f}"
            f"\n  Change: {item['score_change']:+.2f}"
            f"\n  Location: {item['location_id']}"
        )

    print("\nSaved:")
    print(output_path)
    print(OUTPUT_DIR / "future_rotation_comparison.csv")
    print(report_path)


if __name__ == "__main__":
    main()
