
"""
AERIS | Crop Suitability Engine
Version 1.0

Evaluates crop suitability at each SMAP spatial location.

Inputs:
    environmental/nashik_environmental_features.csv
    agronomy/crop_knowledge_base.py

Outputs:
    crop_suitability_all.csv
    crop_suitability_ranked.csv
    crop_suitability_report.json

Scores are deterministic screening scores, not yield predictions.
"""

from pathlib import Path
import sys
import json
import logging

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "aeris" / "agronomy"))

from crop_knowledge_base import CROPS, validate_database

INPUT = (
    ROOT / "aeris" / "data" / "processed" /
    "environmental" / "nashik_environmental_features.csv"
)

OUTPUT_DIR = (
    ROOT / "aeris" / "data" / "processed" / "crop_suitability"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "crop_suitability.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Factor weights. Scores are normalized over available evidence.
WEIGHTS = {
    "soil_ph": 20,
    "soil_texture": 15,
    "temperature": 15,
    "soil_moisture": 15,
    "rainfall": 20,
    "slope": 10,
    "season": 5
}

SEASON_CLIMATE = {
    "kharif": "monsoon",
    "rabi": "winter",
    "summer": "summer",
    "perennial": "annual"
}

WATER_DEMAND_ORDER = {
    "low": 1,
    "low_medium": 2,
    "medium": 3,
    "medium_high": 4,
    "high": 5,
    "very_high": 6
}


def number(row, key):
    """Return a finite numeric value or None."""
    value = row.get(key, np.nan)

    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def range_score(value, acceptable, tolerance=None):
    """
    Score a value against an acceptable range.

    100 = inside range.
    Outside range, score declines linearly.
    """
    if value is None:
        return None

    low, high = acceptable

    if low <= value <= high:
        return 100.0

    if tolerance is None:
        tolerance = max((high - low) * 0.5, 1.0)

    if value < low:
        distance = low - value
    else:
        distance = value - high

    return max(0.0, 100.0 * (1.0 - distance / tolerance))


def pH_value(row):
    value = number(row, "phh2o_0-5cm")

    if value is None:
        return None

    # SoilGrids pH layers are commonly stored in tenths of pH units.
    if value > 14:
        value /= 10.0

    if not 0 <= value <= 14:
        return None

    return value


def texture_values(row):
    """
    SoilGrids sand and clay are commonly expressed in g/kg.
    Convert to percentages when values exceed 100.
    """
    sand = number(row, "sand_0-5cm")
    clay = number(row, "clay_0-5cm")

    if sand is not None and sand > 100:
        sand /= 10.0

    if clay is not None and clay > 100:
        clay /= 10.0

    return sand, clay


def infer_texture(sand, clay):
    if sand is None or clay is None:
        return None

    if sand < 0 or clay < 0 or sand + clay > 100:
        return None

    silt = 100 - sand - clay

    # Simplified screening classes, not a full USDA texture triangle.
    if clay >= 40:
        return "clay"

    if clay >= 27:
        return "clay_loam"

    if sand >= 70:
        return "sandy_loam"

    if sand >= 43 and clay < 20:
        return "sandy_loam"

    if clay >= 20:
        return "clay_loam"

    if silt >= 50:
        return "loam"

    return "loam"


def score_soil_ph(row, crop):
    value = pH_value(row)

    score = range_score(value, crop["soil_ph"], tolerance=1.5)

    if score is None:
        return None, "Soil pH unavailable."

    return score, (
        f"Surface soil pH is {value:.1f}; "
        f"crop preference is {crop['soil_ph'][0]}–{crop['soil_ph'][1]}."
    )


def score_texture(row, crop):
    sand, clay = texture_values(row)
    texture = infer_texture(sand, clay)

    if texture is None:
        return None, "Soil texture unavailable."

    accepted = crop["soil_texture"]

    if texture in accepted:
        score = 100.0
        reason = f"Estimated texture ({texture}) matches crop preference."
    elif texture == "loam" or "loam" in accepted:
        score = 65.0
        reason = f"Estimated texture ({texture}) is a partial match."
    else:
        score = 30.0
        reason = f"Estimated texture ({texture}) differs from crop preference."

    return score, reason


def score_temperature(row, crop, season):
    climate_season = SEASON_CLIMATE[season]

    if climate_season == "annual":
        key = "T2M_annual_mean"
    else:
        key = f"T2M_{climate_season}_mean"

    value = number(row, key)

    if value is None:
        return None, f"Temperature data unavailable for {season}."

    score = range_score(
        value,
        crop["temperature_c"],
        tolerance=8.0
    )

    return score, (
        f"Mean {season} temperature is {value:.1f}°C; "
        f"crop range is {crop['temperature_c'][0]}–"
        f"{crop['temperature_c'][1]}°C."
    )


def score_moisture(row, crop):
    value = number(row, "smap_mean")

    if value is None:
        return None, "SMAP root-zone moisture unavailable."

    # Broad screening ranges for long-term root-zone moisture.
    # This is NOT a crop-specific water balance or irrigation estimate.
    demand = crop["water_demand"]

    ranges = {
        "low": (0.12, 0.30),
        "low_medium": (0.15, 0.34),
        "medium": (0.18, 0.38),
        "medium_high": (0.22, 0.43),
        "high": (0.28, 0.48),
        "very_high": (0.32, 0.55)
    }

    score = range_score(value, ranges[demand], tolerance=0.18)

    return score, (
        f"Long-term SMAP root-zone moisture mean is {value:.3f} m³/m³. "
        "This is a moisture proxy, not measured irrigation availability."
    )


def score_rainfall(row, crop, season):
    climate_season = SEASON_CLIMATE[season]

    if climate_season == "annual":
        key = "annual_rainfall_mean_mm"
    else:
        key = f"PRECTOTCORR_{climate_season}_total_mean"

    value = number(row, key)

    if value is None:
        return None, f"Rainfall data unavailable for {season}."

    # Approximate seasonal rainfall screening bands.
    # These are broad planning assumptions, not crop water requirements.
    bands = {
        "low": (100, 450),
        "low_medium": (150, 650),
        "medium": (250, 900),
        "medium_high": (350, 1200),
        "high": (500, 1800),
        "very_high": (700, 2500)
    }

    low, high = bands[crop["water_demand"]]

    score = range_score(value, (low, high), tolerance=500)

    return score, (
        f"Mean seasonal rainfall is {value:.0f} mm; "
        f"broad screening band is {low}–{high} mm."
    )


def score_slope(row):
    value = number(row, "slope_degrees")

    if value is None:
        return None, "Terrain slope unavailable."

    if value <= 3:
        score = 100
    elif value <= 8:
        score = 85
    elif value <= 15:
        score = 60
    elif value <= 25:
        score = 30
    else:
        score = 10

    return float(score), f"Terrain slope is {value:.1f}°."


def score_season(crop, season):
    if season in crop["season"]:
        return 100.0, f"Crop is listed for the {season} season."

    return 0.0, f"Crop is not listed for the {season} season."


def classify(score, coverage):
    if coverage < 0.60:
        return "insufficient_evidence"

    if score >= 80:
        return "high"
    if score >= 65:
        return "moderate"
    if score >= 45:
        return "marginal"

    return "low"


def evaluate_crop(row, crop_id, crop, season):
    factors = {}
    reasons = []
    warnings = []

    factor_functions = {
        "soil_ph": lambda: score_soil_ph(row, crop),
        "soil_texture": lambda: score_texture(row, crop),
        "temperature": lambda: score_temperature(row, crop, season),
        "soil_moisture": lambda: score_moisture(row, crop),
        "rainfall": lambda: score_rainfall(row, crop, season),
        "slope": lambda: score_slope(row),
        "season": lambda: score_season(crop, season)
    }

    weighted_total = 0.0
    available_weight = 0.0

    for factor, function in factor_functions.items():
        value, explanation = function()
        factors[factor] = value

        if value is None:
            warnings.append(explanation)
            continue

        weighted_total += value * WEIGHTS[factor]
        available_weight += WEIGHTS[factor]

        if value >= 75:
            reasons.append(explanation)
        elif value < 45:
            warnings.append(explanation)

    score = (
        weighted_total / available_weight
        if available_weight > 0 else None
    )

    coverage = available_weight / sum(WEIGHTS.values())

    # Seasonal eligibility is a hard constraint for annual crops.
    # Perennial crops are assessed separately using annual climate.
    season_eligible = season in crop["season"]

    if not season_eligible and crop["crop_group"] != "fruit":
        score = 0.0

    classification = (
        classify(score, coverage)
        if score is not None else "insufficient_evidence"
    )

    if not season_eligible and crop["crop_group"] != "fruit":
        classification = "out_of_season"
        warnings.append(
            f"{crop['name']} is not listed for the {season} season."
        )

    return {
        "crop_id": crop_id,
        "crop_name": crop["name"],
        "crop_group": crop["crop_group"],
        "family": crop["family"],
        "season": season,
        "suitability_score": (
            round(float(score), 2) if score is not None else None
        ),
        "evidence_coverage": round(float(coverage), 3),
        "classification": classification,
        "factor_scores": factors,
        "reasons": reasons,
        "warnings": warnings
    }


def main():
    print("=" * 58)
    print("AERIS | CROP SUITABILITY ENGINE")
    print("=" * 58)

    validation = validate_database()

    if not validation["valid"]:
        raise RuntimeError(
            f"Crop knowledge base invalid: {validation['errors']}"
        )

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = ["latitude", "longitude", "smap_mean"]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    all_results = []
    ranked_results = []

    seasons = ["kharif", "rabi", "summer"]

    print(f"Locations: {len(df)}")
    print(f"Crops: {len(CROPS)}")
    print(f"Seasons: {', '.join(seasons)}\n")

    for location_index, row in df.iterrows():
        location_id = f"LOC_{location_index + 1:03d}"

        for season in seasons:
            location_results = []

            for crop_id, crop in CROPS.items():
                result = evaluate_crop(row, crop_id, crop, season)

                record = {
                    "location_id": location_id,
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    **{
                        k: v for k, v in result.items()
                        if k not in ["reasons", "warnings", "factor_scores"]
                    },
                    "factor_scores_json": json.dumps(
                        result["factor_scores"]
                    ),
                    "reasons": " | ".join(result["reasons"]),
                    "warnings": " | ".join(result["warnings"])
                }

                all_results.append(record)
                location_results.append(record)

            eligible = [
                result for result in location_results
                if result["classification"] not in [
                    "out_of_season", "insufficient_evidence"
                ]
                and result["suitability_score"] is not None
            ]

            eligible.sort(
                key=lambda x: x["suitability_score"],
                reverse=True
            )

            for rank, result in enumerate(eligible, start=1):
                ranked_results.append({
                    **result,
                    "rank": rank
                })

    all_df = pd.DataFrame(all_results)
    ranked_df = pd.DataFrame(ranked_results)

    all_path = OUTPUT_DIR / "crop_suitability_all.csv"
    ranked_path = OUTPUT_DIR / "crop_suitability_ranked.csv"

    all_df.to_csv(all_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)

    report = {
        "engine_version": "1.0",
        "database_version": "1.0",
        "region": "Nashik-area bounding box",
        "locations": int(len(df)),
        "crop_count": len(CROPS),
        "seasons_evaluated": seasons,
        "evaluations": len(all_df),
        "ranked_recommendations": len(ranked_df),
        "weights": WEIGHTS,
        "score_type": "weighted environmental screening score",
        "not_a_yield_prediction": True,
        "outputs": {
            "all": str(all_path),
            "ranked": str(ranked_path)
        },
        "classification_counts": {
            str(k): int(v)
            for k, v in all_df["classification"].value_counts().items()
        },
        "average_evidence_coverage": float(
            all_df["evidence_coverage"].mean()
        )
    }

    report_path = OUTPUT_DIR / "crop_suitability_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== ENGINE REPORT ==========")
    print(f"Evaluations: {len(all_df)}")
    print(f"Ranked recommendations: {len(ranked_df)}")
    print(
        f"Average evidence coverage: "
        f"{report['average_evidence_coverage']:.1%}"
    )

    print("\nClassification counts:")
    for label, count in report["classification_counts"].items():
        print(f"  {label}: {count}")

    print("\nExample: top Kharif crops at first location")

    example = ranked_df[
        (ranked_df["location_id"] == "LOC_001")
        & (ranked_df["season"] == "kharif")
    ].head(5)

    if example.empty:
        print("No crops passed the evidence and season filters.")
    else:
        for _, item in example.iterrows():
            print(
                f"  #{item['rank']} {item['crop_name']}: "
                f"{item['suitability_score']:.1f}/100 "
                f"({item['classification']})"
            )

    print("\nSaved:")
    print(all_path)
    print(ranked_path)
    print(report_path)


if __name__ == "__main__":
    main()
