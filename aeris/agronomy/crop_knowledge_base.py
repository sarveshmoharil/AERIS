
"""
AERIS — Crop Knowledge Base
Version: 1.0

Purpose:
    Structured agronomic knowledge for crop suitability and rotation planning
    in Nashik district, Maharashtra.

IMPORTANT:
    Values are approximate planning ranges, not field prescriptions.
    SoilGrids and NASA environmental data are modelled observations.
    Validate recommendations with local KVK/agriculture officers and
    field-specific soil tests before practical use.
"""

from copy import deepcopy


DATABASE_VERSION = "1.0"
REGION = "Nashik, Maharashtra, India"

SEASONS = {
    "kharif": {
        "months": [6, 7, 8, 9, 10],
        "description": "Monsoon cropping season"
    },
    "rabi": {
        "months": [10, 11, 12, 1, 2, 3],
        "description": "Post-monsoon and winter cropping season"
    },
    "summer": {
        "months": [2, 3, 4, 5],
        "description": "Summer irrigated cropping season"
    },
    "perennial": {
        "months": list(range(1, 13)),
        "description": "Long-duration perennial cultivation"
    }
}


# Water demand is an ordinal planning category, NOT mm of crop water.
# pH and temperature ranges are broad suitability screening ranges.
# Duration is approximate days from sowing/transplanting to harvest.
CROPS = {

    "rice": {
        "name": "Rice",
        "scientific_name": "Oryza sativa",
        "family": "Poaceae",
        "crop_group": "cereal",
        "rotation_group": "cereal",
        "season": ["kharif"],
        "duration_days": [110, 150],
        "temperature_c": [20, 35],
        "soil_ph": [5.0, 7.5],
        "soil_texture": ["clay", "clay_loam", "loam"],
        "drainage": "water_retentive",
        "water_demand": "very_high",
        "waterlogging_tolerance": "high",
        "nitrogen_fixing": False,
        "residue_value": "medium",
        "preferred_zones": ["ghat", "high_rainfall", "assured_irrigation"],
        "rotation_constraints": ["requires_reliable_water_supply"],
        "rotation_benefits": ["can_follow_short_duration_pulses"],
        "evidence_level": "regional_crop",
        "notes": "Most appropriate in wetter Nashik zones with suitable water availability."
    },

    "maize": {
        "name": "Maize",
        "scientific_name": "Zea mays",
        "family": "Poaceae",
        "crop_group": "cereal",
        "rotation_group": "cereal",
        "season": ["kharif", "rabi"],
        "duration_days": [90, 120],
        "temperature_c": [18, 35],
        "soil_ph": [5.5, 7.5],
        "soil_texture": ["loam", "clay_loam", "sandy_loam"],
        "drainage": "well_drained",
        "water_demand": "medium_high",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "high",
        "preferred_zones": ["plain", "assured_rainfall", "assured_irrigation"],
        "rotation_constraints": ["avoid_repeated_cereal_rotation"],
        "rotation_benefits": ["benefits_from_preceding_legume"],
        "evidence_level": "regional_crop",
        "notes": "Important Nashik crop. Avoid prolonged waterlogging and repeated monocropping."
    },

    "bajra": {
        "name": "Pearl Millet",
        "scientific_name": "Pennisetum glaucum",
        "family": "Poaceae",
        "crop_group": "cereal",
        "rotation_group": "millet",
        "season": ["kharif"],
        "duration_days": [75, 100],
        "temperature_c": [25, 38],
        "soil_ph": [5.5, 8.0],
        "soil_texture": ["sandy_loam", "loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "low",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "medium",
        "preferred_zones": ["scarcity", "low_rainfall", "plain"],
        "rotation_constraints": ["avoid_repeated_cereal_rotation"],
        "rotation_benefits": ["suitable_for_low_rainfall_sequences"],
        "evidence_level": "regional_crop",
        "notes": "Suitable candidate for lower-rainfall parts of Nashik."
    },

    "ragi": {
        "name": "Finger Millet",
        "scientific_name": "Eleusine coracana",
        "family": "Poaceae",
        "crop_group": "cereal",
        "rotation_group": "millet",
        "season": ["kharif"],
        "duration_days": [100, 120],
        "temperature_c": [20, 30],
        "soil_ph": [5.0, 7.5],
        "soil_texture": ["loam", "sandy_loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "low_medium",
        "waterlogging_tolerance": "medium",
        "nitrogen_fixing": False,
        "residue_value": "medium",
        "preferred_zones": ["ghat", "high_rainfall", "hilly"],
        "rotation_constraints": ["avoid_repeated_cereal_rotation"],
        "rotation_benefits": ["diversifies_cereal_systems"],
        "evidence_level": "regional_crop",
        "notes": "Relevant to wetter and hilly Nashik areas."
    },

    "soybean": {
        "name": "Soybean",
        "scientific_name": "Glycine max",
        "family": "Fabaceae",
        "crop_group": "oilseed",
        "rotation_group": "legume",
        "season": ["kharif"],
        "duration_days": [90, 115],
        "temperature_c": [20, 32],
        "soil_ph": [6.0, 7.5],
        "soil_texture": ["loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": True,
        "residue_value": "high",
        "preferred_zones": ["plain", "assured_rainfall", "medium_soil"],
        "rotation_constraints": ["avoid_repeated_legume_rotation"],
        "rotation_benefits": ["biological_nitrogen_fixation", "breaks_cereal_sequence"],
        "evidence_level": "regional_crop",
        "notes": "Inoculation and suitable drainage are important for successful cultivation."
    },

    "tur": {
        "name": "Pigeon Pea",
        "scientific_name": "Cajanus cajan",
        "family": "Fabaceae",
        "crop_group": "pulse",
        "rotation_group": "legume",
        "season": ["kharif"],
        "duration_days": [150, 200],
        "temperature_c": [20, 35],
        "soil_ph": [5.5, 8.0],
        "soil_texture": ["loam", "clay_loam", "sandy_loam"],
        "drainage": "well_drained",
        "water_demand": "low_medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": True,
        "residue_value": "high",
        "preferred_zones": ["plain", "scarcity", "medium_soil"],
        "rotation_constraints": ["long_duration_crop", "avoid_waterlogged_fields"],
        "rotation_benefits": ["biological_nitrogen_fixation", "deep_rooting"],
        "evidence_level": "regional_crop",
        "notes": "Long-duration pulse suitable for appropriately drained fields."
    },

    "mung": {
        "name": "Green Gram",
        "scientific_name": "Vigna radiata",
        "family": "Fabaceae",
        "crop_group": "pulse",
        "rotation_group": "legume",
        "season": ["kharif", "summer"],
        "duration_days": [60, 80],
        "temperature_c": [25, 35],
        "soil_ph": [6.0, 7.5],
        "soil_texture": ["sandy_loam", "loam"],
        "drainage": "well_drained",
        "water_demand": "low_medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": True,
        "residue_value": "medium",
        "preferred_zones": ["plain", "scarcity", "assured_irrigation"],
        "rotation_constraints": ["avoid_repeated_legume_rotation"],
        "rotation_benefits": ["short_duration", "biological_nitrogen_fixation"],
        "evidence_level": "regional_crop",
        "notes": "Short-duration pulse that can fit between major crop seasons."
    },

    "urad": {
        "name": "Black Gram",
        "scientific_name": "Vigna mungo",
        "family": "Fabaceae",
        "crop_group": "pulse",
        "rotation_group": "legume",
        "season": ["kharif", "summer"],
        "duration_days": [70, 90],
        "temperature_c": [25, 35],
        "soil_ph": [5.5, 7.5],
        "soil_texture": ["loam", "sandy_loam"],
        "drainage": "well_drained",
        "water_demand": "low_medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": True,
        "residue_value": "medium",
        "preferred_zones": ["plain", "scarcity", "assured_irrigation"],
        "rotation_constraints": ["avoid_repeated_legume_rotation"],
        "rotation_benefits": ["short_duration", "biological_nitrogen_fixation"],
        "evidence_level": "regional_crop",
        "notes": "Short-duration pulse; field drainage is important."
    },

    "groundnut": {
        "name": "Groundnut",
        "scientific_name": "Arachis hypogaea",
        "family": "Fabaceae",
        "crop_group": "oilseed",
        "rotation_group": "legume",
        "season": ["kharif"],
        "duration_days": [100, 130],
        "temperature_c": [22, 32],
        "soil_ph": [5.5, 7.5],
        "soil_texture": ["sandy_loam", "loam"],
        "drainage": "well_drained",
        "water_demand": "medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": True,
        "residue_value": "high",
        "preferred_zones": ["plain", "scarcity", "light_soil"],
        "rotation_constraints": ["avoid_repeated_legume_rotation"],
        "rotation_benefits": ["biological_nitrogen_fixation"],
        "evidence_level": "regional_crop",
        "notes": "Loose, well-drained soil supports peg penetration and pod development."
    },

    "chickpea": {
        "name": "Chickpea",
        "scientific_name": "Cicer arietinum",
        "family": "Fabaceae",
        "crop_group": "pulse",
        "rotation_group": "legume",
        "season": ["rabi"],
        "duration_days": [90, 120],
        "temperature_c": [15, 30],
        "soil_ph": [6.0, 8.0],
        "soil_texture": ["loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "low",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": True,
        "residue_value": "medium",
        "preferred_zones": ["plain", "scarcity", "residual_moisture"],
        "rotation_constraints": ["avoid_repeated_legume_rotation"],
        "rotation_benefits": ["biological_nitrogen_fixation", "uses_residual_moisture"],
        "evidence_level": "regional_crop",
        "notes": "Rabi pulse suited to residual soil moisture and limited irrigation."
    },

    "wheat": {
        "name": "Wheat",
        "scientific_name": "Triticum aestivum",
        "family": "Poaceae",
        "crop_group": "cereal",
        "rotation_group": "cereal",
        "season": ["rabi"],
        "duration_days": [110, 140],
        "temperature_c": [10, 25],
        "soil_ph": [6.0, 8.0],
        "soil_texture": ["loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "medium",
        "preferred_zones": ["plain", "assured_irrigation", "medium_soil"],
        "rotation_constraints": ["avoid_repeated_cereal_rotation"],
        "rotation_benefits": ["can_follow_short_duration_kharif_crops"],
        "evidence_level": "regional_crop",
        "notes": "Requires timely sowing and suitable winter temperatures."
    },

    "onion": {
        "name": "Onion",
        "scientific_name": "Allium cepa",
        "family": "Amaryllidaceae",
        "crop_group": "vegetable",
        "rotation_group": "vegetable",
        "season": ["kharif", "rabi", "summer"],
        "duration_days": [100, 140],
        "temperature_c": [13, 30],
        "soil_ph": [6.0, 7.5],
        "soil_texture": ["sandy_loam", "loam"],
        "drainage": "well_drained",
        "water_demand": "medium_high",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "low",
        "preferred_zones": ["plain", "assured_irrigation", "medium_soil"],
        "rotation_constraints": ["avoid_repeated_allium_crops", "requires_market_and_storage_planning"],
        "rotation_benefits": ["high_value_crop"],
        "evidence_level": "regional_crop",
        "notes": "Important Nashik commercial crop. Irrigation scheduling and storage matter."
    },

    "cotton": {
        "name": "Cotton",
        "scientific_name": "Gossypium spp.",
        "family": "Malvaceae",
        "crop_group": "fibre",
        "rotation_group": "fibre",
        "season": ["kharif"],
        "duration_days": [150, 180],
        "temperature_c": [21, 35],
        "soil_ph": [5.5, 8.0],
        "soil_texture": ["clay", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "medium",
        "preferred_zones": ["scarcity", "deep_black_soil"],
        "rotation_constraints": ["long_duration_crop", "pest_management_required"],
        "rotation_benefits": ["can_follow_legume_break_crop"],
        "evidence_level": "regional_crop",
        "notes": "Consider only where local conditions and cultivar recommendations support it."
    },

    "tomato": {
        "name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae",
        "crop_group": "vegetable",
        "rotation_group": "solanaceae",
        "season": ["kharif", "rabi", "summer"],
        "duration_days": [100, 150],
        "temperature_c": [18, 30],
        "soil_ph": [5.5, 7.5],
        "soil_texture": ["loam", "sandy_loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "medium_high",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "low",
        "preferred_zones": ["plain", "assured_irrigation"],
        "rotation_constraints": ["avoid_repeated_solanaceae", "disease_management_required"],
        "rotation_benefits": ["diversifies_crop_income"],
        "evidence_level": "regional_crop",
        "notes": "Commercial vegetable crop; disease and market risk must be considered."
    },

    "grapes": {
        "name": "Grapes",
        "scientific_name": "Vitis vinifera",
        "family": "Vitaceae",
        "crop_group": "fruit",
        "rotation_group": "perennial",
        "season": ["perennial"],
        "duration_days": [365, 365],
        "temperature_c": [15, 35],
        "soil_ph": [6.0, 8.0],
        "soil_texture": ["sandy_loam", "loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "medium_high",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "low",
        "preferred_zones": ["assured_irrigation", "plain", "medium_soil"],
        "rotation_constraints": ["perennial_establishment", "requires_specialized_management"],
        "rotation_benefits": [],
        "evidence_level": "regional_crop",
        "notes": "Major Nashik horticultural crop. Not treated as an annual rotation crop."
    },

    "pomegranate": {
        "name": "Pomegranate",
        "scientific_name": "Punica granatum",
        "family": "Lythraceae",
        "crop_group": "fruit",
        "rotation_group": "perennial",
        "season": ["perennial"],
        "duration_days": [365, 365],
        "temperature_c": [20, 38],
        "soil_ph": [6.0, 8.0],
        "soil_texture": ["sandy_loam", "loam", "clay_loam"],
        "drainage": "well_drained",
        "water_demand": "low_medium",
        "waterlogging_tolerance": "low",
        "nitrogen_fixing": False,
        "residue_value": "low",
        "preferred_zones": ["scarcity", "low_rainfall", "assured_irrigation"],
        "rotation_constraints": ["perennial_establishment", "requires_specialized_management"],
        "rotation_benefits": [],
        "evidence_level": "regional_crop",
        "notes": "Relevant to drier Nashik zones; orchard suitability needs water-quality and disease assessment."
    }
}


# Rotation compatibility is a starting agronomic rule set.
# It does not imply that every sequence is suitable in every field.
ROTATION_RULES = {
    "same_family_penalty": 20,
    "same_rotation_group_penalty": 12,
    "consecutive_cereal_penalty": 15,
    "legume_after_non_legume_bonus": 8,
    "legume_nitrogen_credit": "qualitative_only",
    "perennial_rotation_allowed": False,
    "max_consecutive_same_family": 1,
    "max_sequence_length": 3
}


def get_crop(crop_id):
    """Return a copy of a crop record."""
    crop_id = crop_id.lower().strip()

    if crop_id not in CROPS:
        raise KeyError(f"Unknown crop: {crop_id}")

    return deepcopy(CROPS[crop_id])


def list_crops(season=None, crop_group=None, include_perennials=True):
    """Filter crops by season and crop group."""
    results = {}

    for crop_id, crop in CROPS.items():
        if not include_perennials and crop["crop_group"] == "fruit":
            continue

        if season and season not in crop["season"]:
            continue

        if crop_group and crop["crop_group"] != crop_group:
            continue

        results[crop_id] = deepcopy(crop)

    return results


def get_rotation_penalty(previous_crop_id, next_crop_id):
    """Calculate a simple, explainable crop-sequence penalty."""
    previous = get_crop(previous_crop_id)
    following = get_crop(next_crop_id)

    if previous["crop_group"] == "fruit" or following["crop_group"] == "fruit":
        return {
            "penalty": None,
            "valid": False,
            "reasons": ["Perennial crops are excluded from annual rotation scoring."]
        }

    penalty = 0
    reasons = []

    if previous["family"] == following["family"]:
        penalty += ROTATION_RULES["same_family_penalty"]
        reasons.append("Consecutive crops belong to the same botanical family.")

    if previous["rotation_group"] == following["rotation_group"]:
        penalty += ROTATION_RULES["same_rotation_group_penalty"]
        reasons.append("Consecutive crops belong to the same rotation group.")

    if (
        previous["crop_group"] == "cereal"
        and following["crop_group"] == "cereal"
    ):
        penalty += ROTATION_RULES["consecutive_cereal_penalty"]
        reasons.append("Consecutive cereal crops may increase nutrient and pest pressure.")

    if following["nitrogen_fixing"] and not previous["nitrogen_fixing"]:
        penalty -= ROTATION_RULES["legume_after_non_legume_bonus"]
        reasons.append("A legume follows a non-legume crop.")

    return {
        "penalty": max(0, penalty),
        "valid": True,
        "reasons": reasons or ["No basic sequence penalty triggered."]
    }


def validate_database():
    """Check completeness and consistency of the knowledge base."""
    required_fields = [
        "name",
        "scientific_name",
        "family",
        "crop_group",
        "rotation_group",
        "season",
        "duration_days",
        "temperature_c",
        "soil_ph",
        "soil_texture",
        "water_demand",
        "nitrogen_fixing"
    ]

    errors = []

    for crop_id, crop in CROPS.items():
        for field in required_fields:
            if field not in crop:
                errors.append(f"{crop_id}: missing {field}")

        if crop.get("duration_days", [0, 0])[0] > crop.get("duration_days", [0, 0])[1]:
            errors.append(f"{crop_id}: invalid duration range")

        if crop.get("soil_ph", [0, 0])[0] > crop.get("soil_ph", [0, 0])[1]:
            errors.append(f"{crop_id}: invalid soil pH range")

    return {
        "valid": len(errors) == 0,
        "crop_count": len(CROPS),
        "errors": errors
    }


if __name__ == "__main__":
    print("=" * 55)
    print("AERIS | AGRONOMY KNOWLEDGE BASE")
    print("=" * 55)

    report = validate_database()

    print(f"Database version: {DATABASE_VERSION}")
    print(f"Region: {REGION}")
    print(f"Total crops: {report['crop_count']}")
    print(f"Database valid: {report['valid']}")

    if report["errors"]:
        for error in report["errors"]:
            print("[ERROR]", error)

    print("\nCrop inventory:")

    for crop_id, crop in CROPS.items():
        duration = crop["duration_days"]
        print(
            f"  {crop_id:14} | "
            f"{crop['name']:18} | "
            f"{'/'.join(crop['season']):18} | "
            f"{duration[0]}-{duration[1]} days"
        )

    print("\nExample rotation: soybean -> wheat")
    print(get_rotation_penalty("soybean", "wheat"))
