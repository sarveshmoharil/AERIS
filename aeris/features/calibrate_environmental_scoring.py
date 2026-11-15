
"""
AERIS | Environmental Scoring Calibration
Version 1.0

Diagnostic-only script.
Inspects actual environmental distributions and existing
crop suitability scores before calibration.

Does not modify existing datasets.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "aeris" / "data" / "processed"

ENV_FILE = DATA / "environmental" / "nashik_environmental_features.csv"
SUITABILITY_FILE = DATA / "crop_suitability" / "crop_suitability_all.csv"

OUTPUT_DIR = DATA / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def describe_column(df, column):
    if column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return None

    return {
        "count": int(values.count()),
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
        "std": round(float(values.std()), 4),
        "min": round(float(values.min()), 4),
        "p10": round(float(values.quantile(0.10)), 4),
        "p25": round(float(values.quantile(0.25)), 4),
        "p75": round(float(values.quantile(0.75)), 4),
        "p90": round(float(values.quantile(0.90)), 4),
        "max": round(float(values.max()), 4)
    }


def inspect_environment(env):
    print("\n========== ENVIRONMENT DISTRIBUTIONS ==========")

    columns = [
        "smap_mean",
        "smap_median",
        "smap_min",
        "smap_max",
        "phh2o_0-5cm",
        "phh2o_5-15cm",
        "phh2o_15-30cm",
        "sand_0-5cm",
        "clay_0-5cm",
        "elevation_m",
        "slope_degrees",
        "annual_rainfall_mean_mm",
        "PRECTOTCORR_monsoon_total_mean",
        "PRECTOTCORR_post_monsoon_total_mean",
        "PRECTOTCORR_winter_total_mean",
        "PRECTOTCORR_summer_total_mean",
        "T2M_monsoon_mean",
        "T2M_post_monsoon_mean",
        "T2M_winter_mean",
        "T2M_summer_mean",
        "T2M_annual_mean"
    ]

    results = {}

    for column in columns:
        stats = describe_column(env, column)

        if stats is None:
            print(f"\n{column}: NOT AVAILABLE")
            continue

        results[column] = stats

        print(f"\n{column}")
        for key, value in stats.items():
            print(f"  {key:>8}: {value}")

    return results


def inspect_rainfall(suitability):
    print("\n========== RAINFALL SCORE DIAGNOSTICS ==========")

    df = suitability.copy()

    df["rainfall_score"] = pd.to_numeric(
        df["factor_scores_json"].apply(
            lambda value: json.loads(value).get("rainfall")
            if isinstance(value, str) else None
        ),
        errors="coerce"
    )

    if "season" in df.columns:
        result = (
            df.groupby("season")["rainfall_score"]
            .agg(["count", "mean", "min", "median", "max"])
            .round(2)
        )

        print(result.to_string())

        return result.reset_index().to_dict("records")

    return []


def inspect_factor_scores(suitability):
    print("\n========== FACTOR SCORE DIAGNOSTICS ==========")

    factors = [
        "soil_ph",
        "soil_texture",
        "temperature",
        "soil_moisture",
        "rainfall",
        "slope",
        "season"
    ]

    records = []

    for factor in factors:
        scores = []

        for value in suitability["factor_scores_json"].dropna():
            try:
                parsed = json.loads(value)
                score = parsed.get(factor)

                if score is not None:
                    scores.append(float(score))
            except (ValueError, TypeError):
                continue

        if not scores:
            continue

        scores = np.array(scores)

        record = {
            "factor": factor,
            "mean": round(float(scores.mean()), 2),
            "median": round(float(np.median(scores)), 2),
            "minimum": round(float(scores.min()), 2),
            "maximum": round(float(scores.max()), 2),
            "zero_count": int(np.sum(scores == 0)),
            "below_45_count": int(np.sum(scores < 45)),
            "above_90_count": int(np.sum(scores > 90))
        }

        records.append(record)

        print(
            f"{factor:20} "
            f"mean={record['mean']:6.2f} "
            f"median={record['median']:6.2f} "
            f"zero={record['zero_count']:5} "
            f"below45={record['below_45_count']:5}"
        )

    return records


def inspect_crop_patterns(suitability):
    print("\n========== CROP-SPECIFIC RAINFALL SCORES ==========")

    df = suitability.copy()

    df["rainfall_score"] = df["factor_scores_json"].apply(
        lambda value: (
            json.loads(value).get("rainfall")
            if isinstance(value, str) else None
        )
    )

    df["rainfall_score"] = pd.to_numeric(
        df["rainfall_score"],
        errors="coerce"
    )

    result = (
        df.groupby(["crop_name", "season"])
        .agg(
            mean_suitability=("suitability_score", "mean"),
            mean_rainfall_score=("rainfall_score", "mean"),
            evaluations=("location_id", "count")
        )
        .round(2)
        .reset_index()
    )

    result = result.sort_values(
        ["season", "mean_rainfall_score"]
    )

    print(result.to_string(index=False))

    return result


def inspect_duplicate_climate(env):
    print("\n========== CLIMATE SPATIAL VARIATION ==========")

    climate_columns = [
        col for col in env.columns
        if col.startswith("T2M_")
        or col.startswith("PRECTOTCORR_")
    ]

    results = {}

    for column in climate_columns:
        values = pd.to_numeric(
            env[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        results[column] = {
            "unique_values": int(values.nunique()),
            "spatially_constant": bool(values.nunique() <= 1)
        }

        print(
            f"{column:45} "
            f"unique values: {values.nunique()}"
        )

    return results


def main():
    print("=" * 60)
    print("AERIS | ENVIRONMENTAL CALIBRATION DIAGNOSTICS")
    print("=" * 60)

    env = load_data(ENV_FILE)
    suitability = load_data(SUITABILITY_FILE)

    print(f"Environmental locations: {len(env)}")
    print(f"Suitability evaluations: {len(suitability)}")

    environmental = inspect_environment(env)
    rainfall = inspect_rainfall(suitability)
    factors = inspect_factor_scores(suitability)
    crop_patterns = inspect_crop_patterns(suitability)
    climate_variation = inspect_duplicate_climate(env)

    crop_patterns.to_csv(
        OUTPUT_DIR / "crop_rainfall_diagnostics.csv",
        index=False
    )

    report = {
        "version": "1.0",
        "diagnostic_only": True,
        "environmental_statistics": environmental,
        "rainfall_by_season": rainfall,
        "factor_score_statistics": factors,
        "climate_spatial_variation": climate_variation,
        "locations": int(len(env)),
        "evaluations": int(len(suitability)),
        "recommendation": (
            "Review actual environmental distributions and crop-specific "
            "agronomic thresholds before changing suitability weights."
        )
    }

    output = OUTPUT_DIR / "calibration_diagnostics.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== DIAGNOSTICS COMPLETE ==========")
    print(f"Saved: {output}")
    print(
        f"Saved: {OUTPUT_DIR / 'crop_rainfall_diagnostics.csv'}"
    )

    print("\nNo original datasets or scores were modified.")


if __name__ == "__main__":
    main()
