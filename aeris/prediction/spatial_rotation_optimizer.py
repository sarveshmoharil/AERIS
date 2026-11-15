
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

SUITABILITY_FILE = (
    DATA / "future_rotations" / "future_crop_suitability.csv"
)
ASSIGNMENT_FILE = (
    DATA / "spatial" / "sample_grid_assignments.csv"
)
ENVIRONMENT_FILE = (
    DATA / "spatial" / "grid_cell_environmental_profiles.csv"
)

OUTPUT_DIR = DATA / "spatial_optimization"

SEASONS = ["kharif", "rabi", "summer"]
GRID_CELLS = [
    f"G{r}0{c}"
    for r in range(1, 5)
    for c in range(1, 3)
]

POPULATION_SIZE = 70
GENERATIONS = 50
MUTATION_RATE = 0.20
ELITE_COUNT = 8
CANDIDATES_PER_SEASON = 8
RANDOM_SEED = 42

TRANSFORM = Transformer.from_crs(
    "EPSG:4326", "EPSG:32643", always_xy=True
)

# Annual crop IDs only. Perennials are excluded from annual rotations.
PERENNIALS = {"grapes", "pomegranate"}

LEGUMES = {
    "soybean", "tur", "mung", "urad",
    "groundnut", "chickpea"
}

CEREALS = {
    "rice", "maize", "bajra", "ragi", "wheat"
}


def load_inputs():
    for path in [
        SUITABILITY_FILE,
        ASSIGNMENT_FILE,
        ENVIRONMENT_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    suitability = pd.read_csv(SUITABILITY_FILE)
    assignments = pd.read_csv(ASSIGNMENT_FILE)
    environment = pd.read_csv(ENVIRONMENT_FILE)

    required = {
        "location_id", "latitude", "longitude",
        "crop_id", "crop_name", "family", "crop_group",
        "season", "future_suitability_score",
        "evidence_coverage", "classification"
    }

    missing = required - set(suitability.columns)

    if missing:
        raise ValueError(
            f"Suitability data missing columns: {sorted(missing)}"
        )

    return suitability, assignments, environment


def assign_future_locations(suitability, assignments):
    # Reconstruct the exact projected grid used by SEAE-01.
    sx, sy = TRANSFORM.transform(
        assignments["longitude"].values,
        assignments["latitude"].values
    )

    xmin = float(np.min(sx))
    ymin = float(np.min(sy))
    xmax = float(np.max(sx))
    ymax = float(np.max(sy))

    cell_size = max(
        (xmax - xmin) / 2,
        (ymax - ymin) / 4
    )

    x, y = TRANSFORM.transform(
        suitability["longitude"].values,
        suitability["latitude"].values
    )

    columns = np.floor((x - xmin) / cell_size).astype(int)
    rows = np.floor((y - ymin) / cell_size).astype(int)

    columns = np.clip(columns, 0, 1)
    rows = np.clip(rows, 0, 3)

    result = suitability.copy()

    result["grid_cell"] = [
        f"G{r + 1}0{c + 1}"
        for r, c in zip(rows, columns)
    ]

    return result


def prepare_candidates(df):
    df = df.copy()

    df["future_suitability_score"] = pd.to_numeric(
        df["future_suitability_score"], errors="coerce"
    )

    df["evidence_coverage"] = pd.to_numeric(
        df["evidence_coverage"], errors="coerce"
    )

    df = df[
        ~df["classification"].isin([
            "out_of_season",
            "insufficient_evidence"
        ])
    ]

    df = df.dropna(
        subset=["future_suitability_score", "crop_id"]
    )

    df = df[~df["crop_id"].isin(PERENNIALS)]

    # Aggregate individual location evaluations into each cell.
    grouped = (
        df.groupby([
            "grid_cell", "season", "crop_id",
            "crop_name", "family", "crop_group"
        ])
        .agg(
            suitability=("future_suitability_score", "mean"),
            evidence=("evidence_coverage", "mean"),
            location_count=("location_id", "nunique")
        )
        .reset_index()
    )

    return grouped


def build_pools(grouped, cell=None):
    if cell is None:
        data = grouped.copy()
    else:
        data = grouped[grouped["grid_cell"] == cell].copy()

    pools = {}

    for season in SEASONS:
        subset = data[data["season"].str.lower() == season]

        subset = subset.sort_values(
            ["suitability", "evidence"],
            ascending=False
        ).head(CANDIDATES_PER_SEASON)

        pools[season] = subset.to_dict("records")

    return pools


def get_crop_record(pools, season, crop_id):
    for crop in pools[season]:
        if crop["crop_id"] == crop_id:
            return crop

    raise KeyError(f"Crop {crop_id} missing from {season} pool")


def rotation_penalty(sequence, pools):
    penalty = 0.0

    for i in range(len(sequence) - 1):
        first = get_crop_record(
            pools, SEASONS[i], sequence[i]
        )
        second = get_crop_record(
            pools, SEASONS[i + 1], sequence[i + 1]
        )

        # Same botanical family in consecutive seasons.
        if first["family"] == second["family"]:
            penalty += 20

        # Repeated crop group.
        if first["crop_group"] == second["crop_group"]:
            penalty += 12

        # Consecutive cereals.
        if (
            sequence[i] in CEREALS
            and sequence[i + 1] in CEREALS
        ):
            penalty += 15

        # Legume following a non-legume.
        if (
            sequence[i] not in LEGUMES
            and sequence[i + 1] in LEGUMES
        ):
            penalty = max(0, penalty - 8)

    return penalty


def evaluate(sequence, pools):
    records = [
        get_crop_record(pools, season, crop_id)
        for season, crop_id in zip(SEASONS, sequence)
    ]

    suitability = float(np.mean([
        crop["suitability"] for crop in records
    ]))

    evidence = float(np.mean([
        crop["evidence"] for crop in records
    ]))

    penalty = rotation_penalty(sequence, pools)

    compatibility = max(
        0, min(100, 100 - penalty)
    )

    score = (
        suitability * 0.75
        + compatibility * 0.25
    )

    return {
        "score": round(score, 3),
        "suitability": round(suitability, 3),
        "compatibility": round(compatibility, 3),
        "penalty": round(penalty, 3),
        "evidence": round(evidence, 3),
        "records": records
    }


def random_chromosome(pools):
    return tuple(
        random.choice(pools[season])["crop_id"]
        for season in SEASONS
    )


def tournament(population, pools):
    contestants = random.sample(
        population,
        min(4, len(population))
    )

    return max(
        contestants,
        key=lambda item: evaluate(item, pools)["score"]
    )


def crossover(parent_a, parent_b):
    # One crossover point preserves the seasonal order.
    point = random.randint(1, 2)

    return (
        parent_a[:point] + parent_b[point:],
        parent_b[:point] + parent_a[point:]
    )


def mutate(chromosome, pools):
    result = list(chromosome)

    for index, season in enumerate(SEASONS):
        if random.random() < MUTATION_RATE:
            choices = [
                crop["crop_id"]
                for crop in pools[season]
                if crop["crop_id"] != result[index]
            ]

            if choices:
                result[index] = random.choice(choices)

    return tuple(result)


def run_genetic_algorithm(pools, cell_name):
    if any(not pools[season] for season in SEASONS):
        missing = [
            season for season in SEASONS
            if not pools[season]
        ]
        print(f"Skipping {cell_name}: no candidates for {missing}")
        return [], []

    random.seed(RANDOM_SEED + sum(map(ord, cell_name)))

    population = [
        random_chromosome(pools)
        for _ in range(POPULATION_SIZE)
    ]

    history = []
    all_evaluated = set()

    for generation in range(GENERATIONS):
        scored = sorted(
            population,
            key=lambda chromosome: evaluate(
                chromosome, pools
            )["score"],
            reverse=True
        )

        for chromosome in scored:
            all_evaluated.add(chromosome)

        best = evaluate(scored[0], pools)

        history.append({
            "target": cell_name,
            "generation": generation + 1,
            "best_fitness": best["score"],
            "mean_fitness": round(
                float(np.mean([
                    evaluate(c, pools)["score"]
                    for c in population
                ])), 3
            ),
            "population_size": len(population),
            "unique_sequences": len(set(population)),
            "best_sequence": list(scored[0])
        })

        elites = scored[:ELITE_COUNT]
        new_population = list(elites)

        while len(new_population) < POPULATION_SIZE:
            parent_a = tournament(population, pools)
            parent_b = tournament(population, pools)

            child_a, child_b = crossover(
                parent_a, parent_b
            )

            child_a = mutate(child_a, pools)
            child_b = mutate(child_b, pools)

            new_population.append(child_a)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child_b)

        population = new_population

    # Include final population.
    all_evaluated.update(population)

    # Keep the highest-scoring distinct sequences.
    ranked = sorted(
        all_evaluated,
        key=lambda chromosome: evaluate(
            chromosome, pools
        )["score"],
        reverse=True
    )

    selected = []
    used_crops = set()

    for chromosome in ranked:
        if chromosome in selected:
            continue

        result = evaluate(chromosome, pools)

        # Ensure the alternatives are not mere permutations
        # with almost identical crop composition.
        if any(
            len(set(chromosome) & set(existing)) == 3
            for existing in selected
        ):
            continue

        selected.append(chromosome)

        if len(selected) == 3:
            break

    # If fewer than three diverse candidates exist,
    # fill with the next distinct valid sequences.
    for chromosome in ranked:
        if len(selected) >= 3:
            break
        if chromosome not in selected:
            selected.append(chromosome)

    return selected, history


def make_explanation(result):
    records = result["records"]

    parts = []

    for season, crop in zip(SEASONS, records):
        parts.append(
            f"{season.title()}: {crop['crop_name']} "
            f"(suitability {crop['suitability']:.1f}/100)"
        )

    parts.append(
        f"Rotation compatibility: {result['compatibility']:.1f}/100"
    )

    parts.append(
        f"Mean evidence coverage: {result['evidence'] * 100:.1f}%"
    )

    return " | ".join(parts)


def main():
    print("=" * 65)
    print("AERIS | SRO-01 SPATIAL ROTATION OPTIMIZER")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suitability, assignments, environment = load_inputs()

    suitability = assign_future_locations(
        suitability, assignments
    )

    grouped = prepare_candidates(suitability)

    results = []
    history = []

    targets = GRID_CELLS + ["NASHIK_REGION"]

    for target in targets:
        print(f"\nOptimizing: {target}")

        pools = build_pools(
            grouped,
            None if target == "NASHIK_REGION" else target
        )

        candidates, generations = run_genetic_algorithm(
            pools, target
        )

        history.extend(generations)

        for rank, chromosome in enumerate(candidates, start=1):
            result = evaluate(chromosome, pools)
            records = result.pop("records")

            result.update({
                "target": target,
                "rank": rank,
                "rotation": " -> ".join(
                    crop["crop_name"] for crop in records
                ),
                "crop_ids": list(chromosome),
                "crop_names": [
                    crop["crop_name"] for crop in records
                ],
                "seasonal_suitability": [
                    round(crop["suitability"], 2)
                    for crop in records
                ],
                "explanation": make_explanation({
                    **result, "records": records
                })
            })

            results.append(result)

        print(f"  Alternatives generated: {len(candidates)}")

    results_df = pd.DataFrame(results)

    results_path = OUTPUT_DIR / "rotation_optimizer_candidates.csv"
    history_path = OUTPUT_DIR / "optimization_generation_history.json"
    report_path = OUTPUT_DIR / "rotation_optimizer_report.json"

    results_df.to_csv(results_path, index=False)

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    report = {
        "engine": "SRO-01",
        "name": "Spatial Rotation Optimizer",
        "method": "Genetic algorithm",
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "mutation_rate": MUTATION_RATE,
        "targets": targets,
        "candidate_count": len(results_df),
        "environmental_profiles_loaded": len(environment),
        "spatial_samples": len(assignments),
        "future_suitability_rows": len(suitability),
        "outputs": {
            "candidates": str(results_path),
            "generation_history": str(history_path)
        },
        "limitations": [
            "This is a genetic algorithm, not a deep-learning model.",
            "Fitness is based on heuristic crop suitability and rotation rules.",
            "It does not predict yield or farm profitability.",
            "Cell suitability is aggregated from sampled locations.",
            "Rainfall inputs are shared Nashik city climate data.",
            "Three alternatives are optimization candidates, not field-validated prescriptions."
        ]
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n========== OPTIMIZER REPORT ==========")
    print(f"Targets processed: {len(targets)}")
    print(f"Candidate rotations: {len(results_df)}")

    for target in targets:
        subset = results_df[results_df["target"] == target]

        print(f"\n{target}")

        if subset.empty:
            print("No complete rotation alternatives available.")
            continue

        for _, row in subset.iterrows():
            print(
                f"  #{int(row['rank'])} "
                f"{row['rotation']} "
                f"| {row['score']:.2f}/100"
            )

    print("\nSaved:")
    print(results_path)
    print(history_path)
    print(report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
