"""
AERIS District Crop & Rotation Engine (DCR-01)
Runs crop screening and an independent genetic algorithm for each district-grid cell.

Inputs:
  data/processed/district_grid/nashik_district_grid_environmental_profiles.csv
  data/processed/rainfall_prediction/seasonal_rainfall_forecast.csv (optional)
  aeris/agronomy/crop_knowledge_base.py

Outputs:
  data/processed/district_grid/crop_suitability_district.csv
  data/processed/district_grid/rotation_candidates_district.csv
  data/processed/district_grid/ga_history_district.json
  data/processed/district_grid/district_crop_rotation_report.json

Scores are transparent screening/optimization scores, NOT yield or profit predictions.
SoilGrids raw scale handling follows the existing AERIS conventions:
  pH raw values >14 are divided by 10; sand/clay values >100 are divided by 10.
Verify source metadata/scale factors before treating these conversions as laboratory values.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "aeris" / "agronomy"))
from crop_knowledge_base import CROPS, validate_database

DATA = ROOT / "aeris" / "data" / "processed"
GRID_DIR = DATA / "district_grid"
PROFILE_FILE = GRID_DIR / "nashik_district_grid_environmental_profiles.csv"
FORECAST_FILE = DATA / "rainfall_prediction" / "seasonal_rainfall_forecast.csv"
RAIN_REPORT_FILE = DATA / "rainfall_prediction" / "rainfall_prediction_report.json"

SUITABILITY_OUT = GRID_DIR / "crop_suitability_district.csv"
CANDIDATES_OUT = GRID_DIR / "rotation_candidates_district.csv"
HISTORY_OUT = GRID_DIR / "ga_history_district.json"
REPORT_OUT = GRID_DIR / "district_crop_rotation_report.json"

SEASONS = ("kharif", "rabi", "summer")
WEIGHTS = {"soil_ph": 45.0, "soil_texture": 35.0, "slope": 20.0}
POPULATION = 80
GENERATIONS = 60
ELITE = 8
TOP_PER_SEASON = 8
TOP_ROTATIONS = 10
MUTATION_RATE = 0.22
CROSSOVER_RATE = 0.75
SUITABILITY_WEIGHT = 0.75
COMPATIBILITY_WEIGHT = 0.25
WATER_DEMAND = {"low":1,"low_medium":2,"medium":3,"medium_high":4,"high":5,"very_high":6}

def finite(row, key):
    try:
        v = float(row.get(key, np.nan))
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def coverage(row, key):
    v = finite(row, key)
    return float(np.clip(v / 100.0, 0, 1)) if v is not None else 0.0

def pH_value(row):
    v = finite(row, "phh2o_0-5cm_mean")
    if v is None: return None
    if v > 14: v /= 10.0
    return v if 0 <= v <= 14 else None

def texture_values(row):
    sand, clay = finite(row, "sand_0-5cm_mean"), finite(row, "clay_0-5cm_mean")
    if sand is not None and sand > 100: sand /= 10.0
    if clay is not None and clay > 100: clay /= 10.0
    if sand is None or clay is None or sand < 0 or clay < 0 or sand + clay > 100:
        return None, None, None
    silt = 100.0 - sand - clay
    if clay >= 40: texture = "clay"
    elif clay >= 27: texture = "clay_loam"
    elif sand >= 70 or (sand >= 43 and clay < 20): texture = "sandy_loam"
    elif clay >= 20: texture = "clay_loam"
    elif silt >= 50: texture = "loam"
    else: texture = "loam"
    return sand, clay, texture

def range_score(value, acceptable, tolerance):
    if value is None: return None
    lo, hi = acceptable
    if lo <= value <= hi: return 100.0
    distance = lo - value if value < lo else value - hi
    return float(max(0.0, 100.0 * (1.0 - distance / tolerance)))

def evaluate_cell_crop(row, cell, crop_id, crop, season, rainfall_context):
    ph = pH_value(row)
    sand, clay, texture = texture_values(row)
    slope = finite(row, "slope_degrees_mean")
    ph_score = range_score(ph, crop["soil_ph"], 1.5)
    texture_score = None
    if texture is not None:
        if texture in crop["soil_texture"]: texture_score = 100.0
        elif texture == "loam" or "loam" in crop["soil_texture"]: texture_score = 65.0
        else: texture_score = 30.0
    slope_score = None
    if slope is not None:
        slope_score = 100.0 if slope <= 3 else 85.0 if slope <= 8 else 60.0 if slope <= 15 else 30.0 if slope <= 25 else 10.0

    vals = {"soil_ph": ph_score, "soil_texture": texture_score, "slope": slope_score}
    weighted, available = 0.0, 0.0
    for factor, score in vals.items():
        if score is not None:
            weighted += score * WEIGHTS[factor]
            available += WEIGHTS[factor]
    base = weighted / available if available else np.nan

    # Evidence coverage measures spatial raster evidence only; shared climate is excluded.
    spatial_cov = (
        WEIGHTS["soil_ph"] * coverage(row, "phh2o_0-5cm_coverage_pct")
        + WEIGHTS["soil_texture"] * min(
            coverage(row, "sand_0-5cm_coverage_pct"),
            coverage(row, "clay_0-5cm_coverage_pct"))
        + WEIGHTS["slope"] * coverage(row, "slope_degrees_coverage_pct")
    ) / sum(WEIGHTS.values())

    rainfall_adjustment = 0.0
    rainfall_note = "No district-cell rainfall forecast adjustment applied."
    if season in rainfall_context:
        ctx = rainfall_context[season]
        demand = WATER_DEMAND.get(crop.get("water_demand", "medium"), 3)
        anomaly = ctx["rainfall_ratio"] - 1.0
        rainfall_adjustment = float(np.clip(anomaly * (demand - 3) * 6.0, -6.0, 6.0))
        rainfall_note = (
            f"Shared Nashik-point {season} outlook {ctx['predicted_mm']:.1f} mm "
            f"vs historical seasonal mean {ctx['historical_mean_mm']:.1f} mm; "
            f"heuristic water-demand adjustment {rainfall_adjustment:+.1f}."
        )
    adjusted = float(np.clip(base + rainfall_adjustment, 0, 100)) if np.isfinite(base) else np.nan

    explicit_status = str(row.get("evidence_status", "UNKNOWN"))
    if (not np.isfinite(base) or spatial_cov < 0.40
            or explicit_status == "INSUFFICIENT_RASTER_EVIDENCE"):
        classification = "insufficient_evidence"
    elif spatial_cov < 0.80 or explicit_status == "PARTIAL_RASTER_EVIDENCE":
        classification = "provisional_partial_evidence"
    elif adjusted >= 80: classification = "high"
    elif adjusted >= 65: classification = "moderate"
    elif adjusted >= 45: classification = "marginal"
    else: classification = "low"

    if season not in crop["season"] and season != "perennial":
        classification = "out_of_season"
        adjusted = 0.0
        rainfall_adjustment = 0.0

    reasons, warnings = [], []
    if ph_score is not None:
        reasons.append(f"Surface pH screening value {ph:.2f}; crop preference {crop['soil_ph'][0]}–{crop['soil_ph'][1]}.")
    else: warnings.append("Surface soil pH unavailable.")
    if texture is not None:
        reasons.append(f"Estimated surface texture: {texture} (sand {sand:.1f}%, clay {clay:.1f}%).")
    else: warnings.append("Surface texture could not be inferred from sand/clay raster values.")
    if slope_score is not None:
        reasons.append(f"Mean terrain slope {slope:.2f}°; screening score {slope_score:.0f}/100.")
    else: warnings.append("Terrain slope unavailable.")
    if spatial_cov < 0.80:
        warnings.append(f"Spatial raster evidence coverage is {spatial_cov:.1%}; treat ranking as provisional.")
    warnings.append("Temperature and root-zone moisture are not available as cell-specific layers in this district profile.")
    warnings.append("This is a screening score, not yield, profitability, irrigation sufficiency, or a field prescription.")

    return {
        "cell_id": cell, "crop_id": crop_id, "crop_name": crop["name"],
        "crop_group": crop["crop_group"], "family": crop["family"],
        "season": season, "suitability_score": round(adjusted,2) if np.isfinite(adjusted) else np.nan,
        "baseline_soil_terrain_score": round(base,2) if np.isfinite(base) else np.nan,
        "rainfall_adjustment": round(rainfall_adjustment,2),
        "evidence_coverage": round(float(spatial_cov),3),
        "evidence_status": str(row.get("evidence_status","UNKNOWN")),
        "classification": classification,
        "factor_scores_json": json.dumps(vals),
        "reasons": " | ".join(reasons),
        "warnings": " | ".join(warnings),
        "rainfall_context": rainfall_note,
        "soil_ph_value_used": ph, "estimated_texture": texture,
        "slope_degrees_mean": slope,
    }

def load_rainfall_context():
    if not FORECAST_FILE.exists():
        return {}, {"available": False, "reason": f"Forecast file not found: {FORECAST_FILE}"}
    df = pd.read_csv(FORECAST_FILE)
    required = {"season","forecast_mm","historical_mean_mm"}
    if not required.issubset(df.columns):
        return {}, {"available": False, "reason": "Forecast CSV lacks required columns."}
    d = df.set_index("season")
    missing = [s for s in ("kharif","post_monsoon","winter","summer") if s not in d.index]
    if missing:
        return {}, {"available": False, "reason": f"Missing RIE seasons: {missing}"}
    context = {}
    for season in ("kharif","summer"):
        pred, hist = float(d.loc[season,"forecast_mm"]), float(d.loc[season,"historical_mean_mm"])
        context[season] = {"predicted_mm":pred,"historical_mean_mm":hist,"rainfall_ratio":pred/hist if hist>0 else 1.0}
    pred = float(d.loc["post_monsoon","forecast_mm"]) + float(d.loc["winter","forecast_mm"])
    hist = float(d.loc["post_monsoon","historical_mean_mm"]) + float(d.loc["winter","historical_mean_mm"])
    context["rabi"] = {"predicted_mm":pred,"historical_mean_mm":hist,"rainfall_ratio":pred/hist if hist>0 else 1.0}
    return context, {"available":True,"source":"RIE-01 seasonal rainfall forecast","spatial_scope":"single Nashik climate point shared across all 15 cells"}

def pair_penalty(a, b):
    ca, cb = CROPS[a], CROPS[b]
    penalty, reasons = 0, []
    if ca["family"] == cb["family"]:
        penalty += 20; reasons.append("Same botanical family in consecutive seasons (+20).")
    if ca["rotation_group"] == cb["rotation_group"]:
        penalty += 12; reasons.append("Same rotation group in consecutive seasons (+12).")
    if ca["crop_group"] == "cereal" and cb["crop_group"] == "cereal":
        penalty += 15; reasons.append("Consecutive cereal crops (+15).")
    if cb.get("nitrogen_fixing") and not ca.get("nitrogen_fixing"):
        penalty -= 8; reasons.append("Legume follows non-legume (−8 bonus).")
    return max(0, penalty), reasons or ["No basic sequence penalty triggered."]

def evaluate_sequence(ids, lookup):
    rows = [lookup[(season, crop_id)] for season, crop_id in zip(SEASONS, ids)]
    scores = [float(r["suitability_score"]) for r in rows]
    penalties, reasons = 0.0, []
    for a,b in zip(ids[:-1], ids[1:]):
        p, why = pair_penalty(a,b); penalties += p; reasons.extend(why)
    avg_penalty = penalties / 2.0
    compatibility = float(np.clip(100.0 - avg_penalty * 2.0, 0, 100))
    suitability = float(np.mean(scores))
    fitness = suitability * SUITABILITY_WEIGHT + compatibility * COMPATIBILITY_WEIGHT
    return {
        "rotation": " -> ".join(CROPS[c]["name"] for c in ids),
        "crop_ids": list(ids), "crop_names": [CROPS[c]["name"] for c in ids],
        "seasonal_suitability": scores, "average_suitability": round(suitability,2),
        "total_rotation_penalty": round(penalties,2),
        "rotation_compatibility": round(compatibility,2),
        "fitness_score": round(fitness,2),
        "explanation": " | ".join(
            [f"{r['crop_name']} ({r['season']}): {r['suitability_score']:.1f}/100; {r['rainfall_context']}" for r in rows]
            + reasons
        ),
    }

def optimize_cell(cell, group, seed):
    if "evidence_status" in group.columns and group["evidence_status"].astype(str).eq("INSUFFICIENT_RASTER_EVIDENCE").all():
        return [], [], {"status":"BLOCKED_INSUFFICIENT_EVIDENCE","reason":"District-grid builder marked this cell as insufficient raster evidence; no rotation is generated."}
    season_pools = {}
    for season in SEASONS:
        eligible = group[
            (group.season == season)
            & (~group.classification.isin(["insufficient_evidence", "out_of_season"]))
            & group.suitability_score.notna()
        ].sort_values("suitability_score", ascending=False)
        eligible = eligible[eligible.crop_id.map(lambda c: CROPS[c]["crop_group"] != "fruit")]
        season_pools[season] = eligible.head(TOP_PER_SEASON).to_dict("records")
    if any(not season_pools[s] for s in SEASONS):
        return [], [], {"status":"NO_COMPLETE_ROTATION","reason":"At least one season has no crop with usable spatial evidence."}

    lookup = {(r["season"],r["crop_id"]):r for s in SEASONS for r in season_pools[s]}
    rng = np.random.default_rng(seed)
    pools = [[r["crop_id"] for r in season_pools[s]] for s in SEASONS]
    def random_individual(): return tuple(rng.choice(pool) for pool in pools)
    def fitness(ind): return evaluate_sequence(ind, lookup)["fitness_score"]

    population = list(dict.fromkeys(random_individual() for _ in range(POPULATION)))
    while len(population) < POPULATION: population.append(random_individual())
    history = []
    for gen in range(1, GENERATIONS+1):
        scored = sorted(((fitness(ind),ind) for ind in population), reverse=True, key=lambda x:x[0])
        unique = len(set(ind for _,ind in scored))
        history.append({
            "cell_id":cell,"generation":gen,
            "best_fitness":round(float(scored[0][0]),4),
            "mean_fitness":round(float(np.mean([s for s,_ in scored])),4),
            "unique_sequences":unique,"population_size":len(population),
            "best_sequence":" -> ".join(CROPS[c]["name"] for c in scored[0][1]),
        })
        next_pop = [ind for _,ind in scored[:ELITE]]
        while len(next_pop) < POPULATION:
            # Tournament selection
            def tournament():
                choices = rng.choice(len(scored), size=3, replace=False)
                return scored[int(max(choices, key=lambda i: scored[i][0]))][1]
            p1,p2 = tournament(), tournament()
            child = list(p1)
            if rng.random() < CROSSOVER_RATE:
                cut = int(rng.integers(1,3))
                child = list(p1[:cut] + p2[cut:])
            for j in range(3):
                if rng.random() < MUTATION_RATE:
                    child[j] = str(rng.choice(pools[j]))
            next_pop.append(tuple(child))
        population = next_pop

    # Exhaustive ranking of the bounded seasonal candidate pools ensures the displayed
    # shortlist is not dependent on the final stochastic population alone.
    import itertools
    all_sequences = []
    for ids in itertools.product(*pools):
        all_sequences.append(evaluate_sequence(ids, lookup))
    all_sequences.sort(key=lambda x:x["fitness_score"], reverse=True)
    seen, candidates = set(), []
    for item in all_sequences:
        key = tuple(item["crop_ids"])
        if key in seen: continue
        seen.add(key)
        candidates.append(item)
        if len(candidates) >= TOP_ROTATIONS: break
    cell_evidence = str(group["evidence_status"].iloc[0]) if "evidence_status" in group.columns else "UNKNOWN"
    cell_coverage = float(group["evidence_coverage"].iloc[0]) if "evidence_coverage" in group.columns else 0.0
    for rank, item in enumerate(candidates,1):
        item.update({"cell_id":cell,"rank":rank,"optimizer":"DCR-01","generations":GENERATIONS,
                     "population_size":POPULATION,"final_generation_best":history[-1]["best_fitness"],
                     "evidence_status":cell_evidence,"evidence_coverage":cell_coverage})
    return candidates, history, {"status":"COMPLETED","season_pool_sizes":{s:len(pools[i]) for i,s in enumerate(SEASONS)}}

def main():
    if not validate_database()["valid"]:
        raise RuntimeError(f"Crop KB invalid: {validate_database()['errors']}")
    if not PROFILE_FILE.exists(): raise FileNotFoundError(PROFILE_FILE)
    profiles = pd.read_csv(PROFILE_FILE)
    required = {"grid_cell","phh2o_0-5cm_mean","sand_0-5cm_mean","clay_0-5cm_mean","slope_degrees_mean"}
    missing = sorted(required - set(profiles.columns))
    if missing: raise ValueError(f"District profile missing required fields: {missing}")
    rainfall_context, rainfall_meta = load_rainfall_context()

    suitability = []
    for _, row in profiles.iterrows():
        cell = str(row["grid_cell"])
        for season in SEASONS:
            for crop_id, crop in CROPS.items():
                if crop["crop_group"] == "fruit":
                    continue
                suitability.append(evaluate_cell_crop(row,cell,crop_id,crop,season,rainfall_context))
        for crop_id, crop in CROPS.items():
            if crop["crop_group"] == "fruit":
                suitability.append(evaluate_cell_crop(row,cell,crop_id,crop,"perennial",rainfall_context))
    suit = pd.DataFrame(suitability)
    # Rank eligible annual crops separately by cell and season.
    suit["rank"] = np.nan
    for (cell,season), idx in suit.groupby(["cell_id","season"]).groups.items():
        eligible = suit.loc[idx].sort_values("suitability_score",ascending=False)
        eligible = eligible[~eligible.classification.isin(["insufficient_evidence", "out_of_season"])]
        suit.loc[eligible.index,"rank"] = np.arange(1,len(eligible)+1)
    suit.to_csv(SUITABILITY_OUT,index=False)

    candidates, history, cell_status = [], [], {}
    for i, (cell, group) in enumerate(suit.groupby("cell_id")):
        c,h,status = optimize_cell(cell,group,20261115+i*101)
        candidates.extend(c); history.extend(h); cell_status[cell]=status
    pd.DataFrame(candidates).to_csv(CANDIDATES_OUT,index=False)
    HISTORY_OUT.write_text(json.dumps(history,indent=2,ensure_ascii=False),encoding="utf-8")

    summary = {
        "engine":"AERIS District Crop & Rotation Engine","version":"1.0","optimizer":"DCR-01",
        "cells_processed":int(profiles.grid_cell.nunique()),"crop_count":len(CROPS),
        "suitability_evaluations":int(len(suit)),"rotation_candidates":len(candidates),
        "ga_population_size":POPULATION,"ga_generations":GENERATIONS,
        "fitness":{"suitability_weight":SUITABILITY_WEIGHT,"rotation_compatibility_weight":COMPATIBILITY_WEIGHT,
                   "formula":"0.75 * mean seasonal suitability + 0.25 * rotation compatibility"},
        "rainfall_context":rainfall_meta,
        "cell_status":cell_status,
        "evidence_status_counts":profiles.evidence_status.value_counts().to_dict() if "evidence_status" in profiles else {},
        "limitations":[
            "Scores are heuristic environmental screening scores, not yield or profit predictions.",
            "Cell-specific inputs are SoilGrids surface pH, sand/clay texture proxies, and SRTM slope.",
            "Temperature and root-zone moisture are not available as district-cell layers in the supplied profile CSV.",
            "Rainfall outlook, when present, is a single Nashik climate point shared across all cells; its crop adjustment is heuristic.",
            "SoilGrids raw scale factors and units require metadata verification; existing AERIS conversion conventions are used for pH/sand/clay.",
            "Management cells are research grid units, not administrative or soil survey zones.",
            "Low-evidence cells may have no complete rotation candidates; no values are fabricated to fill gaps.",
            "GA history records the evolutionary search; candidate shortlist is exhaustively ranked within the top seasonal candidate pools."
        ],
        "outputs":{"suitability":str(SUITABILITY_OUT),"candidates":str(CANDIDATES_OUT),"ga_history":str(HISTORY_OUT),"report":str(REPORT_OUT)}
    }
    REPORT_OUT.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("AERIS DCR-01 COMPLETE")
    print(f"Cells: {summary['cells_processed']} | Crops: {len(CROPS)} | Suitability rows: {len(suit)}")
    print(f"Rotation candidates: {len(candidates)} | GA records: {len(history)}")
    print("Cell results:")
    for cell,status in cell_status.items():
        print(f"  {cell}: {status['status']} | candidates={sum(1 for x in candidates if x['cell_id']==cell)}")
    print(f"Outputs: {GRID_DIR}")

if __name__ == "__main__":
    main()
