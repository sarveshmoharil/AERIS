"""
AERIS | RIE-01
Rainfall Intelligence Engine
Version 1.0

Input:
    data/processed/nashik_nasa_power_daily.csv

Outputs:
    historical_seasonal_rainfall.csv
    seasonal_rainfall_forecast.csv
    historical_dry_spells.csv
    historical_monsoon_timing.csv
    rainfall_backtest.csv
    rainfall_prediction_report.json

Forecasting method:
    Recent climatology + restrained linear trend + bootstrap uncertainty.

Note:
    This is a statistical outlook, not a live weather forecast.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "aeris" / "data" / "processed"
    / "nashik_nasa_power_daily.csv"
)

OUTPUT = (
    ROOT / "aeris" / "data" / "processed"
    / "rainfall_prediction"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2027
RANDOM_SEED = 42
SIMULATIONS = 10000

RAIN = "PRECTOTCORR"

SEASONS = {
    "kharif": [6, 7, 8, 9],
    "post_monsoon": [10, 11],
    "winter": [12, 1, 2],
    "summer": [3, 4, 5],
}


# ---------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[RAIN] = pd.to_numeric(df[RAIN], errors="coerce")

    df = df.dropna(subset=["date", RAIN])
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date")

    df[RAIN] = df[RAIN].clip(lower=0)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# 2. BUILD COMPLETE SEASONAL RAINFALL HISTORY
# ---------------------------------------------------------

def get_seasonal_year(df, season):

    result = df[df["month"].isin(SEASONS[season])].copy()

    if season == "winter":
        result["season_year"] = np.where(
            result["month"] == 12,
            result["year"] + 1,
            result["year"]
        )
    else:
        result["season_year"] = result["year"]

    return result


def build_seasonal_history(df):

    records = []

    for season in SEASONS:

        seasonal = get_seasonal_year(df, season)

        for year, group in seasonal.groupby("season_year"):

            year = int(year)

            if season == "winter":
                start = pd.Timestamp(year - 1, 12, 1)
                end = pd.Timestamp(year, 2, 1) + pd.offsets.MonthEnd(0)

            else:
                months = SEASONS[season]
                start = pd.Timestamp(year, min(months), 1)
                end = (
                    pd.Timestamp(year, max(months), 1)
                    + pd.offsets.MonthEnd(0)
                )

            expected_days = len(
                pd.date_range(start, end, freq="D")
            )

            actual_days = group["date"].dt.normalize().nunique()

            if actual_days < expected_days * 0.95:
                continue

            records.append({
                "year": year,
                "season": season,
                "rainfall_mm": float(group[RAIN].sum()),
                "days_observed": int(actual_days),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 3. RAINFALL FORECAST MODEL
# ---------------------------------------------------------

def forecast_season(values, seed):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 8:
        raise ValueError(
            "At least 8 complete historical seasons are required."
        )

    recent = values[-min(15, len(values)):]

    recent_mean = float(np.mean(recent))

    x = np.arange(len(values), dtype=float)

    slope, intercept = np.polyfit(x, values, 1)

    trend_value = float(intercept + slope * len(values))

    # Limit extreme extrapolation.
    trend_value = np.clip(
        trend_value,
        recent_mean * 0.70,
        recent_mean * 1.30
    )

    prediction = (
        0.70 * recent_mean
        + 0.30 * trend_value
    )

    fitted = intercept + slope * x
    residuals = values - fitted

    rng = np.random.default_rng(seed)

    sampled_residuals = rng.choice(
        residuals,
        size=SIMULATIONS,
        replace=True
    )

    simulations = np.maximum(
        0,
        trend_value + sampled_residuals
    )

    simulations = (
        0.70 * simulations
        + 0.30 * recent_mean
    )

    return {
        "forecast_mm": round(float(prediction), 2),
        "lower_80_mm": round(
            float(np.quantile(simulations, 0.10)), 2
        ),
        "upper_80_mm": round(
            float(np.quantile(simulations, 0.90)), 2
        ),
        "historical_mean_mm": round(
            float(np.mean(values)), 2
        ),
        "recent_15_year_mean_mm": round(
            recent_mean, 2
        ),
        "trend_component_mm": round(
            float(trend_value), 2
        ),
        "historical_observations": int(len(values)),
    }


def build_forecasts(history):

    forecasts = []

    for index, season in enumerate(SEASONS):

        subset = history[
            history["season"] == season
        ].sort_values("year")

        result = forecast_season(
            subset["rainfall_mm"].values,
            RANDOM_SEED + index
        )

        forecasts.append({
            "target_year": TARGET_YEAR,
            "season": season,
            **result,
        })

    return pd.DataFrame(forecasts)


# ---------------------------------------------------------
# 4. DRY-SPELL ANALYSIS
# ---------------------------------------------------------

def longest_dry_spell(values):

    longest = 0
    current = 0

    for value in values:

        if value < 1.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def analyse_dry_spells(df):

    monsoon = df[
        df["month"].isin([6, 7, 8, 9])
    ].copy()

    records = []

    for year, group in monsoon.groupby("year"):

        start = pd.Timestamp(int(year), 6, 1)
        end = pd.Timestamp(int(year), 9, 30)

        daily = (
            group.set_index("date")[RAIN]
            .resample("D")
            .sum()
            .reindex(pd.date_range(start, end))
        )

        if daily.notna().sum() < len(daily) * 0.95:
            continue

        longest = longest_dry_spell(
            daily.fillna(0).values
        )

        records.append({
            "year": int(year),
            "maximum_dry_spell_days": int(longest),
            "dry_spell_5_days": bool(longest >= 5),
            "dry_spell_7_days": bool(longest >= 7),
            "dry_spell_10_days": bool(longest >= 10),
        })

    result = pd.DataFrame(records)

    if result.empty:
        return result, {}

    summary = {
        "years_analysed": int(len(result)),
        "probability_5_day_dry_spell": round(
            float(result["dry_spell_5_days"].mean()), 3
        ),
        "probability_7_day_dry_spell": round(
            float(result["dry_spell_7_days"].mean()), 3
        ),
        "probability_10_day_dry_spell": round(
            float(result["dry_spell_10_days"].mean()), 3
        ),
        "average_maximum_dry_spell_days": round(
            float(result["maximum_dry_spell_days"].mean()), 2
        ),
    }

    return result, summary


# ---------------------------------------------------------
# 5. MONSOON ONSET AND WITHDRAWAL ANALYSIS
# ---------------------------------------------------------

def analyse_monsoon_timing(df):

    records = []

    for year in sorted(df["year"].unique()):

        year = int(year)

        group = df[
            (df["year"] == year)
            & (df["month"].isin([6, 7, 8, 9, 10, 11]))
        ]

        daily = (
            group.set_index("date")[RAIN]
            .resample("D")
            .sum()
        )

        daily = daily.reindex(
            pd.date_range(
                f"{year}-06-01",
                f"{year}-11-15",
                freq="D"
            )
        )

        if daily.notna().mean() < 0.95:
            continue

        daily = daily.fillna(0)

        # Onset: first 3-day rainfall total >=20 mm.
        onset_period = daily.loc[
            f"{year}-06-01":f"{year}-07-31"
        ]

        onset_rolling = onset_period.rolling(3).sum()

        onset_candidates = onset_rolling[
            onset_rolling >= 20
        ]

        onset = None

        if not onset_candidates.empty:
            onset = onset_candidates.index[0] - pd.Timedelta(days=2)

        # Withdrawal proxy: first 10-day total <10 mm
        # after September 15.
        withdrawal_period = daily.loc[
            f"{year}-09-15":f"{year}-11-15"
        ]

        withdrawal_rolling = withdrawal_period.rolling(10).sum()

        withdrawal_candidates = withdrawal_rolling[
            withdrawal_rolling < 10
        ]

        withdrawal = None

        if not withdrawal_candidates.empty:
            withdrawal = (
                withdrawal_candidates.index[0]
                - pd.Timedelta(days=9)
            )

        records.append({
            "year": year,
            "onset_date": (
                onset.strftime("%Y-%m-%d")
                if onset is not None else None
            ),
            "withdrawal_proxy_date": (
                withdrawal.strftime("%Y-%m-%d")
                if withdrawal is not None else None
            ),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------
# 6. WALK-FORWARD BACKTESTING
# ---------------------------------------------------------

def run_backtest(history):

    records = []

    for season in SEASONS:

        subset = history[
            history["season"] == season
        ].sort_values("year")

        values = subset["rainfall_mm"].values
        years = subset["year"].values

        for i in range(10, len(values)):

            training = values[max(0, i - 15):i]

            prediction = float(np.mean(training))
            actual = float(values[i])

            records.append({
                "year": int(years[i]),
                "season": season,
                "actual_mm": actual,
                "predicted_mm": round(prediction, 2),
                "absolute_error_mm": round(
                    abs(actual - prediction), 2
                ),
            })

    result = pd.DataFrame(records)

    if result.empty:
        return result, {}

    summary = {
        "test_observations": int(len(result)),
        "mean_absolute_error_mm": round(
            float(result["absolute_error_mm"].mean()), 2
        ),
        "baseline": "Trailing 15-season historical average",
    }

    return result, summary


# ---------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("AERIS | RIE-01 RAINFALL INTELLIGENCE ENGINE")
    print("=" * 60)

    df = load_data()

    print(f"Observations: {len(df)}")
    print(f"Coverage: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Target year: {TARGET_YEAR}")

    # Historical seasonal rainfall.
    history = build_seasonal_history(df)

    history.to_csv(
        OUTPUT / "historical_seasonal_rainfall.csv",
        index=False
    )

    # Forecast.
    forecasts = build_forecasts(history)

    forecasts.to_csv(
        OUTPUT / "seasonal_rainfall_forecast.csv",
        index=False
    )

    # Dry spells.
    dry_history, dry_summary = analyse_dry_spells(df)

    dry_history.to_csv(
        OUTPUT / "historical_dry_spells.csv",
        index=False
    )

    # Monsoon timing.
    timing = analyse_monsoon_timing(df)

    timing.to_csv(
        OUTPUT / "historical_monsoon_timing.csv",
        index=False
    )

    # Backtest.
    backtest, backtest_summary = run_backtest(history)

    backtest.to_csv(
        OUTPUT / "rainfall_backtest.csv",
        index=False
    )

    # Historical timing summaries.
    onset = pd.to_datetime(
        timing["onset_date"], errors="coerce"
    ).dropna()

    withdrawal = pd.to_datetime(
        timing["withdrawal_proxy_date"], errors="coerce"
    ).dropna()

    timing_summary = {
        "median_onset_day_of_year": (
            int(onset.dt.dayofyear.median())
            if len(onset) else None
        ),
        "median_withdrawal_day_of_year": (
            int(withdrawal.dt.dayofyear.median())
            if len(withdrawal) else None
        ),
        "onset_rule": "First 3-day rainfall total >=20 mm between June 1 and July 31.",
        "withdrawal_rule": "First 10-day rainfall total <10 mm after September 15.",
        "note": "Historical rainfall-based indicators, not future date predictions.",
    }

    # Report.
    report = {
        "engine": "AERIS RIE-01",
        "version": "1.0",
        "target_year": TARGET_YEAR,
        "data_source": "NASA POWER",
        "data_start": str(df["date"].min().date()),
        "data_end": str(df["date"].max().date()),
        "spatial_scope": "Nashik city climate point, shared across locations.",
        "forecast_method": (
            "70% recent 15-season climatology + "
            "30% restrained linear trend, with bootstrap uncertainty."
        ),
        "forecast_type": "Statistical seasonal outlook, not live weather forecast.",
        "seasonal_forecasts": forecasts.to_dict(orient="records"),
        "dry_spell_analysis": dry_summary,
        "monsoon_timing": timing_summary,
        "backtesting": backtest_summary,
        "limitations": [
            "Historical statistical outlook, not a live meteorological forecast.",
            "Bootstrap intervals represent historical variability, not guaranteed bounds.",
            "Climate data represents one Nashik point.",
            "Onset and withdrawal are simplified rainfall-based indicators.",
        ],
    }

    with open(
        OUTPUT / "rainfall_prediction_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(report, f, indent=4)

    # Console output.
    print("\n========== SEASONAL RAINFALL OUTLOOK ==========")

    for _, row in forecasts.iterrows():

        print(
            f"{row['season']:15} "
            f"{row['forecast_mm']:8.1f} mm "
            f"[{row['lower_80_mm']:.1f} - "
            f"{row['upper_80_mm']:.1f} mm]"
        )

    print("\n========== DRY-SPELL RISK ==========")

    for key, value in dry_summary.items():
        print(f"{key}: {value}")

    print("\n========== BACKTEST ==========")

    for key, value in backtest_summary.items():
        print(f"{key}: {value}")

    print("\n========== OUTPUTS ==========")
    print(OUTPUT)
    print("\nRIE-01 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()