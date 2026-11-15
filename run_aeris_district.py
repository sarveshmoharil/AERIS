"""
AERIS one-command district pipeline.
Run from E:\TrialShit:
    .\.venv\Scripts\python.exe .\run_aeris_district.py
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREDICTION = ROOT / "aeris" / "prediction"
DATA = ROOT / "aeris" / "data" / "processed"
FORECAST = DATA / "rainfall_prediction" / "seasonal_rainfall_forecast.csv"
DAILY_RAIN = DATA / "nashik_nasa_power_daily.csv"
RIE = PREDICTION / "rainfall_prediction_engine.py"
DCR = PREDICTION / "district_crop_rotation_engine.py"
UI = ROOT / "aeris" / "ui" / "scientist_workbench.py"

def run(script, label):
    if not script.exists():
        raise FileNotFoundError(f"{label} script missing: {script}")
    print(f"\n{'='*68}\n{label}\n{'='*68}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

def main():
    if not FORECAST.exists():
        if RIE.exists() and DAILY_RAIN.exists():
            run(RIE, "RIE-01 — Rainfall Intelligence")
        else:
            print(
                "NOTICE: No RIE-01 forecast was regenerated because the "
                "forecast CSV or NASA POWER daily input is absent. "
                "DCR-01 will run without rainfall adjustment; the UI will state this.",
                flush=True
            )
    run(DCR, "DCR-01 — District Crop Suitability + Per-cell Genetic Algorithm")
    run(UI, "AERIS Scientist Workbench")

if __name__ == "__main__":
    main()
