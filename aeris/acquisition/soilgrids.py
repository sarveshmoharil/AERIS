
from pathlib import Path
from datetime import datetime, timezone
import json
import requests
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "aeris" / "data" / "raw" / "soilgrids"
REPORT = ROOT / "aeris" / "data" / "metadata" / "soilgrids_report.json"

OUT.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

# Nashik-area geographic bounding box
WEST, SOUTH, EAST, NORTH = 73.5, 19.5, 74.1, 20.5

# SoilGrids native projection
SOIL_CRS = CRS.from_proj4("+proj=igh +datum=WGS84 +units=m +no_defs")
WCS_CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/152160"

transformer = Transformer.from_crs(
    "EPSG:4326", SOIL_CRS, always_xy=True
)

corners = [
    transformer.transform(WEST, SOUTH),
    transformer.transform(WEST, NORTH),
    transformer.transform(EAST, SOUTH),
    transformer.transform(EAST, NORTH),
]

MINX = min(p[0] for p in corners)
MAXX = max(p[0] for p in corners)
MINY = min(p[1] for p in corners)
MAXY = max(p[1] for p in corners)

BASE_URL = "https://maps.isric.org/mapserv"

PROPERTIES = {
    "phh2o": ["0-5cm", "5-15cm", "15-30cm"],
    "soc": ["0-5cm", "5-15cm", "15-30cm"],
    "clay": ["0-5cm", "5-15cm", "15-30cm"],
    "sand": ["0-5cm", "5-15cm", "15-30cm"],
    "nitrogen": ["0-5cm", "5-15cm", "15-30cm"],
    "cec": ["0-5cm", "5-15cm", "15-30cm"],
}


def download_layer(prop, depth):
    filename = f"{prop}_{depth}.tif"
    destination = OUT / filename

    if destination.exists() and destination.stat().st_size > 1000:
        print(f"[SKIP] {filename}")
        return {"file": filename, "status": "exists"}

    params = [
        ("map", f"/map/{prop}.map"),
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{prop}_{depth}_Q0.5"),
        ("SUBSETTINGCRS", WCS_CRS_URI),
        ("OUTPUTCRS", WCS_CRS_URI),
        ("SUBSET", f"X({MINX},{MAXX})"),
        ("SUBSET", f"Y({MINY},{MAXY})"),
        ("RESX", "250"),
        ("RESY", "250"),
        ("FORMAT", "image/tiff"),
    ]

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=240
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:600]}"
            )

        # GeoTIFF magic bytes: little-endian or big-endian TIFF
        if response.content[:4] not in (b"II*\x00", b"MM\x00*"):
            raise RuntimeError(
                f"Not a GeoTIFF: {response.text[:500]}"
            )

        temporary = destination.with_suffix(".part")
        temporary.write_bytes(response.content)
        temporary.replace(destination)

        size = destination.stat().st_size / 1024
        print(f"[OK] {filename} ({size:.1f} KB)")

        return {"file": filename, "status": "downloaded"}

    except Exception as exc:
        print(f"[FAIL] {filename}: {exc}")
        return {
            "file": filename,
            "status": "failed",
            "error": str(exc)
        }


def main():
    print("AERIS | SoilGrids acquisition")
    print(f"Projected bounds: {MINX:.2f}, {MINY:.2f}, "
          f"{MAXX:.2f}, {MAXY:.2f}")
    print()

    results = []

    for prop, depths in PROPERTIES.items():
        for depth in depths:
            results.append(download_layer(prop, depth))

    report = {
        "source": "ISRIC SoilGrids",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "geographic_bbox": [WEST, SOUTH, EAST, NORTH],
        "projected_bbox": [MINX, MINY, MAXX, MAXY],
        "projection": "EPSG:152160",
        "resolution_m": 250,
        "quantile": "Q0.5",
        "results": results,
        "downloaded": sum(x["status"] == "downloaded" for x in results),
        "existing": sum(x["status"] == "exists" for x in results),
        "failed": sum(x["status"] == "failed" for x in results),
    }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== REPORT ==========")
    print(f"Downloaded: {report['downloaded']}")
    print(f"Existing:   {report['existing']}")
    print(f"Failed:     {report['failed']}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
