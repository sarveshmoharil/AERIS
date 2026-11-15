
"""
AERIS v1.0
NASA SMAP Granule Manifest Builder

Inspects all discovered granules.
Does not download HDF5 files.
Identifies duplicate observation timestamps.
"""

import earthaccess
import logging
import json
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("AERIS.SMAP")

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "aeris" / "data" / "metadata"
OUTPUT.mkdir(parents=True, exist_ok=True)

BBOX = (73.5, 19.5, 74.1, 20.5)

START = "2025-10-01"
END = "2025-10-02"


def main():

    print("\nAERIS | SMAP GRANULE MANIFEST")

    auth = earthaccess.login(
        strategy="interactive",
        persist=True
    )

    if not auth.authenticated:
        raise RuntimeError("Authentication failed")

    granules = earthaccess.search_data(
        short_name="SPL4SMGP",
        version="008",
        temporal=(START, END),
        bounding_box=BBOX,
        count=100
    )

    print(f"\nDiscovered: {len(granules)} granules")

    records = []

    for granule in granules:

        metadata = dict(granule)

        umm = metadata.get("umm", {})

        granule_name = umm.get("GranuleUR", "")

        temporal = umm.get(
            "TemporalExtent", {}
        ).get("RangeDateTime", {})

        beginning = temporal.get(
            "BeginningDateTime"
        )

        ending = temporal.get(
            "EndingDateTime"
        )

        distribution = umm.get(
            "DataGranule", {}
        ).get(
            "ArchiveAndDistributionInformation", []
        )

        hdf_size = next(
            (
                item.get("SizeInBytes")
                for item in distribution
                if item.get("Name", "").endswith(".h5")
            ),
            None
        )

        processing = re.search(
            r"_Vv(\d+)_",
            granule_name
        )

        processing_version = (
            processing.group(1)
            if processing else "unknown"
        )

        records.append({
            "granule": granule_name,
            "beginning": beginning,
            "ending": ending,
            "processing_version": processing_version,
            "size_MB": round(
                hdf_size / (1024 ** 2), 2
            ) if hdf_size else None,
            "download_url": next(
                (
                    item["URL"]
                    for item in umm.get("RelatedUrls", [])
                    if item.get("Type") == "GET DATA"
                    and item["URL"].endswith(".h5")
                ),
                None
            )
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No granules available")

    df["beginning"] = pd.to_datetime(
        df["beginning"], utc=True
    )

    df.sort_values(
        ["beginning", "processing_version"],
        ascending=[True, False],
        inplace=True
    )

    df["duplicate_timestamp"] = df.duplicated(
        subset=["beginning"],
        keep=False
    )

    csv_path = OUTPUT / "smap_granule_manifest.csv"
    json_path = OUTPUT / "smap_granule_manifest.json"

    df.to_csv(csv_path, index=False)
    df.to_json(
        json_path,
        orient="records",
        indent=4,
        date_format="iso"
    )

    print("\nGRANULE MANIFEST")
    print("-" * 100)

    print(
        df[[
            "granule",
            "beginning",
            "processing_version",
            "size_MB",
            "duplicate_timestamp"
        ]].to_string(index=False)
    )

    print("\nSUMMARY")
    print("-" * 40)
    print(f"Total granules: {len(df)}")
    print(f"Unique timestamps: {df['beginning'].nunique()}")
    print(f"Duplicate entries: {df['duplicate_timestamp'].sum()}")
    print(
        f"Total archive size: "
        f"{df['size_MB'].sum():.2f} MB"
    )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    print("\nMANIFEST COMPLETED")


if __name__ == "__main__":
    main()
