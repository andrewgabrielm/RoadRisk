
from pathlib import Path
from datetime import datetime, timezone
import json
import os

import pandas as pd
import requests

# 1. Configuration
API_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"

START_DATE = "2019-01-01"
END_DATE = "2026-01-01"  # Exclusive: includes all of 2025
PAGE_SIZE = 10000

COLUMNS = [
    "collision_id",
    "crash_date",
    "crash_time",
    "borough",
    "latitude",
    "longitude",
    "on_street_name",
]

OUTPUT = Path("data/raw/manhattan_collisions.csv")
METADATA = Path("data/raw/download_metadata.json")

DATE_FILTER = (
    f"crash_date >= '{START_DATE}T00:00:00' AND "
    f"crash_date < '{END_DATE}T00:00:00'"
)

WHERE = f"({DATE_FILTER}) AND borough = 'MANHATTAN'"

session = requests.Session()

# Optional: use an app token if you have one
token = os.getenv("NYC_OPEN_DATA_APP_TOKEN")
if token:
    session.headers.update({"X-App-Token": token})


def fetch(params):
    """Request data from NYC Open Data."""
    response = session.get(
        API_URL, params=params, timeout=120
    )
    response.raise_for_status()
    return response.json()


def count_records(where):
    """Count records matching a query."""
    result = fetch({
        "$select": "count(*) as n",
        "$where": where,
    })
    return int(result[0]["n"])


def download_collisions():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    expected = count_records(WHERE)
    print(f"Matching records: {expected:,}")

    if expected == 0:
        raise RuntimeError("No records found. Check the query.")

    downloaded = 0

    for offset in range(0, expected, PAGE_SIZE):
        records = fetch({
            "$select": ", ".join(COLUMNS),
            "$where": WHERE,
            "$order": "collision_id ASC",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        })

        if not records:
            raise RuntimeError("Download ended unexpectedly.")

        batch = pd.DataFrame(records).reindex(columns=COLUMNS)

        batch.to_csv(
            OUTPUT,
            mode="w" if offset == 0 else "a",
            header=(offset == 0),
            index=False,
        )

        downloaded += len(batch)
        print(f"Downloaded: {downloaded:,} / {expected:,}")

    if downloaded != expected:
        raise RuntimeError(
            "Record count changed during download. Please retry."
        )

    metadata = {
        "source": API_URL,
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "start_date": START_DATE,
        "end_date_exclusive": END_DATE,
        "where_filter": WHERE,
        "rows_downloaded": downloaded,
    }

    METADATA.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved to: {OUTPUT}")

    # Check for missing borough labels near Manhattan.
    # This box also contains areas outside Manhattan.
    missing_borough_query = (
        f"({DATE_FILTER}) AND borough IS NULL "
        "AND latitude BETWEEN 40.68 AND 40.90 "
        "AND longitude BETWEEN -74.05 AND -73.88"
    )

    candidates = count_records(missing_borough_query)
    print(f"Unlabelled records in broad area: {candidates:,}")
    print("These have not been confirmed as Manhattan records.")


if __name__ == "__main__":
    download_collisions()
