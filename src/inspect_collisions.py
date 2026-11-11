
import pandas as pd

PATH = "data/raw/manhattan_collisions.csv"

df = pd.read_csv(PATH)

# Convert date and coordinates
df["crash_date"] = pd.to_datetime(
    df["crash_date"], errors="coerce"
)

df["latitude"] = pd.to_numeric(
    df["latitude"], errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"], errors="coerce"
)

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\n--- DATE COVERAGE ---")
print("Earliest:", df["crash_date"].min())
print("Latest:", df["crash_date"].max())

print("\n--- DATA QUALITY ---")
print("Missing dates:", df["crash_date"].isna().sum())
print("Duplicate collision IDs:", df["collision_id"].duplicated().sum())

missing_coords = (
    df["latitude"].isna() | df["longitude"].isna()
)

print("Missing coordinate pairs:", missing_coords.sum())
print("Missing coordinate %:", round(missing_coords.mean() * 100, 2))

invalid_coords = (
    ~df["latitude"].between(40.5, 41.0)
    | ~df["longitude"].between(-74.3, -73.6)
)

print(
    "Coordinates outside broad NYC bounds:",
    (invalid_coords & ~missing_coords).sum()
)

print("\n--- COLLISIONS BY YEAR ---")
print(
    df["crash_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\n--- CHECKS ---")
assert len(df) > 0
assert df["collision_id"].is_unique
assert df["crash_date"].notna().all()
assert df["crash_date"].ge(pd.Timestamp("2019-01-01")).all()
assert df["crash_date"].lt(pd.Timestamp("2026-01-01")).all()
assert df["borough"].eq("MANHATTAN").all()

print("Initial integrity checks passed.")
