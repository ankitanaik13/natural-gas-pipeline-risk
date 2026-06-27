"""
Engineer all features for the ML model from cleaned incident data.
Run: python -m src.features.build_features
Output: data/processed/pipeline_features.csv
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

HIGH_INCIDENT_STATES = {"TX", "LA", "OK", "KS", "WV", "PA", "CA", "CO", "WY", "OH"}


# ── Temporal features ──────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["incident_year"] = pd.to_numeric(df["IYEAR"], errors="coerce")
    df["pipeline_age_at_incident"] = df["incident_year"] - pd.to_numeric(
        df["INSTALLATION_YEAR"], errors="coerce"
    )
    # Clip implausible ages
    df["pipeline_age_at_incident"] = df["pipeline_age_at_incident"].clip(0, 150)

    def decade(year):
        if pd.isna(year):
            return "UNKNOWN"
        y = int(year)
        return f"{(y // 10) * 10}s"

    df["decade_installed"] = df["INSTALLATION_YEAR"].apply(
        lambda y: decade(pd.to_numeric(y, errors="coerce"))
    )
    return df


# ── Pipeline characteristic features ──────────────────────────────────────────

def add_pipeline_features(df: pd.DataFrame) -> pd.DataFrame:
    # Diameter: PHMSA stores as strings like "12 INCH" or just numeric
    raw_diam = df["PIPE_DIAMETER"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
    df["pipe_diameter_numeric"] = pd.to_numeric(raw_diam, errors="coerce")

    mat = df["PIPE_MATERIAL"].fillna("").str.upper()
    df["is_steel"] = mat.str.contains("STEEL", na=False).astype(int)
    df["is_cast_iron"] = mat.str.contains("CAST IRON|CAST-IRON", na=False).astype(int)
    df["is_plastic"] = mat.str.contains("PLASTIC|PE |POLYETHYLENE|PVC", na=False).astype(int)
    return df


# ── Cause one-hot encoding ─────────────────────────────────────────────────────

CAUSE_CATEGORIES = [
    "CORROSION",
    "EXCAVATION DAMAGE",
    "NATURAL FORCE",
    "EQUIPMENT FAILURE",
    "INCORRECT OPERATION",
    "OTHER",
]


def add_cause_features(df: pd.DataFrame) -> pd.DataFrame:
    cause_col = "CAUSE_CLEAN" if "CAUSE_CLEAN" in df.columns else "CAUSE"
    for cat in CAUSE_CATEGORIES:
        col_name = "cause_" + cat.lower().replace(" ", "_")
        df[col_name] = (df[cause_col].str.upper() == cat).astype(int)
    return df


# ── Spatial features ───────────────────────────────────────────────────────────

def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute county-level incident history and state-level rate features."""

    # State-level flag
    df["STATE_UPPER"] = df["STATE"].astype(str).str.upper().str.strip()
    df["is_high_density_state"] = df["STATE_UPPER"].isin(HIGH_INCIDENT_STATES).astype(int)

    # State incident rate (incidents per 100 rows as a proxy; replace with
    # mileage from annual report if available)
    state_counts = df.groupby("STATE_UPPER").size().rename("_state_total")
    df = df.merge(state_counts, on="STATE_UPPER", how="left")
    df["state_incident_rate"] = (df["_state_total"] / df["_state_total"].sum()) * 100
    df.drop(columns=["_state_total"], inplace=True)

    # County incident history — requires county shapefile
    county_shp = RAW_DIR / "us_counties" / "cb_2025_us_county_500k" / "cb_2025_us_county_500k.shp"
    if county_shp.exists():
        print("County shapefile found — running spatial join...")
        gdf_incidents = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df["LOCATION_LONGITUDE"], df["LOCATION_LATITUDE"]
            ),
            crs="EPSG:4326",
        )
        counties = gpd.read_file(county_shp).to_crs("EPSG:4326")
        # Rename NAME → COUNTY_NAME before join to avoid conflict with
        # the operator NAME column already in the incidents dataframe.
        counties = counties[["GEOID", "NAME", "geometry"]].rename(
            columns={"NAME": "COUNTY_NAME", "GEOID": "COUNTY_GEOID"}
        )
        joined = gpd.sjoin(gdf_incidents, counties,
                           how="left", predicate="within")
        df["COUNTY_GEOID"] = joined["COUNTY_GEOID"].values
        df["COUNTY_NAME"] = joined["COUNTY_NAME"].values

        county_counts = (
            df.groupby("COUNTY_GEOID").size().rename("county_incident_history").reset_index()
        )
        df = df.merge(county_counts, on="COUNTY_GEOID", how="left")
        df["county_incident_history"] = df["county_incident_history"].fillna(0)
        print("Spatial join complete.")
    else:
        print(
            "County shapefile not found at data/raw/us_counties/cb_2025_us_county_500k/. "
            "Skipping spatial join — set county_incident_history to 0."
        )
        df["COUNTY_GEOID"] = None
        df["COUNTY_NAME"] = None
        df["county_incident_history"] = 0

    return df


# ── Drop non-feature columns ───────────────────────────────────────────────────

NON_FEATURE_COLS = [
    "INCIDENT_DATE",
    "LOCATION_LATITUDE",
    "LOCATION_LONGITUDE",
    "OPERATOR_ID",
    "CAUSE",
    "SUBCAUSE",
    "STATE",
    "STATE_UPPER",
    "PIPE_MATERIAL",
    "COUNTY_GEOID",
    "COUNTY_NAME",
    "decade_installed",  # keep as string label — drop before model, used in EDA
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the final feature matrix ready for model training."""
    drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    feature_df = df.drop(columns=drop)
    # Drop remaining object columns
    obj_cols = feature_df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        print(f"Dropping remaining object columns: {obj_cols}")
        feature_df = feature_df.drop(columns=obj_cols)
    return feature_df


# ── Main ───────────────────────────────────────────────────────────────────────

def build(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_pipeline_features(df)
    df = add_cause_features(df)
    df = add_spatial_features(df)
    return df


def main() -> pd.DataFrame:
    incidents_path = PROCESSED_DIR / "incidents_clean.csv"
    if not incidents_path.exists():
        raise FileNotFoundError(
            "Run src/data/clean_data.py first to generate incidents_clean.csv"
        )

    df = pd.read_csv(incidents_path, low_memory=False)
    print(f"Loaded {len(df):,} clean incidents")

    df = build(df)
    feature_df = select_features(df)

    print(f"\nFinal feature matrix: {feature_df.shape}")
    print("Columns:", feature_df.columns.tolist())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "pipeline_features.csv"
    feature_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Also save the full enriched df (with lat/lon) for visualization
    geo_path = PROCESSED_DIR / "incidents_enriched.csv"
    df.to_csv(geo_path, index=False)
    print(f"Saved enriched (with geometry columns) → {geo_path}")

    return feature_df


if __name__ == "__main__":
    main()
