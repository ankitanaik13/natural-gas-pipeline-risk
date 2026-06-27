"""
Clean and standardize raw PHMSA incident data.
Run: python -m src.data.clean_data
Output: data/processed/incidents_clean.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.load_data import load_incidents

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

CAUSE_MAP = {
    # Corrosion
    "CORROSION": "CORROSION",
    "INTERNAL CORROSION": "CORROSION",
    "EXTERNAL CORROSION": "CORROSION",
    # Excavation / outside force
    "EXCAVATION DAMAGE": "EXCAVATION DAMAGE",
    "OTHER OUTSIDE FORCE DAMAGE": "EXCAVATION DAMAGE",
    "PREVIOUS DAMAGE": "EXCAVATION DAMAGE",
    # Natural force
    "NATURAL FORCE DAMAGE": "NATURAL FORCE",
    "EARTH MOVEMENT": "NATURAL FORCE",
    "HEAVY RAINS/FLOODS": "NATURAL FORCE",
    "LIGHTNING": "NATURAL FORCE",
    "TEMPERATURE": "NATURAL FORCE",
    # Equipment
    "EQUIPMENT FAILURE": "EQUIPMENT FAILURE",
    "PUMP/COMPRESSOR FAILURE": "EQUIPMENT FAILURE",
    "VALVE FAILURE": "EQUIPMENT FAILURE",
    "PIPE BODY FAILURE": "EQUIPMENT FAILURE",
    "SEAL/PACKING FAILURE": "EQUIPMENT FAILURE",
    "FITTING FAILURE": "EQUIPMENT FAILURE",
    # Incorrect operation
    "INCORRECT OPERATION": "INCORRECT OPERATION",
    "OPERATOR ERROR": "INCORRECT OPERATION",
    # Other / unknown
    "ALL OTHER CAUSES": "OTHER",
    "OTHER": "OTHER",
    "MISCELLANEOUS": "OTHER",
}


def standardize_cause(series: pd.Series) -> pd.Series:
    """Map raw CAUSE strings to 6 canonical categories."""
    upper = series.str.upper().str.strip()
    mapped = upper.map(CAUSE_MAP)
    # Fallback: partial match for corrosion / excavation
    mask = mapped.isna()
    mapped[mask & upper.str.contains("CORROSION", na=False)] = "CORROSION"
    mapped[mask & upper.str.contains("EXCAVAT", na=False)] = "EXCAVATION DAMAGE"
    mapped[mask & upper.str.contains("NATURAL|EARTH|FLOOD|LIGHTNING|SEISMIC", na=False)] = "NATURAL FORCE"
    mapped[mask & upper.str.contains("EQUIPMENT|VALVE|PUMP|SEAL|FITTING", na=False)] = "EQUIPMENT FAILURE"
    mapped[mask & upper.str.contains("INCORRECT|OPERATOR ERROR", na=False)] = "INCORRECT OPERATION"
    mapped = mapped.fillna("OTHER")
    return mapped


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Input rows: {len(df):,}")

    # ── Remove rows with no coordinates ──────────────────────────────────────
    df = df.dropna(subset=["LOCATION_LATITUDE", "LOCATION_LONGITUDE"])
    df = df[
        (df["LOCATION_LATITUDE"] != 0) & (df["LOCATION_LONGITUDE"] != 0)
    ]
    print(f"After dropping missing/zero coordinates: {len(df):,}")

    # ── Parse incident date ───────────────────────────────────────────────────
    df["IYEAR"] = pd.to_numeric(df["IYEAR"], errors="coerce")
    df["IMONTH"] = pd.to_numeric(df["IMONTH"], errors="coerce")
    df["IDAY"] = pd.to_numeric(df["IDAY"], errors="coerce")

    df["INCIDENT_DATE"] = pd.to_datetime(
        {
            "year": df["IYEAR"],
            "month": df["IMONTH"].clip(1, 12),
            "day": df["IDAY"].clip(1, 28),
        },
        errors="coerce",
    )

    # ── Numeric conversions ───────────────────────────────────────────────────
    df["PRPTY_DAMAGE_COSTS"] = pd.to_numeric(
        df["PRPTY_DAMAGE_COSTS"].astype(str).str.replace(",", ""), errors="coerce"
    ).fillna(0)

    df["FATALITIES"] = pd.to_numeric(df["FATALITIES"], errors="coerce").fillna(0)
    df["INJURE"] = pd.to_numeric(df["INJURE"], errors="coerce").fillna(0)
    df["COMMODITY_RELEASED_QUANTITY"] = pd.to_numeric(
        df["COMMODITY_RELEASED_QUANTITY"], errors="coerce"
    ).fillna(0)
    df["INSTALLATION_YEAR"] = pd.to_numeric(df["INSTALLATION_YEAR"], errors="coerce")

    # ── Standardize CAUSE ────────────────────────────────────────────────────
    if "CAUSE" in df.columns:
        df["CAUSE_CLEAN"] = standardize_cause(df["CAUSE"].fillna("OTHER"))
    else:
        df["CAUSE_CLEAN"] = "OTHER"

    # ── Binary severity target ───────────────────────────────────────────────
    df["high_severity"] = (
        (df["FATALITIES"] > 0) | (df["PRPTY_DAMAGE_COSTS"] > 50_000)
    ).astype(int)

    print(f"High-severity incidents: {df['high_severity'].sum():,} "
          f"({df['high_severity'].mean() * 100:.1f}%)")

    # ── Drop duplicates ───────────────────────────────────────────────────────
    df = df.drop_duplicates()

    # ── Coordinate sanity check ───────────────────────────────────────────────
    df = df[
        df["LOCATION_LATITUDE"].between(-90, 90)
        & df["LOCATION_LONGITUDE"].between(-180, 180)
    ]

    print(f"Final clean rows: {len(df):,}")
    return df.reset_index(drop=True)


def main() -> pd.DataFrame:
    df_raw = load_incidents()
    df_clean = clean(df_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "incidents_clean.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    return df_clean


if __name__ == "__main__":
    main()
