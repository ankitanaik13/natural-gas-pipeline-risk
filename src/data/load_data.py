"""
Load, normalize, and validate raw PHMSA pipeline incident data.
Combines two files with different schemas into one standard dataframe.
Run: python -m src.data.load_data
"""

import re
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

FILE_2010_PRESENT = RAW_DIR / "phmsa_incidents_2010_present.txt"
FILE_2002_2009    = RAW_DIR / "phmsa_incidents_2002_2009.txt"

# Columns expected after normalization (used for validation report)
REQUIRED_COLUMNS = [
    "IYEAR",
    "IMONTH",
    "IDAY",
    "LOCATION_LATITUDE",
    "LOCATION_LONGITUDE",
    "CAUSE",
    "SUBCAUSE",
    "FATALITIES",
    "INJURE",
    "PRPTY_DAMAGE_COSTS",
    "COMMODITY_RELEASED_QUANTITY",
    "INSTALLATION_YEAR",
    "PIPE_DIAMETER",
    "PIPE_MATERIAL",
    "OPERATOR_ID",
    "STATE",
]


# ── Coordinate parsing ────────────────────────────────────────────────────────

def _coord_to_decimal(value) -> float | None:
    """Convert any observed PHMSA lat/lon format to decimal degrees.

    Handles all formats seen in the 2002-2009 file:
      "N27 43.413", "N37 59' 34", "29 12' 54.70", "29 20 10.3694",
      "46 04' 59N", "42.011172", "46 26.3'", "-88.170544",
      "45DEG41MINN", "32^42'43", "60 40' 32W", etc.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()

    # Fast path: already decimal degrees
    try:
        return float(s)
    except ValueError:
        pass

    su = s.upper()

    # Determine sign from cardinal direction letter
    negative = ("S" in su) or ("W" in su)

    # Strip cardinal letters, degree/minute/second markers
    su = re.sub(r"[NSEW]", " ", su)
    su = re.sub(r'DEG|MIN|[°^\'\"°]', " ", su)
    su = re.sub(r"\s+", " ", su).strip()

    parts = su.split()
    try:
        nums = [float(p) for p in parts if p]
        if not nums:
            return None
        dec = nums[0]
        if len(nums) >= 2:
            dec += nums[1] / 60.0
        if len(nums) >= 3:
            dec += nums[2] / 3600.0
        return -dec if negative else dec
    except (ValueError, IndexError):
        return None


# ── Per-file normalization ────────────────────────────────────────────────────

def _normalize_2010_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and derive columns for the 2010-present PHMSA file."""
    renames = {
        "FATAL":                        "FATALITIES",
        "ONSHORE_STATE_ABBREVIATION":   "STATE",
        "PRPTY":                        "PRPTY_DAMAGE_COSTS",
        "MATERIAL_INVOLVED":            "PIPE_MATERIAL",
        "CAUSE_DETAILS":                "SUBCAUSE",
        "UNINTENTIONAL_RELEASE":        "COMMODITY_RELEASED_QUANTITY",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # Parse IMONTH and IDAY from LOCAL_DATETIME ("5/3/2026 10:53")
    if "LOCAL_DATETIME" in df.columns:
        dt = pd.to_datetime(df["LOCAL_DATETIME"], errors="coerce")
        df["IMONTH"] = dt.dt.month
        df["IDAY"]   = dt.dt.day
    else:
        df["IMONTH"] = np.nan
        df["IDAY"]   = np.nan

    # STATE fallback: some incidents are offshore
    if "STATE" in df.columns and "OPERATOR_STATE_ABBREVIATION" in df.columns:
        df["STATE"] = df["STATE"].fillna(df["OPERATOR_STATE_ABBREVIATION"])

    # INSTALLATION_YEAR: "UNKNOWN" → NaN
    if "INSTALLATION_YEAR" in df.columns:
        df["INSTALLATION_YEAR"] = pd.to_numeric(
            df["INSTALLATION_YEAR"], errors="coerce"
        )

    return df


def _normalize_2002_2009(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, convert coordinates, and derive columns for the 2002-2009 file."""

    # Convert DMS lat/lon to decimal degrees
    df["LOCATION_LATITUDE"]  = df["LATITUDE"].apply(_coord_to_decimal)
    df["LOCATION_LONGITUDE"] = df["LONGITUDE"].apply(_coord_to_decimal)
    # West longitudes should be negative; the DMS parser handles N/W prefixes
    # but bare positive values (no direction letter) need sign correction
    mask_lon_pos = (
        df["LOCATION_LONGITUDE"].notna()
        & (df["LOCATION_LONGITUDE"] > 0)
        & ~df["LONGITUDE"].astype(str).str.upper().str.contains("E", na=False)
    )
    df.loc[mask_lon_pos, "LOCATION_LONGITUDE"] *= -1

    # Use text cause (CAUSE_TEXT) — the numeric CAUSE codes are not useful
    if "CAUSE_TEXT" in df.columns:
        df["CAUSE"] = df["CAUSE_TEXT"]

    renames = {
        "FATAL":             "FATALITIES",
        "NPS":               "PIPE_DIAMETER",
        "PRTYR":             "INSTALLATION_YEAR",
        "PRPTY":             "PRPTY_DAMAGE_COSTS",
        "CAUSE_DETAILS_TEXT":"SUBCAUSE",
        "GASPRP":            "COMMODITY_RELEASED_QUANTITY",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # STATE: use accident state (ACSTATE) with operator state (OPSTATE) as fallback
    if "ACSTATE" in df.columns:
        state = df["ACSTATE"]
        if "OPSTATE" in df.columns:
            state = state.fillna(df["OPSTATE"])
        df["STATE"] = state
    elif "OPSTATE" in df.columns:
        df["STATE"] = df["OPSTATE"]

    # IMONTH and IDAY from IDATE ("1/26/2002")
    if "IDATE" in df.columns:
        dt = pd.to_datetime(df["IDATE"], errors="coerce")
        df["IMONTH"] = dt.dt.month
        df["IDAY"]   = dt.dt.day
    else:
        df["IMONTH"] = np.nan
        df["IDAY"]   = np.nan

    # No PIPE_MATERIAL equivalent in the older format
    if "PIPE_MATERIAL" not in df.columns:
        df["PIPE_MATERIAL"] = np.nan

    return df


# ── Public loader ─────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected PHMSA data file not found: {path}\n"
            "Download Gas Transmission Incident Data from\n"
            "https://www.phmsa.dot.gov/data-and-statistics/pipeline/"
            "distribution-transmission-gathering-lng-and-liquid-accident-and-incident-data"
        )
    print(f"Loading: {path.name}")
    df = pd.read_csv(path, sep="\t", encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    print(f"  → {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_incidents(path: Path | None = None) -> pd.DataFrame:
    """Load, normalize, and combine both PHMSA incident files.

    Combines:
      data/raw/phmsa_incidents_2010_present.txt  (modern schema, 600+ cols)
      data/raw/phmsa_incidents_2002_2009.txt     (legacy schema, 196 cols)

    Returns a single dataframe with unified column names ready for clean_data.py.
    Pass `path` to load a single file (skips normalization, used for testing).
    """
    if path is not None:
        df = pd.read_csv(path, sep="\t", encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip().str.upper()
        return df

    df_new = _normalize_2010_plus(_load_txt(FILE_2010_PRESENT))
    df_old = _normalize_2002_2009(_load_txt(FILE_2002_2009))

    df = pd.concat([df_new, df_old], ignore_index=True, sort=False)
    print(f"Combined: {len(df):,} total rows")
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> dict:
    report = {}
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["shape"] = df.shape
    report["missing_required_columns"] = missing_cols
    report["all_columns"] = df.columns.tolist()

    null_counts = {}
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            null_counts[col] = int(df[col].isna().sum())
    report["null_counts"] = null_counts

    no_coords = (
        df[df["LOCATION_LATITUDE"].isna() | df["LOCATION_LONGITUDE"].isna()]
        if "LOCATION_LATITUDE" in df.columns
        else pd.DataFrame()
    )
    report["non_mappable_rows"] = len(no_coords)
    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("PHMSA DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Shape: {report['shape'][0]:,} rows × {report['shape'][1]} columns")

    if report["missing_required_columns"]:
        print(f"\n⚠  Missing required columns: {report['missing_required_columns']}")
    else:
        print("\n✓  All required columns present")

    print(f"\nNon-mappable rows (missing lat/lon): {report['non_mappable_rows']:,}")
    print("\nNull counts per required column:")
    for col, cnt in report["null_counts"].items():
        flag = "  ⚠" if cnt > 0 else "  ✓"
        print(f"  {flag} {col}: {cnt:,}")


def save_report(report: dict, out_path: Path | None = None) -> None:
    if out_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "validation_report.txt"
    with open(out_path, "w") as f:
        f.write("PHMSA DATA VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Shape: {report['shape'][0]:,} rows × {report['shape'][1]} columns\n\n")
        f.write(f"Missing required columns: {report['missing_required_columns']}\n\n")
        f.write(f"Non-mappable rows: {report['non_mappable_rows']:,}\n\n")
        f.write("Null counts:\n")
        for col, cnt in report["null_counts"].items():
            f.write(f"  {col}: {cnt:,}\n")
        f.write("\nAll columns:\n")
        for col in report["all_columns"]:
            f.write(f"  {col}\n")
    print(f"\nValidation report saved → {out_path}")


if __name__ == "__main__":
    df = load_incidents()
    report = validate(df)
    print_report(report)
    save_report(report)
    sys.exit(0 if not report["missing_required_columns"] else 1)
