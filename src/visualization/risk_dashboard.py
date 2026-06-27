"""
Build Kepler.gl interactive risk dashboard.
Run: python -m src.visualization.risk_dashboard
Output: outputs/kepler_risk_dashboard.html

Requires: pip install keplergl
Note: keplergl works best in Jupyter; this script exports a standalone HTML.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"


def build_kepler_map() -> None:
    try:
        from keplergl import KeplerGl
    except ImportError:
        print("keplergl not installed. Run: pip install keplergl")
        return

    incidents_path = PROCESSED_DIR / "incidents_enriched.csv"
    if not incidents_path.exists():
        incidents_path = PROCESSED_DIR / "incidents_clean.csv"

    df = pd.read_csv(incidents_path, low_memory=False)

    # Kepler expects lat/lon columns named explicitly
    df_kepler = df[
        ["LOCATION_LATITUDE", "LOCATION_LONGITUDE", "high_severity",
         "PRPTY_DAMAGE_COSTS", "FATALITIES", "CAUSE_CLEAN", "STATE", "IYEAR"]
    ].rename(columns={
        "LOCATION_LATITUDE": "latitude",
        "LOCATION_LONGITUDE": "longitude",
        "PRPTY_DAMAGE_COSTS": "damage_cost",
        "FATALITIES": "fatalities",
        "CAUSE_CLEAN": "cause",
        "IYEAR": "year",
    }).dropna(subset=["latitude", "longitude"])

    kepler_config = {
        "version": "v1",
        "config": {
            "mapState": {
                "latitude": 38.5,
                "longitude": -96.0,
                "zoom": 4,
            },
            "visState": {
                "layers": [
                    {
                        "type": "point",
                        "config": {
                            "dataId": "incidents",
                            "label": "Pipeline Incidents",
                            "color": [213, 47, 47],
                            "columns": {
                                "lat": "latitude",
                                "lng": "longitude",
                            },
                            "isVisible": True,
                            "visConfig": {
                                "radius": 6,
                                "opacity": 0.8,
                                "colorRange": {
                                    "colors": [
                                        "#1A237E", "#0D47A1", "#F57F17",
                                        "#E65100", "#B71C1C",
                                    ]
                                },
                            },
                        },
                    }
                ]
            },
        },
    }

    m = KeplerGl(height=600, config=kepler_config)
    m.add_data(data=df_kepler, name="incidents")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "kepler_risk_dashboard.html"
    m.save_to_html(file_name=str(out_path))
    print(f"Kepler.gl dashboard saved → {out_path}")


if __name__ == "__main__":
    build_kepler_map()
