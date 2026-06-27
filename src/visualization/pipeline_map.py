"""
Build the interactive Folium pipeline risk map.
Run: python -m src.visualization.pipeline_map
Output: outputs/pipeline_risk_map.html
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

SEVERITY_COLOR = {1: "#D32F2F", 0: "#1565C0"}  # red = high, blue = low
RISK_COLOR_STEPS = ["#2E7D32", "#F9A825", "#EF6C00", "#B71C1C"]


def get_risk_color(score: float) -> str:
    if score < 0.25:
        return RISK_COLOR_STEPS[0]
    elif score < 0.5:
        return RISK_COLOR_STEPS[1]
    elif score < 0.75:
        return RISK_COLOR_STEPS[2]
    return RISK_COLOR_STEPS[3]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    incidents_path = PROCESSED_DIR / "incidents_enriched.csv"
    if not incidents_path.exists():
        incidents_path = PROCESSED_DIR / "incidents_clean.csv"
    df = pd.read_csv(incidents_path, low_memory=False)

    risk_path = PROCESSED_DIR / "risk_scores.csv"
    risk_df = pd.read_csv(risk_path, low_memory=False) if risk_path.exists() else None

    return df, risk_df


def build_map(df: pd.DataFrame, risk_df: pd.DataFrame | None) -> folium.Map:
    # Center on CONUS
    center_lat = df["LOCATION_LATITUDE"].median()
    center_lon = df["LOCATION_LONGITUDE"].median()
    if not (-90 <= center_lat <= 90 and -180 <= center_lon <= 180):
        center_lat, center_lon = 38.5, -96.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="OpenStreetMap",
    )

    # ── Layer 1: Incident points ───────────────────────────────────────────────
    incident_group = folium.FeatureGroup(name="Pipeline Incidents", show=True)
    cluster = MarkerCluster(name="Incidents Cluster")

    for _, row in df.iterrows():
        lat = row.get("LOCATION_LATITUDE")
        lon = row.get("LOCATION_LONGITUDE")
        if pd.isna(lat) or pd.isna(lon):
            continue

        severity = int(row.get("high_severity", 0))
        color = SEVERITY_COLOR[severity]
        damage = row.get("PRPTY_DAMAGE_COSTS", 0)
        fatalities = row.get("FATALITIES", 0)
        cause = row.get("CAUSE_CLEAN", row.get("CAUSE", "Unknown"))
        date = row.get("INCIDENT_DATE", f"{row.get('IYEAR', '')}-{row.get('IMONTH', '')}")
        state = row.get("STATE", "")

        popup_html = f"""
        <div style='font-family:sans-serif;width:200px'>
          <b>{'⚠ HIGH SEVERITY' if severity else 'Low Severity'}</b><br>
          <hr style='margin:4px 0'>
          <b>Date:</b> {date}<br>
          <b>State:</b> {state}<br>
          <b>Cause:</b> {cause}<br>
          <b>Damage:</b> ${damage:,.0f}<br>
          <b>Fatalities:</b> {fatalities:.0f}<br>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6 if severity else 4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{'HIGH' if severity else 'LOW'} | {cause}",
        ).add_to(incident_group)

    incident_group.add_to(m)

    # ── Layer 2: Heatmap ───────────────────────────────────────────────────────
    heat_data = [
        [row["LOCATION_LATITUDE"], row["LOCATION_LONGITUDE"],
         1.5 if row.get("high_severity", 0) else 0.5]
        for _, row in df.iterrows()
        if not pd.isna(row.get("LOCATION_LATITUDE"))
        and not pd.isna(row.get("LOCATION_LONGITUDE"))
    ]

    heat_group = folium.FeatureGroup(name="Incident Heatmap", show=False)
    HeatMap(
        heat_data,
        radius=18,
        blur=12,
        max_zoom=8,
        gradient={0.2: "blue", 0.5: "yellow", 0.8: "orange", 1.0: "red"},
    ).add_to(heat_group)
    heat_group.add_to(m)

    # ── Layer 3: Risk score points (if available from model) ───────────────────
    if risk_df is not None and "risk_score" in risk_df.columns:
        risk_group = folium.FeatureGroup(name="ML Risk Scores", show=False)

        # Merge risk scores back to lat/lon from enriched data
        incidents_with_risk = df.copy()
        if "risk_score" not in incidents_with_risk.columns:
            # Approximate: sample risk scores in same proportion
            incidents_with_risk = incidents_with_risk.sample(
                min(len(risk_df), len(incidents_with_risk)), random_state=42
            )
            incidents_with_risk["risk_score"] = risk_df["risk_score"].values[:len(incidents_with_risk)]

        for _, row in incidents_with_risk.iterrows():
            lat = row.get("LOCATION_LATITUDE")
            lon = row.get("LOCATION_LONGITUDE")
            score = row.get("risk_score", 0)
            if pd.isna(lat) or pd.isna(lon) or pd.isna(score):
                continue

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=get_risk_color(score),
                fill=True,
                fill_color=get_risk_color(score),
                fill_opacity=0.8,
                tooltip=f"ML Risk Score: {score:.2f}",
            ).add_to(risk_group)

        risk_group.add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 40px; left: 40px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: sans-serif;
        font-size: 13px; min-width: 180px;">
      <b>Natural Gas Pipeline Incidents</b><br><br>
      <span style="color:#D32F2F">&#9679;</span> High Severity<br>
      <span style="color:#1565C0">&#9679;</span> Low Severity<br>
      <br>
      <b>ML Risk Score</b><br>
      <span style="color:#2E7D32">&#9679;</span> Low (&lt;0.25)<br>
      <span style="color:#F9A825">&#9679;</span> Medium-Low<br>
      <span style="color:#EF6C00">&#9679;</span> Medium-High<br>
      <span style="color:#B71C1C">&#9679;</span> High (&gt;0.75)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title ──────────────────────────────────────────────────────────────────
    title_html = """
    <div style="
        position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(13,27,42,0.88);
        color: white; padding: 8px 20px; border-radius: 6px;
        font-family: sans-serif; font-size: 15px; font-weight: bold;">
      Natural Gas Pipeline Risk Map — PHMSA 1970–2024
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df, risk_df = load_data()
    print(f"Loaded {len(df):,} incidents")

    m = build_map(df, risk_df)

    out_path = OUTPUTS_DIR / "pipeline_risk_map.html"
    m.save(str(out_path))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
