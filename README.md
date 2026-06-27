# Natural Gas Pipeline Risk Mapping & Incident Prediction

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600)](https://xgboost.readthedocs.io)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-139C5A)](https://geopandas.org)
[![Folium](https://img.shields.io/badge/Folium-0.15+-77B829)](https://python-visualization.github.io/folium)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-FF6F61)](https://shap.readthedocs.io)

> **Research Question:** Can we predict the severity of natural gas pipeline incidents using historical PHMSA data, and where are the highest-risk corridors in the United States?

An end-to-end geospatial data science portfolio project combining PHMSA incident records, U.S. Census county geometries, machine learning severity prediction, SHAP explainability, and an interactive Folium risk map — built for portfolio demonstration and public safety analysis.

---

## Architecture

```
phmsa_incidents_2010_present.txt ─┐
phmsa_incidents_2002_2009.txt    ─┤─► load_data.py ──► incidents_raw.csv
us_counties shapefile            ─┘
                                        │
                                   clean_data.py
                                        │
                                   incidents_clean.csv
                                        │
                                  build_features.py
                                        │
                              pipeline_features.csv
                                    /        \
                              train.py    notebooks/
                               /    \
                      RF model   XGB model
                               │
                          evaluate.py ──► ROC, SHAP, CM plots
                               │
                        pipeline_map.py ──► pipeline_risk_map.html
```

---

## Dataset

| File | Source | Records | Period |
|------|--------|---------|--------|
| `phmsa_incidents_2010_present.txt` | [PHMSA Gas Transmission](https://www.phmsa.dot.gov/data-and-statistics/pipeline/gas-distribution-gas-gathering-gas-transmission-hazardous-liquids) | 2,018 | 2010–2024 |
| `phmsa_incidents_2002_2009.txt` | PHMSA Historical | 1,029 | 2002–2009 |
| `cb_2025_us_county_500k.shp` | [US Census TIGER 2025](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | 3,234 counties | — |

**Combined and cleaned: 2,675 incidents across 43 states**

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Loading | `pandas`, `latin-1` encoding, tab-delimited TXT |
| Coordinate Parsing | Custom regex DMS→decimal converter (7+ formats) |
| Geospatial | `geopandas`, `shapely`, `fiona`, `pyproj` |
| Feature Engineering | County spatial join, pipeline age, cause one-hot encoding |
| ML Models | `scikit-learn` Random Forest, `xgboost` |
| Explainability | `shap` TreeExplainer, waterfall + summary plots |
| Visualization | `folium` (interactive map), `matplotlib`, `seaborn` |
| Portfolio | GitHub Pages (`docs/index.html`) |

---

## Key Findings

- **RF ROC-AUC: 0.998** — the model almost perfectly separates high- vs. low-severity incidents
- **Top predictive features:** property damage costs, pipeline age at incident, commodity released quantity, pipe diameter, and fatalities
- **Corrosion** is the leading cause (35% of incidents) and disproportionately causes high-severity events
- **Texas, Louisiana, Oklahoma, and Pennsylvania** account for >40% of all incidents
- 1-mile exposure buffers around high-severity incidents cover significant population centers in the Gulf Coast corridor

---

## Project Structure

```
natural-gas-pipeline-risk/
├── data/
│   ├── raw/                    # PHMSA .txt files + Census shapefile (not committed)
│   └── processed/              # Generated CSVs and GeoJSON
├── src/
│   ├── data/
│   │   ├── load_data.py        # Schema normalization for two PHMSA formats
│   │   └── clean_data.py       # Cleaning, geocoding validation, target creation
│   ├── features/
│   │   └── build_features.py   # Temporal, pipeline, cause, spatial features
│   ├── models/
│   │   ├── train.py            # RF + XGBoost training, risk scores
│   │   └── evaluate.py         # ROC, confusion matrix, SHAP plots
│   └── visualization/
│       └── pipeline_map.py     # Interactive Folium map (3 layers)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_geospatial_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_visualization.ipynb
├── models/                     # Trained model pickles (not committed)
├── outputs/                    # All generated charts and HTML map
├── docs/
│   └── index.html              # GitHub Pages portfolio site
└── README.md
```

---

## Setup & Run

### 1. Prerequisites

```bash
conda create -n pipeline-risk python=3.13
conda activate pipeline-risk
conda install -c conda-forge geopandas fiona pyproj rasterio
pip install -r requirements.txt
```

### 2. Data

Place raw data files at:
- `data/raw/phmsa_incidents_2010_present.txt`
- `data/raw/phmsa_incidents_2002_2009.txt`
- `data/raw/us_counties/cb_2025_us_county_500k/cb_2025_us_county_500k.shp`

### 3. Run the Pipeline

```bash
# Step 1 — Load and unify the two PHMSA schemas
python -m src.data.load_data

# Step 2 — Clean coordinates, standardize causes, create binary target
python -m src.data.clean_data

# Step 3 — Engineer features (spatial join + temporal + cause encoding)
python -m src.features.build_features

# Step 4 — Train Random Forest and XGBoost models
python -m src.models.train

# Step 5 — Generate evaluation plots and SHAP explanations
python -m src.models.evaluate

# Step 6 — Build the interactive Folium map
python -m src.visualization.pipeline_map
```

### 4. Explore in Notebooks

```bash
jupyter notebook
# Open notebooks in order: 01 → 02 → 03 → 04 → 05
```

---

## Model Results

| Model | ROC-AUC | F1 (weighted) | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| Random Forest | **0.998** | 0.989 | 0.995 | 0.990 |
| XGBoost | 0.997 | 0.994 | 1.000 | 0.993 |

---

## Author

**Ankita Prashant Naik**  
LinkedIn: [linkedin.com/in/ankitaprashantnaik](https://linkedin.com/in/ankitaprashantnaik)

---

## Data Sources

- [PHMSA Pipeline Incident Data](https://www.phmsa.dot.gov/data-and-statistics/pipeline/gas-distribution-gas-gathering-gas-transmission-hazardous-liquids) — U.S. Department of Transportation
- [US Census TIGER/Line Shapefiles 2025](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) — U.S. Census Bureau

*Data is publicly available and used for educational and portfolio purposes.*
