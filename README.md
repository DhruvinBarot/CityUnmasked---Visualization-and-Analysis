# 🏙️ CityUnmasked — Visualization and Analysis
### Track 3 — Urban Data Analysis | City of Syracuse Datathon 2026

---

## 📌 Project Overview

CityUnmasked investigates the relationship between **urban decay** (unfit properties, vacant properties, code violations) and **crime patterns** in Syracuse, NY using four real municipal datasets. The project identifies where blight and crime co-occur, measures the strength of that co-occurrence, classifies every zip code by decay type, and predicts where crime will concentrate next.

**Core Thesis:**
> *We are NOT claiming crime creates blight or blight creates crime. We ARE showing that when they find each other in the same geography, they don't let go — and Syracuse's own data shows us exactly where that grip is tightening right now.*

**Core Question:**
> *Do neighborhoods with more unfit and vacant properties experience disproportionately higher crime rates — and can we predict where intervention is most needed?*

---

---

## 📁 Project Structure

```
CityUnmasked/
├── dashboard.py                    ← Main Streamlit app (run this)
│
├── analysis/                       ← Data processing and chart functions
│   ├── __init__.py
│   ├── crime.py                    ← Crime data loading and charts
│   ├── unfit.py                    ← Unfit properties loading and charts
│   ├── vacant.py                   ← Vacant properties loading and charts
│   ├── code_violations.py          ← Code violations loading, filtering, tiering
│   ├── decay_index.py              ← Spatial join, A/B/C classification, Urban Decay Index
│   ├── models.py                   ← Granger causality, Random Forest
│   ├── map_builder.py              ← Folium map construction
│   └── crime_risk_dev.py           ← Multi-year crime hotspot prediction model ← NEW
│
├── tabs/                           ← Dashboard tab rendering
│   ├── __init__.py
│   ├── tab_crime.py
│   ├── tab_unfit.py
│   ├── tab_vacant.py
│   ├── tab_decay_index.py
│   ├── tab_code_violations.py
│   ├── tab_map.py
│   └── tab_prediction.py          ← Updated with hotspot model UI ← UPDATED
│
├── crime_clean.csv                 ← 25,752 crime incidents (2023–2025)
├── Unfit_Properties.csv            ← 264 unfit violations (2014–2025)
├── Vacant_Properties.csv           ← 1,651 vacant registrations
├── code_violations.csv             ← 140,726 code violations (2017–2026)
├── requirements.txt
└── README.md
```

---

## ✅ What Has Been Built

### Phase 1 — Core Analysis

**Crime Data (25,752 incidents)**
- Top crimes: Larceny, Simple Assault, Criminal Mischief
- 91.5% serious crimes, 8.5% quality-of-life
- Seasonal peak in summer, hourly peak in evenings (6pm–midnight)

**Unfit Properties (264 violations)**
- 73% still Open — unresolved
- 33x growth from 2014 to 2025
- Top zip codes: 13204, 13205, 13208

**Spatial Join (BallTree haversine)**
- 27.4% of crimes within 100m of unfit property — 2x expected rate
- Gradient: 58.3% at 200m, 74.9% at 300m, 89.2% at 500m

### Phase 2 — Extended Analysis

**Vacant Properties (1,615 usable records)**
- 88% still active — higher unresolved rate than unfit
- Brighton (244), Northside (177), Near Westside (120) most affected
- Same zip codes as crime hotspots: 13205, 13204, 13208

**Code Violations (92,790 physical decay records)**
- Filtered from 140,726 total — administrative violations excluded
- Three tiers: Structural/Critical (10,334), Systems Failure (34,427), Environmental Neglect (48,029)
- 10 years of data (2017–2026) powering Granger causality analysis

**Urban Decay Index**
- Combined decay score per zip code (unfit + vacant)
- Neighborhood classification: Type A / B / C (see below)
- Economic abandonment zones identified (high decay, low crime)

**Models**
- Bidirectional Granger Causality (108 months, code violations ↔ crime)
- Random Forest Classifier (predicts high-severity crime from decay + temporal features)

---

## 🔮 New Feature — Multi-Year Crime Hotspot Prediction

### What It Does

The hotspot prediction model uses **2023–2025 crime data** to identify chronic high-risk grid cells across the city and predict which areas are most likely to become Q4 crime clusters.

**Spatial step:**
Each crime incident is snapped to a coarse **~400–500m grid** using lat/lon. Each grid cell represents roughly one city block.

**Temporal step (per year):**
- **History window:** Jan–Sep crimes → used as features
- **Future window:** Oct–Dec crimes → used as labels

**Features per grid cell:**
| Feature | Description |
|---|---|
| `total_crimes` | Number of incidents in Jan–Sep |
| `serious_ratio` | Share of serious crimes (assault, robbery, burglary) in that cell |

**Label:**
A grid cell is labelled **1 (cluster)** if it has 3+ crimes in Oct–Dec, otherwise **0 (no cluster)**.

**Model:**
A **logistic regression** (class-balanced) is trained across all `(grid, year)` rows for 2023, 2024, and 2025. It predicts `P(cluster in Q4)` for each grid cell.

**Multi-year aggregation:**
For each grid cell, predicted risk and observed future crimes are averaged across all three years — surfacing **chronic hotspots** rather than one-off spikes.

### Key Results

The **Top 10 highest-risk grid cells** are concentrated in four chronic hotspot areas:

| Area | ZIP Code |
|---|---|
| Downtown | 13202 |
| Southside | 13207 |
| Eastside / Syracuse University | 13210 |
| Near Westside | 13204 |

These grids show:
- **High total crime counts** in Jan–Sep
- **High proportion of serious offences** — the strongest predictor of becoming a Q4 cluster
- **Consistency across multiple years** — not single-year anomalies

### Why This Matters

The model gives the city a **data-driven tool** to:
1. Prioritize proactive inspections, lighting, and violence-prevention in a small, defined set of blocks
2. Track year-over-year risk score changes to evaluate whether interventions are working
3. Align patrol deployment with predicted Q4 risk before the high-risk period begins

### New Files

| File | Description |
|---|---|
| `analysis/crime_risk_dev.py` | Full model: grid assignment, feature building, logistic regression training, Folium risk heatmap, Top 10 grids |
| `tabs/tab_prediction.py` | Updated prediction tab UI — hotspot map, Top 10 table, model explanation, policy recommendations |

---

## 🏘️ Neighborhood Classification — Type A / B / C

Every zip code is classified into one of three decay types based on crime level and decay level:

| Type | Definition | Intervention |
|---|---|---|
| 🔴 **Type A — Crime-Blight Feedback** | High crime AND high decay. Both problems reinforce each other. | Simultaneous housing intervention AND targeted policing |
| 🔵 **Type B — Economic Abandonment** | High decay, LOW crime. Blight from economic/demographic causes, not crime. | Investment, ownership reform, rehabilitation. NOT increased policing. |
| 🟡 **Type C — Infrastructure Decay** | Unfit-dominant. Structural age and maintenance failure. | Fast-track code enforcement and rehabilitation funding |
| ⚫ **Low Risk / Monitoring** | Below median on both axes. | Monitor, no immediate action |

**The Type B finding is the project's intellectual honesty proof:** blight exists without crime, which means blight has multiple causes, and those causes need different solutions.

---

## 📊 Dashboard — 7 Tabs

| Tab | Contents |
|---|---|
| 📊 Crime Analysis | Top crime types, serious vs QoL split, monthly patterns, hourly distribution |
| 🏚️ Unfit Properties | Annual violation trend, open/closed rate, zip code concentration |
| 🏘️ Vacant Properties | Neighborhood breakdown, active vs resolved, zip distribution |
| 📉 Urban Decay Index | A/B/C classification, scatter quadrant, risk ranking, economic abandonment zones, Granger causality |
| ⚠️ Code Violations | Tier breakdown by year, violation geography, bidirectional Granger test (108 months) |
| 🗺️ Map | Three-layer Folium: crime heatmap + unfit markers + vacant density. Layer toggles. |
| 🔮 Prediction | Hotspot risk heatmap (2023–2025), Top 10 chronic risk grids, logistic regression explanation, policy recommendations |

---

## 📊 Datasets

| Dataset | File | Records | Key Columns |
|---|---|---|---|
| Crime | `crime_clean.csv` | 25,752 | LAT, LON, CRIME_TYPE, SEVERITY, SEASON, TIME_OF_DAY, YEAR, MONTH |
| Unfit Properties | `Unfit_Properties.csv` | 264 | Latitude, Longitude, status_type_name, violation_date, zip |
| Vacant Properties | `Vacant_Properties.csv` | 1,651 | Latitude, Longitude, neighborhood, Zip, VPR_valid |
| Code Violations | `code_violations.csv` | 140,726 (92,790 filtered) | Latitude, Longitude, complaint_type_name, violation, violation_date, Neighborhood |

---

## 🗺️ Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
python -m streamlit run dashboard.py
```

Opens at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit>=1.32.0
streamlit-folium>=0.18.0
folium>=0.16.0
plotly>=5.20.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
statsmodels>=0.14.0
scipy>=1.12.0
geopy>=2.4.0
```

```bash
pip install -r requirements.txt
```

---

## 💡 Key Findings

1. **27.4% of crimes** occur within 100m of an unfit property — **2x the expected rate**
2. **58%** of all crimes fall within 100m of at least one decay point (unfit + vacant combined)
3. Unfit violations grew **33x from 2014 to 2025**, with 73% still unresolved
4. **88% of 1,615 vacant properties** remain active — the highest unresolved rate across all datasets
5. **92,790 physical decay violations** span 10 years — four datasets, same geographic hotspot
6. Zip codes **13204, 13205, 13208** are confirmed Type A zones — highest risk, need dual intervention
7. **Type B zones exist** — high decay, low crime — proving blight has non-criminal causes
8. Chronic crime hotspots in **13202, 13204, 13207, 13210** persist across 2023–2025

---

## 🎯 Policy Recommendations

**Type A Zones (13204, 13205, 13208 — Crime-Blight Feedback):**
Simultaneous housing intervention AND targeted policing. Fixing one without the other breaks only half the cycle.

**Type B Zones (Economic Abandonment):**
Investment programs, ownership enforcement, vacancy rehabilitation. Do NOT increase policing — crime is not the driver.

**Type C Zones (Infrastructure Decay):**
Fast-track code enforcement and rehabilitation funding.

**Hotspot Grid Cells (13202, 13207, 13210, 13204):**
Proactive inspections, lighting upgrades, and violence-prevention programs concentrated in the Top 10 chronic risk blocks before Q4.

**City-wide:**
- Fast-track the 73% of unfit violations still Open
- Address 1,421 active vacant properties
- Target Brighton, Northside, Near Westside for investment
- Use year-over-year risk score changes to evaluate whether interventions are working

---

## 👥 Team

> Add your team member names and GitHub handles here

---

