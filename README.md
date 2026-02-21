# 🏙️ CityUnmasked---Visualization and Analysis
### Track 3 — Urban Data Analysis | City of Syracuse Datathon 2026

---

## 📌 Project Overview

This project analyzes the relationship between **urban decay indicators** (unfit properties, vacant properties) and **crime patterns** in the City of Syracuse, NY. Using real-world municipal datasets, we explore whether deteriorating housing conditions are spatially and statistically linked to higher crime rates — and what the city can do about it.

**Core Question:**
> *Do neighborhoods with more unfit and vacant properties experience disproportionately higher crime rates — and can we predict where intervention is most needed?*

---

## 🏆 Datathon Judging Criteria (25 pts each)

| Award | Our Approach |
|---|---|
| **Best Insight** | Areas within 100m of unfit properties have 2x expected crime concentration |
| **Best Trend** | Unfit property violations grew 33x from 2014 to 2025, 73% still unresolved |
| **Best Visualization** | Interactive Folium heatmap — crime density overlaid with unfit property locations |
| **Best Prediction** | Linear forecast of unfit violations through 2027 + crime risk zone modeling |

---

## 📁 Project Structure

```
datathon/
├── dashboard.py               ← Main Streamlit dashboard (run this)
├── crime_2024.csv             ← 2024 Syracuse crime incidents (6,693 rows)
├── unfit_properties.csv       ← Unfit property violations 2014–2025 (256 rows)
├── vacant_properties.csv      ← 🔲 TODO: Add vacant properties dataset
├── requirements.txt           ← All dependencies
└── README.md                  ← This file
```

---

## ✅ What Has Been Done (Phase 1)

### Data Loaded & Cleaned
- `crime_2024.csv` — 6,693 incidents, dropped 70 rows with missing coordinates
- `unfit_properties.csv` — 256 violations spanning 2014–2025, parsed mixed datetime formats
- Extracted: year, month, hour from datetime columns

### Exploratory Data Analysis
- Top crime types: **Simple Assault (2,165)** and **Criminal Mischief (2,131)** nearly tied at #1
- **83% of crimes are serious** (`QualityOfLife = False`), only 17% are minor incidents
- Crime peaks identified by **month** (seasonal) and **hour of day** (temporal)
- Unfit violations are heavily concentrated in zip codes **13204, 13205, 13208** (west/south Syracuse)

### Key Findings So Far

**Finding 1 — Spatial Overlap (100m radius)**
Using a BallTree haversine spatial join, we found that unfit properties cover ~12% of Syracuse's land area yet **27% of all 2024 crimes occurred within 100m** of one — roughly 2x the expected rate by chance alone.

| Radius | % of Crimes Nearby |
|---|---|
| 100m | 27.4% |
| 200m | 58.3% |
| 300m | 74.9% |
| 500m | 89.2% |

**Finding 2 — Unresolved Violations Crisis**
- 187 out of 256 violations (73%) are still **Open** today
- Violations grew from just 3 in 2014 to 100 already in 2025 — a **33x increase**
- Steepest acceleration began post-2021

**Finding 3 — Geographic Concentration**
Top zip codes for unfit properties: 13205 (67), 13204 (67), 13208 (60) — these are Syracuse's historically under-resourced neighborhoods and align with the crime heatmap hotspots.

### Dashboard Built
A full interactive Streamlit dashboard (`dashboard.py`) with 4 tabs:
- 📊 Crime Analysis
- 🏚️ Unfit Properties
- 🗺️ Interactive Map (Folium heatmap + unfit property markers)
- 🔮 Prediction (linear forecast through 2027)

---

## 🔲 What Needs To Be Done (Phase 2 — For Teammates)

### Priority Task: Integrate Vacant Properties Dataset

The next major step is adding the **Vacant Properties** dataset and combining it with unfit properties to create a unified **Urban Decay Index** per neighborhood/zip code.

#### Step 1 — Load and explore vacant properties
Start by understanding the structure of the vacant properties dataset — check columns, date ranges, null values, and whether it has coordinates or address fields that can be used for spatial analysis.

#### Step 2 — Combine unfit + vacant into one decay dataframe
Standardize the column names across both datasets and merge them into a single dataframe with a `decay_type` label so you can distinguish between unfit and vacant records during analysis.

#### Step 3 — Re-run spatial join with the combined decay dataset
Use the same BallTree haversine approach already in the codebase, but this time run it against the combined unfit + vacant dataset instead of unfit alone. Compare the 100m proximity percentage to what we got with unfit only (27.4%).

#### Step 4 — Compare crime patterns across decay types
Investigate whether crime near **only vacant** properties, **only unfit** properties, or **both** differs in type or severity. Areas with both types overlapping are likely the highest risk zones.

#### Step 5 — Build a decay score per zip code
Aggregate the combined dataset by zip code to create a simple count-based decay score. Cross-reference this with crime counts per zip to see if higher decay scores correlate with higher crime rates.

#### Step 6 — Update the dashboard
Once integrated, add a Vacant Properties section to the existing dashboard, update the map with a new color for vacant properties, and update the prediction tab to reflect the combined decay picture.

---

## 🗺️ Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit==1.32.0
streamlit-folium==0.20.0
folium==0.16.0
plotly==5.20.0
scikit-learn==1.4.1
pandas==2.2.1
numpy==1.26.4
```

Install all at once:
```bash
pip install streamlit streamlit-folium folium plotly scikit-learn pandas numpy
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

---

## 📊 Datasets

| Dataset | File | Rows | Key Columns |
|---|---|---|---|
| Crime 2024 | `crime_2024.csv` | 6,693 | LAT, LONG, CODE_DEFINED, DATEEND, QualityOfLife |
| Unfit Properties | `unfit_properties.csv` | 256 | Latitude, Longitude, status_type_name, violation_date, zip |
| Vacant Properties | `vacant_properties.csv` | TBD | TBD — to be explored by teammates |

---

## 💡 Policy Recommendations (Preliminary)

1. **Prioritize zip codes 13204, 13205, 13208** — majority of both unfit properties and crime hotspots
2. **Fast-track open violations** — 73% remain unresolved; faster resolution may reduce nearby crime
3. **Increase enforcement capacity** — violations are growing 10x faster than closures
4. **Expand analysis to vacant properties** — combining both datasets will sharpen intervention targeting
5. **Monitor 100m buffer zones** around open unfit properties for proactive policing

---

## 👥 Team

> Add your team member names and GitHub handles here

---

*City of Syracuse Datathon 2026 | Track 3 — Urban Data Analysis*
