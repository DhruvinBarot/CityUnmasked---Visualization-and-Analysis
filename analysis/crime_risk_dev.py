"""
Multi-year crime hotspot prediction model using crime_clean.

What it does:
- Loads crime_clean via analysis.crime.load_crime_clean()
- Uses all years in [2023, 2025]
- For each year:
    - Jan–Sep → features per grid cell
    - Oct–Dec → labels per grid cell (cluster vs no cluster)
- Trains a logistic regression on all (grid, year) rows
- Aggregates risk per grid across years (mean) to find chronic hotspots

Main entry point for the dashboard:
- run_hotspot_model() -> (folium_map, top10_df)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import folium
from folium.plugins import HeatMap

# from analysis.crime import load_crime_clean

import os

def load_crime_clean_local() -> pd.DataFrame:
    """
    Load crime_clean.csv from the project root.
    Assumes the Streamlit working directory is CityUnmasked-latest.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "crime_clean.csv")
    df = pd.read_csv(path, parse_dates=["DATEEND"])
    return df

# ─────────────────────────────────────────
# Helper: grid assignment
# ─────────────────────────────────────────

def assign_grid(df, lat_col="LAT", lon_col="LON", grid_size=0.005):
    """
    Assign each point to a coarse grid cell based on lat/long.
    grid_size ~0.005 ≈ 400–500m in Syracuse.

    Adds:
      - grid_lat
      - grid_lon
    """
    df = df.copy()
    df["grid_lat"] = np.floor(df[lat_col] / grid_size) * grid_size
    df["grid_lon"] = np.floor(df[lon_col] / grid_size) * grid_size
    return df


# ─────────────────────────────────────────
# Build multi-year, grid-level features + labels
# ─────────────────────────────────────────

def build_spatiotemporal_dataset(
    crime_df: pd.DataFrame,
    grid_size: float = 0.005,
    split_month: int = 9,
    min_crimes_for_cluster: int = 3,
    min_year: int = 2023,
    max_year: int = 2025,
) -> pd.DataFrame:
    """
    Multi-year spatio-temporal dataset.

    For each year Y in [min_year, max_year]:
      - HISTORY (features):  MONTH <= split_month (Jan–Sep)
      - FUTURE (labels):     MONTH > split_month (Oct–Dec)
      - Label = 1 if future_crimes >= min_crimes_for_cluster for that (grid, Y)

    Returns one row per (grid, year).
    """
    crime_grid = assign_grid(crime_df, "LAT", "LON", grid_size)

    years_to_use = sorted(
        y for y in crime_grid["YEAR"].unique()
        if min_year <= y <= max_year
    )

    rows = []

    for y in years_to_use:
        cg_y = crime_grid[crime_grid["YEAR"] == y].copy()
        if cg_y.empty:
            continue

        history = cg_y[cg_y["MONTH"] <= split_month].copy()
        future = cg_y[cg_y["MONTH"] > split_month].copy()

        if history.empty:
            continue

        grp = history.groupby(["grid_lat", "grid_lon"])
        feat = grp.agg(
            total_crimes=("LAT", "size"),
            # QUALITY_OF_LIFE = True means minor; False = serious
            serious_crimes=("QUALITY_OF_LIFE", lambda s: (~s).sum()),
        ).reset_index()

        feat["serious_ratio"] = np.where(
            feat["total_crimes"] > 0,
            feat["serious_crimes"] / feat["total_crimes"],
            0.0,
        )

        future_counts = (
            future.groupby(["grid_lat", "grid_lon"])
            .size()
            .reset_index(name="future_crimes")
        )

        data_y = feat.merge(future_counts, on=["grid_lat", "grid_lon"], how="left")
        data_y["future_crimes"] = data_y["future_crimes"].fillna(0)
        data_y["label_cluster"] = (
            data_y["future_crimes"] >= min_crimes_for_cluster
        ).astype(int)
        data_y["YEAR"] = y

        data_y["lat_center"] = data_y["grid_lat"] + grid_size / 2
        data_y["lon_center"] = data_y["grid_lon"] + grid_size / 2

        rows.append(data_y)

    if not rows:
        raise ValueError("No data built for any year in the specified range.")

    data_all = pd.concat(rows, ignore_index=True)
    return data_all


# ─────────────────────────────────────────
# Train logistic regression & compute risk
# ─────────────────────────────────────────

def train_crime_risk_model(data: pd.DataFrame) -> pd.DataFrame:
    """
    Take the multi-year grid-level dataset and fit a logistic regression model.

    Adds:
      - risk_score = P(cluster in future) for each (grid, year) row.
    """
    feature_cols = ["total_crimes", "serious_ratio"]

    X = data[feature_cols].fillna(0)
    y = data["label_cluster"]

    if y.nunique() < 2:
        data["risk_score"] = 0.0
        return data

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    data["risk_score"] = model.predict_proba(X)[:, 1]
    return data


# ─────────────────────────────────────────
# Build Folium heatmap
# ─────────────────────────────────────────

def build_risk_heatmap(
    risk_df: pd.DataFrame,
    center_lat: float = 43.048,
    center_lon: float = -76.147,
    zoom_start: int = 13,
    highlight_areas=None,
    top_points: pd.DataFrame | None = None,
) -> folium.Map:
    """
    risk_df: all grid cells with columns lat_center, lon_center, risk_score
    highlight_areas: big neighborhood circles (Downtown, Southside, etc.)
    top_points: DataFrame of specific hotspots to mark, with columns:
                lat_center, lon_center, risk_score, avg_future_crimes, (optional) rank
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    heat_data = risk_df[["lat_center", "lon_center", "risk_score"]].values.tolist()
    HeatMap(
        heat_data,
        radius=18,
        blur=15,
        min_opacity=0.3,
        max_opacity=0.9,
    ).add_to(m)

    if highlight_areas:
        for area in highlight_areas:
            folium.Circle(
                location=(area["lat"], area["lon"]),
                radius=area.get("radius", 900),
                color="#ff0000",
                weight=2,
                fill=True,
                fill_opacity=0.08,
                popup=f'{area["name"]} (ZIP {area["zip"]}) – Chronic Hotspot',
                tooltip=f'{area["name"]} – {area["zip"]}',
            ).add_to(m)

    if top_points is not None and not top_points.empty:
        for i, row in top_points.iterrows():
            lat = row["lat_center"]
            lon = row["lon_center"]
            risk = row["risk_score"]
            avg_future = row["avg_future_crimes"]
            rank = row.get("rank", i + 1)

            popup_text = (
                f"Top {rank} hotspot<br>"
                f"Risk score: {risk:.2f}<br>"
                f"Avg future crimes: {avg_future:.1f}"
            )

            folium.CircleMarker(
                location=(lat, lon),
                radius=7,
                color="#0000ff",
                fill=True,
                fill_color="#0000ff",
                fill_opacity=0.9,
                popup=popup_text,
                tooltip=f"Top {rank} hotspot",
            ).add_to(m)

    return m


# ─────────────────────────────────────────
# Public entry point for Streamlit tab
# ─────────────────────────────────────────

def run_hotspot_model():
    """
    Convenience function:
    - loads crime_clean (2023–2025)
    - builds multi-year dataset
    - trains model
    - returns (folium_map, top10_df)
    """
    crime_df = load_crime_clean_local()
    crime_df = crime_df[
        (crime_df["YEAR"] >= 2023) & (crime_df["YEAR"] <= 2025)
    ].copy()

    data = build_spatiotemporal_dataset(crime_df)
    data_with_risk = train_crime_risk_model(data)

    agg = (
        data_with_risk
        .groupby(["grid_lat", "grid_lon"], as_index=False)
        .agg(
            risk_score=("risk_score", "mean"),
            lat_center=("lat_center", "mean"),
            lon_center=("lon_center", "mean"),
            avg_future_crimes=("future_crimes", "mean"),
        )
    )

    top10 = (
        agg.sort_values("risk_score", ascending=False)
        .loc[:, ["lat_center", "lon_center", "risk_score", "avg_future_crimes"]]
        .head(10)
        .reset_index(drop=True)
    )

    top10_for_map = top10.copy()
    top10_for_map["rank"] = top10_for_map.index + 1

    highlight_areas = [
        {"name": "Downtown",     "zip": "13202", "lat": 43.0480, "lon": -76.1470, "radius": 900},
        {"name": "Southside",    "zip": "13207", "lat": 43.0400, "lon": -76.1520, "radius": 1100},
        {"name": "Eastside (SU)","zip": "13210", "lat": 43.0400, "lon": -76.1350, "radius": 1000},
        {"name": "Near Westside","zip": "13204", "lat": 43.0520, "lon": -76.1720, "radius": 1000},
    ]

    risk_map = build_risk_heatmap(
        agg,
        highlight_areas=highlight_areas,
        top_points=top10_for_map,
    )

    return risk_map, top10