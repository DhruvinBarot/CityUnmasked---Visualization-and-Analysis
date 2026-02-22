import folium
from folium.plugins import HeatMap


def build_map(crime, unfit_clean, vacant):
    """
    Three-layer interactive Folium map:
      Layer 1 — Crime heatmap (all years)
      Layer 2 — Unfit property markers (red=open, gray=closed)
      Layer 3 — Vacant property density heatmap (blue)
    """
    m = folium.Map(location=[43.048, -76.147], zoom_start=13,
                   tiles='CartoDB positron')

    # Crime heatmap
    crime_layer = folium.FeatureGroup(name='🌡️ Crime Heatmap')
    HeatMap(crime[['LAT', 'LON']].values.tolist(),
            radius=10, blur=12, min_opacity=0.4).add_to(crime_layer)
    crime_layer.add_to(m)

    # Unfit property markers
    unfit_layer = folium.FeatureGroup(name='🔴 Unfit Properties')
    for _, row in unfit_clean.iterrows():
        color = 'red' if row.get('status_type_name') == 'Open' else 'gray'
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5, color=color, fill=True, fill_opacity=0.85,
            tooltip=f"UNFIT | {row['address']} | {row.get('status_type_name','?')} | {row['year']}"
        ).add_to(unfit_layer)
    unfit_layer.add_to(m)

    # Vacant property heatmap
    vacant_layer = folium.FeatureGroup(name='🔵 Vacant Properties')
    HeatMap(
        vacant[['lat', 'lon']].values.tolist(),
        radius=8, blur=10, min_opacity=0.3,
        gradient={0.4: 'blue', 0.65: 'cyan', 1: 'aqua'}
    ).add_to(vacant_layer)
    vacant_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m