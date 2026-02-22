import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from sklearn.neighbors import BallTree
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

MONTH_MAP = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
             7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    # ── Crime (multi-year, pre-engineered features) ──
    crime = pd.read_csv("crime_clean.csv")
    crime['month_name'] = crime['MONTH'].map(MONTH_MAP)
    crime = crime.dropna(subset=['LAT', 'LON'])

    # ── Unfit Properties ──
    unfit = pd.read_csv("Unfit_Properties.csv")
    unfit['violation_date'] = pd.to_datetime(unfit['violation_date'], format='mixed', utc=True)
    unfit['year']           = unfit['violation_date'].dt.year
    unfit_clean             = unfit.dropna(subset=['Latitude', 'Longitude'])

    # ── Vacant Properties ──
    vacant = pd.read_csv("Vacant_Properties.csv")
    vacant = vacant.dropna(subset=['Latitude', 'Longitude'])
    vacant = vacant.rename(columns={
        'Latitude': 'lat', 'Longitude': 'lon',
        'PropertyAddress': 'address',
        'Zip': 'zip_code',
        'neighborhood': 'neighborhood'
    })
    vacant['zip_code']  = vacant['zip_code'].astype(str).str.strip()
    vacant['is_active'] = vacant['VPR_valid'].isna() | (vacant['VPR_valid'].str.strip() != 'Y')

    # ── Filter for spatial analysis ──
    crime_2024 = crime.copy()

    # ── Spatial Join: near unfit (100m) ──
    c_coords   = np.radians(crime_2024[['LAT', 'LON']].values)
    u_coords   = np.radians(unfit_clean[['Latitude', 'Longitude']].values)
    tree_unfit = BallTree(u_coords, metric='haversine')
    crime_2024['near_unfit'] = (
        tree_unfit.query_radius(c_coords, r=100/6_371_000, count_only=True) > 0
    )

    # ── Spatial Join: near vacant (100m) ──
    v_coords    = np.radians(vacant[['lat', 'lon']].values)
    tree_vacant = BallTree(v_coords, metric='haversine')
    crime_2024['near_vacant'] = (
        tree_vacant.query_radius(c_coords, r=100/6_371_000, count_only=True) > 0
    )

    # ── Decay zone classification ──
    crime_2024['near_decay'] = crime_2024['near_unfit'] | crime_2024['near_vacant']
    crime_2024['decay_zone'] = 'Neither'
    crime_2024.loc[ crime_2024['near_unfit'] & ~crime_2024['near_vacant'], 'decay_zone'] = 'Near Unfit Only'
    crime_2024.loc[~crime_2024['near_unfit'] &  crime_2024['near_vacant'], 'decay_zone'] = 'Near Vacant Only'
    crime_2024.loc[ crime_2024['near_unfit'] &  crime_2024['near_vacant'], 'decay_zone'] = 'Near Both'

    # ── Urban Decay Index ──
    decay = _build_decay_index(unfit_clean, vacant)

    # ── Assign zip codes to crimes via nearest decay centroid ──
    crime_2024 = assign_crime_zip(crime_2024, decay)

    return crime, crime_2024, unfit, unfit_clean, vacant, decay


def _build_decay_index(unfit_clean, vacant):
    unfit_df = pd.DataFrame({
        'lat':        unfit_clean['Latitude'].values,
        'lon':        unfit_clean['Longitude'].values,
        'zip_code':   unfit_clean['zip'].astype(str).str.strip().values,
        'decay_type': 'Unfit Property',
        'is_active':  (unfit_clean['status_type_name'] == 'Open').values
    })
    vacant_df = pd.DataFrame({
        'lat':        vacant['lat'].values,
        'lon':        vacant['lon'].values,
        'zip_code':   vacant['zip_code'].values,
        'decay_type': 'Vacant Property',
        'is_active':  vacant['is_active'].values
    })
    return pd.concat([unfit_df, vacant_df], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI METRICS
# ══════════════════════════════════════════════════════════════════════════════
def get_kpis(crime_2024, unfit, vacant):
    return {
        "total_crimes"    : len(crime_2024),
        "crime_types"     : crime_2024['CRIME_TYPE'].nunique(),
        "total_unfit"     : len(unfit),
        "open_violations" : int((unfit['status_type_name'] == 'Open').sum()),
        "pct_near_unfit"  : f"{crime_2024['near_unfit'].mean()*100:.0f}%",
        "total_vacant"    : len(vacant),
        "active_vacant"   : int(vacant['is_active'].sum()),
        "pct_near_decay"  : f"{crime_2024['near_decay'].mean()*100:.0f}%",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CRIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def fig_top_crimes(crime_2024):
    top = crime_2024['CRIME_TYPE'].value_counts().head(8).reset_index()
    top.columns = ['Crime Type', 'Count']
    fig = px.bar(top, x='Count', y='Crime Type', orientation='h',
                 color='Count', color_continuous_scale='Reds')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      coloraxis_showscale=False, height=380)
    return fig

def fig_qol_pie(crime_2024):
    qol = crime_2024['QUALITY_OF_LIFE'].map(
        {True: 'Quality of Life', False: 'Serious Crime'}).value_counts()
    fig = px.pie(values=qol.values, names=qol.index,
                 color_discrete_sequence=['#f97316', '#334155'], hole=0.45)
    fig.update_layout(height=380)
    return fig

def fig_crime_by_month(crime_2024):
    month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    monthly = crime_2024.groupby('month_name').size()\
                        .reindex(month_order, fill_value=0).reset_index()
    monthly.columns = ['Month', 'Count']
    fig = px.line(monthly, x='Month', y='Count', markers=True,
                  color_discrete_sequence=['#f97316'])
    fig.update_layout(height=320)
    return fig

def fig_crime_by_hour(crime_2024):
    hourly = crime_2024.groupby('HOUR').size().reset_index()
    hourly.columns = ['Hour', 'Count']
    fig = px.bar(hourly, x='Hour', y='Count',
                 color='Count', color_continuous_scale='Oranges')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig

def fig_near_vs_not(crime_2024):
    top_types = crime_2024['CRIME_TYPE'].value_counts().head(6).index
    near      = crime_2024[ crime_2024['near_unfit']]['CRIME_TYPE']\
                    .value_counts().reindex(top_types, fill_value=0)
    not_near  = crime_2024[~crime_2024['near_unfit']]['CRIME_TYPE']\
                    .value_counts().reindex(top_types, fill_value=0)
    fig = go.Figure(data=[
        go.Bar(name='Near Unfit (100m)', x=list(top_types),
               y=near.values,     marker_color='#f97316'),
        go.Bar(name='Not Near Unfit',    x=list(top_types),
               y=not_near.values, marker_color='#334155'),
    ])
    fig.update_layout(barmode='group', height=350)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UNFIT PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════
def fig_unfit_by_year(unfit):
    yearly = unfit.groupby('year').size().reset_index()
    yearly.columns = ['Year', 'Count']
    fig = px.bar(yearly, x='Year', y='Count',
                 color='Count', color_continuous_scale='Oranges')
    fig.add_hline(y=yearly['Count'].mean(), line_dash='dash', line_color='red',
                  annotation_text=f"Avg: {yearly['Count'].mean():.0f}")
    fig.update_layout(height=380, coloraxis_showscale=False)
    return fig

def fig_open_closed_pie(unfit):
    status = unfit['status_type_name'].value_counts()
    fig = px.pie(values=status.values, names=status.index,
                 color_discrete_sequence=['#ef4444', '#22c55e'], hole=0.45)
    fig.update_layout(height=380)
    return fig

def fig_unfit_by_zip(unfit):
    zips = unfit['zip'].value_counts().head(8).reset_index()
    zips.columns = ['Zip Code', 'Count']
    zips['Zip Code'] = zips['Zip Code'].astype(str)
    fig = px.bar(zips, x='Zip Code', y='Count',
                 color='Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig

def fig_open_by_zip(unfit):
    open_zips = unfit[unfit['status_type_name'] == 'Open']['zip']\
                    .value_counts().head(8).reset_index()
    open_zips.columns = ['Zip Code', 'Open Count']
    open_zips['Zip Code'] = open_zips['Zip Code'].astype(str)
    fig = px.bar(open_zips, x='Zip Code', y='Open Count',
                 color='Open Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VACANT PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════
def fig_vacant_by_neighborhood(vacant):
    nbr = vacant['neighborhood'].value_counts().head(8).reset_index()
    nbr.columns = ['Neighborhood', 'Count']
    fig = px.bar(nbr, x='Count', y='Neighborhood', orientation='h',
                 color='Count', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      coloraxis_showscale=False, height=380)
    return fig

def fig_vacant_active_pie(vacant):
    status = vacant['is_active'].map(
        {True: 'Active / Unresolved', False: 'Resolved'}).value_counts()
    fig = px.pie(values=status.values, names=status.index,
                 color_discrete_sequence=['#ef4444', '#22c55e'], hole=0.45)
    fig.update_layout(height=380)
    return fig

def fig_vacant_by_zip(vacant):
    zips = vacant['zip_code'].value_counts().head(8).reset_index()
    zips.columns = ['Zip Code', 'Count']
    fig = px.bar(zips, x='Zip Code', y='Count',
                 color='Count', color_continuous_scale='Blues')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig

def fig_vacant_active_by_zip(vacant):
    active_zips = vacant[vacant['is_active']]['zip_code']\
                      .value_counts().head(8).reset_index()
    active_zips.columns = ['Zip Code', 'Active Count']
    fig = px.bar(active_zips, x='Zip Code', y='Active Count',
                 color='Active Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — URBAN DECAY INDEX
# ══════════════════════════════════════════════════════════════════════════════
def get_proximity_stats(crime_2024):
    near_both = crime_2024['decay_zone'] == 'Near Both'
    return {
        'near_unfit_pct'  : f"{crime_2024['near_unfit'].mean()*100:.1f}%",
        'near_vacant_pct' : f"{crime_2024['near_vacant'].mean()*100:.1f}%",
        'near_decay_pct'  : f"{crime_2024['near_decay'].mean()*100:.1f}%",
        'near_both_pct'   : f"{near_both.mean()*100:.1f}%",
        'near_unfit_n'    : int(crime_2024['near_unfit'].sum()),
        'near_vacant_n'   : int(crime_2024['near_vacant'].sum()),
        'near_decay_n'    : int(crime_2024['near_decay'].sum()),
        'near_both_n'     : int(near_both.sum()),
    }

def fig_decay_by_zip(decay):
    top_zips = decay['zip_code'].value_counts().head(8).index
    by_zip   = decay[decay['zip_code'].isin(top_zips)]\
                   .groupby(['zip_code', 'decay_type']).size().reset_index()
    by_zip.columns = ['Zip Code', 'Decay Type', 'Count']
    fig = px.bar(by_zip, x='Zip Code', y='Count', color='Decay Type',
                 color_discrete_map={
                     'Unfit Property':  '#f97316',
                     'Vacant Property': '#3b82f6'
                 }, barmode='stack')
    fig.update_layout(height=380)
    return fig

def fig_decay_zone_crimes(crime_2024):
    zone_counts = crime_2024['decay_zone'].value_counts().reset_index()
    zone_counts.columns = ['Zone', 'Crime Count']
    color_map = {
        'Near Both':        '#dc2626',
        'Near Unfit Only':  '#f97316',
        'Near Vacant Only': '#3b82f6',
        'Neither':          '#6b7280'
    }
    fig = px.bar(zone_counts, x='Zone', y='Crime Count',
                 color='Zone', color_discrete_map=color_map)
    fig.update_layout(height=380, showlegend=False)
    return fig

def fig_crime_type_by_decay_zone(crime_2024):
    top_types = crime_2024['CRIME_TYPE'].value_counts().head(5).index
    df = crime_2024[crime_2024['CRIME_TYPE'].isin(top_types)]\
             .groupby(['CRIME_TYPE', 'decay_zone']).size().reset_index()
    df.columns = ['Crime Type', 'Zone', 'Count']
    color_map = {
        'Near Both':        '#dc2626',
        'Near Unfit Only':  '#f97316',
        'Near Vacant Only': '#3b82f6',
        'Neither':          '#6b7280'
    }
    fig = px.bar(df, x='Crime Type', y='Count', color='Zone',
                 color_discrete_map=color_map, barmode='group')
    fig.update_layout(height=400)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FOLIUM MAP
# ══════════════════════════════════════════════════════════════════════════════
def build_map(crime_2024, unfit_clean, vacant=None):
    m = folium.Map(location=[43.048, -76.147], zoom_start=13,
                   tiles='CartoDB positron')

    crime_layer = folium.FeatureGroup(name='🌡️ Crime Heatmap (2024)')
    HeatMap(crime_2024[['LAT', 'LON']].values.tolist(),
            radius=10, blur=12, min_opacity=0.4).add_to(crime_layer)
    crime_layer.add_to(m)

    unfit_layer = folium.FeatureGroup(name='🔴 Unfit Properties')
    for _, row in unfit_clean.iterrows():
        color = 'red' if row.get('status_type_name') == 'Open' else 'gray'
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5, color=color, fill=True, fill_opacity=0.85,
            tooltip=f"UNFIT | {row['address']} | {row.get('status_type_name','?')} | {row['year']}"
        ).add_to(unfit_layer)
    unfit_layer.add_to(m)

    if vacant is not None:
        vacant_layer = folium.FeatureGroup(name='🔵 Vacant Properties')
        HeatMap(
            vacant[['lat', 'lon']].values.tolist(),
            radius=8, blur=10, min_opacity=0.3,
            gradient={0.4: 'blue', 0.65: 'cyan', 1: 'aqua'}
        ).add_to(vacant_layer)
        vacant_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def fig_prediction(unfit):
    yearly     = unfit.groupby('year').size().reset_index()
    yearly.columns = ['Year', 'Count']
    yearly_fit = yearly[yearly['Year'] <= 2024]

    model  = LinearRegression().fit(yearly_fit[['Year']], yearly_fit['Count'])
    future = pd.DataFrame({'Year': [2025, 2026, 2027]})
    preds  = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['Count'],
                         name='Actual', marker_color='#f97316'))
    fig.add_trace(go.Scatter(
        x=future['Year'], y=preds,
        mode='lines+markers+text',
        text=[f"{int(p)}" for p in preds],
        textposition='top center',
        name='Predicted',
        line=dict(color='red', dash='dash'),
        marker=dict(size=10)
    ))
    fig.update_layout(title="Unfit Violations: Actual + Forecast",
                      xaxis_title="Year", yaxis_title="Violations", height=420)
    return fig, future['Year'].tolist(), [int(p) for p in preds]


# ══════════════════════════════════════════════════════════════════════════════
# GRANGER CAUSALITY
# ══════════════════════════════════════════════════════════════════════════════
def run_granger_causality(crime, unfit):
    crime = crime.copy()
    crime['period'] = pd.to_datetime(
        crime['YEAR'].astype(str) + '-' + crime['MONTH'].astype(str).str.zfill(2)
    )
    monthly_crime = crime.groupby('period').size().reset_index()
    monthly_crime.columns = ['period', 'crime_count']

    unfit_ts = unfit.copy()
    unfit_ts['violation_date'] = pd.to_datetime(
        unfit_ts['violation_date'], format='mixed', utc=True
    )
    unfit_ts['period'] = unfit_ts['violation_date'].dt.to_period('M').dt.to_timestamp()
    monthly_unfit = unfit_ts.groupby('period').size().reset_index()
    monthly_unfit.columns = ['period', 'unfit_count']

    ts = pd.merge(monthly_crime, monthly_unfit, on='period', how='inner').sort_values('period')
    ts = ts.dropna()

    if len(ts) < 10:
        return None, ts, "Insufficient overlapping time periods for Granger test."

    adf_crime = adfuller(ts['crime_count'], autolag='AIC')
    adf_unfit = adfuller(ts['unfit_count'], autolag='AIC')

    if adf_crime[1] > 0.05:
        ts['crime_count'] = ts['crime_count'].diff()
    if adf_unfit[1] > 0.05:
        ts['unfit_count'] = ts['unfit_count'].diff()
    ts = ts.dropna()

    max_lag = min(4, len(ts) // 4)
    data_for_test = ts[['crime_count', 'unfit_count']].values

    try:
        gc_results = grangercausalitytests(data_for_test, maxlag=max_lag, verbose=False)
    except Exception as e:
        return None, ts, f"Granger test error: {str(e)}"

    lag_results = []
    for lag in range(1, max_lag + 1):
        tests       = gc_results[lag][0]
        p_ssr_ftest = tests['ssr_ftest'][1]
        p_ssr_chi2  = tests['ssr_chi2test'][1]
        lag_results.append({
            'lag_months':     lag,
            'p_value_ftest':  round(p_ssr_ftest, 4),
            'p_value_chi2':   round(p_ssr_chi2, 4),
            'significant':    p_ssr_ftest < 0.05
        })

    results_df       = pd.DataFrame(lag_results)
    significant_lags = results_df[results_df['significant']]['lag_months'].tolist()

    if significant_lags:
        interpretation = (
            f"✅ Granger causality detected at lag(s): {significant_lags} month(s). "
            f"Unfit property violations statistically predict future crime counts — "
            f"supporting the problem statement that decay precedes crime."
        )
    else:
        interpretation = (
            "⚠️ No statistically significant Granger causality detected in this window. "
            "This may reflect the short overlapping time series. "
            "The spatial correlation remains strong, but directional causation requires longer data."
        )

    return results_df, ts, interpretation


def fig_granger_pvalues(results_df):
    if results_df is None:
        return None
    colors = ['#22c55e' if sig else '#ef4444' for sig in results_df['significant']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Lag {l}m" for l in results_df['lag_months']],
        y=results_df['p_value_ftest'],
        marker_color=colors,
        text=[f"p={p}" for p in results_df['p_value_ftest']],
        textposition='outside',
        name='F-test p-value'
    ))
    fig.add_hline(y=0.05, line_dash='dash', line_color='red',
                  annotation_text='p=0.05 threshold', annotation_position='top right')
    fig.update_layout(
        title="Granger Causality: Does Decay Predict Crime?",
        xaxis_title="Lag (months)",
        yaxis_title="p-value (lower = more significant)",
        yaxis=dict(range=[0, max(results_df['p_value_ftest'].max() * 1.3, 0.15)]),
        height=380
    )
    return fig


def fig_granger_timeseries(ts):
    if ts is None or len(ts) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts['period'], y=ts['crime_count'],
        name='Monthly Crime Count',
        line=dict(color='#f97316', width=2), yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=ts['period'], y=ts['unfit_count'],
        name='Monthly Unfit Violations',
        line=dict(color='#3b82f6', width=2, dash='dot'), yaxis='y2'
    ))
    fig.update_layout(
        title="Monthly Crime vs Unfit Violations Over Time",
        xaxis_title="Month",
        yaxis=dict(title=dict(text='Crime Count',      font=dict(color='#f97316'))),
        yaxis2=dict(title=dict(text='Unfit Violations', font=dict(color='#3b82f6')),
                    overlaying='y', side='right'),
        height=380
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
def run_random_forest(crime_2024):
    df = crime_2024.copy()
    df['high_severity'] = (df['SEVERITY'] >= 3).astype(int)
    df = df[df['TIME_OF_DAY'] != 'Unknown']

    season_dummies = pd.get_dummies(df['SEASON'],      prefix='season')
    tod_dummies    = pd.get_dummies(df['TIME_OF_DAY'], prefix='tod')
    day_dummies    = pd.get_dummies(df['DAY_OF_WEEK'], prefix='day')

    feature_df = pd.concat([
        df[['HOUR', 'MONTH', 'IS_WEEKEND',
            'near_unfit', 'near_vacant', 'near_decay']].reset_index(drop=True),
        season_dummies.reset_index(drop=True),
        tod_dummies.reset_index(drop=True),
        day_dummies.reset_index(drop=True)
    ], axis=1)

    for col in feature_df.select_dtypes(include='bool').columns:
        feature_df[col] = feature_df[col].astype(int)

    X = feature_df
    y = df['high_severity'].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = round(model.score(X_test, y_test) * 100, 1)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=['Low Severity', 'High Severity'],
                                     output_dict=True)

    importance_df = pd.DataFrame({
        'Feature':    X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    return model, importance_df, accuracy, cm, report, X.columns.tolist()


def fig_rf_feature_importance(importance_df):
    colors = []
    for f in importance_df['Feature']:
        if 'near' in f.lower():
            colors.append('#dc2626')
        elif 'hour' in f.lower() or 'tod' in f.lower():
            colors.append('#f97316')
        elif 'season' in f.lower() or 'month' in f.lower():
            colors.append('#f59e0b')
        else:
            colors.append('#6b7280')

    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in importance_df['Importance']],
        textposition='outside'
    ))
    for label, color in [('Decay/Spatial', '#dc2626'), ('Time of Day', '#f97316'),
                          ('Seasonal',      '#f59e0b'), ('Other',       '#6b7280')]:
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color,
                             name=label, showlegend=True))
    fig.update_layout(
        title="Random Forest — Feature Importance for Predicting High-Severity Crime",
        xaxis_title="Importance Score",
        yaxis=dict(categoryorder='total ascending'),
        height=480
    )
    return fig


def fig_rf_confusion_matrix(cm):
    labels = ['Low Severity', 'High Severity']
    fig = px.imshow(cm, x=labels, y=labels,
                    color_continuous_scale='Oranges', text_auto=True,
                    labels=dict(x='Predicted', y='Actual'))
    fig.update_layout(title="Confusion Matrix", height=380, coloraxis_showscale=False)
    return fig


def fig_rf_metrics(report):
    classes = ['Low Severity', 'High Severity']
    metrics = ['precision', 'recall', 'f1-score']
    colors  = ['#3b82f6', '#f97316', '#22c55e']
    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=classes,
            y=[round(report[c][metric], 3) for c in classes],
            marker_color=color,
            text=[f"{report[c][metric]:.2f}" for c in classes],
            textposition='outside'
        ))
    fig.update_layout(barmode='group',
                      title="Model Performance — Precision, Recall, F1",
                      yaxis=dict(range=[0, 1.15]), height=380)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# NEIGHBORHOOD CLASSIFICATION (Type A / B / C)
# ══════════════════════════════════════════════════════════════════════════════
def assign_crime_zip(crime_2024, decay):
    zip_centroids = (
        decay.groupby('zip_code')[['lat', 'lon']]
        .mean()
        .reset_index()
        .dropna()
    )
    zip_centroids = zip_centroids[
        zip_centroids['zip_code'].str.match(r'^\d{5}$')
    ].reset_index(drop=True)

    centroid_coords = np.radians(zip_centroids[['lat', 'lon']].values)
    crime_coords    = np.radians(crime_2024[['LAT', 'LON']].values)

    tree     = BallTree(centroid_coords, metric='haversine')
    _, idx   = tree.query(crime_coords, k=1)

    crime_2024 = crime_2024.copy()
    crime_2024['zip_code'] = zip_centroids['zip_code'].iloc[
        idx.flatten()
    ].values

    return crime_2024


def classify_neighborhoods(crime_2024, decay, unfit):
    # ── Crime count per zip ──
    crime_zip = (
        crime_2024.groupby('zip_code')
        .size()
        .reset_index(name='crime_count')
    )

    # ── Decay score per zip ──
    decay_zip = (
        decay.groupby(['zip_code', 'decay_type'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    decay_zip.columns.name = None
    if 'Unfit Property'  not in decay_zip.columns: decay_zip['Unfit Property']  = 0
    if 'Vacant Property' not in decay_zip.columns: decay_zip['Vacant Property'] = 0
    decay_zip['decay_score'] = decay_zip['Unfit Property'] + decay_zip['Vacant Property']
    decay_zip['unfit_ratio'] = (
        decay_zip['Unfit Property'] / decay_zip['decay_score'].replace(0, 1)
    )

    # ── Unresolved rate per zip ──
    unfit_open             = unfit.copy()
    unfit_open['zip_code'] = unfit_open['zip'].astype(str).str.strip()
    unfit_open['is_open']  = unfit_open['status_type_name'] == 'Open'
    unresolved = (
        unfit_open.groupby('zip_code')
        .agg(total_unfit=('is_open', 'count'),
             open_unfit =('is_open', 'sum'))
        .reset_index()
    )
    unresolved['pct_unresolved'] = (
        unresolved['open_unfit'] / unresolved['total_unfit']
    )

    # ── Merge ──
    nbr = decay_zip.merge(crime_zip, on='zip_code', how='left')
    nbr = nbr.merge(
        unresolved[['zip_code', 'pct_unresolved', 'total_unfit', 'open_unfit']],
        on='zip_code', how='left'
    )
    nbr['crime_count']    = nbr['crime_count'].fillna(0)
    nbr['pct_unresolved'] = nbr['pct_unresolved'].fillna(0)

    # ── Thresholds ──
    crime_median          = nbr['crime_count'].median()
    decay_median          = nbr['decay_score'].median()
    unfit_ratio_threshold = 0.4

    def classify(row):
        high_crime  = row['crime_count'] > crime_median
        high_decay  = row['decay_score'] > decay_median
        unfit_heavy = row['unfit_ratio'] > unfit_ratio_threshold
        if high_crime and high_decay:
            return 'Type A — Crime-Blight Feedback'
        elif high_decay and not high_crime:
            return 'Type B — Economic Abandonment'
        elif unfit_heavy and not high_crime:
            return 'Type C — Infrastructure Decay'
        else:
            return 'Low Risk / Monitoring'

    nbr['zone_type'] = nbr.apply(classify, axis=1)

    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    nbr['risk_score'] = (
        norm(nbr['crime_count'])    * 0.40 +
        norm(nbr['decay_score'])    * 0.35 +
        norm(nbr['pct_unresolved']) * 0.25
    ) * 100

    return nbr.sort_values('risk_score', ascending=False)


def get_economic_abandonment_zones(crime_2024, decay):
    crime_zip = (
        crime_2024.groupby('zip_code')
        .size()
        .reset_index(name='crime_count')
    )
    low_crime_threshold = crime_zip['crime_count'].quantile(0.25)
    low_crime_zips      = crime_zip[
        crime_zip['crime_count'] <= low_crime_threshold
    ]['zip_code'].tolist()

    economic_abandoned = decay[
        (decay['decay_type'] == 'Vacant Property') &
        (decay['zip_code'].isin(low_crime_zips))
    ].copy()

    return economic_abandoned, low_crime_zips


def fig_zone_type_breakdown(nbr):
    counts = nbr['zone_type'].value_counts().reset_index()
    counts.columns = ['Zone Type', 'Zip Code Count']
    color_map = {
        'Type A — Crime-Blight Feedback': '#dc2626',
        'Type B — Economic Abandonment':  '#3b82f6',
        'Type C — Infrastructure Decay':  '#f59e0b',
        'Low Risk / Monitoring':          '#6b7280'
    }
    fig = px.bar(counts, x='Zone Type', y='Zip Code Count',
                 color='Zone Type', color_discrete_map=color_map,
                 text='Zip Code Count')
    fig.update_layout(title="Zip Code Count by Decay Zone Classification",
                      showlegend=False, height=380)
    return fig


def fig_risk_score_ranking(nbr):
    top       = nbr.head(10).copy()
    top['zip_code'] = top['zip_code'].astype(str)
    color_map = {
        'Type A — Crime-Blight Feedback': '#dc2626',
        'Type B — Economic Abandonment':  '#3b82f6',
        'Type C — Infrastructure Decay':  '#f59e0b',
        'Low Risk / Monitoring':          '#6b7280'
    }
    fig = px.bar(top, x='risk_score', y='zip_code',
                 orientation='h', color='zone_type',
                 color_discrete_map=color_map,
                 text=top['risk_score'].round(1),
                 labels={'risk_score': 'Risk Score (0–100)',
                         'zip_code':   'Zip Code',
                         'zone_type':  'Zone Type'})
    fig.update_layout(title="Top 10 Zip Codes by Composite Risk Score",
                      yaxis=dict(categoryorder='total ascending'),
                      height=420)
    return fig


def fig_crime_vs_decay_scatter(nbr):
    color_map = {
        'Type A — Crime-Blight Feedback': '#dc2626',
        'Type B — Economic Abandonment':  '#3b82f6',
        'Type C — Infrastructure Decay':  '#f59e0b',
        'Low Risk / Monitoring':          '#6b7280'
    }
    fig = px.scatter(
        nbr, x='decay_score', y='crime_count',
        color='zone_type', color_discrete_map=color_map,
        hover_data=['zip_code', 'pct_unresolved', 'risk_score'],
        size='risk_score', size_max=30,
        labels={
            'decay_score': 'Urban Decay Score (unfit + vacant count)',
            'crime_count': '2024 Crime Count',
            'zone_type':   'Zone Type'
        },
        title="Crime vs Decay Score — Every Zip Code"
    )
    fig.add_vline(x=nbr['decay_score'].median(), line_dash='dash',
                  line_color='gray', annotation_text='Decay median')
    fig.add_hline(y=nbr['crime_count'].median(), line_dash='dash',
                  line_color='gray', annotation_text='Crime median')
    fig.update_layout(height=460)
    return fig


def fig_economic_abandonment(economic_abandoned):
    if len(economic_abandoned) == 0:
        return None
    by_zip = economic_abandoned['zip_code'].value_counts().head(8).reset_index()
    by_zip.columns = ['Zip Code', 'Vacant Properties']
    fig = px.bar(by_zip, x='Zip Code', y='Vacant Properties',
                 color='Vacant Properties', color_continuous_scale='Blues',
                 title="Economically Abandoned Vacancies — Low Crime Zip Codes")
    fig.update_layout(height=340, coloraxis_showscale=False)
    return fig