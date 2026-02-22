# ─────────────────────────────────────────────
# phase2_analysis.py
# CityUnmasked — Phase 2 Analysis Module
# ─────────────────────────────────────────────
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# ── TASK 1: Load & Clean Vacant Properties ──

df_vacant = pd.read_csv("vacant_properties.csv")

# Drop rows missing coordinates
vacant_before = len(df_vacant)
df_vacant = df_vacant.dropna(subset=["Latitude", "Longitude"])
vacant_dropped = vacant_before - len(df_vacant)

# Standardize column names
df_vacant = df_vacant.rename(columns={
    "Latitude": "lat",
    "Longitude": "lon",
    "PropertyAddress": "address",
    "Zip": "zip_code",
    "neighborhood": "neighborhood"
})

# Zip as clean string
df_vacant["zip_code"] = df_vacant["zip_code"].astype(str).str.strip()

# Active vacancy flag
df_vacant["is_active"] = df_vacant["VPR_valid"].isna() | (df_vacant["VPR_valid"].str.strip() != "Y")

# ── Verification Output ──
print(f"\n=== VACANT PROPERTIES ===")
print(f"Total records:       {vacant_before}")
print(f"Dropped (no coords): {vacant_dropped}")
print(f"Usable records:      {len(df_vacant)}")
print(f"Active vacancies:    {df_vacant['is_active'].sum()}")
print(f"Top zip codes:\n{df_vacant['zip_code'].value_counts().head(5)}")
print(f"Top neighborhoods:\n{df_vacant['neighborhood'].value_counts().head(5)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — GRANGER CAUSALITY TEST
# ══════════════════════════════════════════════════════════════════════════════
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

def run_granger_causality(crime, unfit):
    """
    Tests whether monthly unfit violation counts Granger-cause monthly crime counts.
    Returns a dict with results, the time series dataframe, and interpretation.
    """
    # ── Build monthly crime series ──
    crime['period'] = pd.to_datetime(
        crime['YEAR'].astype(str) + '-' + crime['MONTH'].astype(str).str.zfill(2)
    )
    monthly_crime = crime.groupby('period').size().reset_index()
    monthly_crime.columns = ['period', 'crime_count']

    # ── Build monthly unfit series ──
    unfit_ts = unfit.copy()
    unfit_ts['violation_date'] = pd.to_datetime(
        unfit_ts['violation_date'], format='mixed', utc=True
    )
    unfit_ts['period'] = unfit_ts['violation_date'].dt.to_period('M').dt.to_timestamp()
    monthly_unfit = unfit_ts.groupby('period').size().reset_index()
    monthly_unfit.columns = ['period', 'unfit_count']

    # ── Merge on overlapping period ──
    ts = pd.merge(monthly_crime, monthly_unfit, on='period', how='inner').sort_values('period')
    ts = ts.dropna()

    # Need at least 10 periods for meaningful results
    if len(ts) < 10:
        return None, ts, "Insufficient overlapping time periods for Granger test."

    # ── Stationarity check (ADF test) ──
    adf_crime = adfuller(ts['crime_count'], autolag='AIC')
    adf_unfit = adfuller(ts['unfit_count'], autolag='AIC')

    # First-difference if non-stationary (p > 0.05)
    if adf_crime[1] > 0.05:
        ts['crime_count'] = ts['crime_count'].diff().dropna()
    if adf_unfit[1] > 0.05:
        ts['unfit_count'] = ts['unfit_count'].diff().dropna()
    ts = ts.dropna()

    # ── Run Granger test (lags 1–4 months) ──
    max_lag = min(4, len(ts) // 4)
    data_for_test = ts[['crime_count', 'unfit_count']].values

    try:
        gc_results = grangercausalitytests(data_for_test, maxlag=max_lag, verbose=False)
    except Exception as e:
        return None, ts, f"Granger test error: {str(e)}"

    # ── Extract p-values per lag ──
    lag_results = []
    for lag in range(1, max_lag + 1):
        tests = gc_results[lag][0]
        p_ssr_ftest = tests['ssr_ftest'][1]
        p_ssr_chi2  = tests['ssr_chi2test'][1]
        lag_results.append({
            'lag_months': lag,
            'p_value_ftest':  round(p_ssr_ftest, 4),
            'p_value_chi2':   round(p_ssr_chi2, 4),
            'significant':    p_ssr_ftest < 0.05
        })

    results_df = pd.DataFrame(lag_results)

    # ── Interpretation ──
    significant_lags = results_df[results_df['significant']]['lag_months'].tolist()
    if significant_lags:
        interpretation = (
            f"✅ Granger causality detected at lag(s): {significant_lags} month(s). "
            f"This means unfit property violations statistically predict future crime counts "
            f"at those time offsets — supporting the problem statement that decay precedes crime."
        )
    else:
        interpretation = (
            "⚠️ No statistically significant Granger causality detected in this window. "
            "This may reflect the short overlapping time series (2023–2024). "
            "The spatial correlation remains strong, but directional causation requires longer data."
        )

    return results_df, ts, interpretation


def fig_granger_pvalues(results_df):
    """Bar chart of p-values per lag with significance threshold line."""
    if results_df is None:
        return None
    colors = ['#22c55e' if sig else '#ef4444'
              for sig in results_df['significant']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Lag {l}m" for l in results_df['lag_months']],
        y=results_df['p_value_ftest'],
        marker_color=colors,
        text=[f"p={p}" for p in results_df['p_value_ftest']],
        textposition='outside',
        name='F-test p-value'
    ))
    fig.add_hline(
        y=0.05, line_dash='dash', line_color='red',
        annotation_text='p = 0.05 significance threshold',
        annotation_position='top right'
    )
    fig.update_layout(
        title="Granger Causality: Does Decay Predict Crime?",
        xaxis_title="Lag (months)",
        yaxis_title="p-value (lower = more significant)",
        yaxis=dict(range=[0, max(results_df['p_value_ftest'].max() * 1.3, 0.15)]),
        height=380
    )
    return fig


def fig_granger_timeseries(ts):
    """Dual-axis line chart of monthly crime vs unfit violations."""
    if ts is None or len(ts) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts['period'], y=ts['crime_count'],
        name='Monthly Crime Count',
        line=dict(color='#f97316', width=2),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=ts['period'], y=ts['unfit_count'],
        name='Monthly Unfit Violations',
        line=dict(color='#3b82f6', width=2, dash='dot'),
        yaxis='y2'
    ))
    fig.update_layout(
        title="Monthly Crime vs Unfit Violations Over Time",
        xaxis_title="Month",
        yaxis=dict(title='Crime Count',      titlefont=dict(color='#f97316')),
        yaxis2=dict(title='Unfit Violations', titlefont=dict(color='#3b82f6'),
                    overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        height=380
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RANDOM FOREST CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def run_random_forest(crime_2024):
    """
    Trains a Random Forest to predict high-severity crime (SEVERITY >= 3)
    using temporal, spatial, and decay features.
    Returns model, feature names, accuracy, confusion matrix, classification report.
    """
    df = crime_2024.copy()

    # ── Target: binary high severity ──
    df['high_severity'] = (df['SEVERITY'] >= 3).astype(int)

    # ── Drop the one Unknown TIME_OF_DAY row ──
    df = df[df['TIME_OF_DAY'] != 'Unknown']

    # ── Feature engineering ──
    # Encode categoricals
    season_dummies  = pd.get_dummies(df['SEASON'],      prefix='season')
    tod_dummies     = pd.get_dummies(df['TIME_OF_DAY'], prefix='tod')
    day_dummies     = pd.get_dummies(df['DAY_OF_WEEK'], prefix='day')

    feature_df = pd.concat([
        df[['HOUR', 'MONTH', 'IS_WEEKEND',
            'near_unfit', 'near_vacant', 'near_decay']].reset_index(drop=True),
        season_dummies.reset_index(drop=True),
        tod_dummies.reset_index(drop=True),
        day_dummies.reset_index(drop=True)
    ], axis=1)

    # Convert booleans to int
    for col in feature_df.select_dtypes(include='bool').columns:
        feature_df[col] = feature_df[col].astype(int)

    X = feature_df
    y = df['high_severity'].reset_index(drop=True)

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── Train Random Forest ──
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred    = model.predict(X_test)
    accuracy  = round(model.score(X_test, y_test) * 100, 1)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred,
                                      target_names=['Low Severity', 'High Severity'],
                                      output_dict=True)

    # ── Feature importance ──
    importance_df = pd.DataFrame({
        'Feature':   X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    return model, importance_df, accuracy, cm, report, X.columns.tolist()


def fig_rf_feature_importance(importance_df):
    """Horizontal bar chart of top 15 feature importances."""
    colors = []
    for f in importance_df['Feature']:
        if 'near' in f.lower():
            colors.append('#dc2626')   # red — decay features
        elif 'hour' in f.lower() or 'tod' in f.lower() or 'time' in f.lower():
            colors.append('#f97316')   # orange — time features
        elif 'season' in f.lower() or 'month' in f.lower():
            colors.append('#f59e0b')   # amber — seasonal features
        else:
            colors.append('#6b7280')   # gray — other

    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in importance_df['Importance']],
        textposition='outside'
    ))
    fig.update_layout(
        title="Random Forest — Feature Importance for Predicting High-Severity Crime",
        xaxis_title="Importance Score",
        yaxis=dict(categoryorder='total ascending'),
        height=480,
        legend=dict(
            title="Feature Type",
            itemsizing='constant'
        )
    )
    # Add manual legend via invisible traces
    for label, color in [('Decay/Spatial', '#dc2626'),
                          ('Time of Day',   '#f97316'),
                          ('Seasonal',      '#f59e0b'),
                          ('Other',         '#6b7280')]:
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            marker_color=color,
            name=label,
            showlegend=True
        ))
    return fig


def fig_rf_confusion_matrix(cm):
    """Heatmap of the confusion matrix."""
    labels = ['Low Severity', 'High Severity']
    fig = px.imshow(
        cm,
        x=labels, y=labels,
        color_continuous_scale='Oranges',
        text_auto=True,
        labels=dict(x='Predicted', y='Actual')
    )
    fig.update_layout(
        title="Confusion Matrix — Random Forest Predictions",
        height=380,
        coloraxis_showscale=False
    )
    return fig


def fig_rf_metrics(report):
    """Bar chart of precision, recall, F1 for each class."""
    classes = ['Low Severity', 'High Severity']
    metrics = ['precision', 'recall', 'f1-score']
    colors  = ['#3b82f6', '#f97316', '#22c55e']

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=classes,
            y=[round(report[c][metric], 3) for c in classes],
            marker_color=color,
            text=[f"{report[c][metric]:.2f}" for c in classes],
            textposition='outside'
        ))
    fig.update_layout(
        barmode='group',
        title="Model Performance — Precision, Recall, F1",
        yaxis=dict(range=[0, 1.15]),
        height=380
    )
    return fig