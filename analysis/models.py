import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ── Granger Causality ─────────────────────────────────────────────────────────

def run_granger_causality(crime, cv):
    """
    Bidirectional Granger causality test using code violations monthly
    time series (108 months, 2017–2026). Tests:
      Direction 1: violations → crime
      Direction 2: crime → violations
    """
    crime = crime.copy()
    crime['period'] = pd.to_datetime(
        crime['YEAR'].astype(str) + '-' + crime['MONTH'].astype(str).str.zfill(2)
    )
    monthly_crime = crime.groupby('period').size().reset_index()
    monthly_crime.columns = ['period', 'crime_count']

    monthly_cv = cv.groupby('period').size().reset_index()
    monthly_cv.columns = ['period', 'violation_count']

    ts = pd.merge(monthly_crime, monthly_cv, on='period', how='inner').sort_values('period')
    ts = ts.dropna()

    if len(ts) < 24:
        return None, None, ts, "Insufficient overlapping time periods."

    # Stationarity — difference if needed
    ts_diff = ts.copy()
    if adfuller(ts['crime_count'], autolag='AIC')[1] > 0.05:
        ts_diff['crime_count']     = ts_diff['crime_count'].diff()
    if adfuller(ts['violation_count'], autolag='AIC')[1] > 0.05:
        ts_diff['violation_count'] = ts_diff['violation_count'].diff()
    ts_diff = ts_diff.dropna()

    max_lag = min(6, len(ts_diff) // 5)

    def _run_test(data, direction):
        try:
            gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            return [{'lag_months': lag,
                     'p_value':    round(gc[lag][0]['ssr_ftest'][1], 4),
                     'significant': gc[lag][0]['ssr_ftest'][1] < 0.05,
                     'direction':  direction}
                    for lag in range(1, max_lag + 1)]
        except Exception:
            return []

    lags1 = _run_test(ts_diff[['crime_count', 'violation_count']].values,
                      'Violations → Crime')
    lags2 = _run_test(ts_diff[['violation_count', 'crime_count']].values,
                      'Crime → Violations')

    results_df = pd.DataFrame(lags1 + lags2)
    sig1 = [r['lag_months'] for r in lags1 if r['significant']]
    sig2 = [r['lag_months'] for r in lags2 if r['significant']]

    if sig1 and sig2:
        interpretation = (
            f"🔄 Bidirectional feedback detected. Violations predict crime at lag(s) "
            f"{sig1} months AND crime predicts violations at lag(s) {sig2} months. "
            f"This confirms a reinforcing feedback loop."
        )
    elif sig1:
        interpretation = (
            f"✅ Violations → Crime causality at lag(s) {sig1} month(s). "
            f"Physical decay statistically precedes crime increases. "
            f"Reverse direction not significant — decay is the leading signal."
        )
    elif sig2:
        interpretation = (
            f"⚠️ Crime → Violations causality at lag(s) {sig2} month(s). "
            f"Crime increases appear to precede property deterioration — "
            f"consistent with resident flight and accelerated abandonment."
        )
    else:
        interpretation = (
            "⚠️ No significant Granger causality in either direction. "
            "The relationship may be contemporaneous or driven by a shared "
            "underlying factor such as disinvestment or poverty."
        )

    return results_df, (sig1, sig2), ts, interpretation


def fig_granger_pvalues(results_df):
    if results_df is None or len(results_df) == 0:
        return None

    fig = go.Figure()
    for direction, color_sig, color_not in [
        ('Violations → Crime', '#22c55e', '#f97316'),
        ('Crime → Violations', '#22c55e', '#3b82f6')
    ]:
        subset = results_df[results_df['direction'] == direction]
        if subset.empty:
            continue
        colors = [color_sig if s else color_not for s in subset['significant']]
        fig.add_trace(go.Bar(
            name=direction,
            x=[f"Lag {l}m" for l in subset['lag_months']],
            y=subset['p_value'],
            marker_color=colors,
            text=[f"p={p}" for p in subset['p_value']],
            textposition='outside',
            offsetgroup=direction
        ))

    fig.add_hline(y=0.05, line_dash='dash', line_color='red',
                  annotation_text='p=0.05 threshold',
                  annotation_position='top right')
    fig.update_layout(
        title="Granger Causality — Violations ↔ Crime (Both Directions)",
        xaxis_title="Lag (months)",
        yaxis_title="p-value (below 0.05 = significant)",
        barmode='group', height=420
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
        x=ts['period'], y=ts['violation_count'],
        name='Monthly Code Violations',
        line=dict(color='#dc2626', width=2, dash='dot'), yaxis='y2'
    ))
    fig.update_layout(
        title="Monthly Crime vs Code Violations (2017–2026)",
        xaxis_title="Month",
        yaxis=dict(title=dict(text='Crime Count',
                              font=dict(color='#f97316'))),
        yaxis2=dict(title=dict(text='Code Violations',
                               font=dict(color='#dc2626')),
                    overlaying='y', side='right'),
        height=400
    )
    return fig


# ── Random Forest ─────────────────────────────────────────────────────────────

def run_random_forest(crime):
    """
    Random Forest predicting high-severity crime (SEVERITY >= 3).
    Features: temporal (hour, month, season, day, weekend) +
              spatial decay proximity (near_unfit, near_vacant, near_decay) +
              violation density (violation_count, severity_score, critical flag).
    """
    df = crime.copy()
    df['high_severity'] = (df['SEVERITY'] >= 3).astype(int)
    df = df[df['TIME_OF_DAY'] != 'Unknown']

    base_cols = ['HOUR', 'MONTH', 'IS_WEEKEND',
                 'near_unfit', 'near_vacant', 'near_decay']
    for col in ['violation_count', 'violation_severity_score',
                'has_critical_violation']:
        if col in df.columns:
            base_cols.append(col)

    feature_df = pd.concat([
        df[base_cols].reset_index(drop=True),
        pd.get_dummies(df['SEASON'],      prefix='season').reset_index(drop=True),
        pd.get_dummies(df['TIME_OF_DAY'], prefix='tod').reset_index(drop=True),
        pd.get_dummies(df['DAY_OF_WEEK'], prefix='day').reset_index(drop=True)
    ], axis=1)

    for col in feature_df.select_dtypes(include='bool').columns:
        feature_df[col] = feature_df[col].astype(int)
    feature_df = feature_df.fillna(0)

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

    return model, importance_df, accuracy, cm, report


def fig_rf_feature_importance(importance_df):
    colors = []
    for f in importance_df['Feature']:
        f = f.lower()
        if 'near' in f or 'violation' in f or 'critical' in f:
            colors.append('#dc2626')
        elif 'hour' in f or 'tod' in f:
            colors.append('#f97316')
        elif 'season' in f or 'month' in f:
            colors.append('#f59e0b')
        else:
            colors.append('#6b7280')

    fig = go.Figure(go.Bar(
        x=importance_df['Importance'], y=importance_df['Feature'],
        orientation='h', marker_color=colors,
        text=[f"{v:.3f}" for v in importance_df['Importance']],
        textposition='outside'
    ))
    for label, color in [('Decay / Violation', '#dc2626'),
                          ('Time of Day', '#f97316'),
                          ('Seasonal', '#f59e0b'),
                          ('Other', '#6b7280')]:
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color,
                             name=label, showlegend=True))
    fig.update_layout(
        title="Feature Importance — What Predicts High-Severity Crime?",
        xaxis_title="Importance Score",
        yaxis=dict(categoryorder='total ascending'),
        height=480
    )
    return fig


def fig_rf_metrics(report):
    classes = ['Low Severity', 'High Severity']
    fig = go.Figure()
    for metric, color in [('precision', '#3b82f6'),
                           ('recall',    '#f97316'),
                           ('f1-score',  '#22c55e')]:
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