import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MONTH_MAP = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}


def load_crime():
    """Load and clean crime_clean.csv. Returns full multi-year DataFrame."""
    df = pd.read_csv("crime_clean.csv")
    df['month_name'] = df['MONTH'].map(MONTH_MAP)
    df = df.dropna(subset=['LAT', 'LON'])
    return df


# ── Charts ────────────────────────────────────────────────────────────────────

def fig_top_crimes(crime):
    top = crime['CRIME_TYPE'].value_counts().head(8).reset_index()
    top.columns = ['Crime Type', 'Count']
    fig = px.bar(top, x='Count', y='Crime Type', orientation='h',
                 color='Count', color_continuous_scale='Reds')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      coloraxis_showscale=False, height=380)
    return fig


def fig_qol_pie(crime):
    qol = crime['QUALITY_OF_LIFE'].map(
        {True: 'Quality of Life', False: 'Serious Crime'}
    ).value_counts()
    fig = px.pie(values=qol.values, names=qol.index,
                 color_discrete_sequence=['#f97316', '#334155'], hole=0.45)
    fig.update_layout(height=380)
    return fig


def fig_crime_by_month(crime):
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly = (crime.groupby('month_name').size()
               .reindex(month_order, fill_value=0).reset_index())
    monthly.columns = ['Month', 'Count']
    fig = px.line(monthly, x='Month', y='Count', markers=True,
                  color_discrete_sequence=['#f97316'])
    fig.update_layout(height=320)
    return fig


def fig_crime_by_hour(crime):
    hourly = crime.groupby('HOUR').size().reset_index()
    hourly.columns = ['Hour', 'Count']
    fig = px.bar(hourly, x='Hour', y='Count',
                 color='Count', color_continuous_scale='Oranges')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig