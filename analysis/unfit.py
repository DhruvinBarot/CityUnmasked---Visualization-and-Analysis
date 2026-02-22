import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


def load_unfit():
    """Load and clean Unfit_Properties.csv."""
    df = pd.read_csv("Unfit_Properties.csv")
    df['violation_date'] = pd.to_datetime(
        df['violation_date'], format='mixed', utc=True
    )
    df['year'] = df['violation_date'].dt.year
    return df


def get_unfit_clean(unfit):
    """Return rows with valid coordinates."""
    return unfit.dropna(subset=['Latitude', 'Longitude'])


# ── Charts ────────────────────────────────────────────────────────────────────

def fig_unfit_by_year(unfit):
    yearly = unfit.groupby('year').size().reset_index()
    yearly.columns = ['Year', 'Count']
    fig = px.bar(yearly, x='Year', y='Count',
                 color='Count', color_continuous_scale='Oranges')
    fig.add_hline(y=yearly['Count'].mean(), line_dash='dash',
                  line_color='red',
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
    open_zips = (unfit[unfit['status_type_name'] == 'Open']['zip']
                 .value_counts().head(8).reset_index())
    open_zips.columns = ['Zip Code', 'Open Count']
    open_zips['Zip Code'] = open_zips['Zip Code'].astype(str)
    fig = px.bar(open_zips, x='Zip Code', y='Open Count',
                 color='Open Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig


def fig_prediction(unfit):
    """Linear forecast of violations through 2027."""
    yearly = unfit.groupby('year').size().reset_index()
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
                      xaxis_title="Year", yaxis_title="Violations",
                      height=420)
    return fig, future['Year'].tolist(), [int(p) for p in preds]