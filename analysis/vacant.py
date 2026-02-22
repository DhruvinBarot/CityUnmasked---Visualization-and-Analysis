import pandas as pd
import plotly.express as px


def load_vacant():
    """Load and clean Vacant_Properties.csv."""
    df = pd.read_csv("Vacant_Properties.csv")
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df = df.rename(columns={
        'Latitude':        'lat',
        'Longitude':       'lon',
        'PropertyAddress': 'address',
        'Zip':             'zip_code',
        'neighborhood':    'neighborhood'
    })
    df['zip_code']  = df['zip_code'].astype(str).str.strip()
    df['is_active'] = (
        df['VPR_valid'].isna() | (df['VPR_valid'].str.strip() != 'Y')
    )
    return df


# ── Charts ────────────────────────────────────────────────────────────────────

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
        {True: 'Active / Unresolved', False: 'Resolved'}
    ).value_counts()
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
    active = (vacant[vacant['is_active']]['zip_code']
              .value_counts().head(8).reset_index())
    active.columns = ['Zip Code', 'Active Count']
    fig = px.bar(active, x='Zip Code', y='Active Count',
                 color='Active Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig