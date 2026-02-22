import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import BallTree

# ── Filter and tier configuration ─────────────────────────────────────────────

KEEP_COMPLAINT_TYPES = {
    'Property Maintenance-Int',
    'Property Maintenance-Ext',
    'Vacant House',
    'Overgrowth: Private, Occ',
    'Trash/Debris-Private, Occ',
    'Fire Safety',
    'Vacant Lot',
}

TIER_1_KEYWORDS = [
    '107.1.3', 'unfit for human', 'structural members',
    '304.10', 'stairways', '305.4', 'stairs and walking',
    '304.2', 'protective treatment', '27-32 (b)', 'stairs, porches'
]
TIER_2_KEYWORDS = [
    '305.3', 'interior surfaces', '504.1', 'plumbing',
    '304.13', 'window', 'skylight', '605.1', 'installation',
    '603.1', 'mechanical', 'appliances', '309.1', 'infestation',
    '705.1', 'carbon monoxide', '304.15', 'doors', '305.6',
    'interior doors', 'lead abatement', '27-57', 'receptacle',
    '27-32 (d)', 'protective coating', '27-31', 'structural'
]
TIER_3_KEYWORDS = [
    '27-72', 'overgrowth', 'trash', 'debris',
    '308.1', 'rubbish', 'garbage', '27-116', 'vacant property registry'
]
EXCLUDE_KEYWORDS = [
    '27-133 registration', '27-43', 'certification',
    '105.2', 'building permit'
]

TIER_LABELS = {
    3: 'Structural / Critical',
    2: 'Systems Failure',
    1: 'Environmental Neglect'
}

COLOR_MAP_TIER = {
    'Structural / Critical': '#dc2626',
    'Systems Failure':       '#f97316',
    'Environmental Neglect': '#f59e0b'
}


def _assign_tier(violation_text):
    if pd.isna(violation_text):
        return 1
    v = violation_text.lower()
    for kw in EXCLUDE_KEYWORDS:
        if kw in v:
            return 0
    for kw in TIER_1_KEYWORDS:
        if kw.lower() in v:
            return 3
    for kw in TIER_2_KEYWORDS:
        if kw.lower() in v:
            return 2
    for kw in TIER_3_KEYWORDS:
        if kw.lower() in v:
            return 1
    return 1


def load_code_violations():
    """
    Load code_violations.csv filtered to physical decay only.
    Returns clean DataFrame with tier labels and standardized columns.
    """
    df = pd.read_csv("code_violations.csv")

    df['violation_date'] = pd.to_datetime(
        df['violation_date'], format='mixed', utc=True
    )
    df['year']   = df['violation_date'].dt.year
    df['month']  = df['violation_date'].dt.month
    df['period'] = df['violation_date'].dt.to_period('M').dt.to_timestamp()

    df = df[df['complaint_type_name'].isin(KEEP_COMPLAINT_TYPES)].copy()
    df['tier'] = df['violation'].apply(_assign_tier)
    df = df[df['tier'] > 0].copy()
    df = df.dropna(subset=['Latitude', 'Longitude'])

    df = df.rename(columns={
        'complaint_address': 'address',
        'complaint_zip':     'zip_code',
        'Neighborhood':      'neighborhood',
        'Latitude':          'lat',
        'Longitude':         'lon'
    })
    df['zip_code']   = df['zip_code'].astype(str).str.strip()
    df['is_open']    = df['status_type_name'] == 'Open'
    df['tier_label'] = df['tier'].map(TIER_LABELS)

    return df


def get_violation_time_series(cv):
    """Monthly violation counts by tier — used for Granger analysis."""
    monthly = (
        cv.groupby(['period', 'tier_label'])
        .size().unstack(fill_value=0).reset_index()
    )
    monthly['total'] = monthly.drop('period', axis=1).sum(axis=1)
    return monthly.sort_values('period')


def add_violation_features(crime_df, cv):
    """
    Add per-crime violation features using BallTree 100m radius:
      - violation_count: total decay violations nearby
      - violation_severity_score: sum of tier weights
      - has_critical_violation: any Tier 1 structural violation nearby
    """
    if len(cv) == 0:
        crime_df['violation_count']          = 0
        crime_df['violation_severity_score'] = 0
        crime_df['has_critical_violation']   = False
        return crime_df

    c_coords  = np.radians(crime_df[['LAT', 'LON']].values)
    cv_coords = np.radians(cv[['lat', 'lon']].values)
    tree      = BallTree(cv_coords, metric='haversine')
    tiers     = cv['tier'].values

    counts  = tree.query_radius(c_coords, r=100/6_371_000, count_only=True)
    indices = tree.query_radius(c_coords, r=100/6_371_000, count_only=False)

    crime_df = crime_df.copy()
    crime_df['violation_count']          = counts
    crime_df['violation_severity_score'] = [
        tiers[idx].sum() if len(idx) > 0 else 0 for idx in indices
    ]
    crime_df['has_critical_violation'] = [
        bool((tiers[idx] == 3).any()) if len(idx) > 0 else False
        for idx in indices
    ]
    return crime_df


# ── Charts ────────────────────────────────────────────────────────────────────

def fig_violations_by_year_tier(cv):
    yearly = cv.groupby(['year', 'tier_label']).size().reset_index()
    yearly.columns = ['Year', 'Tier', 'Count']
    fig = px.bar(yearly, x='Year', y='Count', color='Tier',
                 color_discrete_map=COLOR_MAP_TIER, barmode='stack')
    fig.update_layout(height=380)
    return fig


def fig_tier_pie(cv):
    counts = cv['tier_label'].value_counts()
    fig = px.pie(values=counts.values, names=counts.index,
                 color=counts.index,
                 color_discrete_map=COLOR_MAP_TIER, hole=0.45)
    fig.update_layout(height=380)
    return fig


def fig_violations_by_zip(cv):
    zips = cv['zip_code'].value_counts().head(8).reset_index()
    zips.columns = ['Zip Code', 'Count']
    fig = px.bar(zips, x='Zip Code', y='Count',
                 color='Count', color_continuous_scale='Reds')
    fig.update_layout(height=320, coloraxis_showscale=False)
    return fig


def fig_violations_by_neighborhood(cv):
    nbr = cv['neighborhood'].value_counts().head(8).reset_index()
    nbr.columns = ['Neighborhood', 'Count']
    fig = px.bar(nbr, x='Count', y='Neighborhood', orientation='h',
                 color='Count', color_continuous_scale='Reds')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      height=320, coloraxis_showscale=False)
    return fig