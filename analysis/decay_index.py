import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import BallTree


# ── Build combined decay dataset ──────────────────────────────────────────────

def build_decay_index(unfit_clean, vacant):
    """Combine unfit and vacant into a single decay DataFrame."""
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


# ── Spatial joins ─────────────────────────────────────────────────────────────

def run_spatial_joins(crime, unfit_clean, vacant):
    """
    Tag each crime with proximity flags:
      near_unfit, near_vacant, near_decay, decay_zone
    Uses BallTree haversine at 100m radius.
    """
    c_coords = np.radians(crime[['LAT', 'LON']].values)

    # Near unfit
    u_coords   = np.radians(unfit_clean[['Latitude', 'Longitude']].values)
    tree_unfit = BallTree(u_coords, metric='haversine')
    crime['near_unfit'] = (
        tree_unfit.query_radius(c_coords, r=100/6_371_000, count_only=True) > 0
    )

    # Near vacant
    v_coords    = np.radians(vacant[['lat', 'lon']].values)
    tree_vacant = BallTree(v_coords, metric='haversine')
    crime['near_vacant'] = (
        tree_vacant.query_radius(c_coords, r=100/6_371_000, count_only=True) > 0
    )

    # Combined zone label
    crime['near_decay'] = crime['near_unfit'] | crime['near_vacant']
    crime['decay_zone'] = 'Neither'
    crime.loc[ crime['near_unfit'] & ~crime['near_vacant'], 'decay_zone'] = 'Near Unfit Only'
    crime.loc[~crime['near_unfit'] &  crime['near_vacant'], 'decay_zone'] = 'Near Vacant Only'
    crime.loc[ crime['near_unfit'] &  crime['near_vacant'], 'decay_zone'] = 'Near Both'

    return crime


def assign_crime_zip(crime, decay):
    """Assign each crime a zip code via nearest decay centroid (BallTree)."""
    zip_centroids = (
        decay.groupby('zip_code')[['lat', 'lon']].mean()
        .reset_index().dropna()
    )
    zip_centroids = zip_centroids[
        zip_centroids['zip_code'].str.match(r'^\d{5}$')
    ].reset_index(drop=True)

    centroid_coords = np.radians(zip_centroids[['lat', 'lon']].values)
    crime_coords    = np.radians(crime[['LAT', 'LON']].values)

    tree   = BallTree(centroid_coords, metric='haversine')
    _, idx = tree.query(crime_coords, k=1)

    crime = crime.copy()
    crime['zip_code'] = zip_centroids['zip_code'].iloc[idx.flatten()].values
    return crime


# ── Neighborhood classification ───────────────────────────────────────────────

def classify_neighborhoods(crime, decay, unfit):
    """
    Classify each zip code into:
      Type A — Crime-Blight Feedback (high crime + high decay)
      Type B — Economic Abandonment  (high decay, low crime)
      Type C — Infrastructure Decay  (unfit-dominant)
      Low Risk / Monitoring
    Returns DataFrame with composite risk score (0–100).
    """
    crime_zip = crime.groupby('zip_code').size().reset_index(name='crime_count')

    decay_zip = (
        decay.groupby(['zip_code', 'decay_type']).size()
        .unstack(fill_value=0).reset_index()
    )
    decay_zip.columns.name = None
    if 'Unfit Property'  not in decay_zip.columns: decay_zip['Unfit Property']  = 0
    if 'Vacant Property' not in decay_zip.columns: decay_zip['Vacant Property'] = 0
    decay_zip['decay_score'] = decay_zip['Unfit Property'] + decay_zip['Vacant Property']
    decay_zip['unfit_ratio'] = (
        decay_zip['Unfit Property'] / decay_zip['decay_score'].replace(0, 1)
    )

    unfit_z = unfit.copy()
    unfit_z['zip_code'] = unfit_z['zip'].astype(str).str.strip()
    unfit_z['is_open']  = unfit_z['status_type_name'] == 'Open'
    unresolved = (
        unfit_z.groupby('zip_code')
        .agg(total_unfit=('is_open', 'count'), open_unfit=('is_open', 'sum'))
        .reset_index()
    )
    unresolved['pct_unresolved'] = unresolved['open_unfit'] / unresolved['total_unfit']

    nbr = decay_zip.merge(crime_zip, on='zip_code', how='left')
    nbr = nbr.merge(
        unresolved[['zip_code', 'pct_unresolved', 'total_unfit', 'open_unfit']],
        on='zip_code', how='left'
    )
    nbr['crime_count']    = nbr['crime_count'].fillna(0)
    nbr['pct_unresolved'] = nbr['pct_unresolved'].fillna(0)

    crime_median = nbr['crime_count'].median()
    decay_median = nbr['decay_score'].median()

    def classify(row):
        high_crime  = row['crime_count'] > crime_median
        high_decay  = row['decay_score'] > decay_median
        unfit_heavy = row['unfit_ratio'] > 0.4
        if high_crime and high_decay:   return 'Type A — Crime-Blight Feedback'
        elif high_decay:                return 'Type B — Economic Abandonment'
        elif unfit_heavy:               return 'Type C — Infrastructure Decay'
        else:                           return 'Low Risk / Monitoring'

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


def get_economic_abandonment_zones(crime, decay):
    """Vacant properties in bottom 25% crime-density zip codes."""
    crime_zip = crime.groupby('zip_code').size().reset_index(name='crime_count')
    threshold = crime_zip['crime_count'].quantile(0.25)
    low_crime_zips = crime_zip[
        crime_zip['crime_count'] <= threshold
    ]['zip_code'].tolist()
    abandoned = decay[
        (decay['decay_type'] == 'Vacant Property') &
        (decay['zip_code'].isin(low_crime_zips))
    ].copy()
    return abandoned, low_crime_zips


def get_proximity_stats(crime):
    near_both = crime['decay_zone'] == 'Near Both'
    return {
        'near_unfit_pct':  f"{crime['near_unfit'].mean()*100:.1f}%",
        'near_vacant_pct': f"{crime['near_vacant'].mean()*100:.1f}%",
        'near_decay_pct':  f"{crime['near_decay'].mean()*100:.1f}%",
        'near_both_pct':   f"{near_both.mean()*100:.1f}%",
        'near_unfit_n':    int(crime['near_unfit'].sum()),
        'near_vacant_n':   int(crime['near_vacant'].sum()),
        'near_decay_n':    int(crime['near_decay'].sum()),
        'near_both_n':     int(near_both.sum()),
    }


# ── Charts ────────────────────────────────────────────────────────────────────

ZONE_COLORS = {
    'Type A — Crime-Blight Feedback': '#dc2626',
    'Type B — Economic Abandonment':  '#3b82f6',
    'Type C — Infrastructure Decay':  '#f59e0b',
    'Low Risk / Monitoring':          '#6b7280'
}

ZONE_COLORS_DECAY = {
    'Near Both':        '#dc2626',
    'Near Unfit Only':  '#f97316',
    'Near Vacant Only': '#3b82f6',
    'Neither':          '#6b7280'
}


def fig_crime_vs_decay_scatter(nbr):
    fig = px.scatter(
        nbr, x='decay_score', y='crime_count',
        color='zone_type', color_discrete_map=ZONE_COLORS,
        hover_data=['zip_code', 'pct_unresolved', 'risk_score'],
        size='risk_score', size_max=30,
        labels={
            'decay_score': 'Urban Decay Score (unfit + vacant)',
            'crime_count': 'Crime Count',
            'zone_type':   'Zone Type'
        },
        title="Crime vs Decay — Every Zip Code"
    )
    fig.add_vline(x=nbr['decay_score'].median(), line_dash='dash',
                  line_color='gray', annotation_text='Decay median')
    fig.add_hline(y=nbr['crime_count'].median(), line_dash='dash',
                  line_color='gray', annotation_text='Crime median')
    fig.update_layout(height=460)
    return fig


def fig_zone_type_breakdown(nbr):
    counts = nbr['zone_type'].value_counts().reset_index()
    counts.columns = ['Zone Type', 'Count']
    fig = px.bar(counts, x='Zone Type', y='Count',
                 color='Zone Type', color_discrete_map=ZONE_COLORS,
                 text='Count')
    fig.update_layout(showlegend=False, height=380,
                      title="Zip Codes by Zone Type")
    return fig


def fig_risk_score_ranking(nbr):
    top = nbr.head(10).copy()
    top['zip_code'] = top['zip_code'].astype(str)
    fig = px.bar(top, x='risk_score', y='zip_code', orientation='h',
                 color='zone_type', color_discrete_map=ZONE_COLORS,
                 text=top['risk_score'].round(1),
                 labels={'risk_score': 'Risk Score (0–100)',
                         'zip_code': 'Zip Code',
                         'zone_type': 'Zone Type'},
                 title="Top 10 Zip Codes by Risk Score")
    fig.update_layout(yaxis=dict(categoryorder='total ascending'), height=420)
    return fig


def fig_decay_zone_crimes(crime):
    counts = crime['decay_zone'].value_counts().reset_index()
    counts.columns = ['Zone', 'Crime Count']
    fig = px.bar(counts, x='Zone', y='Crime Count',
                 color='Zone', color_discrete_map=ZONE_COLORS_DECAY)
    fig.update_layout(height=380, showlegend=False,
                      title="Crime Count by Decay Zone")
    return fig


def fig_crime_type_by_zone(crime):
    top_types = crime['CRIME_TYPE'].value_counts().head(5).index
    df = (crime[crime['CRIME_TYPE'].isin(top_types)]
          .groupby(['CRIME_TYPE', 'decay_zone']).size().reset_index())
    df.columns = ['Crime Type', 'Zone', 'Count']
    fig = px.bar(df, x='Crime Type', y='Count', color='Zone',
                 color_discrete_map=ZONE_COLORS_DECAY, barmode='group',
                 title="Top Crime Types by Decay Zone")
    fig.update_layout(height=400)
    return fig


def fig_economic_abandonment(abandoned):
    if len(abandoned) == 0:
        return None
    by_zip = abandoned['zip_code'].value_counts().head(8).reset_index()
    by_zip.columns = ['Zip Code', 'Vacant Properties']
    fig = px.bar(by_zip, x='Zip Code', y='Vacant Properties',
                 color='Vacant Properties', color_continuous_scale='Blues',
                 title="Economically Abandoned Vacancies — Low Crime Zips")
    fig.update_layout(height=340, coloraxis_showscale=False)
    return fig