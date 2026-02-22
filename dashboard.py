import streamlit as st
from streamlit_folium import st_folium
from crime_unfit_analysis import (
    load_data, get_kpis,
    fig_top_crimes, fig_qol_pie, fig_crime_by_month,
    fig_crime_by_hour, fig_near_vs_not,
    fig_unfit_by_year, fig_open_closed_pie,
    fig_unfit_by_zip, fig_open_by_zip,
    fig_vacant_by_neighborhood, fig_vacant_active_pie,
    fig_vacant_by_zip, fig_vacant_active_by_zip,
    fig_decay_by_zip, fig_decay_zone_crimes,
    fig_crime_type_by_decay_zone, get_proximity_stats,
    run_granger_causality, fig_granger_pvalues, fig_granger_timeseries,
    run_random_forest, fig_rf_feature_importance,
    fig_rf_confusion_matrix, fig_rf_metrics,
    classify_neighborhoods, get_economic_abandonment_zones,
    fig_zone_type_breakdown, fig_risk_score_ranking,
    fig_crime_vs_decay_scatter, fig_economic_abandonment,
    build_map, fig_prediction
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CityUnmasked — Syracuse Urban Analysis",
    page_icon="🏙️",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; color: white; padding: 20px;
        border-radius: 10px; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #f97316; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: bold;
        border-left: 4px solid #f97316;
        padding-left: 10px; margin: 20px 0 10px 0;
    }
    .section-header-blue {
        font-size: 1.4rem; font-weight: bold;
        border-left: 4px solid #3b82f6;
        padding-left: 10px; margin: 20px 0 10px 0;
    }
    .section-header-red {
        font-size: 1.4rem; font-weight: bold;
        border-left: 4px solid #dc2626;
        padding-left: 10px; margin: 20px 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
crime, crime_2024, unfit, unfit_clean, vacant, decay = load_data()

# ── Cached heavy computations ──────────────────────────────────────────────────
@st.cache_data
def get_classification(_crime_2024, _decay, _unfit):
    return classify_neighborhoods(_crime_2024, _decay, _unfit)

@st.cache_data
def get_abandonment(_crime_2024, _decay):
    return get_economic_abandonment_zones(_crime_2024, _decay)

@st.cache_data
def get_granger(_crime, _unfit):
    return run_granger_causality(_crime, _unfit)

@st.cache_data
def get_rf(_crime_2024):
    return run_random_forest(_crime_2024)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🏙️ CityUnmasked — Syracuse Crime & Urban Decay Analysis")
st.markdown("**Track 3 — Urban Data Analysis | City of Syracuse Datathon 2026**")
st.divider()

# ── KPI Row ────────────────────────────────────────────────────────────────────
kpis = get_kpis(crime_2024, unfit, vacant)
cols = st.columns(8)
kpi_data = [
    ("Total Crimes",          kpis["total_crimes"],    "#f97316"),
    ("Crime Types",           kpis["crime_types"],     "#f97316"),
    ("Unfit Properties",      kpis["total_unfit"],     "#f97316"),
    ("Open Violations",       kpis["open_violations"], "#ef4444"),
    ("Near Unfit (100m)",     kpis["pct_near_unfit"],  "#f97316"),
    ("Vacant Properties",     kpis["total_vacant"],    "#3b82f6"),
    ("Active Vacancies",      kpis["active_vacant"],   "#3b82f6"),
    ("Near Any Decay (100m)", kpis["pct_near_decay"],  "#dc2626"),
]
for col, (label, val, color) in zip(cols, kpi_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Crime Analysis",
    "🏚️ Unfit Properties",
    "🏘️ Vacant Properties",
    "📉 Urban Decay Index",
    "🗺️ Map",
    "🔮 Prediction"
])

# ── Tab 1 — Crime Analysis ─────────────────────────────────────────────────────
with tab1:
    st.caption("Analyzing Syracuse crime incidents across all years — patterns by type, time, and proximity to urban decay.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Top Crime Types (All Years)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_top_crimes(crime_2024), use_container_width=True)
        st.caption("📌 The 8 most frequently reported crime types across all years in the dataset. Longer bar = more incidents.")
    with col2:
        st.markdown('<div class="section-header">Serious vs Quality-of-Life Crimes</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_qol_pie(crime_2024), use_container_width=True)
        st.caption("📌 Quality-of-life crimes are minor incidents (noise, loitering). Serious crimes include assault, robbery, and property damage. The vast majority of incidents are serious.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Crime by Month</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_crime_by_month(crime_2024), use_container_width=True)
        st.caption("📌 Crime counts by month aggregated across all years. Summer months (June–August) consistently spike due to increased outdoor activity.")
    with col4:
        st.markdown('<div class="section-header">Crime by Hour of Day</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_crime_by_hour(crime_2024), use_container_width=True)
        st.caption("📌 Which hours see the most crime across all years. The evening window (6pm–midnight) is consistently the most dangerous period.")

    st.markdown('<div class="section-header">Crime Types: Near vs Not Near Unfit Properties (100m)</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_near_vs_not(crime_2024), use_container_width=True)
    st.caption("📌 Each crime type is split into two bars — crimes within 100m of an unfit property (orange) vs. those that weren't (dark). If the orange bar is proportionally larger, that crime type is over-represented in decay zones.")
    st.info("💡 **Insight:** Areas within 100m of unfit properties represent ~12% of city area but account for a disproportionate share of all crimes — roughly **2x the expected rate.**")

# ── Tab 2 — Unfit Properties ───────────────────────────────────────────────────
with tab2:
    st.caption("Unfit properties are buildings formally cited by the city as unsafe or uninhabitable. This tab tracks the violation trend, resolution rate, and geographic concentration.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Violations Filed Per Year</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_unfit_by_year(unfit), use_container_width=True)
        st.caption("📌 How many new unfit property violations were issued each year since 2014. The sharp upward trend post-2021 means the problem is accelerating faster than the city can respond.")
    with col2:
        st.markdown('<div class="section-header">Open vs Closed Violations</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_open_closed_pie(unfit), use_container_width=True)
        st.caption("📌 73% of all violations ever filed are still Open — the city is issuing citations faster than it is resolving them.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Total Unfit Properties by Zip</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_unfit_by_zip(unfit), use_container_width=True)
        st.caption("📌 Which zip codes have the most unfit citations. 13204, 13205, and 13208 — Syracuse's west and south sides — consistently rank highest.")
    with col4:
        st.markdown('<div class="section-header">Open Violations by Zip Code</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_open_by_zip(unfit), use_container_width=True)
        st.caption("📌 Filtered to only unresolved violations — the active problem areas where enforcement is most urgently needed right now.")

    st.warning("⚠️ **Trend Alert:** Unfit violations grew **33x from 2014 to 2025**, with 73% still unresolved. Steepest acceleration began in 2022.")

# ── Tab 3 — Vacant Properties ──────────────────────────────────────────────────
with tab3:
    st.caption("Vacant properties are registered with the city as unoccupied. With 1,615 records and 88% still active, this dataset is 6x larger than unfit properties and reveals a deeper layer of urban decay.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header-blue">Vacancies by Neighborhood</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_by_neighborhood(vacant), use_container_width=True)
        st.caption("📌 The 8 neighborhoods with the most vacant properties. Brighton and Northside lead — historically under-resourced areas on Syracuse's north and south sides.")
    with col2:
        st.markdown('<div class="section-header-blue">Active vs Resolved Vacancies</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_active_pie(vacant), use_container_width=True)
        st.caption("📌 88% of registered vacant properties are still active — even higher than the 73% unresolved rate for unfit properties.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header-blue">Total Vacancies by Zip Code</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_by_zip(vacant), use_container_width=True)
        st.caption("📌 13205 has 521 vacancies alone — more than double the next zip code.")
    with col4:
        st.markdown('<div class="section-header-blue">Active Vacancies by Zip Code</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_active_by_zip(vacant), use_container_width=True)
        st.caption("📌 The ranking barely changes from total to active — confirming that almost nothing is getting resolved in any zip code.")

    st.info("💡 **Insight:** 88% of 1,615 vacant properties remain active. Brighton, Northside, and Near Westside are the most affected neighborhoods, and their top zip codes (13205, 13204, 13208) are the same as the crime hotspots.")

# ── Tab 4 — Urban Decay Index ──────────────────────────────────────────────────
with tab4:
    st.caption("Every zip code in Syracuse classified into one of three decay types — based on the combination of crime levels and blight levels. This is the intellectual core of the project.")

    with st.expander("ℹ️ How are the three zone types defined?"):
        st.markdown("""
        **Type A — Crime-Blight Feedback Zone** 🔴
        High crime AND high decay co-occurring. These neighborhoods need both problems addressed simultaneously — fixing one without the other breaks only half the cycle.

        **Type B — Economic Abandonment Zone** 🔵
        High decay but LOW crime. These properties are vacant or unfit for non-crime reasons: landlord economics failing, population loss, age deterioration, absentee ownership. Increased policing is the WRONG intervention here — these neighborhoods need investment, ownership reform, and rehabilitation funding.

        **Type C — Infrastructure Decay Zone** 🟡
        Structurally cited properties (unfit-dominant) regardless of crime level. Violations driven by age, deferred maintenance, and harsh upstate NY winters. Fast-track code enforcement and rehabilitation funding are the right tools.

        **Low Risk / Monitoring** ⚫
        Below median on both crime and decay. Monitor but no immediate intervention needed.

        **Why this matters:** Showing that blight exists in LOW crime areas (Type B) is the proof of intellectual honesty — blight has multiple causes, and different causes need different solutions.
        """)

    nbr = get_classification(crime_2024, decay, unfit)
    economic_abandoned, low_crime_zips = get_abandonment(crime_2024, decay)

    # ── Zone KPIs ──
    type_counts = nbr['zone_type'].value_counts()
    z1, z2, z3, z4 = st.columns(4)
    zone_kpis = [
        ("Type A — Feedback Zones",    type_counts.get('Type A — Crime-Blight Feedback', 0), "#dc2626"),
        ("Type B — Abandonment Zones", type_counts.get('Type B — Economic Abandonment',  0), "#3b82f6"),
        ("Type C — Infrastructure",    type_counts.get('Type C — Infrastructure Decay',   0), "#f59e0b"),
        ("Low Risk Zip Codes",         type_counts.get('Low Risk / Monitoring',            0), "#6b7280"),
    ]
    for col, (label, val, color) in zip([z1, z2, z3, z4], zone_kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")

    st.markdown('<div class="section-header-red">Crime vs Decay — Every Zip Code Classified</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_crime_vs_decay_scatter(nbr), use_container_width=True)
    st.caption("📌 Each dot is a zip code. Dashed lines are medians dividing the chart into four quadrants. Top-right (red) = Type A feedback zones. Dots with high decay but low crime (blue) = Type B — proof that blight exists without crime.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header-red">Zone Type Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_zone_type_breakdown(nbr), use_container_width=True)
        st.caption("📌 How many zip codes fall into each category. Type B zones outnumbering Type A means most blight in Syracuse is economically driven, not crime-driven.")
    with col2:
        st.markdown('<div class="section-header-red">Top 10 Zip Codes by Risk Score</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_risk_score_ranking(nbr), use_container_width=True)
        st.caption("📌 Composite risk score (0–100) weighted: crime 40%, decay 35%, unresolved violations 25%. Red = Type A, needs simultaneous housing AND policing intervention.")

    st.divider()
    st.markdown("### 🔵 Economic Abandonment Zones — Blight Without Crime")
    st.markdown(f"**{len(economic_abandoned):,} vacant properties** identified in low-crime zip codes (bottom 25% crime density). These are not crime-driven — they require investment and ownership solutions, not policing.")

    econ_fig = fig_economic_abandonment(economic_abandoned)
    if econ_fig:
        st.plotly_chart(econ_fig, use_container_width=True)
        st.caption("📌 Vacant properties in Syracuse's lowest-crime zip codes. Their presence proves vacancy has multiple causes — economic abandonment, population loss, and landlord failure operate independently of crime.")

    st.info(f"💡 **Key finding for intellectual honesty:** {len(low_crime_zips)} zip codes have low crime but significant vacancy. These need investment programs, not enforcement.")

    with st.expander("📊 Full Neighborhood Classification Table"):
        display_cols = ['zip_code', 'zone_type', 'crime_count', 'decay_score', 'pct_unresolved', 'risk_score']
        display_df = nbr[display_cols].copy()
        display_df['pct_unresolved'] = (display_df['pct_unresolved'] * 100).round(1).astype(str) + '%'
        display_df['risk_score']     = display_df['risk_score'].round(1)
        display_df.columns = ['Zip Code', 'Zone Type', 'Crime Count', 'Decay Score', '% Unresolved', 'Risk Score']
        st.dataframe(display_df, use_container_width=True)

    st.divider()
    st.markdown("### 📍 Spatial Proximity — Crimes within 100m of Decay Points")
    stats = get_proximity_stats(crime_2024)
    p1, p2, p3, p4 = st.columns(4)
    proximity_data = [
        ("Near Unfit Only",   stats['near_unfit_pct'],  stats['near_unfit_n'],  "#f97316"),
        ("Near Vacant Only",  stats['near_vacant_pct'], stats['near_vacant_n'], "#3b82f6"),
        ("Near Any Decay",    stats['near_decay_pct'],  stats['near_decay_n'],  "#dc2626"),
        ("Near Both Types",   stats['near_both_pct'],   stats['near_both_n'],   "#7c3aed"),
    ]
    for col, (label, pct, n, color) in zip([p1, p2, p3, p4], proximity_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{pct}</div>
                <div class="metric-label">{label}<br>({n:,} crimes)</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header-red">Urban Decay Score by Zip Code</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_decay_by_zip(decay), use_container_width=True)
        st.caption("📌 Raw decay point count per zip, stacked by type. Orange = unfit, blue = vacant.")
    with col4:
        st.markdown('<div class="section-header-red">Crime Count by Decay Zone</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_decay_zone_crimes(crime_2024), use_container_width=True)
        st.caption("📌 How many crimes fell in each proximity zone across all years. 'Near Both' has the densest concentration.")

    st.markdown('<div class="section-header-red">Crime Types by Decay Zone</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_crime_type_by_decay_zone(crime_2024), use_container_width=True)
    st.caption("📌 For the top 5 crime types, crimes are split by which decay zone they occurred in. Violent crimes clustering in 'Near Both' directly supports the problem statement.")

    st.divider()
    st.markdown("### 📈 Granger Causality — Does Decay Predict Crime?")
    st.caption("Tests whether past unfit violation counts help predict future crime counts better than past crime alone. p < 0.05 = statistically significant.")

    granger_results, ts_df, interpretation = get_granger(crime, unfit)
    st.info(f"**Result:** {interpretation}")

    if granger_results is not None:
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_gc = fig_granger_pvalues(granger_results)
            if fig_gc:
                st.plotly_chart(fig_gc, use_container_width=True)
                st.caption("📌 Green bars below the red line = statistically significant. The lower the bar, the stronger the signal that decay predicts crime at that lag.")
        with gc2:
            fig_ts = fig_granger_timeseries(ts_df)
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True)
                st.caption("📌 Monthly crime (orange) vs unfit violations (blue dotted). If the blue line visually leads the orange, it supports the Granger result.")

        with st.expander("📊 Full Granger Results Table"):
            st.dataframe(granger_results, use_container_width=True)

    st.error("🔴 **Key Finding:** Type A zip codes (13204, 13205, 13208) lead on both decay score AND crime — these need simultaneous housing AND public safety intervention. Type B zones need investment, not policing.")

# ── Tab 5 — Map ────────────────────────────────────────────────────────────────
with tab5:
    st.caption("All three datasets plotted on a single interactive map. Use the layer control in the top-right corner to show or hide each layer.")

    with st.expander("ℹ️ How to read this map"):
        st.markdown("""
        - **Orange/red heatmap** — Crime density across all years. Brighter = more crimes in that area.
        - **Red dots** — Open unfit property violations. Currently cited as unsafe and not yet fixed.
        - **Gray dots** — Closed unfit violations. The property was remediated.
        - **Blue heatmap** — Vacant property density. Brighter blue = more vacant properties clustered in that area.
        - **Where to look:** Find spots where the orange crime heatmap, red unfit dots, and blue vacant heatmap all overlap. Those intersections are the city's highest-priority intervention zones.
        """)

    st.markdown('<div class="section-header">Crime Heatmap + Urban Decay Locations</div>', unsafe_allow_html=True)
    st_folium(build_map(crime_2024, unfit_clean, vacant), width=1100, height=580)
    st.success("🗺️ Areas where red unfit markers and blue vacant density overlap with high crime heatmap intensity are Syracuse's highest-priority intervention zones — concentrated in the northwest and southwest corridors.")

# ── Tab 6 — Prediction ─────────────────────────────────────────────────────────
with tab6:
    st.caption("Two prediction approaches: a linear forecast of future unfit violations, and a Random Forest model that predicts high-severity crime risk from decay and temporal features.")

    st.markdown("### 📉 Unfit Violation Forecast (Linear Trend)")

    with st.expander("ℹ️ How does the linear forecast work?"):
        st.markdown("""
        We fit a linear regression to annual unfit violation counts from 2014–2024 and project forward to 2027.
        This is a conservative estimate — actual growth has been exponential recently, so reality may be worse.
        The value is in showing the city what happens if nothing changes.
        """)

    fig_pred, years, preds = fig_prediction(unfit)
    st.plotly_chart(fig_pred, use_container_width=True)
    st.caption("📌 Orange bars = actual counts per year. Red dashed line = linear projection. The 3 cards below show predicted new violations for each upcoming year.")

    c1, c2, c3 = st.columns(3)
    for col, yr, pred in zip([c1, c2, c3], years, preds):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pred}</div>
                <div class="metric-label">Predicted violations in {yr}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🌲 Random Forest — Predicting High-Severity Crime Risk")
    st.caption("A Random Forest classifier trained on the full crime dataset to predict whether a crime will be high-severity (SEVERITY ≥ 3). Features include time of day, season, day of week, and whether the crime occurred near urban decay points.")

    with st.expander("ℹ️ How does the Random Forest work?"):
        st.markdown("""
        **What it learns:** Trained on 75% of all crime records, tested on 25%. For each crime it sees hour, month, season, day of week, weekend flag, and whether the location is near unfit properties, vacant properties, or both.

        **What it predicts:** Whether that crime will be high-severity (SEVERITY ≥ 3 — assault, robbery, burglary, and above).

        **Why it matters:** If `near_unfit`, `near_vacant`, or `near_decay` rank high in feature importance, proximity to decay is a genuine predictor of serious crime — directly validating that urban decay and crime severity are linked.

        **Class balancing:** `class_weight='balanced'` prevents the model from always predicting the majority class.
        """)

    rf_model, importance_df, accuracy, cm, report, feature_names = get_rf(crime_2024)

    r1, r2, r3, r4 = st.columns(4)
    rf_kpi_data = [
        ("Model Accuracy",          f"{accuracy}%",                                "#22c55e"),
        ("High Severity F1",        f"{report['High Severity']['f1-score']:.2f}",  "#f97316"),
        ("High Severity Recall",    f"{report['High Severity']['recall']:.2f}",    "#3b82f6"),
        ("High Severity Precision", f"{report['High Severity']['precision']:.2f}", "#7c3aed"),
    ]
    for col, (label, val, color) in zip([r1, r2, r3, r4], rf_kpi_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")

    st.markdown('<div class="section-header">Feature Importance — What Predicts High-Severity Crime?</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_rf_feature_importance(importance_df), use_container_width=True)
    st.caption("📌 Longer bar = more important feature. Red bars are decay/spatial features — if they rank highly, proximity to urban decay is a genuine predictor of crime severity. Orange = time-of-day, amber = seasonal.")

    rf_col1, rf_col2 = st.columns(2)
    with rf_col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_rf_confusion_matrix(cm), use_container_width=True)
        st.caption("📌 Rows = actual class, columns = predicted. Top-left and bottom-right are correct predictions.")
    with rf_col2:
        st.markdown('<div class="section-header">Precision, Recall & F1 by Class</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_rf_metrics(report), use_container_width=True)
        st.caption("📌 Precision = of all crimes predicted high-severity, how many actually were. Recall = of all actual high-severity crimes, how many did we catch. F1 balances both.")

    st.error("📈 If current trends continue, Syracuse could see **100+ new violations per year** through 2027, compounding the backlog of already-open cases.")
    st.markdown("---")
    st.markdown("### 🎯 Policy Recommendations")
    st.markdown("""
    **Type A Zones (Crime-Blight Feedback) — 13204, 13205, 13208:**
    Simultaneous housing intervention AND targeted policing. Fixing one without the other breaks only half the cycle.

    **Type B Zones (Economic Abandonment):**
    Investment programs, ownership enforcement, vacancy rehabilitation. Do NOT increase policing — crime is not the driver here.

    **Type C Zones (Infrastructure Decay):**
    Fast-track code enforcement and rehabilitation funding. These violations are structural, not criminal.

    **City-wide:**
    1. Fast-track the 73% of unfit violations still Open — faster resolution reduces the crime proximity effect
    2. Address the vacant property crisis — 88% active rate across 1,615 properties dwarfs the unfit problem in scale
    3. Target Brighton, Northside, and Near Westside for neighborhood investment programs
    4. Increase enforcement capacity — violations are growing 33x faster than closures, a systemic gap
    """)