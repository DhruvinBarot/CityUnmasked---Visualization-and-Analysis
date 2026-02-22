import streamlit as st
from analysis.crime import load_crime
from analysis.unfit import load_unfit, get_unfit_clean
from analysis.vacant import load_vacant
from analysis.code_violations import load_code_violations, add_violation_features
from analysis.decay_index import (
    build_decay_index, run_spatial_joins, assign_crime_zip
)
import tabs.tab_crime           as tab_crime
import tabs.tab_unfit           as tab_unfit
import tabs.tab_vacant          as tab_vacant
import tabs.tab_decay_index     as tab_decay_index
import tabs.tab_code_violations as tab_code_violations
import tabs.tab_map             as tab_map
import tabs.tab_prediction      as tab_prediction

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CityUnmasked — Syracuse Urban Analysis",
    page_icon="🏙️",
    layout="wide"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
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


# ── Load all data (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_all():
    crime = load_crime()
    unfit = load_unfit()
    unfit_clean = get_unfit_clean(unfit)
    vacant = load_vacant()
    cv = load_code_violations()
    decay = build_decay_index(unfit_clean, vacant)
    crime = run_spatial_joins(crime, unfit_clean, vacant)
    crime = assign_crime_zip(crime, decay)
    crime = add_violation_features(crime, cv)
    return crime, unfit, unfit_clean, vacant, cv, decay


crime, unfit, unfit_clean, vacant, cv, decay = load_all()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("#  CityUnmasked — Syracuse Crime & Urban Decay Analysis")
# st.markdown("**Track 3 — Urban Data Analysis | City of Syracuse Datathon 2026**")
st.divider()

# ── KPI row ────────────────────────────────────────────────────────────────────
cols = st.columns(4)
kpi_data = [
    ("Total Crimes",           f"{len(crime):,}",                                              "#f97316"),
    ("Crime Types",            str(crime['CRIME_TYPE'].nunique()),                             "#f97316"),
    ("Unfit Properties",       str(len(unfit)),                                                "#f97316"),
    # ("Open Violations",        str(int((unfit['status_type_name']=='Open').sum())),            "#ef4444"),
    # ("Near Unfit (100m)",      f"{crime['near_unfit'].mean()*100:.0f}%",                      "#f97316"),
    ("Vacant Properties",      f"{len(vacant):,}",                                             "#3b82f6"),
    # ("Active Vacancies",       str(int(vacant['is_active'].sum())),                            "#3b82f6"),
    # ("Near Any Decay (100m)",  f"{crime['near_decay'].mean()*100:.0f}%",                      "#dc2626"),
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚠️ Code Violations",
    "🏚️ Unfit Properties",
    "🏘️ Vacant Properties",
    "📊 Crime Analysis",
    "📉 Urban Decay Index",
    "🗺️ Map",
    "🔮 Prediction"
])

with tab1: tab_code_violations.render(crime, cv)
with tab2: tab_unfit.render(unfit)
with tab3: tab_vacant.render(vacant)
with tab4: tab_crime.render(crime)
with tab5: tab_decay_index.render(crime, decay, unfit)
with tab6: tab_map.render(crime, unfit_clean, vacant)
with tab7: tab_prediction.render(unfit, crime)