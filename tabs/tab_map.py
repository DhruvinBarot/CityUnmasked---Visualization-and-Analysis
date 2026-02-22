import streamlit as st
from streamlit_folium import st_folium
from analysis.map_builder import build_map


def render(crime, unfit_clean, vacant):
    st.caption("All three datasets on one interactive map. Use the layer control (top-right) to toggle layers.")

    with st.expander("ℹ️ How to read this map"):
        st.markdown("""
        - **Orange/red heatmap** — Crime density across all years. Brighter = more crimes.
        - **Red dots** — Open unfit property violations. Currently unsafe and unresolved.
        - **Gray dots** — Closed unfit violations. Property was remediated.
        - **Blue heatmap** — Vacant property density. Brighter = more vacancies clustered there.
        - **Where to look:** Find spots where all three overlap — those are Syracuse's highest-priority intervention zones.
        """)

    st.markdown('<div class="section-header">Crime Heatmap + Urban Decay Locations</div>', unsafe_allow_html=True)
    st_folium(build_map(crime, unfit_clean, vacant), width=1100, height=580)
    st.success("🗺️ Northwest and southwest corridors show the strongest overlap of crime intensity and decay concentration — confirmed Type A intervention zones.")