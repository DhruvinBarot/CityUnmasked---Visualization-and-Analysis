import streamlit as st
from analysis.vacant import (
    fig_vacant_by_neighborhood, fig_vacant_active_pie,
    fig_vacant_by_zip, fig_vacant_active_by_zip
)


def render(vacant):
    st.caption("Registered vacant properties — 6x larger than the unfit dataset, 88% still active.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header-blue">Vacancies by Neighborhood</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_by_neighborhood(vacant), use_container_width=True)
        st.caption("📌 Brighton and Northside lead — historically under-resourced areas on Syracuse's north and south sides.")

    with col2:
        st.markdown('<div class="section-header-blue">Active vs Resolved</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_active_pie(vacant), use_container_width=True)
        st.caption("📌 88% still active — higher unresolved rate than even unfit properties (73%).")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header-blue">Total Vacancies by Zip</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_by_zip(vacant), use_container_width=True)
        st.caption("📌 13205 alone has 521 vacancies — more than double the next zip code.")

    with col4:
        st.markdown('<div class="section-header-blue">Active Vacancies by Zip</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_vacant_active_by_zip(vacant), use_container_width=True)
        st.caption("📌 The ranking is nearly identical to total — confirming almost nothing is getting resolved in any zip code.")

    st.info("💡 **Insight:** The same zip codes (13205, 13204, 13208) dominate vacancies AND crime. Four independent datasets pointing at the same geography.")