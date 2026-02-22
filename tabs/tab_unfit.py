import streamlit as st
from analysis.unfit import (
    fig_unfit_by_year, fig_open_closed_pie,
    fig_unfit_by_zip, fig_open_by_zip
)


def render(unfit):
    st.caption("Buildings formally cited by the city as unsafe or uninhabitable — violation trend, resolution rate, and geographic concentration.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Violations Filed Per Year</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_unfit_by_year(unfit), use_container_width=True)
        st.caption("📌 Annual violation counts since 2014. The post-2021 acceleration means the problem is growing faster than the city can respond.")

    with col2:
        st.markdown('<div class="section-header">Open vs Closed Violations</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_open_closed_pie(unfit), use_container_width=True)
        st.caption("📌 73% of all violations ever filed are still Open — the city issues citations faster than it resolves them.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Total Violations by Zip</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_unfit_by_zip(unfit), use_container_width=True)
        st.caption("📌 13204, 13205, 13208 — Syracuse's west and south sides — consistently rank highest.")

    with col4:
        st.markdown('<div class="section-header">Open Violations by Zip</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_open_by_zip(unfit), use_container_width=True)
        st.caption("📌 Active unresolved violations right now — where enforcement is most urgently needed.")

    st.warning("⚠️ **Trend Alert:** Unfit violations grew **33x from 2014 to 2025**, with 73% still unresolved. Steepest acceleration began in 2022.")