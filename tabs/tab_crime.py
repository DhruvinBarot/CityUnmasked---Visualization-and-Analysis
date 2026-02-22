import streamlit as st
from analysis.crime import (
    fig_top_crimes, fig_qol_pie,
    fig_crime_by_month, fig_crime_by_hour
)


def render(crime):
    st.caption("Syracuse crime incidents across all years — patterns by type, time, and proximity to urban decay.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Top Crime Types</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_top_crimes(crime), use_container_width=True)
        st.caption("📌 The 8 most frequent crime types across all years. Longer bar = more incidents.")

    with col2:
        st.markdown('<div class="section-header">Serious vs Quality-of-Life</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_qol_pie(crime), use_container_width=True)
        st.caption("📌 Serious crimes (assault, robbery, property damage) vs minor quality-of-life incidents. The vast majority are serious.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Crime by Month</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_crime_by_month(crime), use_container_width=True)
        st.caption("📌 Monthly totals across all years. Summer consistently spikes — more outdoor activity, more opportunity.")

    with col4:
        st.markdown('<div class="section-header">Crime by Hour of Day</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_crime_by_hour(crime), use_container_width=True)
        st.caption("📌 Evening hours (6pm–midnight) are consistently the most dangerous window across all years.")

    st.info("💡 **Insight:** Crime is not random — it has a clear time signature. Evening hours + summer months = peak risk window. Decay zones amplify this.")