import streamlit as st
from analysis.models import (
    run_granger_causality, fig_granger_pvalues, fig_granger_timeseries
)
from analysis.code_violations import (
    fig_violations_by_year_tier, fig_tier_pie,
    fig_violations_by_zip, fig_violations_by_neighborhood
)


def render(crime, cv):
    st.caption("92,790 physical decay violations (2017–2026) — filtered to structural, systems failure, and environmental neglect only. Administrative violations excluded.")

    with st.expander("ℹ️ What are the three violation tiers?"):
        st.markdown("""
        **Tier 1 — Structural / Critical** 🔴
        Direct threats to building integrity: unfit for human occupancy, structural member failures, stairway collapse risk. These are the violations most linked to abandonment.

        **Tier 2 — Systems Failure** 🟠
        Building systems failing: interior surfaces, plumbing, broken windows/doors, electrical hazards, mechanical failures, pest infestation, lead paint, carbon monoxide risk.

        **Tier 3 — Environmental Neglect** 🟡
        Visible abandonment signals: overgrowth, trash/debris, garbage accumulation. The broken windows theory indicators — visible disorder signals an area is unmonitored.

        Administrative violations (registration, permits, certifications) were excluded — they don't indicate physical decay.
        """)

    # ── KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        ("Total Physical Violations", f"{len(cv):,}",                                              "#dc2626"),
        ("Still Open",                f"{cv['is_open'].sum():,} ({cv['is_open'].mean()*100:.0f}%)", "#f97316"),
        ("Structural / Critical",     f"{(cv['tier']==3).sum():,}",                                 "#dc2626"),
        ("Years of Data",             f"{cv['year'].min()}–{cv['year'].max()}",                     "#6b7280"),
    ]
    for col, (label, val, color) in zip([k1, k2, k3, k4], kpi_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header-red">Violations by Year and Tier</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_violations_by_year_tier(cv), use_container_width=True)
        st.caption("📌 Annual violation counts split by tier. Growing red (structural) bars are the most alarming signal.")

    with col2:
        st.markdown('<div class="section-header-red">Tier Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_tier_pie(cv), use_container_width=True)
        st.caption("📌 Environmental neglect dominates — consistent with broken windows theory. Structural violations are the most dangerous fraction.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header-red">Top Zip Codes</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_violations_by_zip(cv), use_container_width=True)
        st.caption("📌 13205, 13204, 13208 dominate — same hotspots as crime, unfit, and vacant. Four datasets confirming the same geography.")

    with col4:
        st.markdown('<div class="section-header-red">Top Neighborhoods</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_violations_by_neighborhood(cv), use_container_width=True)
        st.caption("📌 Northside, Brighton, Near Westside — same neighborhoods as the crime heatmap and vacancy analysis.")

    # ── Granger causality ──
    st.divider()
    st.markdown("### 📈 Granger Causality — Do Violations Predict Crime? (108 Months)")
    st.caption(
        "Bidirectional test using 92,790 code violations across 108 months. "
        "This is the statistically powerful version — far stronger than a 24-month series."
    )

    with st.expander("ℹ️ What does bidirectional Granger testing tell us?"):
        st.markdown("""
        **Violations → Crime:** Past violations predict future crime = physical decay is the leading signal. Supports property intervention as crime prevention.

        **Crime → Violations:** Past crime predicts future violations = crime drives abandonment and neglect. Consistent with residents leaving and landlords stopping maintenance.

        **Both significant:** Confirmed feedback loop — each accelerates the other. The justification for simultaneous Type A intervention.

        **Neither significant:** Relationship may be contemporaneous or driven by a shared cause like poverty or disinvestment.
        """)

    granger_results, _, ts, interpretation = run_granger_causality(crime, cv)
    st.info(f"**Result:** {interpretation}")

    if granger_results is not None:
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_gc = fig_granger_pvalues(granger_results)
            if fig_gc:
                st.plotly_chart(fig_gc, use_container_width=True)
                st.caption("📌 Bars below the red line = statistically significant. Orange = violations predict crime. Blue = crime predicts violations.")
        with gc2:
            fig_ts = fig_granger_timeseries(ts)
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True)
                st.caption("📌 Monthly crime (orange) vs code violations (red dotted) over 9 years. Violation spikes preceding crime spikes support the causal direction.")

        with st.expander("📊 Full Granger Results Table"):
            st.dataframe(granger_results, use_container_width=True)