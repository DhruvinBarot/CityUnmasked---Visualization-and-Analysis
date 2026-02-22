import streamlit as st
from analysis.unfit import fig_prediction
from analysis.models import (
    run_random_forest, fig_rf_feature_importance, fig_rf_metrics
)


def render(unfit, crime):
    st.caption("Linear violation forecast through 2027 and a Random Forest model predicting high-severity crime risk from decay and temporal features.")

    # ── Linear forecast ──
    st.markdown("### 📉 Unfit Violation Forecast (Linear Trend)")
    with st.expander("ℹ️ How does the forecast work?"):
        st.markdown("""
        Linear regression fitted to annual unfit violation counts (2014–2024), projected to 2027.
        Conservative estimate — actual growth has been exponential since 2021, so reality may be worse.
        The value: showing the city the cost of inaction in concrete numbers.
        """)

    fig_pred, years, preds = fig_prediction(unfit)
    st.plotly_chart(fig_pred, use_container_width=True)
    st.caption("📌 Orange = actual counts. Red dashed = projection. If nothing changes, Syracuse is on course for 100+ new violations per year on top of the 187 already unresolved.")

    c1, c2, c3 = st.columns(3)
    for col, yr, pred in zip([c1, c2, c3], years, preds):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{pred}</div>
                <div class="metric-label">Predicted violations in {yr}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Random Forest ──
    st.markdown("### 🌲 Random Forest — Predicting High-Severity Crime Risk")
    with st.expander("ℹ️ How does the Random Forest work?"):
        st.markdown("""
        **What it learns:** Trained on 75% of all crime records, tested on 25%.

        **Features:** Hour, month, season, day of week, weekend flag, proximity to unfit/vacant/decay zones, and — new in Phase 2 — violation count within 100m, violation severity score, and critical structural violation flag.

        **What it predicts:** High-severity crime (SEVERITY ≥ 3 — assault, robbery, burglary, and above).

        **Why feature importance matters:** If decay/violation features rank highly, proximity to urban decay is a genuine predictor of serious crime — not a coincidence. This is the model-based validation of the project thesis.

        **200 trees, class-balanced** to handle the majority-class imbalance in severity levels.
        """)

    model, importance_df, accuracy, cm, report = run_random_forest(crime)

    r1, r2, r3, r4 = st.columns(4)
    for col, (label, val, color) in zip([r1, r2, r3, r4], [
        ("Model Accuracy",          f"{accuracy}%",                                "#22c55e"),
        ("High Severity F1",        f"{report['High Severity']['f1-score']:.2f}",  "#f97316"),
        ("High Severity Recall",    f"{report['High Severity']['recall']:.2f}",    "#3b82f6"),
        ("High Severity Precision", f"{report['High Severity']['precision']:.2f}", "#7c3aed"),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_rf_feature_importance(importance_df), use_container_width=True)
    st.caption("📌 Red bars = decay/violation features. If these rank at the top, the model independently validates the thesis: proximity to urban decay predicts serious crime severity.")

    st.markdown('<div class="section-header">Precision, Recall & F1</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_rf_metrics(report), use_container_width=True)
    st.caption("📌 Precision = of all predicted high-severity, how many actually were. Recall = of all actual high-severity, how many did we catch. F1 balances both.")

    st.divider()
    st.markdown("### 🎯 Policy Recommendations")
    st.markdown("""
    **Type A Zones — Crime-Blight Feedback (13204, 13205, 13208):**
    Simultaneous housing intervention AND targeted policing. Fixing one without the other breaks only half the cycle.

    **Type B Zones — Economic Abandonment:**
    Investment, ownership enforcement, vacancy rehabilitation. Do NOT increase policing — crime is not the driver.

    **Type C Zones — Infrastructure Decay:**
    Fast-track code enforcement and rehabilitation funding. These violations are structural, not criminal.

    **City-wide priorities:**
    - Fast-track the 73% of unfit violations still Open
    - Address 1,421 active vacant properties — 88% unresolved rate is worse than unfit
    - Target Brighton, Northside, Near Westside for concentrated investment
    - Increase enforcement capacity — violations growing 33x faster than resolution
    """)