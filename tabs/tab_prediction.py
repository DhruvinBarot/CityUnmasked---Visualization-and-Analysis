import streamlit as st
from streamlit_folium import st_folium

from analysis.crime_risk_dev import run_hotspot_model

@st.cache_data
def get_hotspot_results():
    """Train hotspot model once and cache results for the session."""
    return run_hotspot_model()


def render(unfit, crime):
    """
    Prediction tab: multi-year hotspot model built from crime_clean (2023–2025).

    We show:
    - A risk heatmap over Syracuse
    - Top 10 highest-risk grid cells
    - Interpretation + policy recommendations
    """

    st.caption(
        "Multi-year spatio-temporal hotspot model using crime_clean (2023–2025). "
        "For each year, we learn where October–December crime clusters will form "
        "based on January–September patterns, then aggregate risk across years to "
        "find persistent hotspots."
    )

    # ── Run / load model ──
    risk_map, top10 = get_hotspot_results()

    # ── Map + table layout ──
    st.markdown("### 🔥 Predicted Crime Hotspots (2023–2025)")

    col_map, col_table = st.columns([2, 1])

    with col_map:
        st.markdown("**Risk Heatmap — Syracuse Grid (~400–500m cells)**")
        st_folium(risk_map, width=900, height=600)
        st.caption(
            "Colors show the predicted probability that each 400–500m grid cell "
            "becomes a crime cluster in Q4 (Oct–Dec). Red circles outline "
            "chronic hotspot neighborhoods: Downtown (13202), Southside (13207), "
            "Eastside / SU (13210), and Near Westside (13204). Blue dots mark the "
            "top 10 highest-risk grid cells."
        )

    with col_table:
        st.markdown("**Top 10 Highest-Risk Grid Cells**")

        display = top10.copy()
        display["risk_score"] = display["risk_score"].round(2)
        display["avg_future_crimes"] = display["avg_future_crimes"].round(1)
        display.insert(0, "Rank", range(1, len(display) + 1))

        display.columns = [
            "Rank",
            "Lat Center",
            "Lon Center",
            "Risk Score",
            "Avg Future Crimes (Oct–Dec)",
        ]

        st.dataframe(display, use_container_width=True)
        st.caption(
            "Risk Score ≈ modelled probability that this grid becomes a Q4 crime "
            "cluster in a given year. Avg Future Crimes is the mean number of "
            "crimes observed in those Q4 windows across 2023–2025."
        )

    st.divider()

    # ── Interpretation ──
    st.markdown("### 🔍 What the Model is Doing")

    with st.expander("See the modelling logic in plain English"):
        st.markdown(
            """
            **1. Space:**  
            We snap every incident to a coarse latitude/longitude grid (~400–500m).  
            Each grid cell becomes a small “block” of the city.

            **2. Time:**  
            For each year (2023, 2024, 2025), we split the data into:
            - **History:** January–September  
            - **Future:** October–December  

            **3. Features (X):** per grid cell in Jan–Sep  
            - Total crime count  
            - Share of serious crimes (assault, robbery, burglary, etc.)  

            **4. Label (y):** per grid cell in Oct–Dec  
            - 1 = “cluster” if future crimes ≥ threshold  
            - 0 = otherwise  

            **5. Model:**  
            Logistic Regression (class-balanced) learns the relationship between
            historical intensity/seriousness and whether that cell turns into a Q4 cluster.

            **6. Multi-year aggregation:**  
            We repeat this for 2023, 2024, 2025 and average the predicted risk per grid.
            Cells that are high-risk in multiple years emerge as **chronic hotspots**.
            """
        )

    st.markdown("### 🧠 Key Takeaways")

    st.markdown(
        """
        - **Risk is not uniform:** the model consistently concentrates high risk in:
          - **Downtown (13202)**  
          - **Southside (13207)**  
          - **Eastside / SU (13210)**  
          - **Near Westside (13204)**  

        - **Chronic vs one-off:** by averaging risk across three years, we highlight
          **structurally risky blocks**, not just single-year spikes.

        - **Severity matters:** grids with a high share of serious crimes in Jan–Sep
          are much more likely to become Q4 hotspots.
        """
    )

    st.divider()

    # ── Policy framing – tuned to these results ──
    st.markdown("### 🎯 Policy Recommendations Based on the Hotspot Model")

    st.markdown(
        """
        **Zone A – Crime–Blight Feedback Loops (Southside 13207, Near Westside 13204)**  
        - Pair **housing repairs + vacancy remediation** with targeted violence-prevention.  
        - Use the top-risk grids as priority blocks for inspections, lighting, and outreach.  

        **Zone B – Economic & Institutional Centers (Downtown 13202, Eastside / SU 13210)**  
        - Focus on **late-night safety** (lighting, transit, guardianship) and  
          **code enforcement on chronically problematic parcels**.  
        - Partner with anchor institutions (SU, hospitals, major employers) to co-fund interventions.  

        **City-wide priorities informed by the model:**  
        - Build a **hotspot list** of 10–20 grids that recur as high-risk and track quarterly metrics.  
        - Tie **code enforcement, vacant property follow-up, and police patrol planning** to these grids.  
        - Use year-over-year changes in risk scores as a way to monitor whether interventions are working.
        """
    )