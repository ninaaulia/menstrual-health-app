
import streamlit as st
import sys
sys.path.append('menstrual_health_app')
from utils.style import apply_global_style

st.set_page_config(
    page_title="Menstrual Health — Monitoring & Risk Detection",
    page_icon="🩸",
    layout="wide"
)

apply_global_style()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Menstrual Health")
    st.markdown("Monitoring & Risk Detection")
    st.markdown("---")
    st.markdown("### Links")
    st.markdown("[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/akshayas02/menstrual-cycle-data-with-factors-dataset/data)")
    st.markdown("[![GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/ninaaulia)")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/ninaaulia/)")
    st.markdown("---")
    st.caption("Built with Streamlit · 2026")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.title("Menstrual Health — Monitoring & Risk Detection")
st.markdown("A machine learning project for analyzing menstrual health patterns, detecting cycle irregularities, and identifying anomalies.")
st.markdown("---")

# ── Metrics ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users",    "100")
col2.metric("Features",       "11")
col3.metric("Models Trained", "3")
col4.metric("Anomaly Rate",   "10%")

st.markdown("---")

# ── Navigation cards ───────────────────────────────────────────────────────────
st.markdown("### Navigate")
col1, col2 = st.columns(2)

with col1:
    st.info("""
    **EDA Dashboard**

    Explore menstrual health data through Plotly visualizations and pattern analysis.
    """)

with col2:
    st.warning("""
    **ML Prediction**

    Predict cycle patterns and detect anomalies using machine learning models.
    """)

st.markdown("---")

# ── Project Overview ───────────────────────────────────────────────────────────
st.markdown("### Project Overview")

st.markdown("""
Menstruation is a natural biological process experienced by more than **2 billion people** every
month, yet menstrual health inequality remains a global issue. According to **UNICEF**, around
**1.5 billion people** still lack access to basic sanitation facilities, while millions of girls
and women struggle to afford menstrual products, access clean water, or receive proper menstrual
education.

In many countries, menstruation is still surrounded by stigma and misinformation. In
**Afghanistan**, more than half of girls experience their first menstruation without prior
knowledge, causing fear, shame, and isolation. Around **30% of schoolgirls** miss school during
menstruation due to inadequate sanitation facilities, while nearly **10% are absent daily**
because of menstruation-related challenges. In **Nepal**, harmful cultural myths can isolate
menstruating women from their families and communities, sometimes resulting in life-threatening
conditions.

Menstrual health challenges are not limited to low-income countries. Even in high-income nations,
many adolescents still experience untreated menstrual pain, poor menstrual literacy, and delayed
diagnosis of reproductive health conditions such as **PCOS** and **endometriosis**. Research also
shows that menstrual health significantly impacts fertility, productivity, physical performance,
and long-term quality of life.

This project aims to explore how **data science and machine learning** can support menstrual
health awareness and early detection through a menstrual health analytics system. Using synthetic
menstrual cycle data, the project combines Exploratory Data Analysis (EDA), classification
modeling, time-series forecasting, and anomaly detection to identify irregular menstrual patterns
and provide data-driven insights.
""")

st.markdown("**The system includes:**")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    - Interactive data visualization using Plotly
    - Classification models to detect irregular menstrual cycles
    - Time-series forecasting using ARIMA and LSTM
    """)
with col2:
    st.markdown("""
    - Anomaly detection using Isolation Forest and PCA visualization
    - A Streamlit dashboard for accessible and beginner-friendly interaction
    """)

st.markdown("---")

# ── References ─────────────────────────────────────────────────────────────────
st.markdown("### References")

st.markdown("""
<div class="ref-box">

1. Hennegan, J., Swe, Z. Y., Than, K. K., Smith, C., Sol, L., Alberda, H., ... & Azzopardi, P. S. (2022).
Monitoring menstrual health knowledge: awareness of menstruation at menarche as an indicator.
<i>Frontiers in Global Women's Health, 3</i>, Article 832549.
<a href="https://doi.org/10.3389/fgwh.2022.832549" target="_blank">https://doi.org/10.3389/fgwh.2022.832549</a>

<br><br>

2. Holmes, K., Curry, C., Sherry, N., Ferfolja, T., Parry, K., Smith, C., ... & Armour, M. (2021).
Adolescent menstrual health literacy in low, middle and high-income countries: a narrative review.
<i>International Journal of Environmental Research and Public Health, 18</i>(5), Article 2260.
<a href="https://doi.org/10.3390/ijerph18052260" target="_blank">https://doi.org/10.3390/ijerph18052260</a>

<br><br>

3. Masuda, H., Okada, S., Shiozawa, N., Sakaue, Y., Manno, M., Makikawa, M., & Isaka, T. (2025).
Machine learning model for menstrual cycle phase classification and ovulation day detection based
on sleeping heart rate under free-living conditions.
<i>Computers in Biology and Medicine, 187</i>, Article 109705.
<a href="https://doi.org/10.1016/j.compbiomed.2024.109705" target="_blank">https://doi.org/10.1016/j.compbiomed.2024.109705</a>

<br><br>

4. UN Women. (n.d.). Period poverty: Why millions of girls and women cannot afford their periods.
<a href="https://www.unwomen.org/en/articles/explainer/period-poverty-why-millions-of-girls-and-women-cannot-afford-their-periods" target="_blank">
https://www.unwomen.org/en/articles/explainer/period-poverty-why-millions-of-girls-and-women-cannot-afford-their-periods</a>

<br><br>

5. UNICEF. (n.d.). Breaking taboos.
<a href="https://www.unicef.org/rosa/stories/breaking-taboos" target="_blank">
https://www.unicef.org/rosa/stories/breaking-taboos</a>

<br><br>

6. UNICEF USA. (n.d.). Fighting menstruation myths keeps girls in school.
<a href="https://www.unicefusa.org/stories/fighting-menstruation-myths-keeps-girls-school" target="_blank">
https://www.unicefusa.org/stories/fighting-menstruation-myths-keeps-girls-school</a>

<br><br>

7. UNICEF USA. (n.d.). How good menstrual hygiene keeps girls learning.
<a href="https://www.unicefusa.org/stories/how-good-menstrual-hygiene-keeps-girls-learning" target="_blank">
https://www.unicefusa.org/stories/how-good-menstrual-hygiene-keeps-girls-learning</a>

</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Menstrual Health Monitoring & Risk Detection · Built with Streamlit · Dataset from Kaggle")
