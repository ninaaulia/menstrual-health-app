
import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)
sys.path.append('menstrual_health_app')

from utils.style import apply_global_style
from utils.load_models import load_data

st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide"
)

apply_global_style()

# Load data
df = load_data()

# Feature engineering
df['bmi_categories'] = pd.cut(
    df['BMI'],
    bins=[0,18.5,24.9,29.9,100],
    labels=['Underweight','Normal','Overweight','Obese']
)

df['sleep_category'] = pd.cut(
    df['Sleep Hours'],
    bins=[0,5,8,24],
    labels=['Low','Normal','High']
)

df['period_category'] = pd.cut(
    df['Period Length'],
    bins=[0,4,6,100],
    labels=['Normal','At Risk','High Risk']
)

df['cycle_category'] = pd.cut(
    df['Cycle Length'],
    bins=[0,20,35,100],
    labels=['Short','Normal','Long']
)

# Title
st.title("EDA Dashboard")

st.markdown(
    "Interactive exploration of menstrual health patterns."
)

st.markdown("---")

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", len(df))
col2.metric("Features", len(df.columns))
col3.metric("AVG Age", round(df["Age"].mean(),1))
col4.metric("AVG BMI", round(df["BMI"].mean(),1))

st.markdown("---")

# Feature Distribution
st.markdown("## Feature Distribution")

columns = [
    "BMI",
    "Cycle Length",
    "Period Length",
    "Stress Level",
    "Age",
    "Sleep Hours"
]

colors = [
    "#FF407D",
    "#FFCAD4",
    "#40679E",
    "#1B3C73",
    "#B4D3D9",
    "#FA6868"
]

fig = go.Figure()

for i, col in enumerate(columns):

    fig.add_trace(
        go.Histogram(
            x=df[col],
            name=col,
            marker_color=colors[i],
            visible=(i == 0)
        )
    )

buttons = []

for i, col in enumerate(columns):

    visible = [False] * len(columns)
    visible[i] = True

    buttons.append(dict(
        label=col,
        method="update",
        args=[
            {"visible": visible},
            {"title": f"{col} Distribution"}
        ]
    ))

fig.update_layout(
    updatemenus=[dict(buttons=buttons)],
    title="BMI Distribution",
    plot_bgcolor="#F1E9E9",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Menstrual Metrics vs BMI Category
st.markdown("## Menstrual Metrics vs BMI Category")

fig = go.Figure()

fig.add_trace(
    go.Box(
        x=df["bmi_categories"],
        y=df["Cycle Length"],
        name="Cycle Length",
        marker_color="#FF407D",
        boxmean=True,
        visible=True
    )
)

fig.add_trace(
    go.Box(
        x=df["bmi_categories"],
        y=df["Period Length"],
        name="Period Length",
        marker_color="#1B3C73",
        boxmean=True,
        visible=False
    )
)

fig.update_layout(
    updatemenus=[
        dict(
            buttons=[
                dict(
                    label="Cycle Length",
                    method="update",
                    args=[{"visible":[True,False]}]
                ),
                dict(
                    label="Period Length",
                    method="update",
                    args=[{"visible":[False,True]}]
                ),
            ]
        )
    ],
    plot_bgcolor="#F1E9E9",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Symptom Frequency
st.markdown("## Symptom Frequency")

symptoms_count = df["Symptoms"].value_counts().reset_index()

symptoms_count.columns = [
    "Symptoms",
    "Count"
]

fig = px.bar(
    symptoms_count,
    x="Symptoms",
    y="Count",
    color="Count",
    color_continuous_scale="Burg",
    text="Count"
)

fig.update_layout(
    plot_bgcolor="#F1E9E9",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# First rows
st.markdown("---")

st.subheader("First 5 Rows")

st.dataframe(
    df.head(),
    use_container_width=True
)
