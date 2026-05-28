
import streamlit as st

def apply_global_style():
    st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; }

        h1, h2, h3, h4, h5, h6 { color: #D6336C !important; }

        [data-testid="stMetricLabel"] { color: #D6336C !important; }
        [data-testid="stMetricValue"] { color: #D6336C !important; }

        [data-testid="stSidebar"] {
            background-color: #FFB6C1 !important;
        }
        [data-testid="stSidebar"] * {
            color: #021A54 !important;
        }
        [data-testid="stSidebar"] a {
            color: #021A54 !important;
            font-weight: 500;
        }

        hr { border-color: #F8BBD0; }

        .result-card {
            background-color: #FFF0F5;
            border-left: 4px solid #D6336C;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 12px;
        }
        .result-card h4 { color: #D6336C; margin-bottom: 6px; }
        .result-card p  { color: #333; margin: 0; font-size: 15px; }

        .regular-card {
            background-color: #F0FFF4;
            border-left: 4px solid #2E7D32;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 12px;
        }
        .irregular-card {
            background-color: #FFF3E0;
            border-left: 4px solid #E65100;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 12px;
        }

        .ref-box {
            background-color: #FFF0F5;
            border-left: 4px solid #D6336C;
            border-radius: 8px;
            padding: 16px 20px;
            font-size: 13px;
            color: #555;
            line-height: 1.8;
        }
        .ref-box a { color: #D6336C; }

        button[data-baseweb="tab"] {
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #D6336C !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            border-bottom: 3px solid #D6336C !important;
        }
    </style>
    """, unsafe_allow_html=True)
