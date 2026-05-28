
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('menstrual_health_app')

from utils.style import apply_global_style
from utils.load_models import (
    load_classifier, load_clf_scaler, load_clf_columns,
    load_arima, load_anomaly_pipeline, load_data
)

st.set_page_config(page_title="ML Prediction — Menstrual Health", layout="wide")
apply_global_style()

with st.sidebar:
    st.markdown("## Menstrual Health")
    st.markdown("Monitoring & Risk Detection")
    st.markdown("---")
    st.markdown("### ML Models")
    st.markdown("- Classification (Logistic Regression)")
    st.markdown("- Forecasting (ARIMA)")
    st.markdown("- Anomaly Detection (Isolation Forest)")
    st.markdown("---")
    st.caption("Built with Streamlit · 2026")

st.title("ML Prediction Dashboard")
st.markdown("---")

clf_model                      = load_classifier()
clf_scaler                     = load_clf_scaler()
clf_columns                    = load_clf_columns()
arima_model, arima_meta        = load_arima()
anomaly_pipeline, anomaly_meta = load_anomaly_pipeline()
df                             = load_data()

tab1, tab2, tab3 = st.tabs(["Classification", "Forecasting", "Anomaly Detection"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Cycle Irregularity Classification")
    st.markdown("Predict whether a menstrual cycle is regular or irregular based on lifestyle and health factors.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### User Input")
        age    = st.slider("Age", min_value=8, max_value=60, value=25)
        bmi    = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
        stress = st.selectbox("Stress Level", options=[1, 2, 3, 4, 5], index=2,
                              help="1 = Low, 5 = High")
        sleep  = st.number_input("Sleep Hours", min_value=3, max_value=24, value=7, step=1)
        period_length = st.number_input("Period Length (days)", min_value=1, max_value=100,
                                        value=5, step=1)
        exercise = st.selectbox("Exercise Frequency",
                                options=['Low', 'Moderate', 'High'], index=1)
        diet     = st.selectbox("Diet",
                                options=['Balanced', 'High Sugar', 'Low Carb', 'Vegetarian'],
                                index=0)
        predict_btn = st.button("Run Classification", use_container_width=True)

    with col_result:
        st.markdown("#### Prediction Result")

        if predict_btn:
            num_cols   = ['Age', 'BMI', 'Stress Level', 'Sleep Hours', 'Period Length']
            input_dict = {
                'Age': age, 'BMI': bmi, 'Stress Level': stress,
                'Sleep Hours': sleep, 'Period Length': period_length
            }
            for col in clf_columns:
                if col.startswith('Exercise Frequency_'):
                    cat = col.replace('Exercise Frequency_', '')
                    input_dict[col] = 1 if exercise == cat else 0
            for col in clf_columns:
                if col.startswith('Diet_'):
                    cat = col.replace('Diet_', '')
                    input_dict[col] = 1 if diet == cat else 0

            input_df           = pd.DataFrame([input_dict])[clf_columns]
            input_df[num_cols] = clf_scaler.transform(input_df[num_cols])

            prediction     = clf_model.predict(input_df)[0]
            proba          = clf_model.predict_proba(input_df)[0]
            confidence     = round(max(proba) * 100, 1)
            prob_irregular = round(proba[1] * 100, 1)
            prob_regular   = round(proba[0] * 100, 1)

            if prediction == 0:
                st.markdown("""
                <div class="regular-card">
                    <h4 style="color:#2E7D32;">Regular Cycle</h4>
                    <p>The predicted cycle falls within the normal range (21–35 days).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="irregular-card">
                    <h4 style="color:#E65100;">Irregular Cycle Detected</h4>
                    <p>The predicted cycle falls outside the normal range.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Confidence Score",      f"{confidence}%")
            m2.metric("Irregular Probability", f"{prob_irregular}%")

            fig = go.Figure(go.Bar(
                x=[prob_regular, prob_irregular],
                y=['Regular', 'Irregular'],
                orientation='h',
                marker_color=['#2E7D32', '#E65100'],
                text=[f"{prob_regular}%", f"{prob_irregular}%"],
                textposition='auto'
            ))
            fig.update_layout(
                title="Prediction Probability", xaxis=dict(range=[0, 100]),
                height=220, margin=dict(t=40, b=20, l=10, r=10),
                plot_bgcolor='white', paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Input Summary")
            summary_df = pd.DataFrame({
                'Feature': ['Age', 'BMI', 'Stress Level', 'Sleep Hours',
                            'Period Length', 'Exercise', 'Diet'],
                'Value':   [age, bmi, stress, sleep, period_length, exercise, diet]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div class="result-card">
                <h4>Awaiting Input</h4>
                <p>Fill in the user data and click <b>Run Classification</b>.</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Menstrual Cycle Trend Forecasting")
    st.markdown("Visualize and explore predicted future cycle patterns based on historical data trends.")

    st.info(
        "This forecast is generated using historical menstrual cycle patterns from the dataset. "
    )
    st.markdown("---")

    col_ctrl, col_chart = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### Forecast Horizon")
        st.markdown(
            "Select how many upcoming menstrual cycles you want to predict "
            "based on historical dataset trends."
        )

        forecast_steps = st.slider(
            "Number of upcoming cycles to predict",
            min_value=1, max_value=12, value=6,
            help="Each step represents one full menstrual cycle (~28 days on average)"
        )

        show_ci = st.checkbox(
            "Show 95% Confidence Interval",
            value=True,
            help="The shaded area shows the range within which the actual cycle length is likely to fall"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        forecast_btn = st.button("Generate Forecast", use_container_width=True)

        # Info box about CI
        if show_ci:
            st.markdown("""
            <div style="background:#F8F9FA; border-left:3px solid #ADB5BD;
                        border-radius:6px; padding:10px 14px; margin-top:12px;">
                <p style="margin:0; font-size:12px; color:#555;">
                    <b>About the confidence interval:</b><br>
                    The shaded band represents the 95% prediction range.
                    A wider band indicates greater uncertainty in the forecast.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.markdown("#### Forecast Result")

        if forecast_btn:
            # ── ARIMA forecast (model logic unchanged) ─────────────────────────
            forecast     = arima_model.get_forecast(steps=forecast_steps)
            pm           = forecast.predicted_mean
            future_mean  = pm.values if hasattr(pm, 'values') else np.array(pm)
            ci           = forecast.conf_int(alpha=0.05)
            future_ci    = ci.values if hasattr(ci, 'values') else np.array(ci)
            future_lower = future_ci[:, 0]
            future_upper = future_ci[:, 1]

            ts     = df['Cycle Length'].dropna().values if 'Cycle Length' in df.columns \
                     else np.random.normal(28, 3, 80).round(1)
            n_hist = len(ts)
            x_hist = list(range(n_hist))
            x_fut  = list(range(n_hist, n_hist + forecast_steps))

            # ── Stability metrics (integer — no decimals) ──────────────────────
            next_cycle      = int(round(float(future_mean[0])))
            avg_forecast    = int(round(float(future_mean.mean())))
            min_forecast    = int(round(float(future_mean.min())))
            max_forecast    = int(round(float(future_mean.max())))
            ci_width        = round(float((future_upper - future_lower).mean()), 1)
            irregular_count = sum(1 for v in future_mean if v < 21 or v > 35)

            if irregular_count == 0 and ci_width < 8:
                stability        = "Stable"
                stability_color  = "#2E7D32"
                stability_bg     = "#F0FFF4"
                stability_border = "#2E7D32"
                stability_note   = (
                    "All forecasted cycles fall within the clinically normal range of 21–35 days. "
                    "The historical data suggests a consistent and predictable cycle pattern."
                )
            elif irregular_count <= forecast_steps // 2 or ci_width < 14:
                stability        = "Moderate Variability"
                stability_color  = "#E65100"
                stability_bg     = "#FFF3E0"
                stability_border = "#E65100"
                stability_note   = (
                    f"{irregular_count} out of {forecast_steps} predicted cycles may fall outside "
                    "the normal range. Lifestyle factors such as stress, sleep quality, and diet "
                    "may contribute to this variability."
                )
            else:
                stability        = "High Variability"
                stability_color  = "#B71C1C"
                stability_bg     = "#FFEBEE"
                stability_border = "#B71C1C"
                stability_note   = (
                    f"{irregular_count} out of {forecast_steps} predicted cycles show signs of "
                    "irregularity. This pattern warrants closer attention. Consulting a healthcare "
                    "provider is recommended if this reflects personal experience."
                )

            # ── Summary card ───────────────────────────────────────────────────
            st.markdown(f"""
            <div style="background:{stability_bg}; border-left:4px solid {stability_border};
                        border-radius:8px; padding:18px 22px; margin-bottom:16px;">
                <div style="display:flex; justify-content:space-between;
                            align-items:flex-start; gap:16px;">
                    <div>
                        <p style="margin:0; font-size:11px; color:#888;
                                  text-transform:uppercase; letter-spacing:0.05em;">
                            Next predicted cycle length
                        </p>
                        <p style="margin:4px 0 0; font-size:32px; font-weight:700;
                                  color:{stability_color}; line-height:1;">
                            {next_cycle} days
                        </p>
                        <p style="margin:6px 0 0; font-size:12px; color:#666;">
                            Forecasting <b>{forecast_steps} cycles ahead</b> &nbsp;·&nbsp;
                            Dataset average: <b>{avg_forecast} days</b> &nbsp;·&nbsp;
                            Uncertainty range: <b>±{round(ci_width/2, 1)} days</b>
                        </p>
                    </div>
                    <div style="text-align:right; flex-shrink:0;">
                        <p style="margin:0; font-size:11px; color:#888;
                                  text-transform:uppercase; letter-spacing:0.05em;">
                            Trend stability
                        </p>
                        <p style="margin:4px 0 0; font-size:20px; font-weight:700;
                                  color:{stability_color};">
                            {stability}
                        </p>
                    </div>
                </div>
                <hr style="border:none; border-top:1px solid {stability_border};
                           opacity:0.25; margin:14px 0;">
                <p style="margin:0; font-size:13px; color:#444; line-height:1.6;">
                    {stability_note}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Plotly chart ───────────────────────────────────────────────────
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=x_hist, y=ts,
                mode='lines',
                name='Historical data',
                line=dict(color='#021A54', width=1.5),
                hovertemplate='Cycle %{x}<br>Length: %{y} days<extra></extra>'
            ))

            if show_ci:
                fig.add_trace(go.Scatter(
                    x=x_fut + x_fut[::-1],
                    y=list(future_upper) + list(future_lower)[::-1],
                    fill='toself',
                    fillcolor='rgba(214,51,108,0.12)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% prediction range',
                    hoverinfo='skip'
                ))

            fig.add_trace(go.Scatter(
                x=x_fut, y=future_mean,
                mode='lines+markers',
                name='Predicted cycles',
                line=dict(color='#D6336C', width=2.5, dash='dot'),
                marker=dict(size=9, color='#D6336C', symbol='circle',
                            line=dict(color='white', width=1.5)),
                hovertemplate='Predicted cycle %{x}<br>Length: %{y:.0f} days<extra></extra>'
            ))

            fig.add_vline(
                x=n_hist - 0.5,
                line_dash='dash', line_color='#ADB5BD', line_width=1.5,
                annotation_text='Forecast begins here',
                annotation_position='top right',
                annotation_font_color='#888',
                annotation_font_size=11
            )

            fig.add_hrect(
                y0=21, y1=35,
                fillcolor='rgba(0,180,100,0.05)',
                line_width=0,
                annotation_text='Normal cycle range (21–35 days)',
                annotation_position='right',
                annotation_font_size=10,
                annotation_font_color='#2E7D32'
            )

            fig.update_layout(
                title=dict(
                    text=f"Cycle Length Trend — {n_hist} Historical + {forecast_steps} Predicted Cycles",
                    font=dict(size=14, color='#333')
                ),
                xaxis=dict(title='Cycle number', showgrid=True,
                           gridcolor='#F0F0F0', zeroline=False),
                yaxis=dict(title='Cycle length (days)', showgrid=True,
                           gridcolor='#F0F0F0', zeroline=False),
                height=420,
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                legend=dict(
                    orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1,
                    font=dict(size=12)
                ),
                margin=dict(t=60, b=40, l=50, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Forecast table ─────────────────────────────────────────────────
            st.markdown("#### Predicted Cycle Details")
            st.caption(
                "Each row represents one predicted menstrual cycle. "
                "The prediction range shows the lower and upper bounds of the 95% confidence interval."
            )

            def cycle_note(v):
                if v < 21:
                    return "Short cycle — below normal"
                elif v > 35:
                    return "Long cycle — above normal"
                elif v <= 25:
                    return "Short-normal range"
                elif v >= 31:
                    return "Long-normal range"
                else:
                    return "Within normal range"

            forecast_df = pd.DataFrame({
                'Cycle':                [f"Cycle +{i+1}" for i in range(forecast_steps)],
                'Predicted length':     [f"{int(round(v))} days" for v in future_mean],
                'Prediction range':     [
                    f"{int(round(lo))}–{int(round(hi))} days"
                    for lo, hi in zip(future_lower, future_upper)
                ],
                'Interpretation':       [cycle_note(v) for v in future_mean],
                'Clinical status':      [
                    'Regular' if 21 <= v <= 35 else 'Irregular'
                    for v in future_mean
                ]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            # ── Metric row ─────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Next cycle",    f"{next_cycle} days")
            m2.metric("Average",       f"{avg_forecast} days")
            m3.metric("Shortest",      f"{min_forecast} days")
            m4.metric("Longest",       f"{max_forecast} days")

            st.markdown("---")
            st.caption(
                "Forecast generated using ARIMA time series model trained on historical cycle data. "
            )

        else:
            st.markdown("""
            <div class="result-card">
                <h4>No forecast generated yet</h4>
                <p>Select the forecast horizon on the left and click
                <b>Generate Forecast</b> to see predicted cycle trends.</p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Anomaly Detection")
    st.markdown("Detect abnormal menstrual patterns using Isolation Forest with PCA visualization.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")
    FEATURES = anomaly_meta['features']

    with col_form:
        st.markdown("#### User Input")
        cycle_len  = st.slider("Cycle Length (days)",  min_value=10, max_value=100, value=28)
        period_len = st.slider("Period Length (days)", min_value=1,  max_value=100, value=5)

        # FIX: changed from slider min=1 max=10 → selectbox 1–5
        stress_ad  = st.selectbox("Stress Level", options=[1, 2, 3, 4, 5], index=2,
                                  help="1 = Low, 5 = High", key="stress_ad")

        bmi_ad     = st.number_input("BMI", min_value=10.0, max_value=50.0,
                                     value=22.5, step=0.1, key="bmi_ad")
        sleep_ad   = st.slider("Sleep Hours", min_value=3, max_value=12, value=7)
        anomaly_btn = st.button("Run Anomaly Detection", use_container_width=True)

    with col_result:
        st.markdown("#### Detection Result")

        if anomaly_btn:
            X_new      = np.array([[cycle_len, period_len, stress_ad, bmi_ad, sleep_ad]])
            X_scaled   = anomaly_pipeline['scaler'].transform(X_new)
            flag       = anomaly_pipeline['iso_forest'].predict(X_scaled)[0]
            score      = anomaly_pipeline['iso_forest'].decision_function(X_scaled)[0]
            pca_pt     = anomaly_pipeline['pca'].transform(X_scaled)[0]
            is_anomaly = flag == -1

            if not is_anomaly:
                severity        = "Normal"
                severity_color  = "#2E7D32"
                severity_bg     = "#F0FFF4"
                severity_border = "#2E7D32"
                severity_desc   = "No abnormal pattern detected. Your menstrual health indicators are within the expected range."
            elif score >= -0.05:
                severity        = "Low Risk Anomaly"
                severity_color  = "#F9A825"
                severity_bg     = "#FFFDE7"
                severity_border = "#F9A825"
                severity_desc   = "A mild irregularity has been detected. This may be a transient variation — monitor over the next 1–2 cycles."
            elif score >= -0.10:
                severity        = "Moderate Risk Anomaly"
                severity_color  = "#E65100"
                severity_bg     = "#FFF3E0"
                severity_border = "#E65100"
                severity_desc   = "A moderate anomaly has been detected across multiple health indicators. Consider reviewing your lifestyle habits."
            else:
                severity        = "High Risk Anomaly"
                severity_color  = "#B71C1C"
                severity_bg     = "#FFEBEE"
                severity_border = "#B71C1C"
                severity_desc   = "A significant anomaly has been detected. This pattern deviates substantially from normal. Consulting a healthcare provider is recommended."

            st.markdown(f"""
            <div style="background:{severity_bg}; border-left:4px solid {severity_border};
                        border-radius:8px; padding:16px 20px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <p style="margin:0; font-size:12px; color:#666;">Detection status</p>
                        <p style="margin:0; font-size:22px; font-weight:700;
                                  color:{severity_color};">{severity}</p>
                    </div>
                    <div style="text-align:right;">
                        <p style="margin:0; font-size:12px; color:#666;">Anomaly score</p>
                        <p style="margin:0; font-size:22px; font-weight:700;
                                  color:{severity_color};">{round(score, 4)}</p>
                        <p style="margin:0; font-size:10px; color:#999;">
                            more negative = more anomalous
                        </p>
                    </div>
                </div>
                <hr style="border-color:{severity_border}; opacity:0.3; margin:10px 0;">
                <p style="margin:0; font-size:13px; color:#444;">{severity_desc}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Contributing factors ───────────────────────────────────────────
            factors = []
            if cycle_len < 21 or cycle_len > 35:
                factors.append(("Abnormal cycle length",
                                f"{cycle_len} days is outside the normal range of 21–35 days."))
            if stress_ad >= 4:
                factors.append(("High stress level",
                                f"Stress level {stress_ad}/5 may disrupt hormonal balance and cycle regularity."))
            if sleep_ad < 6:
                factors.append(("Poor sleep",
                                f"Only {sleep_ad} hours of sleep per night — insufficient rest can affect cycle patterns."))
            if bmi_ad < 18.5:
                factors.append(("Low BMI",
                                f"BMI of {bmi_ad} is below the healthy range and may affect menstrual regularity."))
            elif bmi_ad > 29.9:
                factors.append(("High BMI",
                                f"BMI of {bmi_ad} is above the healthy range and may contribute to cycle irregularities."))
            if period_len < 2 or period_len > 7:
                factors.append(("Unusual period length",
                                f"{period_len} days is outside the typical range of 2–7 days."))

            if factors:
                st.markdown("#### Possible Contributing Factors")
                for title, detail in factors:
                    st.markdown(f"""
                    <div style="background:#FFF0F5; border-left:3px solid #D6336C;
                                border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                        <p style="margin:0; font-size:13px; font-weight:600;
                                  color:#D6336C;">{title}</p>
                        <p style="margin:2px 0 0; font-size:12px; color:#555;">{detail}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#F0FFF4; border-left:3px solid #2E7D32;
                            border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                    <p style="margin:0; font-size:13px; color:#2E7D32;">
                        No individual risk factors identified. The anomaly may reflect
                        a subtle multi-feature interaction rather than a single cause.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ── PCA visualization ──────────────────────────────────────────────
            st.markdown("#### PCA Visualization")
            feat_cols = [f for f in FEATURES if f in df.columns]
            if len(feat_cols) == len(FEATURES):
                X_all     = anomaly_pipeline['scaler'].transform(df[FEATURES].dropna())
                X_pca_all = anomaly_pipeline['pca'].transform(X_all)
                flags_all = anomaly_pipeline['iso_forest'].predict(X_all)

                pca_df = pd.DataFrame({
                    'PC1':    X_pca_all[:, 0],
                    'PC2':    X_pca_all[:, 1],
                    'Status': ['Anomaly' if f == -1 else 'Normal' for f in flags_all]
                })
                fig = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Status',
                    color_discrete_map={'Normal': '#021A54', 'Anomaly': '#D6336C'},
                    title='PCA — Anomaly Distribution', opacity=0.6
                )
                fig.add_trace(go.Scatter(
                    x=[pca_pt[0]], y=[pca_pt[1]], mode='markers',
                    marker=dict(size=16, color='#FFB6C1', symbol='star',
                                line=dict(color='#D6336C', width=2)),
                    name='Your Input'
                ))
                fig.update_layout(
                    height=380, plot_bgcolor='white', paper_bgcolor='white',
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.02, xanchor='right', x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The star marker shows where your input falls relative to the dataset. Points closer to the anomaly cluster indicate higher risk.")

            # ── Anomaly score timeline ─────────────────────────────────────────
            st.markdown("#### Anomaly Score Timeline")
            if len(feat_cols) == len(FEATURES):
                scores   = anomaly_pipeline['iso_forest'].decision_function(X_all)
                flags_tl = anomaly_pipeline['iso_forest'].predict(X_all)
                timeline_df = pd.DataFrame({
                    'Index':  list(range(len(scores))),
                    'Score':  scores,
                    'Status': ['Anomaly' if f == -1 else 'Normal' for f in flags_tl]
                })
                fig2 = px.scatter(
                    timeline_df, x='Index', y='Score', color='Status',
                    color_discrete_map={'Normal': '#021A54', 'Anomaly': '#D6336C'},
                    title='Anomaly Score Timeline — All Records', opacity=0.7
                )
                fig2.add_hline(y=0, line_dash='dash', line_color='gray',
                               annotation_text='Decision threshold (0)',
                               annotation_position='top right')
                fig2.add_hline(y=score, line_dash='dot', line_color='#D6336C',
                               annotation_text=f'Your score ({round(score, 3)})',
                               annotation_position='bottom right')
                fig2.update_layout(
                    height=300, xaxis_title='Record Index',
                    yaxis_title='Anomaly Score',
                    plot_bgcolor='white', paper_bgcolor='white'
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Scores below 0 are classified as anomalies. Your score is shown as a dotted line for comparison.")

            st.markdown("---")
            st.info("**Disclaimer:** This analysis is based on a machine learning model trained on synthetic data. Results are for educational and awareness purposes only and should not be used as a substitute for professional medical advice.")

        else:
            st.markdown("""
            <div class="result-card">
                <h4>Awaiting Input</h4>
                <p>Fill in the user data and click <b>Run Anomaly Detection</b>.</p>
            </div>
            """, unsafe_allow_html=True)
