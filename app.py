import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Detector", page_icon="🩺", layout="centered")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0f1117; }
    .title  { font-size: 2.2rem; font-weight: 700; color: #ffffff; text-align: center; margin-bottom: 0; }
    .sub    { font-size: 1rem; color: #8b9ab0; text-align: center; margin-bottom: 2rem; }
    .card   { background: #1e2130; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.2rem; }
    .result-yes { background: #3b1a1a; border: 1.5px solid #e05c5c; border-radius: 14px; padding: 1.2rem 1.6rem; text-align: center; color: #f87171; font-size: 1.3rem; font-weight: 600; }
    .result-no  { background: #1a2e25; border: 1.5px solid #34d399; border-radius: 14px; padding: 1.2rem 1.6rem; text-align: center; color: #34d399; font-size: 1.3rem; font-weight: 600; }
    .section-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8b9ab0; margin-bottom: 0.4rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
model = joblib.load("diabetes_model.pkl")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🩺 Diabetes Risk Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Enter patient details below and click Predict</div>', unsafe_allow_html=True)

# ── Input fields ───────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.slider("Pregnancies",        0,    17,   1)
    glucose     = st.slider("Glucose (mg/dL)",   50,   200, 120)
    bp          = st.slider("Blood Pressure",     40,   122,  70)
    skin        = st.slider("Skin Thickness (mm)", 0,    60,  20)
with col2:
    insulin     = st.slider("Insulin (μU/ml)",    0,   850,  80)
    bmi         = st.slider("BMI",              15.0,  67.0, 25.0)
    dpf         = st.slider("Diabetes Pedigree", 0.07, 2.50, 0.47)
    age         = st.slider("Age",               21,    81,  30)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
predict = st.button("⚡ Predict Now", use_container_width=True)

if predict:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    proba      = model.predict_proba(input_data)[0]
    risk_pct   = round(proba[1] * 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result banner ──────────────────────────────────────────────────────────
    if prediction[0] == 1:
        st.markdown(f'<div class="result-yes">⚠️ Higher Diabetes Risk — {risk_pct}% probability</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-no">✅ Lower Diabetes Risk — {risk_pct}% probability</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ── Risk gauge ─────────────────────────────────────────────────────────────
    with col_a:
        st.markdown("**Risk Gauge**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "gray"},
                "bar":  {"color": "#e05c5c" if prediction[0] == 1 else "#34d399"},
                "bgcolor": "#1e2130",
                "steps": [
                    {"range": [0,  40], "color": "#1a2e25"},
                    {"range": [40, 70], "color": "#2e2a1a"},
                    {"range": [70,100], "color": "#3b1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": risk_pct}
            }
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(t=20, b=10, l=20, r=20),
            paper_bgcolor="#0f1117", font_color="white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Feature bar chart ──────────────────────────────────────────────────────
    with col_b:
        st.markdown("**Your Values vs Normal Range**")
        labels  = ["Glucose", "BMI", "Age", "Insulin"]
        values  = [glucose,   bmi,   age,   insulin]
        normals = [100,       22,    35,    85]
        colors  = ["#e05c5c" if v > n else "#34d399" for v, n in zip(values, normals)]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Normal", x=labels, y=normals,
            marker_color="#2a3040", text=normals, textposition="outside"
        ))
        fig_bar.add_trace(go.Bar(
            name="Yours", x=labels, y=values,
            marker_color=colors, text=values, textposition="outside"
        ))
        fig_bar.update_layout(
            barmode="group", height=220,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="white", legend=dict(font=dict(color="white")),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("**Key Risk Factors**")
    features   = ["Glucose", "BMI", "Age", "DPF", "Insulin", "Pregnancies", "BP", "Skin"]
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    fig_imp = go.Figure(go.Bar(
        x=importance[sorted_idx],
        y=[features[i] for i in sorted_idx],
        orientation="h",
        marker_color=["#6366f1" if i == sorted_idx[-1] else "#2a3040" for i in range(len(features))]
    ))
    fig_imp.update_layout(
        height=250, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="white",
        xaxis=dict(showgrid=False, title="Importance"),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.caption("⚠️ For educational purposes only. Always consult a qualified healthcare professional.")
