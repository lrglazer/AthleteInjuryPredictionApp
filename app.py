import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Athlete Injury Risk Predictor",
    page_icon="🏃",
    layout="wide"
)

model = joblib.load("injury_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

numerical_cols = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "Training_Intensity",
    "Training_Hours_Per_Week",
    "Recovery_Days_Per_Week",
    "Match_Count_Per_Week",
    "Rest_Between_Events_Days",
    "Fatigue_Score",
    "Performance_Score",
    "Team_Contribution_Score",
    "Load_Balance_Score",
    "ACL_Risk_Score"
]

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #25143a 0%, #131525 45%, #0b1020 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100% !important;
}

html, body, p, div, span, label, h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
}

.main-card {
    background: linear-gradient(180deg, rgba(32,36,66,0.95) 0%, rgba(20,24,46,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 1.35rem;
    box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    margin-bottom: 1rem;
}

.section-card {
    background: linear-gradient(180deg, rgba(31,35,66,0.95) 0%, rgba(18,22,43,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 1.15rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
    height: 100%;
}

.title-text {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 0.35rem;
}

.subtitle-text {
    color: #cbd5e1 !important;
    font-size: 1rem;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 0.9rem;
}

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    color: #ffffff !important;
}

div[data-baseweb="select"] * {
    color: #ffffff !important;
}

div[role="listbox"] {
    background-color: #111827 !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
}

div[role="option"] {
    background-color: #111827 !important;
    color: #ffffff !important;
}

div[role="option"]:hover {
    background-color: #1f2937 !important;
    color: #ffffff !important;
}

.stNumberInput input {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
}

.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 16px;
    border: none;
    font-size: 1rem;
    font-weight: 800;
    color: white !important;
    background: linear-gradient(90deg, #ff4fd8 0%, #b84dff 100%) !important;
    box-shadow: 0 10px 25px rgba(184,77,255,0.35);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ff6ae0 0%, #ca6aff 100%) !important;
    color: white !important;
}

.result-box-low, .result-box-med, .result-box-high {
    border-radius: 20px;
    padding: 1.25rem;
    text-align: center;
    font-weight: 800;
    font-size: 1.35rem;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}

.result-box-low {
    background: linear-gradient(135deg, #0f766e, #14b8a6);
    color: white !important;
}

.result-box-med {
    background: linear-gradient(135deg, #a16207, #f59e0b);
    color: white !important;
}

.result-box-high {
    background: linear-gradient(135deg, #be123c, #f43f5e);
    color: white !important;
}

.helper-text {
    color: #cbd5e1 !important;
    font-size: 0.95rem;
}

.flag-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.85rem;
    margin-bottom: 0.6rem;
    color: #f8fafc !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-card">
    <div class="title-text">🏃 Athlete Injury Risk Predictor</div>
    <div class="subtitle-text">
        Estimate athlete injury risk using training load, recovery, fatigue, and performance data.
    </div>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1.55, 1], gap="medium")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Athlete Inputs</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        age = st.number_input("Age", min_value=16, max_value=40, value=20)
        height = st.number_input("Height (cm)", min_value=140, max_value=230, value=175)
        weight = st.number_input("Weight (kg)", min_value=40, max_value=140, value=70)
        training_intensity = st.slider("Training Intensity", 1, 10, 6)
        training_hours = st.slider("Training Hours Per Week", 0, 30, 10)

    with c2:
        recovery_days = st.slider("Recovery Days Per Week", 0, 7, 2)
        match_count = st.slider("Match Count Per Week", 0, 7, 2)
        rest_days = st.slider("Rest Between Events (Days)", 0, 7, 2)
        fatigue = st.slider("Fatigue Score", 1, 10, 5)
        performance = st.slider("Performance Score", 0, 100, 80)

    with c3:
        team_contribution = st.slider("Team Contribution Score", 0, 100, 75)
        load_balance = st.slider("Load Balance Score", 0, 100, 70)
        acl_risk = st.slider("ACL Risk Score", 0, 100, 40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        position = st.selectbox("Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"])

    predict = st.button("Predict Injury Risk")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How to Use</div>', unsafe_allow_html=True)
    st.markdown('<div class="helper-text">Enter the athlete profile, training load, recovery, and performance data, then click <b>Predict Injury Risk</b> to generate a model-based estimate.</div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="helper-text"><b>Risk levels:</b><br>Low: under 33%<br>Medium: 33% to under 66%<br>High: 66% and above</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if predict:
    input_data = {
        "Age": age,
        "Height_cm": height,
        "Weight_kg": weight,
        "Training_Intensity": training_intensity,
        "Training_Hours_Per_Week": training_hours,
        "Recovery_Days_Per_Week": recovery_days,
        "Match_Count_Per_Week": match_count,
        "Rest_Between_Events_Days": rest_days,
        "Fatigue_Score": fatigue,
        "Performance_Score": performance,
        "Team_Contribution_Score": team_contribution,
        "Load_Balance_Score": load_balance,
        "ACL_Risk_Score": acl_risk,
        "Gender": gender,
        "Position": position
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    prob = model.predict_proba(df)[0][1]
    prob_percent = prob * 100

    flags = []
    if fatigue >= 8:
        flags.append("High fatigue score may increase injury risk.")
    if recovery_days <= 1:
        flags.append("Very low recovery time may reduce readiness.")
    if training_intensity >= 8:
        flags.append("High training intensity may raise physical strain.")
    if training_hours >= 18:
        flags.append("High weekly training volume may increase load.")
    if rest_days <= 1:
        flags.append("Limited rest between events may increase accumulation of fatigue.")
    if acl_risk >= 70:
        flags.append("Elevated ACL risk score may deserve extra monitoring.")
    if load_balance <= 45:
        flags.append("Low load balance score may indicate uneven workload distribution.")

    result_col, flag_col = st.columns([1.15, 1], gap="medium")

    with result_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        st.progress(min(int(prob_percent), 100))
        st.write(f"**Predicted injury probability:** {prob_percent:.2f}%")

        if prob < 0.33:
            st.markdown(
                f'<div class="result-box-low">LOW RISK<br><span style="font-size:1rem; font-weight:500;">Estimated probability: {prob_percent:.2f}%</span></div>',
                unsafe_allow_html=True
            )
        elif prob < 0.66:
            st.markdown(
                f'<div class="result-box-med">MEDIUM RISK<br><span style="font-size:1rem; font-weight:500;">Estimated probability: {prob_percent:.2f}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box-high">HIGH RISK<br><span style="font-size:1rem; font-weight:500;">Estimated probability: {prob_percent:.2f}%</span></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="helper-text">This score is generated from your trained logistic regression model using the saved preprocessing pipeline.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with flag_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Risk Flags</div>', unsafe_allow_html=True)

        if flags:
            for flag in flags:
                st.markdown(f'<div class="flag-box">{flag}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="flag-box">No major red flags from the entered values.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Athlete Summary</div>', unsafe_allow_html=True)

    summary_df = pd.DataFrame({
        "Metric": [
            "Age", "Height_cm", "Weight_kg", "Training_Intensity",
            "Training_Hours_Per_Week", "Recovery_Days_Per_Week",
            "Match_Count_Per_Week", "Rest_Between_Events_Days",
            "Fatigue_Score", "Performance_Score",
            "Team_Contribution_Score", "Load_Balance_Score",
            "ACL_Risk_Score", "Gender", "Position"
        ],
        "Value": [
            age, height, weight, training_intensity,
            training_hours, recovery_days, match_count, rest_days,
            fatigue, performance, team_contribution, load_balance,
            acl_risk, gender, position
        ]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("Built with Streamlit, scikit-learn, and your athlete injury prediction model.")