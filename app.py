import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Mental Health Score Predictor", layout="centered")

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "mental_health_model.pkl"
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color:#4CAF50;'>Mental Health Score Predictor</h1>
        <p style='font-size:18px;'>Estimate your mental wellness score based on your screen time and exercise habits.</p>
    </div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("### Enter Your Daily Habits")
    col1, col2 = st.columns(2)

    with col1:
        screen_time = st.slider("Screen Time (hours/day)", 0.0, 12.0, 4.0, step=0.5)

    with col2:
        exercise = st.slider("Exercise (minutes/day)", 0, 120, 30)

if st.button("Predict My Score"):
    input_df = pd.DataFrame([[
        screen_time, exercise
    ]], columns=['Screen_Time_Hours', 'Exercise_Minutes'])

    raw_score = model.predict(input_df)[0]
    score = max(0, min(100, raw_score))

    if score >= 80:
        color = "#C8E6C9"   
    elif score >= 60:
        color = "#FFF9C4"   
    else:
        color = "#FFCDD2"   

    st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
            <h2 style='color:#000000; font-size: 24px; margin-bottom: 10px;'>Your Predicted Mental Health Score: <strong>{score:.1f} / 100</strong></h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with a simple linear regression model. This app is for educational/demo purposes only.")

