import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Mental Health Score Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      html, body, .stApp, [data-testid="stAppViewContainer"] { background: #ffffff !important; color: #000000 !important; }
      .card { padding: 20px; border-radius: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); background: #ffffff; }
      .result { padding: 22px; border-radius: 16px; text-align: center; box-shadow: 0 6px 20px rgba(0,0,0,0.10); }
      .title { text-align:center; padding: 8px 0 0 0; }
      .title h1 { color:#4CAF50; margin:0; }
      .subtitle { font-size:18px; opacity:0.85; margin-top:6px; }
      .stButton>button { border-radius: 999px; padding: 0.6rem 1.2rem; font-weight: 600; border: 1px solid #4CAF50; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "mental_health_model.pkl"
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.markdown(
    """
    <div class='title'>
        <h1>Mental Health Score Predictor</h1>
        <div class='subtitle'>Estimate your mental wellness score based on your screen time and exercise habits.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Enter Your Daily Habits")
    col1, col2 = st.columns(2)
    with col1:
        screen_time = st.slider("Screen Time (hours/day)", 0.0, 12.0, 4.0, step=0.5)
    with col2:
        exercise = st.slider("Exercise (minutes/day)", 0, 120, 30)
    go = st.button("Predict My Score")
    st.markdown("</div>", unsafe_allow_html=True)

if go:
    input_df = pd.DataFrame([[screen_time, exercise]], columns=["Screen_Time_Hours", "Exercise_Minutes"])
    raw_score = model.predict(input_df)[0]
    score = max(0, min(100, raw_score))

    if score >= 80:
        bg = "#E8F5E9"; border = "#2E7D32"; text = "#1B5E20"  
    elif score >= 60:
        bg = "#FFF8E1"; border = "#F9A825"; text = "#9E7700"   
    else:
        bg = "#FFEBEE"; border = "#C62828"; text = "#B71C1C"   

    st.markdown(
        f"""
        <div class='result' style="
            background:{bg} !important;
            border: 2px solid {border} !important;
        ">
            <h2 style="color:{text}; font-size:26px; margin:0 0 6px 0;">Your Predicted Mental Health Score</h2>
            <div style="font-size:34px; font-weight:800; color:{text};">{score:.1f} / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("Built with a simple linear regression model. This app is for educational/demo purposes only.")


st.markdown("---")
st.caption("Built with a simple linear regression model. This app is for educational/demo purposes only.")


