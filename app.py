import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Mental Health Score Predictor", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"],
      .block-container, [data-testid="stHeader"] {
        background:#ffffff !important; color:#000000 !important; box-shadow:none !important;
      }
      /* Tighten top spacing so no stray bar appears */
      .block-container { padding-top: 0.5rem !important; }
      /* Simple, clean cards */
      .card { background:#ffffff; border:1px solid #ECECEC; border-radius:16px; padding:18px; box-shadow: 0 4px 18px rgba(0,0,0,.06); }
      .result { border-radius:16px; padding:22px; text-align:center; box-shadow: 0 6px 22px rgba(0,0,0,.08); border:2px solid transparent; }
      .score-badge { display:inline-block; padding:10px 18px; border-radius:999px; font-weight:800; font-size:28px; }
      .muted { color:#222; opacity:.8; }
      .section-title { font-weight:700; margin: 0 0 6px 0; font-size:18px; }
      .stButton>button { border-radius: 10px; padding: .6rem 1.1rem; font-weight:600; border: 1px solid #4CAF50; width: 100%; }
      /* Centered hero without extra wrappers that create ghost bars */
      .hero { padding: 18px 8px 6px 8px; text-align:center; }
      .hero h1 { margin: 0; font-size: 34px; letter-spacing: 0.2px; color:#0F5132; }
      .hero p { margin: 6px auto 0 auto; font-size: 16px; opacity: .85; max-width: 720px; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    p = Path(__file__).parent / "mental_health_model.pkl"
    if not p.exists():
        st.error(f"Model file not found at: {p}")
        st.stop()
    return joblib.load(p)

def prepare_features(model, screen_hours: float, exercise_hours: float) -> pd.DataFrame:
    ex_minutes = exercise_hours * 60.0
    names = getattr(model, "feature_names_in_", None) or ["Screen_Time_Hours", "Exercise_Minutes"]
    features = {}
    for name in names:
        key = name.lower()
        if key == "screen_time_hours":
            features[name] = float(screen_hours)
        elif key == "exercise_minutes":
            features[name] = float(ex_minutes)
        elif key == "exercise_hours":
            features[name] = float(exercise_hours)
        else:
            features[name] = 0.0
    return pd.DataFrame([features], columns=list(names))

model = load_model()


st.markdown(
    """
    <div class="hero">
        <h1>Mental Health Score Predictor</h1>
        <p>Estimate your mental wellness score based on screen time and exercise (both in hours/day).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Enter Your Daily Habits</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    screen_time = st.slider("Screen Time (hours/day)", 0.0, 12.0, 4.0, step=0.5)
with col2:
    exercise = st.slider("Exercise (hours/day)", 0.0, 4.0, 0.5, step=0.25)
go = st.button("Predict My Score")
st.markdown("</div>", unsafe_allow_html=True)

def palette(val: float):
    if val >= 80:
        return {"bg": "#E8F5E9", "border": "#2E7D32", "text": "#1B5E20"}
    if val >= 60:
        return {"bg": "#FFF8E1", "border": "#F9A825", "text": "#9E7700"}
    return {"bg": "#FFEBEE", "border": "#C62828", "text": "#B71C1C"}

if go:
    X = prepare_features(model, screen_time, exercise)
    y_pred = model.predict(X)[0]
    score = max(0.0, min(100.0, float(y_pred)))
    pal = palette(score)

    st.markdown(
        f"""
        <div class='result' style="background:{pal['bg']}; border-color:{pal['border']};">
            <div class="muted" style="margin-bottom:6px;">Your Predicted Mental Health Score</div>
            <div class="score-badge" style="color:{pal['text']}; border:2px solid {pal['border']}; background:#ffffff;">{score:.1f} / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("Built with a simple linear regression model. This app is for educational/demo purposes only.")






