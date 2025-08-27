import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Mental Health Score Predictor", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      html, body, .stApp, [data-testid="stAppViewContainer"] { background:#ffffff !important; color:#000000 !important; }
      .wrap { max-width: 820px; margin: 0 auto; }
      .hero { padding: 18px 8px 0 8px; text-align:center; }
      .hero h1 { margin: 0; font-size: 34px; letter-spacing: 0.2px; color:#0F5132; }
      .hero p { margin: 6px 0 0 0; font-size: 16px; opacity: .85; }
      .card { background:#ffffff; border:1px solid #ECECEC; border-radius:16px; padding:18px; box-shadow: 0 4px 18px rgba(0,0,0,.06); }
      .result { border-radius:16px; padding:22px; text-align:center; box-shadow: 0 6px 22px rgba(0,0,0,.08); border:2px solid transparent; }
      .score-badge { display:inline-block; padding:10px 18px; border-radius:999px; font-weight:800; font-size:28px; }
      .muted { color:#222; opacity:.8; }
      .section-title { font-weight:700; margin: 0 0 6px 0; font-size:18px; }
      .stButton>button { border-radius: 10px; padding: .6rem 1.1rem; font-weight:600; border: 1px solid #4CAF50; }
      .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
      @media (max-width: 680px) { .grid { grid-template-columns: 1fr; } }
      .tip { font-size:14px; padding:12px 14px; border-radius:12px; border:1px dashed #E0E0E0; background:#FAFAFA; }
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
    features = {}
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = ["Screen_Time_Hours", "Exercise_Minutes"]
    for name in names:
        if name.lower() == "screen_time_hours":
            features[name] = float(screen_hours)
        elif name.lower() == "exercise_minutes":
            features[name] = float(ex_minutes)
        elif name.lower() == "exercise_hours":
            features[name] = float(exercise_hours)
        else:
            features[name] = 0.0
    return pd.DataFrame([features], columns=list(names))

model = load_model()

st.markdown("<div class='wrap'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <h1>Mental Health Score Predictor</h1>
        <p>Estimate your mental wellness score based on screen time and exercise (both in hours/day).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Enter Your Daily Habits</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        screen_time = st.slider("Screen Time (hours/day)", 0.0, 12.0, 4.0, step=0.5)
    with col2:
        exercise = st.slider("Exercise (hours/day)", 0.0, 4.0, 0.5, step=0.25)
    go = st.button("Predict My Score", use_container_width=True)
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

    # ---------------------------
    # Personalized Recommendations (added)
    # ---------------------------
    target_exercise_per_day = 0.5  # 30 minutes/day
    low_exercise_threshold = 0.25  # 15 minutes/day

    st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Personalized Recommendations</div>", unsafe_allow_html=True)

    # Exercise recommendations
    if exercise == 0:
        st.markdown(
            """
            <div class="tip">
              <strong>Start moving gradually:</strong><br>
              • Begin with 10–15 minutes of easy activity daily (walking, light stretching, or beginner yoga).<br>
              • Add 5 minutes every few days until you reach about 30 minutes per day on at least 5 days/week.<br>
              • Keep it simple: short walks after meals, take stairs, light bodyweight exercises at home.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif exercise < low_exercise_threshold:
        st.markdown(
            f"""
            <div class="tip">
              <strong>Increase activity to reach a healthy baseline:</strong><br>
              • You're doing about {exercise:.2f} h/day. Aim for at least {target_exercise_per_day:.2f} h/day (≈30 minutes).<br>
              • Add ~10 minutes per day each week and schedule 5 active days/week.<br>
              • Mix cardio (walking, cycling) with light strength work (squats, push-ups, bands).
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif exercise < target_exercise_per_day:
        st.markdown(
            f"""
            <div class="tip">
              <strong>You're close to the recommended level:</strong><br>
              • Current: {exercise:.2f} h/day. Target: {target_exercise_per_day:.2f} h/day (≈30 minutes).<br>
              • Add one more short session or extend two sessions by 10 minutes.<br>
              • Keep one rest day and focus on consistency over intensity.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="tip">
              <strong>Maintain what's working:</strong><br>
              • You're at or above 30 minutes/day. Keep a routine you enjoy and rotate activities to prevent burnout.<br>
              • Include recovery: gentle stretching or a light day each week.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Screen-time recommendations
    if screen_time > 6:
        st.markdown(
            f"""
            <div class="tip" style="margin-top:10px;">
              <strong>Reduce high screen time:</strong><br>
              • Current: {screen_time:.1f} h/day. Try cutting 30–60 minutes by batching notifications and using app limits.<br>
              • Insert short no-screen breaks every 60–90 minutes and avoid screens 60 minutes before bedtime.<br>
              • Swap with low-effort activities: short walk, stretching, reading, or a quick call with a friend.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif screen_time > 3:
        st.markdown(
            f"""
            <div class="tip" style="margin-top:10px;">
              <strong>Fine-tune screen habits:</strong><br>
              • Current: {screen_time:.1f} h/day. A small reduction (15–30 minutes) can improve sleep and mood.<br>
              • Use scheduled focus modes and cluster social/app checks into set windows.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="tip" style="margin-top:10px;">
              <strong>Good balance on screen use:</strong><br>
              • Keep protecting off-screen time and keep evenings calm for better sleep quality.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("Built with a simple linear regression model. This app is for educational/demo purposes only.")







