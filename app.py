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

def palette(val: float):
    if val >= 80:
        return {"bg": "#E8F5E9", "border": "#2E7D32", "text": "#1B5E20"}
    if val >= 60:
        return {"bg": "#FFF8E1", "border": "#F9A825", "text": "#9E7700"}
    return {"bg": "#FFEBEE", "border": "#C62828", "text": "#B71C1C"}

# ---------- Profession-specific tips ----------
PROFESSIONS = [
    "Custom",
    "School Kid",
    "College Student",
    "Software Engineer",
    "Doctor",
    "Nurse",
    "Teacher",
    "Driver",
    "Retail Worker",
    "Construction Worker",
    "Business Owner",
    "Homemaker",
    "Freelancer",
    "Artist",
    "Athlete (Beginner)",
    "Night-shift Worker",
    "Senior Citizen",
]

PRO_TIPS = {
    "School Kid": [
        "Keep screen breaks every class period; aim for short outdoor play after school.",
        "Prioritize consistent bedtime and reading time away from screens."
    ],
    "College Student": [
        "Batch screen use for study blocks and schedule short campus walks between classes.",
        "Try intramural activities or short bodyweight circuits in dorm/hostel rooms."
    ],
    "Software Engineer": [
        "Use 50/10 focus cycles and a standing desk if possible.",
        "Short mobility routines for neck, shoulders, and hips during breaks."
    ],
    "Doctor": [
        "Leverage stair use and brisk corridor walks between rounds.",
        "Protect 20–30 minutes on non-call days for light cardio."
    ],
    "Nurse": [
        "Gentle post-shift stretching to reduce tension.",
        "On lighter days, insert a short walk or cycling session."
    ],
    "Teacher": [
        "Walk the corridor during free periods and add brief stretching between classes.",
        "Afternoon 20-minute brisk walk before grading/planning."
    ],
    "Driver": [
        "Every stop: 3–5 minutes of walking and gentle back stretches.",
        "Maintain hydration; avoid screens during rest time to improve sleep quality."
    ],
    "Retail Worker": [
        "Foot and calf care after shifts; 10–15 minutes light mobility.",
        "On days off, a 30-minute walk or low-impact cardio."
    ],
    "Construction Worker": [
        "Prioritize recovery and mobility to protect joints.",
        "Keep screen time low at night to improve sleep recovery."
    ],
    "Business Owner": [
        "Block a non-negotiable 25–30 minute activity window in calendar.",
        "Batch notifications and use focus modes during deep work."
    ],
    "Homemaker": [
        "Convert chores into intentional movement with posture breaks.",
        "Short indoor walking or beginner yoga videos when time permits."
    ],
    "Freelancer": [
        "Pomodoro cycles with a 5-minute walk or mobility drill each break.",
        "Separate work devices from leisure to reduce evening screen use."
    ],
    "Artist": [
        "Posture breaks every 45 minutes; gentle wrist and back care.",
        "Outdoor sketches or walks to mix creativity with movement."
    ],
    "Athlete (Beginner)": [
        "Build base with low-impact cardio and light strength twice a week.",
        "Keep one full recovery day; avoid late-night screen use."
    ],
    "Night-shift Worker": [
        "Protect pre-sleep wind-down without screens; use blue-light filters.",
        "Short daylight walks after shift support circadian rhythm."
    ],
    "Senior Citizen": [
        "Prioritize safety: balance, gentle walking, or chair exercises.",
        "Keep screen use low in the evening to aid sleep."
    ],
}

# ---------- App ----------
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

    # Presets and Profession
    col_top1, col_top2 = st.columns([1, 1])
    with col_top1:
        preset = st.selectbox(
            "Presets (optional)",
            ("Custom", "Office routine", "Active routine", "Minimal screen"),
            index=0,
        )
    with col_top2:
        profession = st.selectbox("Profession", PROFESSIONS, index=0)

    # Session defaults so presets can control sliders
    if "screen_time_val" not in st.session_state:
        st.session_state.screen_time_val = 4.0
    if "exercise_val" not in st.session_state:
        st.session_state.exercise_val = 0.5

    # Apply preset values
    if preset == "Office routine":
        st.session_state.screen_time_val = 7.5
        st.session_state.exercise_val = 0.25
    elif preset == "Active routine":
        st.session_state.screen_time_val = 3.5
        st.session_state.exercise_val = 1.0
    elif preset == "Minimal screen":
        st.session_state.screen_time_val = 1.5
        st.session_state.exercise_val = 0.5

    col1, col2 = st.columns(2)
    with col1:
        screen_time = st.slider("Screen Time (hours/day)", 0.0, 12.0, float(st.session_state.screen_time_val), step=0.5)
    with col2:
        exercise = st.slider("Exercise (hours/day)", 0.0, 4.0, float(st.session_state.exercise_val), step=0.25)

    go = st.button("Predict My Score", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
    # Personalized Recommendations
    # ---------------------------
    target_exercise_per_day = 0.5  # 30 minutes/day
    low_exercise_threshold = 0.25  # 15 minutes/day

    st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Personalized Recommendations</div>", unsafe_allow_html=True)

    # Exercise recommendations based on input level
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
              • Add about 10 minutes per day each week and schedule 5 active days/week.<br>
              • Mix cardio (walking, cycling) with light strength work (squats, push-ups, resistance bands).
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
              • Swap with low-effort activities: short walk, stretching, reading, or a call with a friend.
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

    # Profession-specific tips
    tips = PRO_TIPS.get(profession, [])
    if tips:
        st.markdown(
            f"""
            <div class="tip" style="margin-top:10px;">
              <strong>Suggestions for {profession}:</strong><br>
              • {tips[0]}<br>
              • {tips[1]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Critical-condition exercise plans (based on predicted score)
    # Define severity tiers
    severe_cut = 40.0
    moderate_cut = 60.0
    if score < severe_cut:
        st.markdown(
            """
            <div class="tip" style="margin-top:10px;">
              <strong>When your score is low, start with low-impact plans:</strong><br>
              • 10–15 minutes/day of easy walking or chair exercises; focus on gentle range-of-motion.<br>
              • 1–2 beginner yoga or breathing sessions per week (10–20 minutes).<br>
              • Prioritize sleep routine and daytime light exposure.<br>
              • If you have pain, dizziness, or chronic conditions, speak with a qualified professional before progressing.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif score < moderate_cut:
        st.markdown(
            """
            <div class="tip" style="margin-top:10px;">
              <strong>Build steadily with moderate-intensity options:</strong><br>
              • Aim for 25–30 minutes/day on 5 days/week: brisk walk, cycling, or low-impact aerobics.<br>
              • Add short strength circuits twice weekly (squats, wall push-ups, hip hinges).<br>
              • Keep one lighter day for recovery and stretching.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="tip" style="margin-top:10px;">
              <strong>Maintain and refine:</strong><br>
              • Keep your 30 minutes/day baseline, consider variety (intervals, hills, cycling).<br>
              • Add mobility and core work 2–3 times per week for resilience.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # What-if Analysis
    # ---------------------------
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>What-if Analysis</div>", unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        delta_ex = st.slider("Adjust exercise (± hours/day)", -0.5, 1.0, 0.0, step=0.25)
    with col_w2:
        delta_sc = st.slider("Adjust screen time (± hours/day)", -2.0, 2.0, 0.0, step=0.25)

    sim_ex = clamp(exercise + delta_ex, 0.0, 4.0)
    sim_sc = clamp(screen_time + delta_sc, 0.0, 12.0)

    X_sim = prepare_features(model, sim_sc, sim_ex)
    sim_score = max(0.0, min(100.0, float(model.predict(X_sim)[0])))
    diff = sim_score - score

    st.markdown(
        f"""
        <div class="tip">
          <strong>Simulated score:</strong> {sim_score:.1f} / 100
          <br>Change vs current: {diff:+.1f}
          <br>New inputs → Exercise: {sim_ex:.2f} h/day, Screen: {sim_sc:.2f} h/day
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # Reach a Target Score (linear estimate)
    # ---------------------------
    st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Reach a Target Score</div>", unsafe_allow_html=True)

    target = st.slider("Choose a target score", 50, 90, 70, step=1)

    def estimate_extra_exercise_hours(target_score: float):
        coef = getattr(model, "coef_", None)
        names = getattr(model, "feature_names_in_", None)
        if coef is None or names is None:
            return None  # non-linear or missing metadata

        name_map = {n.lower(): n for n in names}
        coef_map = {n.lower(): float(c) for n, c in zip(names, coef)}

        # Current prediction
        row = prepare_features(model, screen_time, exercise)
        current_pred = float(model.predict(row)[0])

        # Determine which exercise feature is used
        ex_key = None
        scale_to_hours = 1.0
        if "exercise_hours" in name_map and "exercise_hours" in coef_map:
            ex_key = "exercise_hours"
            scale_to_hours = 1.0
        elif "exercise_minutes" in name_map and "exercise_minutes" in coef_map:
            ex_key = "exercise_minutes"
            scale_to_hours = 60.0
        else:
            return None

        ex_coef = coef_map.get(ex_key, 0.0)
        if abs(ex_coef) < 1e-9:
            return None

        delta_feature = (target_score - current_pred) / ex_coef
        delta_hours = float(delta_feature) / scale_to_hours
        return delta_hours

    delta_needed = estimate_extra_exercise_hours(target)
    if delta_needed is None:
        st.markdown(
            "<div class='tip'>A precise estimate is not available for this model. Try the What-if Analysis sliders above to explore improvements.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="tip">
              To reach a score of <strong>{target}</strong>, you would need approximately
              <strong>{abs(delta_needed)*60:.0f} additional minutes/day</strong> of exercise (estimate).
              Use the What-if Analysis to verify.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # Session History & Download
    # ---------------------------
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append(
        {
            "score": round(score, 1),
            "screen_time_h": round(screen_time, 2),
            "exercise_h": round(exercise, 2),
            "profession": profession,
        }
    )

    hist_df = pd.DataFrame(st.session_state.history)

    st.markdown("<div class='card' style='margin-top:14px;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Session History</div>", unsafe_allow_html=True)
    st.dataframe(hist_df, use_container_width=True)

    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download session as CSV", csv, "mental_health_session.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("Built with a simple linear regression model. This app is for educational/demo purposes only.")








