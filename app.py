import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Mental Health Score Predictor", page_icon="🧠", layout="centered")


model = joblib.load('/Users/shreyasvikrantdewangswami/MentalHealthApp/mental_health_model.pkl')


st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color:#4CAF50;'>🧠 Mental Health Score Predictor</h1>
        <p style='font-size:18px;'>Estimate your mental wellness score based on your screen time and exercise habits.</p>
    </div>
""", unsafe_allow_html=True)


with st.container():
    st.markdown("### 📥 Enter Your Daily Habits")
    col1, col2 = st.columns(2)

    with col1:
        screen_time = st.slider("📱 Screen Time (hours/day)", 0.0, 12.0, 4.0, step=0.5)

    with col2:
        exercise = st.slider("🏃 Exercise (minutes/day)", 0, 120, 30)

# 🎯 Prediction Button
if st.button("✨ Predict My Score"):
    input_df = pd.DataFrame([[
        screen_time, exercise
    ]], columns=['Screen_Time_Hours', 'Exercise_Minutes'])

    #  Prediction
    raw_score = model.predict(input_df)[0]
    score = max(0, min(100, raw_score))

    # 🟢 Feedback Message
    if score >= 80:
        emoji = "🟢"
        msg = "Excellent! Your mental health appears strong. Keep it up! 💪"
        color = "#C8E6C9"
    elif score >= 60:
        emoji = "🟡"
        msg = "You're doing okay, but there's room to improve. Try reducing screen time or moving more! 🚶"
        color = "#FFF9C4"
    else:
        emoji = "🔴"
        msg = "Your score is low. Consider healthier routines. You're not alone—start small! ❤️"
        color = "#FFCDD2"

    #  Result Card
    st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:15px; text-align:center;">
            <h2 style='color:#333;'>{emoji} Your Predicted Mental Health Score: <strong>{score:.1f} / 100</strong></h2>
            <p style='font-size:17px;'>{msg}</p>
        </div>
    """, unsafe_allow_html=True)

# ℹ Footer
st.markdown("---")
st.caption("💡 Built with a simple linear regression model. This app is for educational/demo purposes only.")
