Mental Health Score Predictor

Overview

This is a simple web-based application that predicts a person's mental health score using two very common lifestyle habits: how much screen time they get and how much they exercise daily. It gives an easy-to-understand score out of 100 and shows a color-based message that reflects their mental wellness level.

The app looks clean, friendly, and simple for any user to try. It is not meant to be a professional medical tool, but it gives a meaningful insight into how our habits might affect our mental health.



Why this topic?

Mental health is becoming increasingly important, especially with rising screen time and changing lifestyles. I chose this topic to create something simple yet meaningful that helps raise awareness and encourages healthier daily habits. It’s a way to connect machine learning with real-life wellness in a friendly, accessible format.



 How It Works

The app uses a machine learning model trained on example data. It takes in two inputs:

1. Screen Time** (hours per day)
2. Exercise (minutes per day)

Once the user enters these, the app calculates and shows:

A mental health score between 0 and 100**
A colored circle with a message:

   🟢 Green: Doing great
   🟡 Yellow: Room to improve
   🔴 Red: Habits may need serious changes



What Makes This App Useful

Even though it's a simple app, it shows how powerful even small machine learning models can be. It connects daily life with technology and can help beginners understand how data can give insights into health and behavior.



 Libraries Used

Streamlit: To create a beautiful and interactive web interface.
Pandas: To handle user input and structure the data for prediction.
Scikit-learn: Used to build and load the machine learning model.
Joblib: To save and load the trained model quickly.



Machine Learning and Preprocessing

The dataset used includes samples of screen time and exercise data along with **mental health scores**.
The model used is Linear Regression, chosen because it's simple, effective, and works well with just two input features.
Data was **cleaned and normalized** to remove any noise.
Features (screen time and exercise) were **scaled* to keep predictions reliable.
The model was trained using this clean data and saved as a `.pkl` file.


Why Linear Regression?

Linear Regression was selected because it performs well when the relationship between the inputs and output is continuous and fairly linear. Since the goal is to generate a numeric score from just two simple inputs, it's the best fit without introducing unnecessary complexity.



About the Dataset

The dataset is synthetic (demo) and includes three columns:

Screen Time** (hours per day)
Exercise Time** (minutes per day)
Mental Health Score** (0–100)

Each row represents a different individual’s daily habits. The model learns patterns from this and applies it to predict a new user’s score.



 Color-Based Results

After the model gives a score, the app displays a feedback message with a colored indicator:

 🟢 Green (Score 80–100):

You are following very healthy habits. Keep it up. You likely feel emotionally balanced and energized.

 🟡 Yellow (Score 60–79):

You are doing okay, but could feel better. Try small changes like less screen time or more exercise.

 🔴 Red (Score below 60):

Your habits might be affecting your mental well-being. Try to take breaks from screens, get some movement, and improve your routine gradually.



Conclusion

This project is a great example of how small data projects can be meaningful. It blends machine learning, health, and daily behavior into one interactive tool. While it’s not for medical use, it helps people reflect on their habits and gives students a great way to understand ML and app development in a real-world way.

This project is ideal for beginners learning:

Machine learning
Streamlit and UI creation
Connecting models to real-life inputs
Data preprocessing and scoring





