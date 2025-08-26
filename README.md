
Mental Health Score Predictor

OVERVIEW
This is a simple web application that predicts a person’s mental health score using two lifestyle habits: daily screen time and daily exercise. The app outputs a score out of 100 with a short message indicating the user’s wellness range. The interface is clean and beginner friendly. This is not a medical tool; it is an educational project to help people reflect on how habits may relate to well-being.

LIVE APP
Streamlit deployment: [https://mentalhealthpredictor-enkctfobpwl9dknp3urkah.streamlit.app/](https://mentalhealthpredictor-enkctfobpwl9dknp3urkah.streamlit.app/)

REPOSITORY
GitHub: [https://github.com/dewangshree/MentalHealthPredictor](https://github.com/dewangshree/MentalHealthPredictor)

WHY THIS PROJECT
Mental health is a growing concern with increasing screen use and changing routines. This project demonstrates how a basic machine learning model can provide simple, interpretable feedback about lifestyle habits. It aims to raise awareness while serving as an accessible example for beginners in machine learning and app development.

HOW IT WORKS
Inputs (both in hours per day):

1. Screen Time (0–12)
2. Exercise Time (0–4)

Outputs:

1. Predicted Mental Health Score (0–100)
2. A short interpretation mapped to a color range: Green (doing great), Yellow (room to improve), Red (habits may need significant changes)

WHY THESE INPUT RANGES AND WHAT THEY INDICATE
Screen Time capped at 12 hours per day:

* Rationale: Twelve hours represents an extreme but realistic upper bound when combining work, study, and entertainment. Consistently exceeding half of the day on screens is uncommon for most people and often reflects potentially unbalanced digital habits.
* Indication: Higher screen time values generally suggest increased risk of digital fatigue and reduced recovery time, which can correlate with lower wellness scores in this educational model.

Exercise Time capped at 4 hours per day:

* Rationale: Four hours is a practical upper bound for the general population. While athletes may exceed this, daily exercise beyond four hours is atypical and can introduce edge cases that are not representative for beginners using this app.
* Indication: Greater exercise time up to this cap tends to indicate healthier activity habits and recovery, which can correlate with higher wellness scores in this educational model.

WHAT MAKES THIS USEFUL
With only two inputs and a simple model, the app shows how data and machine learning can provide understandable, actionable feedback. It is an approachable starting point for students to learn about preprocessing, model training, deployment, and building an interactive interface.

TECHNOLOGIES AND LIBRARIES

 Streamlit: web user interface
 Pandas: data handling
 Scikit-learn: model training and prediction
 Joblib: model serialization and loading

MACHINE LEARNING DETAILS

Task: Predict a continuous mental health score from two numeric features (screen time and exercise), both measured in hours.
 Model: Linear Regression, chosen for simplicity, speed, and suitability for approximately linear relationships with a continuous target.
Data: Synthetic demonstration dataset with three columns:

   Screen Time (hours/day, 0–12)
   Exercise Time (hours/day, 0–4)
   Mental Health Score (0–100)
 Preprocessing:

   Data cleaning to remove invalid or noisy rows
   Feature scaling to keep magnitudes comparable and stabilize training
  Model artifact:

  Trained model saved as a .pkl file via joblib for fast loading in the app

COLOR-BASED INTERPRETATION

 Green (Score 80–100): Strong and balanced habits
 Yellow (Score 60–79): Generally okay; consider modest changes such as reducing screen time or increasing exercise
 Red (Score below 60): Habits may be affecting well-being; consider limiting continuous screen exposure and adding more movement gradually

INSTALLATION

1. Install Python 3.9 or later.
2. Create and activate a virtual environment.
3. Install dependencies:

   
4. Ensure the trained model file (.pkl) is present where the app expects it.

RUNNING THE APP LOCALLY

```bash
streamlit run app.py
```

DATASET NOTES

* Screen Time is limited to 12 hours/day as a realistic upper bound for most users.
* Exercise Time is limited to 4 hours/day as a practical maximum for typical daily routines.
* Mental Health Score ranges from 0 to 100 and represents overall wellness in this educational context.

LIMITATIONS

* The dataset is synthetic and simplified for demonstration.
* Predictions are approximate and for educational use only; they are not clinical assessments.

DISCLAIMER
This project is educational and informational. It does not provide medical advice or diagnosis. For mental health concerns, consult qualified professionals or local support resources.

CONTRIBUTING
Fork the repository, create a feature branch, make improvements with clear commits, and open a pull request with a concise description and testing notes.

CONTACT
For issues or feature requests, open an issue in the GitHub repository.










