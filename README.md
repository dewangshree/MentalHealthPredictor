Mental Health Score Predictor

This project is a simple but meaningful web application that predicts a person’s mental health score based on two daily lifestyle habits: screen time and exercise. The application gives a score out of 100 along with a short interpretation of what that score means for overall well-being. It is designed to be easy to use, clean, and approachable for beginners. While it is not a medical or diagnostic tool, it is intended to encourage reflection on everyday habits and how they may relate to mental health.

The app is live and can be accessed online through Streamlit here:
[https://mentalhealthpredictor-enkctfobpwl9dknp3urkah.streamlit.app/](https://mentalhealthpredictor-enkctfobpwl9dknp3urkah.streamlit.app/)

The full source code and resources are available on GitHub:
[https://github.com/dewangshree/MentalHealthPredictor](https://github.com/dewangshree/MentalHealthPredictor)



Why this project

Mental health is becoming more important than ever, with many people spending long hours on screens and struggling to maintain balanced routines. This project was created to raise awareness about how lifestyle habits, even simple ones like screen use and exercise, may influence our wellness. At the same time, it serves as an educational example for students and beginners who want to learn about data science, machine learning, and deployment.

By focusing only on two inputs—daily screen time and daily exercise—the project makes the concept easy to understand. Users can instantly see how these two factors combine to generate a score and get quick feedback on whether their habits look balanced, need a little adjustment, or could benefit from more serious changes.



How it works

The app asks for two inputs: how many hours a person spends daily on screens (capped at 12 hours) and how many hours they exercise each day (capped at 4 hours). Based on these numbers, the model predicts a mental health score between 0 and 100.

The score is then grouped into three categories for better interpretation. A score of 80 or higher is considered strong and balanced, shown in green. Scores between 60 and 79 are considered okay but with room for improvement, shown in yellow. Any score below 60 is shown in red, suggesting that habits might be affecting well-being in a more serious way.

The limits for input values were chosen to reflect realistic ranges. Twelve hours of screen time is already half of the day and is considered an extreme but still possible upper bound. Similarly, exercising for more than four hours daily is uncommon for most people, so the limit was set there. These choices make the predictions simple, relatable, and practical for everyday users.



The machine learning model

The project uses a Linear Regression model, which was chosen for its simplicity and interpretability. The dataset is synthetic and contains three features: screen time, exercise time, and the resulting mental health score. Data preprocessing included cleaning invalid entries and scaling features to keep training stable. Once trained, the model was saved with joblib into a lightweight file so it could be easily loaded by the Streamlit app.

This makes the project a helpful starting point for anyone who wants to learn how to train a model, save it, and use it inside a deployed web application. It is not about achieving perfect accuracy, but about demonstrating the full pipeline in a way that is clear and accessible.



Technologies used

The app is built with Streamlit for the interface. Pandas is used for handling data, scikit-learn is used for training the regression model, and joblib is used to save and load the trained model efficiently. These libraries were chosen because they are lightweight, widely used, and beginner-friendly.



Installation and usage

To run the project locally, you need Python 3.9 or newer. Create and activate a virtual environment, then install the dependencies from the requirements file using:

```
pip install -r requirements.txt
```

Make sure the trained model file is in the correct location, then start the app with:

```
streamlit run app.py
```

This will open the app in your browser.



Dataset notes

The dataset is synthetic and built for demonstration. Screen time values range from 0 to 12 hours per day, exercise time ranges from 0 to 4 hours per day, and mental health scores are scaled between 0 and 100. The goal is not to represent real-world clinical data, but to provide an approachable example of how habits can be mapped to a simple predictive model.



Limitations

Since the dataset is synthetic, the predictions are approximate and simplified. The app does not replace professional advice and should not be seen as a diagnostic tool. Mental health is complex and cannot be fully explained by just two factors. This project is meant as an educational exercise to demonstrate the connection between data, machine learning, and real-world applications.



Contributing

Contributions are welcome. If you want to add features, improve the model, or refine the interface, you can fork the repository, create a branch, and submit a pull request with your changes. Clear explanations and testing notes will make it easier to review contributions.



Disclaimer

This project is for educational purposes only. It does not provide medical advice or diagnosis. If you have concerns about mental health, please seek guidance from qualified professionals or local support services.



















