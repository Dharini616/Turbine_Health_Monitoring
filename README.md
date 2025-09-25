Turbine Health Monitoring and Failure Forecasting

Turbine Health Monitoring is a machine learning project that predicts wind turbine failures using real-time sensor data.
This project is designed to monitor the health of wind turbines and predict failures before they occur.
Using Random Forest Machine Learning models, it performs:
-Binary Classification → Predict whether a turbine will fail (Yes/No).
- Multiclass Classification → Identify the type of failure (gearbox, rotor, generator, etc.).

The workflow covers data preprocessing, EDA, model training, hyperparameter optimization, evaluation, model saving, and live user input testing.

** Features **
1. Dataset Handling
- Cleans dataset (drops TurbineID, handles missing values).
- Encodes categorical fault types.

2. Exploratory Data Analysis (EDA)
- Histogram of wind speed.
- Pie chart of fault type distribution.
- Bar chart of failure vs non-failure.
- Scatter plot of wind speed vs rotor speed.

3. Machine Learning Models
- Random Forest for binary failure prediction.
- Random Forest for multiclass fault type prediction.
- Hyperparameter tuning with GridSearchCV.

4. Evaluation Metrics
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Accuracy Score
- Feature Importance visualization

5. Model Saving and Loading
- Models saved as .pkl using joblib.
- Load models for real-time predictions.

6. User Interaction
- Accepts live numerical input for turbine parameters.
- Predicts Failure (0/1) and, if failed, the specific fault type

Work Flow of Project:-

<img width="500" height="1000" alt="image" src="https://github.com/user-attachments/assets/6e984ce5-bdd5-439e-b845-19e9d286df9f" />
