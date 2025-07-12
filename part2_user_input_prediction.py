# Part 2: Loading the Model and Testing with Live User Input

# Step 1: Import Required Libraries
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("\n Part 2: Model Testing with Live User Input")

# Step 2: Load the original dataset to re-initialize the LabelEncoder
try:
    data_for_encoder = pd.read_csv(r"C:\Users\Dharini\OneDrive\Documents\Data science project  submission\wind turbine data set.csv")
    data_for_encoder['FaultType'] = data_for_encoder['FaultType'].fillna('None')
    encoder = LabelEncoder()
    encoder.fit(data_for_encoder['FaultType'])
    print("LabelEncoder initialized successfully from dataset.")
except FileNotFoundError:
    print("Error: 'wind turbine data set.csv' not found. Cannot initialize LabelEncoder.")
    print("Please ensure the dataset is in the same directory.")
    exit()

# Step 3: Load the trained Random Forest models
try:
    # Loading two Random Forest models. 
    failure_model = joblib.load("turbine_failure_random_forest_model.pkl")
    fault_type_model = joblib.load("fault_type_random_forest_model.pkl")
    print("Random Forest models loaded successfully.")
   
except FileNotFoundError:
    print("Error: Model files not found.")
    print("Please ensure 'turbine_failure_random_forest_model.pkl' and 'fault_type_random_forest_model.pkl' (or their XGBoost equivalents) are in the same directory.")
    exit()

# Step 4: Taking  user input for features
print("\nPlease enter values for each feature (numerical values only)")

# Define the features that the model expects (same as used during training)
features = [
    'WindSpeed', 'RotorSpeed', 'BladeAngle', 'GearboxTemp',
    'GeneratorTemp', 'Vibration', 'PowerOutput', 'Humidity',
    'AmbientTemperature', 'TimeSinceLastMaintenance'
]

user_input_data = {}
for feature in features:
    while True:
        try:
            value = float(input(f"Enter value for {feature}: "))
            user_input_data[feature] = value
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

# Convert the input values to a DataFrame: Converts the collected input into a format the model can use for prediction
input_df = pd.DataFrame([user_input_data])

print(f"\nUser Input Data for Prediction---\n{input_df}")

# Step 5: Make predictions using the loaded models
# The output will be 0 (no failure) or 1 (failure).
failure_prediction = failure_model.predict(input_df)[0]

print("\nTurbine Status Prediction")
if failure_prediction == 1:
    print("Turbine Status: FAILURE PREDICTED (1)")
    fault_type_prediction_encoded = fault_type_model.predict(input_df)[0]
    predicted_fault_type = encoder.inverse_transform([fault_type_prediction_encoded])[0]
    print(f"Predicted Fault Type: {predicted_fault_type}")
else:
    print("Turbine Status: NO FAILURE PREDICTED (0)")
    print("No specific fault type prediction needed as no failure is predicted.")

print("\nModel Testing Complete ")