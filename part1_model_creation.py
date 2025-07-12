# Part 1: Full EDA to ML Training, Evaluation, Optimization, and Model Saving

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier # Using RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib # For saving models

print("   Part 1: Training and Model Saving Process")

# Step 2: Loading of the Dataset
try:
    data = pd.read_csv(r"C:\Users\Dharini\OneDrive\Documents\Data science project  submission\wind turbine data set.csv")
    print("Dataset loaded successfully!")
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nDataset Info:")
    data.info()
except FileNotFoundError:
    print("Error: 'wind turbine data set.csv' not found. Please ensure it's in the same directory.")
    exit()

# Step 3: Data Cleaning and Preparation of data
print("\n Step 3: Data Cleaning and Preparation")
if 'TurbineID' in data.columns:
    data = data.drop(columns=['TurbineID'])
    print("Dropped 'TurbineID' column.")
else:
    print("'TurbineID' column not found, skipping drop.")

if 'FaultType' in data.columns:
    initial_nan_count = data['FaultType'].isnull().sum()
    data['FaultType'] = data['FaultType'].fillna('None')
    if initial_nan_count > 0:
        print(f"Filled {initial_nan_count} missing values in 'FaultType' with 'None'.")
    else:
        print("No missing values found in 'FaultType'.")
else:
    print("'FaultType' column not found, skipping fillna.")

encoder = LabelEncoder()
if 'FaultType' in data.columns:
    data['FaultTypeEncoded'] = encoder.fit_transform(data['FaultType'])
    print("'FaultType' column encoded to 'FaultTypeEncoded'.")
    print(f"Original Fault Types: {encoder.classes_}")
else:
    print("Cannot encode 'FaultType' as the column is missing.")

print("\nData preparation complete. First 5 rows after preparation:")
print(data.head())


# Step 4: Simple Exploratory Data Analysis (EDA)
print("\n Step 4: Simple Exploratory Data Analysis (EDA) ")
# Histogram --- wind speed distribution
plt.figure(figsize=(8, 5))
data['WindSpeed'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Wind Speed')
plt.xlabel('Wind Speed')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('wind_speed_Histogram.png')
print("Generated 'wind_speed_Histogram.png'")
plt.close()

# pie chart ---- percentage of turbine falls into each fault category
if 'FaultType' in data.columns:
    fault_counts = data['FaultType'].value_counts()
    plt.figure(figsize=(8, 8))
    fault_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Fault Type Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('fault_type_Pie_chart.png')
    print("Generated 'fault_type_Pie_chart.png'")
    plt.close()
# Bar chart ------- for failure vs non-failure
if 'Failure' in data.columns:
    failure_counts = data['Failure'].value_counts()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=failure_counts.index, y=failure_counts.values, palette=['green', 'red'])
    plt.title('Failure vs No Failure')
    plt.xticks(ticks=[0, 1], labels=['No Failure (0)', 'Failure (1)'], rotation=0)
    plt.xlabel('Failure Status')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('failure_counts_Bar_chart.png')
    print("Generated 'failure_counts_Bar_chart.png'")
    plt.close()
# Scatter plot wind speed vs rotor speed
'''
if 'WindSpeed' in data.columns and 'RotorSpeed' in data.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='WindSpeed', y='RotorSpeed', data=data, alpha=0.05, color='orange', edgecolor=None)
    plt.title('Wind Speed vs Rotor Speed')
    plt.xlabel('Wind Speed')
    plt.ylabel('Rotor Speed')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('wind_speed_Vs_rotor_speed_scatter.png')
    print("Generated 'wind_speed_vs_rotor_speed_scatter.png'")
    plt.close()
'''
print("\nData Analysis (EDA) complete.")


# Step 5: Predicting Failure (Binary Classification) with Random Forest
print("\nStep 5: Predicting Failure (Binary Classification) with Random Forest")
X = data.drop(columns=['Failure', 'FaultType', 'FaultTypeEncoded'])
y = data['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_model_failure = RandomForestClassifier(random_state=42)

param_grid_failure = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("\nPerforming GridSearchCV for Failure Prediction (finding best parameters)")
grid_search_failure = GridSearchCV(estimator=rf_model_failure, param_grid=param_grid_failure, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_failure.fit(X_train, y_train)

print(f"\nBest parameters for Failure Prediction: {grid_search_failure.best_params_}")
print(f"Best cross-validation accuracy for Failure Prediction: {grid_search_failure.best_score_:.4f}")
best_rf_model_failure = grid_search_failure.best_estimator_
#Retrieves the best model found (best_estimator_)

y_pred_failure = best_rf_model_failure.predict(X_test)
#Uses .predict() to make predictions on the test set

print("\n Failure Prediction - Final Evaluation on Test Set")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_failure))
print("\nClassification Report:\n", classification_report(y_test, y_pred_failure))
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred_failure):.4f}")


# Step 6: Predicting Type of Failure (Multiclass Classification) with Random Forest
print("\nStep 6: Predicting Type of Failure (Multiclass Classification) with Random Forest")
failure_cases = data[data['Failure'] == 1].copy()
X_fault = failure_cases.drop(columns=['Failure', 'FaultType', 'FaultTypeEncoded'])
y_fault = failure_cases['FaultTypeEncoded']
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fault, y_fault, test_size=0.2, random_state=42, stratify=y_fault)
rf_model_fault = RandomForestClassifier(random_state=42)

param_grid_fault = {
    'n_estimators': [50, 100, 150],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("\nPerforming GridSearchCV for Fault Type Prediction (finding best parameters)")
grid_search_fault = GridSearchCV(estimator=rf_model_fault, param_grid=param_grid_fault, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search_fault.fit(Xf_train, yf_train)

print(f"\nBest parameters for Fault Type Prediction: {grid_search_fault.best_params_}")
print(f"Best cross-validation accuracy for Fault Type Prediction: {grid_search_fault.best_score_:.4f}")
# Displays best parameters and performance score
best_rf_model_fault = grid_search_fault.best_estimator_
# Predicts the fault type on the test data.
yf_pred_fault = best_rf_model_fault.predict(Xf_test)
print("\nFault Type Prediction - Final Evaluation on Test Set")
print("Confusion Matrix:\n", confusion_matrix(yf_test, yf_pred_fault))
# Collects all the labels (actual + predicted) to ensure completeness in the report
unique_labels = np.unique(np.concatenate([yf_test, yf_pred_fault]))
print("\nClassification Report:\n", classification_report(yf_test, yf_pred_fault, labels=unique_labels, target_names=encoder.inverse_transform(unique_labels))) #Shows a full report on how well the model predicted each fault type.
print(f"Test Set Accuracy: {accuracy_score(yf_test, yf_pred_fault):.4f}")# Displays overall accuracy for fault type prediction


# Step 7: Bar Chart - Feature Importance : showing which features influenced the prediction most.
print("\nStep 7: Bar Chart - Feature Importance for Failure Prediction (Random Forest)")
importances_failure = best_rf_model_failure.feature_importances_
features_failure = X.columns
importance_df_failure = pd.DataFrame({'Feature': features_failure, 'Importance': importances_failure})
importance_df_failure = importance_df_failure.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=importance_df_failure, palette='viridis')
plt.xlabel('Importance')
plt.title('Feature Importance for Predicting Failures (Random Forest)')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('feature_importance_failure.png')
print("Generated 'feature_importance_failure.png'")
plt.close()

# Step 8: Save Trained Random Forest Models
joblib.dump(best_rf_model_failure, "turbine_failure_random_forest_model.pkl")
joblib.dump(best_rf_model_fault, "fault_type_random_forest_model.pkl")
print("\nNew Random Forest Models saved successfully!")

print("\n Part 1: All tasks completed successfully")


























'''
# Part 2: Loading the Model and Testing with Live User Input

# Step 1: Import Required Libraries
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("\n Part 2: Model Testing with Live User Input")

# Step 2: Load the original dataset to re-initialize the LabelEncoder
try:
    data_for_encoder = pd.read_csv("wind turbine data set.csv")
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
    # Loading Random Forest models. If you trained XGBoost, change these filenames.
    failure_model = joblib.load("turbine_failure_random_forest_model.pkl")
    fault_type_model = joblib.load("fault_type_random_forest_model.pkl")
    print("Random Forest models loaded successfully.")
    # For XGBoost Models:
    # failure_model = joblib.load("turbine_failure_xgboost_model.pkl")
    # fault_type_model = joblib.load("fault_type_xgboost_model.pkl")
    # print("XGBoost models loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found.")
    print("Please ensure 'turbine_failure_random_forest_model.pkl' and 'fault_type_random_forest_model.pkl' (or their XGBoost equivalents) are in the same directory.")
    exit()

# Step 4: Take live user input for features
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

# Convert the input values to a DataFrame
input_df = pd.DataFrame([user_input_data])

print(f"\nUser Input Data for Prediction---\n{input_df}")

# Step 5: Make predictions using the loaded models
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
'''