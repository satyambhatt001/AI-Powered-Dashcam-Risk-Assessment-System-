import numpy as np
import pandas as pd
import joblib
import os
from imblearn.combine import SMOTETomek  # type: ignore # Better oversampling+undersampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier # type: ignore

# Load dataset
data = pd.read_csv("data/sensor_data.csv")

# Feature Engineering: Add acceleration and relative speed
data["acceleration"] = data["velocity"].diff().fillna(0)
data["relative_speed"] = data["velocity"] - data["vehicle_speed"]

# Define features and target
feature_columns = ["object_distance", "velocity", "trajectory_angle", "vehicle_speed", "acceleration", "relative_speed"]
X = data[feature_columns]
y = data["collision_risk"]

# Handle missing high-risk samples
if 2 not in y.values:
    print("⚠️ No high-risk samples found. Generating synthetic data...")
    high_risk_samples = data[data["collision_risk"] == 1].sample(500, replace=True)
    high_risk_samples["collision_risk"] = 2
    data = pd.concat([data, high_risk_samples], ignore_index=True)
    y = data["collision_risk"]
    X = data[feature_columns]

# Handle class imbalance using SMOTETomek (better than just SMOTE)
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning (optimized manually)
model = XGBClassifier(
    max_depth=7,
    learning_rate=0.08,
    n_estimators=250,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,  # Reduce overfitting
    min_child_weight=3,  # Prevent learning too many patterns in noise
    reg_lambda=1.5,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained with accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\nFeature Importances:\n", feature_importance)

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/risk_assessment_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully!")
