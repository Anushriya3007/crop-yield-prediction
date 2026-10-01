import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load merged dataset
df = pd.read_csv("data/processed/crop_weather_dataset.csv")

# Features and target
X = df[
    [
        "Crop",
        "Year",
        "Avg_Temperature",
        "Total_Rainfall",
        "Avg_Humidity",
        "Avg_Solar_Radiation"
    ]
]

y = df["Yield"]

# Categorical and numerical features
categorical_features = ["Crop"]
numerical_features = [
    "Year",
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
    "Avg_Solar_Radiation"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Time-based split
train_df = df[df["Year"] <= 2020]
test_df = df[df["Year"] > 2020]

X_train = train_df[X.columns]
y_train = train_df["Yield"]

X_test = test_df[X.columns]
y_test = test_df["Yield"]

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Baseline Random Forest Results")
print("--------------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))