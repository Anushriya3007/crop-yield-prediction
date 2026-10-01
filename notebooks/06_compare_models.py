import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/processed/crop_weather_dataset.csv")

# Features
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

# Target
y = df["Yield"]


# Time-based split
train_df = df[df["Year"] <= 2020]
test_df = df[df["Year"] > 2020]

X_train = train_df[X.columns]
y_train = train_df["Yield"]

X_test = test_df[X.columns]
y_test = test_df["Yield"]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Crop"]
        )
    ],
    remainder="passthrough"
)


# Models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )
}


results = []


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# Results table
results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)