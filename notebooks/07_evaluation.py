import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/processed/crop_weather_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Time-based train/test split
# --------------------------------------------------

train_df = df[df["Year"] <= 2020].copy()
test_df = df[df["Year"] > 2020].copy()

print("\nTraining years:", train_df["Year"].min(), "-", train_df["Year"].max())
print("Testing years:", test_df["Year"].min(), "-", test_df["Year"].max())
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))


# --------------------------------------------------
# 3. Features and target
# --------------------------------------------------

features = [
    "Crop",
    "Year",
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
    "Avg_Solar_Radiation"
]

target = "Yield"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

categorical_features = ["Crop"]
numeric_features = [
    "Year",
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
    "Avg_Solar_Radiation"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "crop",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# --------------------------------------------------
# 5. Models
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------

results = []
predictions = {}

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    predictions[model_name] = y_pred


# --------------------------------------------------
# 7. Model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("================")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 8. Best model based on RMSE
# --------------------------------------------------

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

print("\nBest model based on RMSE:")
print(best_model_name)


# --------------------------------------------------
# 9. Actual vs predicted
# --------------------------------------------------

best_predictions = predictions[best_model_name]

comparison_df = test_df[
    ["Crop", "Year", "Yield"]
].copy()

comparison_df["Predicted_Yield"] = best_predictions

comparison_df["Absolute_Error"] = abs(
    comparison_df["Yield"] -
    comparison_df["Predicted_Yield"]
)

print("\nActual vs Predicted")
print("===================")

print(
    comparison_df
    .sort_values(["Crop", "Year"])
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# 10. Average error by crop
# --------------------------------------------------

crop_error = (
    comparison_df
    .groupby("Crop")["Absolute_Error"]
    .mean()
    .sort_values()
)

print("\nAverage Absolute Error by Crop")
print("==============================")
print(crop_error)


# --------------------------------------------------
# 11. Save evaluation results
# --------------------------------------------------

results_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

comparison_df.to_csv(
    "data/processed/test_predictions.csv",
    index=False
)

print("\nEvaluation files saved successfully.")