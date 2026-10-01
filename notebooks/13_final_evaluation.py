import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
file_path = "data/processed/crop_weather_satellite_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset shape:")
print(df.shape)


# --------------------------------------------------
# 2. Time-based split
# --------------------------------------------------
train_df = df[df["Year"] <= 2020].copy()
test_df = df[df["Year"] > 2020].copy()

print("\nTraining years:", train_df["Year"].min(),
      "to", train_df["Year"].max())

print("Testing years:", test_df["Year"].min(),
      "to", test_df["Year"].max())

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))


# --------------------------------------------------
# 3. Features and target
# --------------------------------------------------
features = [
    "Crop",
    "Avg_Temperature",
    "Total_Rainfall",
    "Avg_Humidity",
    "Avg_Solar_Radiation",
    "NDVI",
    "EVI"
]

target = "Yield"

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df[target]
y_test = test_df[target]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------
categorical_features = ["Crop"]

numerical_features = [
    feature
    for feature in features
    if feature != "Crop"
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
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 5. Gradient Boosting model
# --------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            GradientBoostingRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# 6. Train
# --------------------------------------------------
print("\nTraining final model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Predictions
# --------------------------------------------------
predictions = model.predict(X_test)


# --------------------------------------------------
# 8. Overall metrics
# --------------------------------------------------
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# MAPE
mape = (
    abs(
        (y_test - predictions)
        / y_test
    ).mean()
    * 100
)


# SMAPE
smape = (
    100
    * (
        2
        * abs(y_test - predictions)
        / (
            abs(y_test)
            + abs(predictions)
        )
    ).mean()
)


print("\nFinal Model Performance")
print("=======================")

print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.4f}")
print(f"MAPE  : {mape:.2f}%")
print(f"SMAPE : {smape:.2f}%")


# --------------------------------------------------
# 9. Prediction dataframe
# --------------------------------------------------
results_df = test_df[
    ["Country", "Crop", "Year", "Yield"]
].copy()

results_df["Predicted_Yield"] = predictions

results_df["Absolute_Error"] = (
    abs(
        results_df["Yield"]
        - results_df["Predicted_Yield"]
    )
)

results_df["Percentage_Error"] = (
    results_df["Absolute_Error"]
    / results_df["Yield"]
    * 100
)


# --------------------------------------------------
# 10. Crop-wise evaluation
# --------------------------------------------------
crop_results = (
    results_df
    .groupby("Crop")
    .agg(
        Actual_Yield=("Yield", "mean"),
        Predicted_Yield=("Predicted_Yield", "mean"),
        MAE=("Absolute_Error", "mean"),
        MAPE=("Percentage_Error", "mean")
    )
    .reset_index()
)

print("\nCrop-wise Performance")
print("=====================")

print(
    crop_results.to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. Save predictions
# --------------------------------------------------
predictions_file = (
    "data/processed/final_test_predictions.csv"
)

results_df.to_csv(
    predictions_file,
    index=False
)


# --------------------------------------------------
# 12. Save crop-wise metrics
# --------------------------------------------------
crop_metrics_file = (
    "data/processed/crop_wise_metrics.csv"
)

crop_results.to_csv(
    crop_metrics_file,
    index=False
)


print(
    f"\nSaved predictions to: "
    f"{predictions_file}"
)

print(
    f"Saved crop-wise metrics to: "
    f"{crop_metrics_file}"
)