import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
file_path = "data/processed/crop_weather_satellite_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset shape:")
print(df.shape)


# --------------------------------------------------
# 2. Time-based train/test split
# --------------------------------------------------
train_df = df[df["Year"] <= 2020].copy()
test_df = df[df["Year"] > 2020].copy()

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))


# --------------------------------------------------
# 3. Target
# --------------------------------------------------
target = "Yield"


# --------------------------------------------------
# 4. Environmental feature sets
# --------------------------------------------------
feature_sets = {

    "Crop + Weather": [
        "Crop",
        "Avg_Temperature",
        "Total_Rainfall",
        "Avg_Humidity",
        "Avg_Solar_Radiation"
    ],

    "Crop + Satellite": [
        "Crop",
        "NDVI",
        "EVI"
    ],

    "Crop + Weather + Satellite": [
        "Crop",
        "Avg_Temperature",
        "Total_Rainfall",
        "Avg_Humidity",
        "Avg_Solar_Radiation",
        "NDVI",
        "EVI"
    ]
}


# --------------------------------------------------
# 5. Train and evaluate models
# --------------------------------------------------
results = []

for name, features in feature_sets.items():

    X_train = train_df[features]
    X_test = test_df[features]

    y_train = train_df[target]
    y_test = test_df[target]

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    results.append({
        "Feature Set": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------
results_df = pd.DataFrame(results)

print("\nEnvironmental Feature Analysis")
print("==============================")
print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------
output_file = (
    "data/processed/environmental_model_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(
    f"\nSaved results to: {output_file}"
)