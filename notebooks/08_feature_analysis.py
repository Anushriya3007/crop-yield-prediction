import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/processed/crop_weather_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Time-based split
# --------------------------------------------------

train_df = df[df["Year"] <= 2020].copy()
test_df = df[df["Year"] > 2020].copy()


# --------------------------------------------------
# 3. Define feature sets
# --------------------------------------------------

feature_sets = {

    "Crop + Year": [
        "Crop",
        "Year"
    ],

    "Crop + Weather": [
        "Crop",
        "Avg_Temperature",
        "Total_Rainfall",
        "Avg_Humidity",
        "Avg_Solar_Radiation"
    ],

    "Crop + Year + Weather": [
        "Crop",
        "Year",
        "Avg_Temperature",
        "Total_Rainfall",
        "Avg_Humidity",
        "Avg_Solar_Radiation"
    ]
}


# --------------------------------------------------
# 4. Target
# --------------------------------------------------

target = "Yield"

y_train = train_df[target]
y_test = test_df[target]


# --------------------------------------------------
# 5. Run experiments
# --------------------------------------------------

results = []

for experiment_name, features in feature_sets.items():

    categorical_features = ["Crop"]

    numeric_features = [
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
                numeric_features
            )
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train = train_df[features]
    X_test = test_df[features]

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Feature Set": experiment_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nFeature Ablation Results")
print("========================")

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

output_file = "data/processed/feature_analysis.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\nSaved results to:")
print(output_file)