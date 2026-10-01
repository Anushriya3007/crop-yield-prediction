import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
file_path = "data/processed/crop_weather_satellite_dataset.csv"
df = pd.read_csv(file_path)


# --------------------------------------------------
# 2. Training data
# --------------------------------------------------
train_df = df[df["Year"] <= 2020].copy()


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
y_train = train_df[target]


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
# 5. Final model
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
print("Training final model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Save model
# --------------------------------------------------
output_file = "models/crop_yield_model.pkl"

joblib.dump(model, output_file)

print(f"Model saved to: {output_file}")