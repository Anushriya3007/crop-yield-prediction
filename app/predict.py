import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/crop_yield_model.pkl")


# Example input
input_data = pd.DataFrame([{
    "Crop": "Wheat",
    "Avg_Temperature": 25.0,
    "Total_Rainfall": 900.0,
    "Avg_Humidity": 65.0,
    "Avg_Solar_Radiation": 18.0,
    "NDVI": 0.45,
    "EVI": 0.25
}])


# Predict yield
prediction = model.predict(input_data)[0]


print("\nCrop Yield Prediction")
print("====================")
print(f"Crop: {input_data['Crop'].iloc[0]}")
print(f"Predicted Yield: {prediction:.2f} kg/ha")