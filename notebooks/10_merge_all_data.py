import pandas as pd

# Load datasets
crop_weather_file = "data/processed/crop_weather_dataset.csv"
satellite_file = "data/processed/satellite_annual.csv"

crop_weather_df = pd.read_csv(crop_weather_file)
satellite_df = pd.read_csv(satellite_file)

print("Crop + Weather shape:")
print(crop_weather_df.shape)

print("\nSatellite shape:")
print(satellite_df.shape)

# Merge using Year
final_df = pd.merge(
    crop_weather_df,
    satellite_df,
    on="Year",
    how="inner"
)

# Sort
final_df = final_df.sort_values(
    ["Crop", "Year"]
).reset_index(drop=True)

print("\nFinal dataset shape:")
print(final_df.shape)

print("\nColumns:")
print(final_df.columns.tolist())

print("\nMissing values:")
print(final_df.isnull().sum())

print("\nFirst 10 rows:")
print(final_df.head(10))

# Save
output_file = "data/processed/crop_weather_satellite_dataset.csv"
final_df.to_csv(output_file, index=False)

print(f"\nSaved final dataset to: {output_file}")