import pandas as pd

# Load processed datasets
crop_file = "data/processed/crop_yield_clean.csv"
weather_file = "data/processed/weather_annual.csv"

crop_df = pd.read_csv(crop_file)
weather_df = pd.read_csv(weather_file)

print("Crop dataset shape:")
print(crop_df.shape)

print("\nWeather dataset shape:")
print(weather_df.shape)

# Merge using Year
merged_df = pd.merge(
    crop_df,
    weather_df,
    on="Year",
    how="inner"
)

# Sort the dataset
merged_df = merged_df.sort_values(
    ["Crop", "Year"]
).reset_index(drop=True)

print("\nMerged dataset shape:")
print(merged_df.shape)

print("\nColumns:")
print(merged_df.columns.tolist())

print("\nFirst 10 rows:")
print(merged_df.head(10))

print("\nMissing values:")
print(merged_df.isnull().sum())

# Save merged dataset
output_file = "data/processed/crop_weather_dataset.csv"

merged_df.to_csv(output_file, index=False)

print(f"\nSaved merged dataset to: {output_file}")