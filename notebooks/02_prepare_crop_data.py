import pandas as pd

# Load FAOSTAT dataset
file_path = "data/raw/FAOSTAT_data_en_10-1-2026.csv"

df = pd.read_csv(file_path)

# Crops selected for the project
selected_crops = [
    "Wheat",
    "Maize (corn)",
    "Barley",
    "Millet",
    "Sorghum",
    "Soya beans",
    "Groundnuts, excluding shelled",
    "Chick peas, dry",
    "Potatoes",
    "Sugar cane",
    "Sunflower seed",
    "Lentils, dry"
]

# Filter selected crops
crop_df = df[df["Item"].isin(selected_crops)].copy()

# Keep only useful columns
crop_df = crop_df[
    ["Area", "Item", "Year", "Unit", "Value"]
]

# Rename columns
crop_df = crop_df.rename(columns={
    "Area": "Country",
    "Item": "Crop",
    "Value": "Yield"
})

# Sort data
crop_df = crop_df.sort_values(
    ["Crop", "Year"]
).reset_index(drop=True)

# Display results
print("Clean dataset shape:")
print(crop_df.shape)

print("\nSelected crops:")
print(crop_df["Crop"].unique())

print("\nYear coverage:")
print(
    crop_df.groupby("Crop")["Year"]
    .agg(["min", "max", "count"])
)

print("\nFirst 10 rows:")
print(crop_df.head(10))

# Save processed dataset
output_path = "data/processed/crop_yield_clean.csv"

crop_df.to_csv(output_path, index=False)

print(f"\nSaved cleaned dataset to: {output_path}")