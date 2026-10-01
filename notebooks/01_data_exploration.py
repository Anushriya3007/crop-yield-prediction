import pandas as pd

# Load FAOSTAT dataset
file_path = "data/raw/FAOSTAT_data_en_10-1-2026.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nUnique Crops:")
print(df["Item"].unique())

print("\nUnique Countries:")
print(df["Area"].unique())

print("\nUnique Elements:")
print(df["Element"].unique())

print("\nYear coverage by crop:")
print(
    df.groupby("Item")["Year"]
      .agg(["min", "max", "count"])
      .sort_values("count", ascending=False)
)