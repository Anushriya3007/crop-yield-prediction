import requests
import pandas as pd

# Representative central coordinate for India
latitude = 22.0
longitude = 79.0

# NASA POWER API
url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=T2M,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN"
    "&community=AG"
    "&longitude=79.0"
    "&latitude=22.0"
    "&start=20000101"
    "&end=20241231"
    "&format=JSON"
)

print("Downloading weather data from NASA POWER...")

response = requests.get(url, timeout=60)
response.raise_for_status()

data = response.json()

parameters = data["properties"]["parameter"]

# Convert NASA POWER data into DataFrame
weather_df = pd.DataFrame({
    "Date": pd.to_datetime(list(parameters["T2M"].keys())),
    "Temperature": list(parameters["T2M"].values()),
    "Rainfall": list(parameters["PRECTOTCORR"].values()),
    "Humidity": list(parameters["RH2M"].values()),
    "Solar_Radiation": list(parameters["ALLSKY_SFC_SW_DWN"].values())
})

# NASA POWER may use fill values such as -999
weather_df = weather_df.replace(-999, pd.NA)

# Extract year
weather_df["Year"] = weather_df["Date"].dt.year

print("\nDaily weather dataset:")
print(weather_df.head())

print("\nShape:")
print(weather_df.shape)

print("\nMissing values:")
print(weather_df.isnull().sum())

# Aggregate daily data into annual features
annual_weather = weather_df.groupby("Year").agg(
    Avg_Temperature=("Temperature", "mean"),
    Total_Rainfall=("Rainfall", "sum"),
    Avg_Humidity=("Humidity", "mean"),
    Avg_Solar_Radiation=("Solar_Radiation", "mean")
).reset_index()

print("\nAnnual weather features:")
print(annual_weather)

# Save
output_path = "data/processed/weather_annual.csv"
annual_weather.to_csv(output_path, index=False)

print(f"\nSaved weather data to: {output_path}")