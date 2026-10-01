import ee
import pandas as pd

# --------------------------------------------------
# 1. Initialize Google Earth Engine
# --------------------------------------------------
ee.Initialize(project="persuasive-net-456701-t1")

print("Earth Engine initialized successfully")

# --------------------------------------------------
# 2. Define representative study location
# --------------------------------------------------
# Representative location for the India-level prototype
latitude = 22.0
longitude = 79.0

point = ee.Geometry.Point([longitude, latitude])

# Use a 10 km buffer instead of a single satellite pixel
region = point.buffer(10000)

# --------------------------------------------------
# 3. Load MODIS vegetation index data
# --------------------------------------------------
collection = (
    ee.ImageCollection("MODIS/061/MOD13Q1")
    .filterBounds(region)
)

print("MODIS collection loaded")

# --------------------------------------------------
# 4. Quality filtering
# --------------------------------------------------
def mask_quality(image):
    qa = image.select("SummaryQA")

    # Keep good and marginal-quality observations
    mask = qa.lte(1)

    return image.updateMask(mask)


collection = collection.map(mask_quality)

# --------------------------------------------------
# 5. Extract annual NDVI and EVI
# --------------------------------------------------
results = []

for year in range(2000, 2025):

    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    yearly_collection = (
        collection
        .filterDate(start_date, end_date)
        .select(["NDVI", "EVI"])
    )

    yearly_mean = yearly_collection.mean()

    # MODIS NDVI/EVI scale factor = 0.0001
    yearly_mean = yearly_mean.multiply(0.0001)

    stats = yearly_mean.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=250,
        maxPixels=1e9
    )

    values = stats.getInfo()

    ndvi = values.get("NDVI")
    evi = values.get("EVI")

    results.append({
        "Year": year,
        "NDVI": ndvi,
        "EVI": evi
    })

    print(
        f"{year}: "
        f"NDVI={ndvi}, "
        f"EVI={evi}"
    )

# --------------------------------------------------
# 6. Create DataFrame
# --------------------------------------------------
satellite_df = pd.DataFrame(results)

print("\nSatellite dataset:")
print(satellite_df)

print("\nMissing values:")
print(satellite_df.isnull().sum())

# --------------------------------------------------
# 7. Save satellite data
# --------------------------------------------------
output_file = "data/processed/satellite_annual.csv"

satellite_df.to_csv(output_file, index=False)

print(f"\nSaved satellite data to: {output_file}")