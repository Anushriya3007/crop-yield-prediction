import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/crop_yield_model.pkl")


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("🌾 Crop Yield Predictor")

st.write(
    "Machine learning-based crop yield prediction using "
    "weather and satellite-derived vegetation features."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About the Model")

    st.write(
        "This application uses a trained **Gradient Boosting "
        "Regression** model to estimate crop yield."
    )

    st.subheader("Data Sources")

    st.write("🌾 FAOSTAT")
    st.write("🌦️ NASA POWER")
    st.write("🛰️ MODIS / Google Earth Engine")

    st.divider()

    st.subheader("Model Evaluation")

    st.metric("MAE", "952.66 kg/ha")
    st.metric("MAPE", "9.98%")
    st.metric("R²", "0.9844")

    st.caption(
        "Evaluation performed on the held-out 2021–2024 period."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🌱 Environmental Conditions")

st.write(
    "Enter the environmental conditions to estimate crop yield."
)


crop_options = [
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


crop = st.selectbox(
    "🌾 Select Crop",
    crop_options
)


# ============================================================
# INPUT VALUES
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    temperature = st.number_input(
        "🌡️ Average Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=25.0,
        step=0.1
    )

    rainfall = st.number_input(
        "🌧️ Total Rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=900.0,
        step=10.0
    )


with col2:

    humidity = st.number_input(
        "💧 Average Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=1.0
    )

    solar = st.number_input(
        "☀️ Average Solar Radiation",
        min_value=0.0,
        max_value=40.0,
        value=18.0,
        step=0.1
    )


with col3:

    ndvi = st.number_input(
        "🛰️ NDVI",
        min_value=-1.0,
        max_value=1.0,
        value=0.45,
        step=0.01
    )

    evi = st.number_input(
        "🛰️ EVI",
        min_value=-1.0,
        max_value=1.0,
        value=0.25,
        step=0.01
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🌱 Predict Crop Yield",
    use_container_width=True
):

    input_data = pd.DataFrame([
        {
            "Crop": crop,
            "Avg_Temperature": temperature,
            "Total_Rainfall": rainfall,
            "Avg_Humidity": humidity,
            "Avg_Solar_Radiation": solar,
            "NDVI": ndvi,
            "EVI": evi
        }
    ])

    prediction = model.predict(input_data)[0]

    st.success("Prediction generated successfully!")

    st.subheader(f"🌾 Predicted Yield for {crop}")

    st.metric(
        label="Estimated Yield",
        value=f"{prediction:,.2f} kg/ha"
    )

    st.info(
        "Prediction generated using the trained "
        "Gradient Boosting regression model."
    )


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.header("🔬 Project Pipeline")

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("🌾 Crop Data")

    st.write(
        "FAOSTAT historical crop-yield data"
    )

    st.caption("2000–2024")


with col2:

    st.subheader("🌦️ Weather Data")

    st.write(
        "NASA POWER environmental variables"
    )

    st.caption(
        "Temperature • Rainfall • Humidity • Solar Radiation"
    )


with col3:

    st.subheader("🛰️ Satellite Data")

    st.write(
        "MODIS through Google Earth Engine"
    )

    st.caption("NDVI • EVI")


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander("📊 Model & Evaluation Details"):

    st.write(
        """
        **Model:** Gradient Boosting Regressor

        **Training period:** 2000–2020

        **Testing period:** 2021–2024

        **Training samples:** 252

        **Testing samples:** 48
        """
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "R²",
                "MAPE",
                "SMAPE"
            ],
            "Value": [
                "952.66 kg/ha",
                "2832.88 kg/ha",
                "0.9844",
                "9.98%",
                "10.80%"
            ]
        }
    )

    st.table(metrics_df)


# ============================================================
# NDVI AND EVI
# ============================================================

with st.expander("🛰️ What are NDVI and EVI?"):

    st.subheader("NDVI")

    st.write(
        """
        NDVI (Normalized Difference Vegetation Index) is a
        satellite-derived vegetation index that indicates
        vegetation greenness and vegetation condition.
        """
    )

    st.subheader("EVI")

    st.write(
        """
        EVI (Enhanced Vegetation Index) is another vegetation
        index designed to improve vegetation sensitivity,
        particularly under dense vegetation conditions.
        """
    )

    st.write(
        "Both features were extracted from MODIS satellite "
        "data using Google Earth Engine."
    )


# ============================================================
# LIMITATIONS
# ============================================================

with st.expander("⚠️ Project Limitations"):

    st.write(
        """
        This is an academic prototype.

        • Weather data currently represents a representative
          Indian location.

        • Satellite features are extracted from a representative
          10 km region.

        • The dataset contains 300 observations across 12 crops.

        • State or district-level environmental data could improve
          future versions.

        • Additional crop-specific and seasonal features could
          be incorporated in future work.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Crop Yield Predictor • Machine Learning + Weather + Satellite Data"
)