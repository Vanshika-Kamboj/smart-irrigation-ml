import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
clf     = joblib.load('models/best_classifier.pkl')
reg     = joblib.load('models/regressor.pkl')
kmeans  = joblib.load('models/kmeans.pkl')
scaler  = joblib.load('models/scaler.pkl')
le_dict = joblib.load('models/label_encoders.pkl')
le_target = joblib.load('models/target_encoder.pkl')
pca     = joblib.load('models/pca.pkl')
features = joblib.load('models/feature_names.pkl')

st.set_page_config(page_title="🌾 Irrigation ML App", layout="wide")

st.title("🌾 Smart Irrigation Prediction System")
st.markdown("### Predict Irrigation Need, Water Usage & Field Clusters")
st.divider()

# ── Sidebar Inputs ──
st.sidebar.header("📋 Enter Field Details")

soil_type     = st.sidebar.selectbox("Soil Type",       ['Clay','Silt','Sandy','Loamy'])
crop_type     = st.sidebar.selectbox("Crop Type",       ['Wheat','Rice','Maize','Cotton','Sugarcane'])
season        = st.sidebar.selectbox("Season",          ['Kharif','Rabi','Zaid'])
region        = st.sidebar.selectbox("Region",          ['North','South','East','West','Central'])
growth_stage  = st.sidebar.selectbox("Growth Stage",    ['Seedling','Vegetative','Flowering','Maturity'])
irr_type      = st.sidebar.selectbox("Irrigation Type", ['Drip','Sprinkler','Flood','Furrow'])
water_source  = st.sidebar.selectbox("Water Source",    ['Canal','Groundwater','Rainwater','River'])
mulching      = st.sidebar.selectbox("Mulching Used",   ['Yes','No'])

soil_ph       = st.sidebar.slider("Soil pH",            4.8, 8.2, 6.5)
soil_moisture = st.sidebar.slider("Soil Moisture (%)",  8.0, 65.0, 35.0)
org_carbon    = st.sidebar.slider("Organic Carbon",     0.1, 3.0, 1.0)
elec_cond     = st.sidebar.slider("Electrical Conductivity", 0.1, 4.0, 1.5)
temperature   = st.sidebar.slider("Temperature (°C)",   10.0, 50.0, 30.0)
humidity      = st.sidebar.slider("Humidity (%)",       10.0, 100.0, 60.0)
rainfall      = st.sidebar.slider("Rainfall (mm)",      0.0, 300.0, 50.0)
sunlight      = st.sidebar.slider("Sunlight Hours",     1.0, 14.0, 7.0)
wind_speed    = st.sidebar.slider("Wind Speed (km/h)",  0.0, 30.0, 10.0)
field_area    = st.sidebar.slider("Field Area (ha)",    0.3, 15.0, 5.0)
prev_irr      = st.sidebar.slider("Previous Irrigation (mm)", 0.0, 120.0, 50.0)

# ── Predict Button ──
if st.sidebar.button("🚀 Predict Now"):

    # Build input row
    input_dict = {
        'Soil_Type'              : soil_type,
        'Soil_pH'                : soil_ph,
        'Soil_Moisture'          : soil_moisture,
        'Organic_Carbon'         : org_carbon,
        'Electrical_Conductivity': elec_cond,
        'Temperature_C'          : temperature,
        'Humidity'               : humidity,
        'Rainfall_mm'            : rainfall,
        'Sunlight_Hours'         : sunlight,
        'Wind_Speed_kmh'         : wind_speed,
        'Crop_Type'              : crop_type,
        'Crop_Growth_Stage'      : growth_stage,
        'Season'                 : season,
        'Irrigation_Type'        : irr_type,
        'Water_Source'           : water_source,
        'Field_Area_hectare'     : field_area,
        'Mulching_Used'          : mulching,
        'Previous_Irrigation_mm' : prev_irr,
        'Region'                 : region
    }

    # Encode categoricals
    cat_cols = ['Soil_Type','Crop_Type','Crop_Growth_Stage','Season',
                'Irrigation_Type','Water_Source','Mulching_Used','Region']

    for col in cat_cols:
        try:
            input_dict[col] = le_dict[col].transform([input_dict[col]])[0]
        except:
            input_dict[col] = 0

    # Create dataframe in correct order
    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)

    # Predictions
    irr_pred    = le_target.inverse_transform(clf.predict(input_scaled))[0]
    water_pred  = reg.predict(input_scaled)[0]
    cluster     = kmeans.predict(input_scaled)[0] + 1

    # ── Show Results ──
    st.subheader("🎯 Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        color = "🔴" if irr_pred == "High" else "🟡" if irr_pred == "Medium" else "🟢"
        st.metric("💧 Irrigation Need", f"{color} {irr_pred}")

    with col2:
        st.metric("💦 Water Usage (mm)", f"{water_pred:.2f} mm")

    with col3:
        st.metric("🌾 Field Cluster", f"Cluster {cluster}")

    st.divider()

    # ── Input Summary ──
    st.subheader("📋 Input Summary")
    summary = {
        'Soil Type': soil_type, 'Crop Type': crop_type,
        'Season': season, 'Region': region,
        'Temperature': f"{temperature}°C", 'Humidity': f"{humidity}%",
        'Rainfall': f"{rainfall}mm", 'Soil Moisture': f"{soil_moisture}%"
    }
    st.table(pd.DataFrame(summary.items(), columns=['Feature', 'Value']))

else:
    st.info("👈 Fill in the details on the left sidebar and click **Predict Now**")
    st.image('models/clusters.png', caption='Field Clusters Discovered', use_column_width=True)