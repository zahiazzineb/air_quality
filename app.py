#!pip install streamlit
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
# Assuming you have your trained models (model_resp, model_cardio) and scaler loaded

# Streamlit app
st.title("Respiratory and Cardiovascular Risk Assessment")

# Input fields for air quality measurements
st.header("Enter Air Quality Measurements:")
co_gt = st.number_input("CO(GT) (µg/m³)", min_value=0.0)
pt08_s1_co = st.number_input("PT08.S1(CO)", min_value=0.0)
nmhc_gt = st.number_input("NMHC(GT) (µg/m³)", min_value=0.0)
c6h6_gt = st.number_input("C6H6(GT) (µg/m³)", min_value=0.0)
pt08_s2_nmhc = st.number_input("PT08.S2(NMHC)", min_value=0.0)
nox_gt = st.number_input("NOx(GT) (µg/m³)", min_value=0.0)
pt08_s3_nox = st.number_input("PT08.S3(NOx)", min_value=0.0)
no2_gt = st.number_input("NO2(GT) (µg/m³)", min_value=0.0)
pt08_s4_no2 = st.number_input("PT08.S4(NO2)", min_value=0.0)
pt08_s5_o3 = st.number_input("PT08.S5(O3)", min_value=0.0)
model_resp = keras.models.load_model('model_air.h5')
model_cardio = keras.models.load_model('model_cardio.h5')
scaler = StandardScaler()
if st.button("Assess Risk"):
    # Create a DataFrame with the input data
    new_data = pd.DataFrame({
        'CO(GT)': [co_gt],
        'PT08.S1(CO)': [pt08_s1_co],
        'NMHC(GT)': [nmhc_gt],
        'C6H6(GT)': [c6h6_gt],
        'PT08.S2(NMHC)': [pt08_s2_nmhc],
        'NOx(GT)': [nox_gt],
        'PT08.S3(NOx)': [pt08_s3_nox],
        'NO2(GT)': [no2_gt],
        'PT08.S4(NO2)': [pt08_s4_no2],
        'PT08.S5(O3)': [pt08_s5_o3]
    })

    # Scale the new data using the trained scaler
    new_data_scaled = scaler.fit_transform(new_data)

    # Make predictions using the trained models
    respiratory_prediction = model_resp.predict(new_data_scaled)
    cardiovascular_prediction = model_cardio.predict(new_data_scaled)

    # Get the predicted class (High or Low)
    respiratory_risk = 'High' if respiratory_prediction[0][0] > respiratory_prediction[0][1] else 'Low'
    cardiovascular_risk = 'High' if cardiovascular_prediction[0][0] > cardiovascular_prediction[0][1] else 'Low'

    # Display the results
    st.header("Assessment Results:")
    st.write(f"**Respiratory Risk:** {respiratory_risk}")
    st.write(f"**Cardiovascular Risk:** {cardiovascular_risk}")
