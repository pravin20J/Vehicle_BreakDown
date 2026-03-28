import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="pravin1214/vehicle_break_down", filename="maintainance_prediction_v1.joblib
.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Engine Failure Prediction")
st.write("""
This application Predicts Engine Failure Data.
""")

# User input

engine_rpm = st.number_input("The number of revolutions per minute", min_value=60, max_value=2500, value=60)
oil_pressure = st.number_input("The pressure of the lubricating oil", min_value=0.01, max_value=10.00, value=0.01)
fuel_pressure = st.number_input("The pressure at which fuel is supplied", min_value=0.01, max_value=25.00, value=0.01)
coolant_pressure = st.number_input("The pressure of the engine coolant", min_value=0.01, max_value=10.00, value=0.01)
lub_oil_temp = st.number_input("The temperature of the lubricating oil", min_value=60.00, max_value=100.00, value=60.00)   
coolant_temp = st.number_input("The temperature of the engine coolant", min_value=60.00, max_value=200.00, value=60.00)                             
                                                       

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': engine_rpm,
    'Lub oil pressure': oil_pressure,
    'Fuel pressure': fuel_pressure,
    'Coolant pressure': coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': coolant_temp,
}])


if st.button("Predict Customer"):
    prediction = model.predict(input_data)[0]
    result = "Failed" if prediction == 1 else "Not Failed"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
