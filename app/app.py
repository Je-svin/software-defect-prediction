import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🐞",
    layout="centered"
)

# Title
st.title("🐞 Software Defect Prediction System")

st.markdown(
    "### Machine Learning Based Software Quality Prediction"
)

st.write(
    "Enter the software metrics below to predict whether "
    "a software module is defective or non-defective."
)

st.divider()

# Input section
st.subheader("📊 Software Metrics")

LOC = st.number_input("LOC", value=0.5)
CYCLO = st.number_input("CYCLO", value=0.5)
LENGTH = st.number_input("LENGTH", value=0.5)
VOLUME = st.number_input("VOLUME", value=0.5)
DIFFICULTY = st.number_input("DIFFICULTY", value=0.5)
INT_FAN_IN = st.number_input("INT_FAN_IN", value=0.5)
INT_FAN_OUT = st.number_input("INT_FAN_OUT", value=0.5)
NUM_OPERATORS = st.number_input("NUM_OPERATORS", value=0.5)
NUM_OPERANDS = st.number_input("NUM_OPERANDS", value=0.5)
BRANCH_COUNT = st.number_input("BRANCH_COUNT", value=0.5)

st.divider()

# Load trained model
model = joblib.load("models/defect_prediction_model.pkl")

# Prediction button
if st.button("🔍 Predict Defect", use_container_width=True):

    input_data = pd.DataFrame([[
        LOC,
        CYCLO,
        LENGTH,
        VOLUME,
        DIFFICULTY,
        INT_FAN_IN,
        INT_FAN_OUT,
        NUM_OPERATORS,
        NUM_OPERANDS,
        BRANCH_COUNT
    ]], columns=[
        "LOC",
        "CYCLO",
        "LENGTH",
        "VOLUME",
        "DIFFICULTY",
        "INT_FAN_IN",
        "INT_FAN_OUT",
        "NUM_OPERATORS",
        "NUM_OPERANDS",
        "BRANCH_COUNT"
    ])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]
    defect_probability = probabilities[1] * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Defective Module")
    else:
        st.success("✅ Non-Defective Module")

    st.metric(
        "Defect Probability",
        f"{defect_probability:.2f}%"
    )

st.divider()

st.caption(
    "Software Defect Prediction System | Machine Learning Project"
)