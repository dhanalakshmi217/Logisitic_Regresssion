import streamlit as st
import pickle
import numpy as np

# Load model
with open("Titanic.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival")

# Inputs
age = st.number_input("Age", min_value=1, max_value=100, value=25)
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents / Children", min_value=0, max_value=6, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, value=50.0)

embarked = st.selectbox("Embarked Port", ["S", "Q", "C"])

# Encoding (same order as training)
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, pclass, sibsp, parch, fare, embarked_q, embarked_s]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger SURVIVED")
    else:
        st.error("❌ Passenger DID NOT SURVIVE")
