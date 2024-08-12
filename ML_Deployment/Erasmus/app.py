import streamlit as st
import joblib
import numpy as np

st.title("Grant Estimation App")

st.divider()

exam_score = st.number_input("Enter Exam Score:", min_value=0.0, max_value=100.0)
department = st.number_input("Dapartment:", min_value=0)


X = [exam_score, department]

sc = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button("Press for predicting the Grant")

st.divider()

if predict_button:

    st.balloons()

    X1 = np.array(X)
    
    X_array = sc.transform([X1])
    
    prediction = model.predict(X_array)[0]

    st.write(f"Grant prediction is {prediction}")


else:
    st.write("Please enter the values and press predict button")

