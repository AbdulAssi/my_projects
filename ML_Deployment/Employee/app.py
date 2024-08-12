import streamlit as st
import joblib
import numpy as np

st.title("Salary Estimation App")

st.divider()

years_at_company = st.number_input("Enter Years at company:", min_value=0, max_value=30)
satisfaction_level = st.number_input("Satisfaction Level:", min_value=0.0)
avarage_monthly_hours = st.number_input("Avarage Monthly Hours:", min_value=0, max_value= 300)


X = [years_at_company, satisfaction_level, avarage_monthly_hours]

sc = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button("Press for predicting the Salary")

st.divider()

if predict_button:

    st.balloons()

    X1 = np.array(X)
    
    X_array = sc.transform([X1])
    
    prediction = model.predict(X_array)[0]

    st.write(f"Salary prediction is {prediction}")


else:
    st.write("Please enter the values and press predict button")


