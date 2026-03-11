import streamlit as st

st.title("Insurance Prediction💰")
st.write("This is a simple insurance prediction app that predicts the insurance premium based on the input features.")
Age=st.number_input("Enter age:")
Annual_Income_LPA=st.number_input("Enter annual income in LPA:",min_value=1,step=1)
Policy_Term_Years=st.number_input("Enter policy term in years:",min_value=1,step=1)
Sum_Assured_Lakhs=st.number_input("Enter sum assured in lakhs:",min_value=1,step=1)
if st.button("Predict"):
    from src.prediction import Insurance_Prediction
    insurance_prediction=Insurance_Prediction()
    result=insurance_prediction.prediction(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    st.success(f"The predicted insurance premium is: {result} thousands")
