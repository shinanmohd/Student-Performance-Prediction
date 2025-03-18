import streamlit as st
import pandas as pd
import joblib

model=joblib.load("LinearRegression.pkl")
scaler=joblib.load("Scaler.pkl")                    
encoder=joblib.load("Encoder.pkl")


# st.title("Students Performence Predictor")
# st.header("Enter Your Details")
# st.subheader("Details")
# st.write("Here are the details")
# a=st.text_input("enter your name")
# st.number_input("Enter Your Age")
# st.selectbox("Choose",("Yes","No"))
# st.radio("Gender",options=["Male","Female"])
# st.button("Submit")

# st.title("CALCULATOR")


# num1=st.number_input("Enter the first number")
# num2=st.number_input("Enter the second number")

# a=st.selectbox("Choose",("Add","Sub","Mult","Div"))

# if a=="Add":
#     if st.button("Submit"):
#         st.success(num1+num2)
# elif a=="Sub":
#     if st.button("Submit"):
#         st.success(num1-num2)
# elif a=="Mult":
#     if st.button("Submit"):
#         st.success(num1*num2)
# elif a=="Div":
#     if st.button("Submit"):
#         if num2==0:
#             st.error("you cant give denominator as 0")
#         else:
#             st.success(num1/num2)


hour_studied=st.number_input("Hour Studied")
prev_score=st.number_input("previous Score")
sleep_hours=st.number_input("sleep Hours")
paper=st.number_input("Sample Question Paper Practiced")
eca=st.selectbox("Extracuricular Activity",("Yes","No"))
eca=encoder.transform([eca])
st.write(eca)
dataframe=pd.DataFrame({"Hours Studied":hour_studied,"Previous Scores":prev_score,"Sleep Hours":sleep_hours,"Sample Question Papers Practiced":paper,"ECA":eca})
# st.write(dataframe)
scaled_data=scaler.transform(dataframe)
# st.write(scaled_data)

if st.button("predict"):
    prediction=model.predict(scaled_data)[0]
    st.write(prediction)
