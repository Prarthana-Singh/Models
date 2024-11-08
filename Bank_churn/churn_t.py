import sklearn
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import time

# By GPT
def main():
    # Add a custom background image for styling
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://example.com/background_image.jpg");  /* Replace with a suitable image link */
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #FFFFFF;  /* Set text color for readability on dark background */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title with Animation
    html_temp = """
        <div style="
            animation: colorChange 8s infinite alternate, borderGlow 2s infinite alternate;
            padding: 20px; 
            border-radius: 10px;
            text-align: center;
            border: 3px solid rgba(255,255,255,0.5);
            box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.6);
        ">
            <h2 style="color: white; font-family: Arial, sans-serif; font-weight: bold; 
                       text-shadow: 0px 0px 10px rgba(255,255,255,0.8); 
                       animation: pulseShadow 2s infinite;">
                🏦 Bank Churn Prediction
            </h2>
        </div>
        <style>
            @keyframes colorChange {
                0% { background-color: #4B183B; }
                25% { background-color: #8A1C4A; }
                50% { background-color: #D41443; }
                75% { background-color: #FF5733; }
                100% { background-color: #FFC300; }
            }
            @keyframes borderGlow {
                0% { box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.5); }
                100% { box-shadow: 0px 0px 40px rgba(255, 255, 255, 1); }
            }
            @keyframes pulseShadow {
                0%, 100% { text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.8); }
                50% { text-shadow: 0px 0px 20px rgba(255, 255, 255, 1); }
            }
        </style>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Load the model
    model = joblib.load('churn_model')

    # Input Fields with Tooltips
    p1 = st.number_input("Enter Your Credit Score 💳", help="Higher values indicate better creditworthiness")
    p2 = st.number_input("Enter Your Age", 18, 100, help="Age of the customer")
    p3 = st.number_input("Enter Tenure", 0, 10, help="Number of years with the bank")
    p4 = st.number_input("Enter Your Balance", help="Account balance in currency")
    p5 = st.slider("Enter NumOfProducts", 1, 4, help="Number of bank products used by the customer")
    p6 = st.slider("Enter HasCrCard", 0, 1, help="Whether the customer has a credit card (1 = Yes, 0 = No)")
    p7 = st.slider("Enter IsActiveMember", 0, 1, help="Whether the customer is an active member (1 = Yes, 0 = No)")
    p8 = st.number_input("Enter Estimated Salary", help="Estimated yearly salary in currency")

    # Geography Input
    p9 = st.selectbox("Geography", ("France", "Spain", "Germany"), help="Country where the customer resides")
    Geography_France = 1 if p9 == "France" else 0
    Geography_Spain = 1 if p9 == "Spain" else 0
    Geography_Germany = 1 if p9 == "Germany" else 0

    # Gender Input
    p10 = st.selectbox("Enter Gender", ("Male", "Female"), help="Customer's gender")
    Gender_Male = 1 if p10 == "Male" else 0

    # Display a summary of inputs
    input_data = pd.DataFrame({
        'Credit Score': [p1],
        'Age': [p2],
        'Tenure': [p3],
        'Balance': [p4],
        'Num of Products': [p5],
        'Credit Card Holder': [p6],
        'Active Member': [p7],
        'Estimated Salary': [p8],
        'Geography': [p9],
        'Gender': [p10]
    })
    st.write("### Customer Data Summary")
    st.table(input_data)

    # Prediction Button with Progress Bar
    if st.button('Predict'):

        # Make Prediction
        pred = model.predict(
            [[p1, p2, p3, p4, p5, p6, p7, p8, Geography_Spain, Geography_Germany, Gender_Male]])

        # Display the result
        if pred[0] == 0:
            st.balloons()
            st.toast("Prediction completed!")
            st.success("🎉 The customer is not likely to churn! 🎉")
        else:
            st.error("⚠️ High risk of customer churn! ⚠️")


        html_temp = """
        <div style="
            animation: borderGlow 2s infinite alternate;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 3px solid rgba(255,255,255,0.5);
            box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.6);
        ">
            <h2 style="color: white; font-family: Arial, sans-serif; font-weight: bold; 
                       text-shadow: 0px 0px 10px rgba(255,255,255,0.8); 
                       animation: pulseShadow 2s infinite;">
                Thank you for using the Bank Churn Prediction tool! 💼
            </h2>
        </div>

        <style>
            /* Keyframes for border glow effect */
            @keyframes borderGlow {
                0% { box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.5); }
                100% { box-shadow: 0px 0px 40px rgba(255, 255, 255, 1); }
            }

            /* Keyframes for text shadow pulsing effect */
            @keyframes pulseShadow {
                0%, 100% { text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.8); }
                50% { text-shadow: 0px 0px 20px rgba(255, 255, 255, 1); }
            }
        </style>
    """

        st.markdown(html_temp, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
