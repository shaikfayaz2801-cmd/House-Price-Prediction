import streamlit as st
import joblib
import numpy as np


model = joblib.load("house_price_model.pkl")


st.set_page_config(page_title="House Price Prediction")


st.title("🏠 House Price Prediction App")

st.write("Enter house details below to predict the house price.")


area = st.number_input(
    "Area (sq ft)",
    min_value=0,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=0,
    step=1
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=0,
    step=1
)

stories = st.number_input(
    "Number of Stories",
    min_value=0,
    step=1
)

mainroad = st.selectbox(
    "Main Road Access",
    ["No", "Yes"]
)

guestroom = st.selectbox(
    "Guest Room",
    ["No", "Yes"]
)

basement = st.selectbox(
    "Basement",
    ["No", "Yes"]
)

hotwaterheating = st.selectbox(
    "Hot Water Heating",
    ["No", "Yes"]
)

airconditioning = st.selectbox(
    "Air Conditioning",
    ["No", "Yes"]
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    step=1
)

prefarea = st.selectbox(
    "Preferred Area",
    ["No", "Yes"]
)

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)


mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

furnishing_map = {
    "Unfurnished": 0,
    "Semi-Furnished": 1,
    "Furnished": 2
}

furnishingstatus = furnishing_map[furnishingstatus]


if st.button("Predict House Price"):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"🏡 Predicted House Price: ₹ {prediction[0]:,.2f}"
    )