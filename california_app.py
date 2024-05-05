import pickle
import streamlit as st
import pandas as pd

df = pd.read_csv(r"C:\Users\admin\PycharmProjects\python_Project\California_Housing_Project\california_housing_data2.csv")

with open(r"C:\Users\admin\PycharmProjects\python_Project\California_Housing_Project\xgbreg_cal_housing_pipe.pkl", "rb") as f:
    model = pickle.load(f)

st.title("House Price Prediction")

st.sidebar.header("Input Parameters")

longitude = st.sidebar.number_input("Longitude", min_value=-124.18, max_value=-114.49)
latitude = st.sidebar.number_input("Latitude", min_value=32.56, max_value=41.92)
housing_median_age = st.sidebar.number_input("Housing Median Age", min_value=1.00, max_value=52.00)
total_rooms = st.sidebar.number_input("Total Rooms", min_value=6.00, max_value=5721.00)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=2.00, max_value=1153.5)
population = st.sidebar.number_input("Population", min_value=5.00, max_value=3186.875)
households = st.sidebar.number_input("Households", min_value=2.00, max_value=1083.625)
median_income = st.sidebar.number_input("Median Income", min_value=0.4999, max_value=7.825187)

features = [[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]]



prediction = model.predict(features)
st.subheader("Prediction:")
st.write(prediction)

st.subheader("Data set")
st.write(df)




