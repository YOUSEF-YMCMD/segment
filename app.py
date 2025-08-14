import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("Kmeans_model.pkl")

# App title
st.title("Prediction App")
st.write("Enter the input features to get a prediction:")

# Input fields
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)
feature5 = st.number_input("Feature 5", value=0.0)
feature6 = st.number_input("Feature 6", value=0.0)

# Predict button
if st.button("Predict"):
    features = np.array([[feature1, feature2, feature3 , feature4 , feature5 , feature6]])
    prediction = model.predict(features)
    st.success(f"Prediction: {prediction[0]}")
