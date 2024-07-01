import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import utlis
import numpy as np

model_path = 'models/random_forest.joblib'
try:
    loaded_model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    loaded_model = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

def predictions(loaded_model, car_name, brand, model_name, age, km, seller, fuel, transmission, Mileage, engine, power, seat):
    if loaded_model is None:
        return "Model not loaded"
    
    input_data = {
        'car_name': [car_name],
        'brand': [brand],
        'model_name': [model_name],
        'age': [age],
        'km_driven': [km],
        'seller_type': [seller],
        'fuel_type': [fuel],
        'transmission_type': [transmission],
        'mileage': [Mileage],
        'engine_cc': [engine],
        'max_power': [power],
        'seats': [seat]
    }

    # Convert input features to DataFrame to match model input format
    input_df = pd.DataFrame(input_data)

    # Initialize StandardScaler and LabelEncoder
    ss = StandardScaler()
    lb = LabelEncoder()

    # # Scale numeric columns
    ss_col =['age', 'km_driven', 'mileage', 'engine_cc', 'max_power']
    for col in ss_col:
        input_df[col] = ss.fit_transform(input_df[[col]])

    # # Encode categorical columns
    lb_col = ['car_name', 'brand', 'model_name', 'seller_type', 'fuel_type', 'transmission_type']
    for col in lb_col:
        input_df[col] = lb.fit_transform(input_df[col])

    features = input_df.values.tolist()[0]
    predicted_price = loaded_model.predict([features])

    print(predicted_price[0])
    return predicted_price[0]

    # # Use the model to predict the price
    # predicted_price = model.predict([car_name,brand,model,age,km,seller,fuel,transmission,Mileage,engine,power,seat])

    # print(predicted_price[0])
    # return predicted_price[0]

def main():
    st.title('Welcome to CarDekho Used Car Prediction')
    st.header("Please enter your details to proceed with your Car selling price")

    car_name = st.selectbox('Car Name', utlis.car_name)
    brand = st.selectbox('Brand', utlis.brand)
    model = st.selectbox("Model Name", utlis.model)
    age = st.number_input("Vehicle Age", min_value=0, max_value=50)
    km = st.number_input("Kilometers driven")
    seller = st.selectbox("Seller Type", utlis.seller)
    fuel = st.selectbox("Fuel Type", utlis.fuel)
    transmission = st.selectbox("Transmission Type", utlis.transmission)
    Mileage = st.number_input("Mileage", min_value=0, max_value=50)
    engine = st.number_input('Engine CC', step=1, min_value=50, max_value=1000)
    power = st.number_input("Max Power")
    seat = st.selectbox("Seats", utlis.seats)

    if st.button('Predict'):
        result = predictions(loaded_model, car_name, brand, model, age, km, seller, fuel, transmission, Mileage, engine, power, seat)
        st.success(f"Predicted Price: {result:.2f} INR")

if __name__ == '__main__':
    main()
