import streamlit as st
from sensors.dht_sensor import read_dht
from sensors.soil_sensor import read_soil
from ai.disease_detection import detect_disease
from ai.pest_detection import detect_pest
from ai.yield_prediction import predict_yield

st.set_page_config(page_title="Smart Farming AI", layout="centered")

st.title("🌱 Edge AI Smart Farming System")

temp, humidity = read_dht()
soil = read_soil()

st.subheader("📊 Sensor Data")
st.write(f"Temperature: {temp} °C")
st.write(f"Humidity: {humidity} %")
st.write(f"Soil Moisture: {soil} %")

image = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

if image:
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    disease = detect_disease(image)
    pest = detect_pest(image)
    yield_pred = predict_yield(temp, humidity, soil)

    st.subheader("🧠 AI Results")
    st.write("Disease Status:", disease)
    st.write("Pest Status:", pest)
    st.write("Estimated Yield:", yield_pred)
