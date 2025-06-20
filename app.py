
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import joblib
from google.colab import files

def add_custom_css():
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: #001f3f;
            color: #ffffff;
        }}
        .title {{
            color: #ffdc00;
            font-size: 40px;
            font-weight: bold;
        }}
        .subtitle {{
            color: #ffdc00;
            font-size: 20px;
            margin-top: -10px;
        }}
        label {{
            color: #ffdc00 !important;
            font-weight: bold;
        }}
        div.stButton > button {{
            background-color: #ffdc00;
            color: #001f3f;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        }}
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    knn = joblib.load('knn_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return knn, le

def ocr_license_plate(image):
    text = pytesseract.image_to_string(image)
    return ''.join(filter(str.isalnum, text)).upper()

def predict_price(knn, le, df, model, year, mileage, tax, mpg, engineSize):
    model_encoded = le.transform([model])[0]
    X_input = np.array([[model_encoded, year, mileage, tax, mpg, engineSize]])
    prediction = knn.predict(X_input)
    return prediction[0]

def main():
    add_custom_css()
    st.markdown('<h1 class="title">Prediksi Harga Toyota Bekas</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload gambar, isi spesifikasi, lalu prediksi harga!</p>', unsafe_allow_html=True)

    df = pd.read_csv("Toyota (1).csv")
    knn, le = load_model()

    car_img = st.file_uploader("Upload Gambar Mobil", type=['jpg', 'jpeg', 'png'])
    if car_img:
        st.image(Image.open(car_img), caption="Gambar Mobil", use_column_width=True)

    plate_img = st.file_uploader("Upload Gambar Plat Nomor", type=['jpg', 'jpeg', 'png'])
    plate_number = ""
    if plate_img:
        img = Image.open(plate_img)
        st.image(img, caption="Gambar Plat Nomor", use_column_width=True)
        plate_number = ocr_license_plate(img)
        st.info(f"Nomor Plat Terdeteksi: {plate_number}")

    st.subheader("Detail Mobil")
    model_input = st.selectbox("Model", sorted(df['model'].unique()))
    year_input = st.number_input("Tahun", min_value=1990, max_value=2025, value=2015)
    mileage_input = st.number_input("Mileage (km)", 0, 300000, value=50000)
    tax_input = st.number_input("Pajak per Tahun (€)", 0, 1000, value=150)
    mpg_input = st.number_input("MPG", 0, 100, value=30)
    engine_size_input = st.number_input("Engine Size (L)", 0.5, 6.0, value=1.6, format="%.1f")

    if st.button("Prediksi Harga"):
        harga = predict_price(knn, le, df, model_input, year_input, mileage_input, tax_input, mpg_input, engine_size_input)
        st.success(f"💰 Prediksi Harga: €{harga:,.2f}")
        if plate_number:
            st.write(f"📛 Nomor Plat: **{plate_number}**")

        with open("hasil_prediksi.txt", "w") as f:
            f.write(
                f"Model: {model_input}\nTahun: {year_input}\nMileage: {mileage_input} km\n"
                f"Tax: {tax_input} €\nMPG: {mpg_input}\nEngine Size: {engine_size_input} L\n"
                f"Nomor Plat: {plate_number}\nPrediksi Harga: €{harga:,.2f}"
            )
        files.download("hasil_prediksi.txt")

if __name__ == "__main__":
    main()
