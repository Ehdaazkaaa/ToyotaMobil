import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import pytesseract
import joblib

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
    # Konversi ke grayscale dan invert warna (opsional, tergantung gambar)
    gray = ImageOps.grayscale(image)
    # Bisa juga coba ImageOps.invert jika background gelap
    # gray = ImageOps.invert(gray)
    # OCR dengan psm 8 (single line)
    text = pytesseract.image_to_string(gray, config='--psm 8')
    plate = ''.join(filter(str.isalnum, text)).upper()
    return plate

def predict_price(knn, le, df, model, year, mileage, tax, mpg, engineSize):
    try:
        model_encoded = le.transform([model])[0]
    except Exception as e:
        st.error(f"Model '{model}' tidak dikenali oleh encoder.")
        return None
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
        st.image(Image.open(car_img), caption="Gambar Mobil", use_container_width=True)

    plate_img = st.file_uploader("Upload Gambar Plat Nomor", type=['jpg', 'jpeg', 'png'])
    plate_number = ""
    if plate_img:
        img = Image.open(plate_img)
        st.image(img, caption="Gambar Plat Nomor", use_container_width=True)
        plate_number = ocr_license_plate(img)
        if plate_number:
            st.info(f"Nomor Plat Terdeteksi: {plate_number}")
        else:
            st.warning("Nomor plat tidak berhasil dideteksi, coba gambar lain atau lebih jelas.")

    st.subheader("Detail Mobil")
    model_input = st.selectbox("Model", sorted(df['model'].unique()))
    year_input = st.number_input("Tahun", min_value=1990, max_value=2025, value=2015, step=1)
    mileage_input = st.number_input("Mileage (km)", 0, 300000, value=50000, step=1000)
    tax_input = st.number_input("Pajak per Tahun (€)", 0, 1000, value=150, step=10)
    mpg_input = st.number_input("MPG", 0, 100, value=30, step=1)
    engine_size_input = st.number_input("Engine Size (L)", 0.5, 6.0, value=1.6, step=0.1, format="%.1f")

    if st.button("Prediksi Harga"):
        harga = predict_price(knn, le, df, model_input, year_input, mileage_input, tax_input, mpg_input, engine_size_input)
        if harga is not None:
            st.success(f"💰 Prediksi Harga: €{harga:,.2f}")
            if plate_number:
                st.write(f"📛 Nomor Plat: **{plate_number}**")

            with open("hasil_prediksi.txt", "w") as f:
                f.write(
                    f"Model: {model_input}\nTahun: {year_input}\nMileage: {mileage_input} km\n"
                    f"Tax: {tax_input} €\nMPG: {mpg_input}\nEngine Size: {engine_size_input} L\n"
                    f"Nomor Plat: {plate_number}\nPrediksi Harga: €{harga:,.2f}"
                )
        else:
            st.error("Prediksi gagal dilakukan.")

if __name__ == "__main__":
    main()
