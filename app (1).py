import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import torch
import easyocr
import cv2

# --- Styling UI ---
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
            margin-bottom: 30px;
        }}
        label, .st-bx {{
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
def load_knn_model():
    knn = joblib.load('knn_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return knn, le

@st.cache_resource
def load_yolo_model():
    # Load YOLOv5 pretrained custom model untuk deteksi plat nomor
    # Ganti 'best_plate_model.pt' dengan path modelmu yang sudah dilatih
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_plate_model.pt', force_reload=False)
    return model

@st.cache_resource
def load_easyocr_reader():
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True kalau pakai GPU
    return reader

# Fungsi deteksi plat nomor menggunakan YOLOv5
def detect_plate_yolo(model, img_cv):
    results = model(img_cv)
    df = results.pandas().xyxy[0]
    if len(df) == 0:
        return None
    # Ambil bbox plat nomor dengan confidence tertinggi
    best = df.sort_values('confidence', ascending=False).iloc[0]
    bbox = best[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
    return bbox

# Crop area plat nomor dari gambar OpenCV
def crop_plate(img_cv, bbox):
    xmin, ymin, xmax, ymax = bbox
    return img_cv[ymin:ymax, xmin:xmax]

# OCR pakai EasyOCR
def ocr_plate_easyocr(reader, plate_img_pil):
    result = reader.readtext(np.array(plate_img_pil))
    plate_text = "".join([res[1] for res in result])
    return plate_text

def predict_price(knn, le, model, year, mileage, tax, mpg, engineSize):
    model_encoded = le.transform([model])[0]
    X_input = np.array([[model_encoded, year, mileage, tax, mpg, engineSize]])
    prediction = knn.predict(X_input)
    return prediction[0]

def main():
    add_custom_css()
    st.markdown('<h1 class="title">Prediksi Harga Toyota Bekas</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload gambar mobil, sistem akan mendeteksi nomor plat dan prediksi harga!</p>', unsafe_allow_html=True)

    df = pd.read_csv("Toyota (1).csv")
    knn, le = load_knn_model()
    yolo_model = load_yolo_model()
    ocr_reader = load_easyocr_reader()

    car_img_file = st.file_uploader("Upload Gambar Mobil", type=['jpg', 'jpeg', 'png'])
    plate_number = ""
    cropped_plate_img_pil = None

    if car_img_file:
        car_img_pil = Image.open(car_img_file).convert('RGB')
        st.image(car_img_pil, caption="Gambar Mobil", use_container_width=True)

        # Convert PIL ke OpenCV (BGR)
        img_cv = cv2.cvtColor(np.array(car_img_pil), cv2.COLOR_RGB2BGR)

        # Deteksi plat nomor dengan YOLO
        bbox = detect_plate_yolo(yolo_model, img_cv)

        if bbox is not None:
            plate_img_cv = crop_plate(img_cv, bbox)
            cropped_plate_img_pil = Image.fromarray(cv2.cvtColor(plate_img_cv, cv2.COLOR_BGR2RGB))
            st.image(cropped_plate_img_pil, caption="Plat Nomor Terdeteksi", use_container_width=True)

            # OCR plat nomor dengan EasyOCR
            plate_number = ocr_plate_easyocr(ocr_reader, cropped_plate_img_pil)
            st.info(f"Nomor Plat Terdeteksi: {plate_number}")
        else:
            st.warning("Plat nomor tidak ditemukan pada gambar mobil.")

    st.subheader("Detail Mobil")
    model_input = st.selectbox("Model", sorted(df['model'].unique()))
    year_input = st.number_input("Tahun", min_value=1990, max_value=2025, value=2015)
    mileage_input = st.number_input("Mileage (km)", 0, 300000, value=50000)
    tax_input = st.number_input("Pajak per Tahun (€)", 0, 1000, value=150)
    mpg_input = st.number_input("MPG", 0, 100, value=30)
    engine_size_input = st.number_input("Engine Size (L)", 0.5, 6.0, value=1.6, format="%.1f")

    if st.button("Prediksi Harga"):
        harga = predict_price(knn, le, model_input, year_input, mileage_input, tax_input, mpg_input, engine_size_input)
        st.success(f"💰 Prediksi Harga: €{harga:,.2f}")
        if plate_number:
            st.write(f"📛 Nomor Plat: **{plate_number}**")

        with open("hasil_prediksi.txt", "w") as f:
            f.write(
                f"Model: {model_input}\nTahun: {year_input}\nMileage: {mileage_input} km\n"
                f"Tax: {tax_input} €\nMPG: {mpg_input}\nEngine Size: {engine_size_input} L\n"
                f"Nomor Plat: {plate_number}\nPrediksi Harga: €{harga:,.2f}"
            )

if __name__ == "__main__":
    main()
