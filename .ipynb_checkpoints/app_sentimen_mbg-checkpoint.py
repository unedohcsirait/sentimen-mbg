import streamlit as st
import re
from joblib import load

# =========================
# CLEANING TEXT
# =========================
def datacleaning(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[?|$|.|@#%^/&*=!_:")(-+,]', '', text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

# =========================
# STREAMLIT UI
# =========================
def main():
    st.title("📊 Analisis Sentimen Kebijakan Makan Bergizi Gratis (MBG)")
    st.write("Masukkan opini / tweet / komentar untuk diprediksi sentimennya")

    input_text = st.text_area("Masukkan teks:")

    if st.button("Prediksi Sentimen"):
        pre_text = datacleaning(input_text)

        # Transform teks menjadi TF-IDF vector
        tfidf_vector = tfidf.transform([pre_text])
        
        # Prediksi
        hasil = model.predict(tfidf_vector)[0]

        # Tampilkan hasil lebih ramah
        if hasil.lower() == 'positif':
            hasil_prediksi = "😊 Positif"
        elif hasil.lower() == 'negatif':
            hasil_prediksi = "😡 Negatif"
        else:
            hasil_prediksi = "😐 Netral"

        st.success(f"Hasil Sentimen: **{hasil_prediksi}**")

# =========================
# LOAD MODEL & TF-IDF
# =========================
if __name__ == "__main__":
    model = load("svm_mbg_sentiment.pkl")
    tfidf = load("tfidf_mbg.pkl")
    main()
