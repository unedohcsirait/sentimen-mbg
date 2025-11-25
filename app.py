# -*- coding: utf-8 -*-
import streamlit as st
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Load models
@st.cache_resource
def load_models():
    tfidf = load("tfidf_mbg.pkl")
    model = load("model_svm_mbg.pkl")
    return tfidf, model

tfidf, model = load_models()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

def clean_text(text):
    """Preprocessing teks sederhana"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[?|$|.|@#%^/&*=!_:")(-+,]', '', text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

def predict_sentiment(text, tfidf_vectorizer, classifier):
    """Prediksi sentimen dengan confidence score"""
    text_tfidf = tfidf_vectorizer.transform([text]).toarray()
    prediction = classifier.predict(text_tfidf)[0]
    
    # Get decision function scores if available
    try:
        decision_scores = classifier.decision_function(text_tfidf)[0]
        if len(decision_scores) == 3:
            # Multi-class classification
            confidence = max(decision_scores)
        else:
            confidence = abs(decision_scores)
    except:
        confidence = 0.0
    
    return prediction, confidence

def main():
    # Header
    st.set_page_config(
        page_title="Analisis Sentimen MBG",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Analisis Sentimen Program MBG")
    st.markdown("### *Makan Bergizi Gratis*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Informasi")
        st.info("""
        **Model:** Support Vector Machine (SVM)  
        **Vectorizer:** TF-IDF  
        **Target:** Sentimen Positif/Negatif/Netral
        """)
        
        st.markdown("---")
        st.header("🔧 Pengaturan")
        show_preprocessing = st.checkbox("Tampilkan teks preprocessing", value=False)
        show_confidence = st.checkbox("Tampilkan confidence score", value=True)
        
        st.markdown("---")
        if st.button("🗑️ Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediksi Tunggal", "📝 Prediksi Batch", "📈 Riwayat & Statistik", "ℹ️ Tentang"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.subheader("Analisis Sentimen Teks Tunggal")
        
        # Example texts
        with st.expander("💡 Lihat Contoh Kalimat"):
            st.markdown("""
            **Contoh Positif:**
            - "Program MBG sangat membantu meningkatkan gizi anak-anak Indonesia"
            - "BGN berhasil menjalankan program dengan baik dan transparan"
            
            **Contoh Negatif:**
            - "Program MBG gagal total dan pemborosan anggaran negara"
            - "Implementasi MBG buruk dan tidak tepat sasaran"
            
            **Contoh Netral:**
            - "Kepala BGN memastikan Program MBG tetap berjalan selama belum ada perintah penghentian dari Presiden Prabowo. BGN terus mengevaluasi sistem agar semakin baik"
            - "BGN melaporkan data pelaksanaan program MBG kepada presiden"
            """)
        
        text = st.text_area(
            "Masukkan teks untuk dianalisis:",
            placeholder="Ketik atau paste teks di sini...",
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            predict_button = st.button("🚀 Prediksi Sentimen", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🔄 Clear", use_container_width=True)
            if clear_button:
                st.rerun()
        
        if predict_button:
            if text.strip() == "":
                st.warning("⚠️ Teks tidak boleh kosong!")
            else:
                with st.spinner("Menganalisis sentimen..."):
                    # Show preprocessing if enabled
                    if show_preprocessing:
                        cleaned = clean_text(text)
                        st.caption(f"**Teks setelah preprocessing:** {cleaned}")
                    
                    # Predict
                    prediction, confidence = predict_sentiment(text, tfidf, model)
                    
                    # Display result
                    st.markdown("### Hasil Analisis")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        if prediction.lower() == "positif":
                            st.success("### 😄 POSITIF")
                        elif prediction.lower() == "negatif":
                            st.error("### 😡 NEGATIF")
                        else:
                            st.info("### 😐 NETRAL")
                    
                    with col_res2:
                        if show_confidence:
                            st.metric("Confidence Score", f"{confidence:.4f}")
                    
                    # Save to history
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': prediction,
                        'confidence': confidence
                    })
    
    # Tab 2: Batch Prediction
    with tab2:
        st.subheader("Analisis Sentimen Batch (Multiple Texts)")
        
        st.markdown("Upload file CSV atau masukkan multiple texts (satu per baris)")
        
        option = st.radio("Pilih metode input:", ["Upload CSV", "Input Manual"])
        
        if option == "Upload CSV":
            st.info("📁 Format CSV: kolom 'text' berisi kalimat yang akan dianalisis")
            uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("❌ CSV harus memiliki kolom 'text'")
                    else:
                        st.write(f"Jumlah data: {len(df)} baris")
                        st.dataframe(df.head())
                        
                        if st.button("🚀 Analisis Semua", type="primary"):
                            with st.spinner("Menganalisis..."):
                                predictions = []
                                confidences = []
                                
                                for text in df['text']:
                                    pred, conf = predict_sentiment(str(text), tfidf, model)
                                    predictions.append(pred)
                                    confidences.append(conf)
                                
                                df['sentimen'] = predictions
                                df['confidence'] = confidences
                                
                                st.success("✅ Analisis selesai!")
                                st.dataframe(df)
                                
                                # Download result
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "📥 Download Hasil",
                                    csv,
                                    "hasil_sentimen_mbg.csv",
                                    "text/csv"
                                )
                                
                                # Show statistics
                                st.markdown("### 📊 Statistik")
                                sentiment_counts = df['sentimen'].value_counts()
                                
                                fig = px.pie(
                                    values=sentiment_counts.values,
                                    names=sentiment_counts.index,
                                    title="Distribusi Sentimen",
                                    color=sentiment_counts.index,
                                    color_discrete_map={
                                        'positif': '#28a745',
                                        'negatif': '#dc3545',
                                        'netral': '#17a2b8'
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        else:  # Input Manual
            batch_text = st.text_area(
                "Masukkan beberapa kalimat (satu per baris):",
                height=200,
                placeholder="Kalimat 1\nKalimat 2\nKalimat 3"
            )
            
            if st.button("🚀 Analisis Semua", type="primary"):
                if batch_text.strip() == "":
                    st.warning("⚠️ Input tidak boleh kosong!")
                else:
                    texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                    
                    with st.spinner("Menganalisis..."):
                        results = []
                        for text in texts:
                            pred, conf = predict_sentiment(text, tfidf, model)
                            results.append({
                                'text': text,
                                'sentimen': pred,
                                'confidence': conf
                            })
                        
                        df_results = pd.DataFrame(results)
                        st.success("✅ Analisis selesai!")
                        st.dataframe(df_results)
                        
                        # Statistics
                        sentiment_counts = df_results['sentimen'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        
                        positif_count = sentiment_counts.get('positif', 0)
                        negatif_count = sentiment_counts.get('negatif', 0)
                        netral_count = sentiment_counts.get('netral', 0)
                        
                        col1.metric("😄 Positif", positif_count)
                        col2.metric("😡 Negatif", negatif_count)
                        col3.metric("😐 Netral", netral_count)
    
    # Tab 3: History & Statistics
    with tab3:
        st.subheader("📈 Riwayat Prediksi & Statistik")
        
        if len(st.session_state.history) == 0:
            st.info("Belum ada riwayat prediksi. Mulai analisis di tab 'Prediksi Tunggal'")
        else:
            df_history = pd.DataFrame(st.session_state.history)
            
            # Statistics
            st.markdown("### 📊 Ringkasan Statistik")
            
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment_counts = df_history['sentiment'].value_counts()
            
            col1.metric("Total Prediksi", len(df_history))
            col2.metric("😄 Positif", sentiment_counts.get('positif', 0))
            col3.metric("😡 Negatif", sentiment_counts.get('negatif', 0))
            col4.metric("😐 Netral", sentiment_counts.get('netral', 0))
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Distribusi Sentimen",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positif': '#28a745',
                        'negatif': '#dc3545',
                        'netral': '#17a2b8'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                fig_bar = px.bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    title="Jumlah per Kategori",
                    labels={'x': 'Sentimen', 'y': 'Jumlah'},
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'positif': '#28a745',
                        'negatif': '#dc3545',
                        'netral': '#17a2b8'
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # History table
            st.markdown("### 📝 Riwayat Detail")
            st.dataframe(
                df_history[['timestamp', 'text', 'sentiment', 'confidence']].sort_values('timestamp', ascending=False),
                use_container_width=True
            )
            
            # Export history
            csv_history = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Riwayat",
                csv_history,
                "riwayat_sentimen_mbg.csv",
                "text/csv"
            )
    
    # Tab 4: About
    with tab4:
        st.subheader("ℹ️ Tentang Aplikasi")
        
        st.markdown("""
        ### 🎯 Analisis Sentimen Program MBG
        
        Aplikasi ini menggunakan Machine Learning untuk menganalisis sentimen teks terkait 
        Program Makan Bergizi Gratis (MBG) Pemerintah Indonesia.
        
        #### 🤖 Teknologi
        - **Model:** Support Vector Machine (SVM)
        - **Vectorizer:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Framework:** Streamlit
        - **Visualisasi:** Plotly
        
        #### 📊 Kategori Sentimen
        - **Positif** 😄: Sentimen mendukung/positif terhadap program
        - **Negatif** 😡: Sentimen menentang/negatif terhadap program  
        - **Netral** 😐: Sentimen objektif/informatif tanpa kecenderungan
        
        #### 🚀 Fitur
        - Prediksi sentimen tunggal dengan confidence score
        - Analisis batch untuk multiple texts
        - Riwayat prediksi dengan statistik
        - Export hasil ke CSV
        - Visualisasi interaktif
        
        #### 📝 Catatan
        Hasil prediksi adalah estimasi berdasarkan model machine learning dan 
        mungkin tidak 100% akurat. Gunakan sebagai referensi analisis awal.
        
        ---
        
        **Dikembangkan untuk analisis sentimen publik terhadap Program MBG**
        """)
        
        st.info("💡 Tips: Gunakan kalimat yang jelas dan kontekstual untuk hasil terbaik")

if __name__ == "__main__":
    main()