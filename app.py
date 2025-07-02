import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.preprocessing import preprocess_face

# Configuración de la página
st.set_page_config(page_title="Juego de Emociones IA", layout="wide")

# Cargar modelos
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_emotion_model():
    return joblib.load("model/emotion_model.pkl")

vader = load_sentiment_analyzer()
emotion_model = load_emotion_model()

# Mapeo de clases simuladas (ajusta según tu dataset real)
emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# App
st.title("😄 Juego de Emociones (Texto + Imagen)")
st.markdown("Esta app analiza emociones desde texto o una imagen de tu rostro")

modo = st.sidebar.radio("Selecciona modo", ["Texto", "Cámara"])

if modo == "Texto":
    st.subheader("📝 Análisis de Emoción por Texto")
    texto = st.text_area("Escribe algo emocional...")
    if st.button("Analizar Texto"):
        vader_score = vader.polarity_scores(texto)
        blob = TextBlob(texto)

        st.write("### Resultado VADER")
        st.json(vader_score)

        st.write("### Resultado TextBlob")
        st.write(f"Polarity: {blob.sentiment.polarity}")
        st.write(f"Subjectivity: {blob.sentiment.subjectivity}")

        # Estimación simple de emoción
        if vader_score['compound'] >= 0.5:
            st.success("Parece una emoción positiva 😊")
        elif vader_score['compound'] <= -0.5:
            st.error("Emoción negativa detectada 😠")
        else:
            st.info("Emoción neutral o mixta 😐")

elif modo == "Cámara":
    st.subheader("📷 Analiza tu Emoción Facial")
    img_data = st.camera_input("Toma una foto con tu expresión emocional")

    if img_data is not None:
        image = Image.open(img_data).convert("RGB")
        st.image(image, caption="Imagen capturada", width=300)

        # Preprocesar rostro
        face = preprocess_face(image)
        if face is not None:
            st.image(face, caption="Rostro detectado", width=150)

            face_flat = face.flatten().reshape(1, -1)
            prediction = emotion_model.predict(face_flat)[0]
            proba = emotion_model.predict_proba(face_flat)[0]

            emotion = emotion_labels.get(prediction, "unknown")
            confidence = proba[prediction]

            st.success(f"Emoción detectada: **{emotion.upper()}**")
            st.write(f"Confianza: {confidence:.2%}")
        else:
            st.warning("No se detectó un rostro válido.")
