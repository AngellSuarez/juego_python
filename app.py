import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import preprocess_face

# Configuración de la página
st.set_page_config(page_title="Juego de Emociones con IA", page_icon="😊", layout="wide")

# Cargar modelo de emociones faciales
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model("emotion_model.h5")

# Cargar analizador de sentimientos
@st.cache_resource
def load_sentiment_analyzers():
    return SentimentIntensityAnalyzer()

# Etiquetas de emociones según el modelo
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Cargar modelos
emotion_model = load_emotion_model()
vader = load_sentiment_analyzers()

# Estado inicial del juego
if 'score' not in st.session_state:
    st.session_state.score = 0
    st.session_state.level = 1
    st.session_state.challenges_completed = 0
    st.session_state.target_emotion = 'happy'
    st.session_state.user_name = ''

# Función para analizar texto
def analyze_emotion_from_text(text):
    vader_scores = vader.polarity_scores(text)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if vader_scores['compound'] >= 0.5:
        return 'Happy', vader_scores['compound']
    elif vader_scores['compound'] <= -0.5:
        return 'Sad', abs(vader_scores['compound'])
    elif polarity > 0.3:
        return 'Happy', polarity
    elif polarity < -0.3:
        return 'Sad', abs(polarity)
    else:
        return 'Neutral', 0.5

# Función para analizar imagen
def analyze_emotion_from_image(image):
    try:
        face = preprocess_face(image)
        predictions = emotion_model.predict(np.array([face]), verbose=0)
        predicted_class = emotion_labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return "Unknown", 0.0

# Interfaz
st.title("🎮 Juego de Emociones con IA")
st.write("Un juego interactivo que usa IA local para reconocer emociones por texto o imagen")

# Configuración del usuario
with st.sidebar:
    st.header("Configuración")
    if not st.session_state.user_name:
        name = st.text_input("Tu nombre:")
        if name:
            st.session_state.user_name = name
            st.rerun()
    st.write(f"Hola, **{st.session_state.user_name}**")

    if st.button("🔄 Reiniciar juego"):
        st.session_state.score = 0
        st.session_state.level = 1
        st.session_state.challenges_completed = 0
        st.rerun()

    st.write(f"📊 **Puntos:** {st.session_state.score}")
    st.write(f"🏅 **Nivel:** {st.session_state.level}")
    st.write(f"🎯 **Desafíos completados:** {st.session_state.challenges_completed}")

# Generar emoción objetivo
def generate_challenge():
    emociones = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Neutral']
    return np.random.choice(emociones)

if st.button("🎯 Nuevo desafío") or 'target_emotion' not in st.session_state:
    st.session_state.target_emotion = generate_challenge()

st.subheader("🎯 Emoción objetivo:")
st.info(f"**{st.session_state.target_emotion.upper()}**")

# Sección de entrada de texto
st.subheader("📝 Escribe algo que exprese la emoción")

user_text = st.text_area("Tu mensaje:")
if st.button("🧠 Analizar texto"):
    emotion, conf = analyze_emotion_from_text(user_text)
    st.write(f"**Emoción detectada:** {emotion}")
    st.write(f"**Confianza:** {conf:.2%}")
    if emotion.lower() == st.session_state.target_emotion.lower():
        st.success("¡Correcto! +10 puntos")
        st.session_state.score += 10
        st.session_state.challenges_completed += 1
    else:
        st.warning(f"Detecté '{emotion}', no coincide con '{st.session_state.target_emotion}'")
        st.session_state.score += 2

# Sección de entrada de imagen
st.subheader("📷 Toma una foto mostrando la emoción")

img = st.camera_input("Captura tu expresión facial")
if img:
    img_pil = Image.open(img).convert("RGB")
    st.image(img_pil, caption="Tu imagen")

    emotion, conf = analyze_emotion_from_image(img_pil)
    st.write(f"**Expresión detectada:** {emotion}")
    st.write(f"**Confianza:** {conf:.2%}")
    if emotion.lower() == st.session_state.target_emotion.lower():
        st.success("¡Perfecto! +15 puntos")
        st.session_state.score += 15
        st.session_state.challenges_completed += 1
    else:
        st.warning(f"Detecté '{emotion}', no coincide con '{st.session_state.target_emotion}'")
        st.session_state.score += 3

# Nivel y progreso
st.subheader("🎮 Progreso del juego")
progress = st.session_state.challenges_completed / 10
st.progress(progress)
if st.session_state.challenges_completed >= 10:
    st.session_state.level += 1
    st.session_state.challenges_completed = 0
    st.success(f"🎉 ¡Subiste al nivel {st.session_state.level}!")

st.write("---")
st.caption("🤖 Hecho con Streamlit y modelos locales")
