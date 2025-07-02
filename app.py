import streamlit as st
import cv2
import numpy as np
from PIL import Image
import speech_recognition as sr
import pyttsx3
import threading
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import io
import base64

# Configuración de la página
st.set_page_config(
    page_title="Juego de Emociones con IA",
    page_icon="😊",
    layout="wide"
)

# Inicialización de modelos de IA local
@st.cache_resource
def load_emotion_model():
    """Carga el modelo de análisis de emociones"""
    try:
        # Modelo liviano para análisis de emociones en texto
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1
        )
        return emotion_classifier
    except:
        # Fallback a un modelo más simple
        return pipeline("sentiment-analysis", device=-1)

@st.cache_resource
def load_face_cascade():
    """Carga el clasificador de rostros de OpenCV"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar modelos
emotion_model = load_emotion_model()
face_cascade = load_face_cascade()

# Estado del juego
if 'game_state' not in st.session_state:
    st.session_state.game_state = {
        'score': 0,
        'level': 1,
        'target_emotion': 'happy',
        'game_mode': 'text',
        'challenges_completed': 0,
        'user_name': ''
    }

# Funciones auxiliares
def analyze_emotion_from_text(text):
    """Analiza la emoción del texto usando el modelo de IA"""
    try:
        result = emotion_model(text)
        if isinstance(result, list) and len(result) > 0:
            emotion = result[0]['label'].lower()
            confidence = result[0]['score']
            return emotion, confidence
    except:
        pass
    return "neutral", 0.5

def detect_faces_in_image(image):
    """Detecta rostros en la imagen"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0, faces

def get_emotion_challenge():
    """Genera un desafío de emoción aleatorio"""
    emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'joy']
    challenges = {
        'happy': "Escribe algo que te haga feliz o sonríe a la cámara",
        'sad': "Describe un momento triste o muestra tristeza",
        'angry': "Expresa algo que te moleste (de forma constructiva)",
        'fear': "Cuenta algo que te dé miedo",
        'surprise': "Reacciona con sorpresa",
        'disgust': "Describe algo desagradable",
        'joy': "Comparte algo que te llene de alegría"
    }
    
    emotion = np.random.choice(emotions)
    return emotion, challenges.get(emotion, "Expresa una emoción")

# Interfaz principal
st.title("🎮 Juego de Emociones con IA")
st.write("Un juego interactivo que usa IA local para reconocer emociones a través de texto, cámara o micrófono")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    if not st.session_state.game_state['user_name']:
        user_name = st.text_input("Tu nombre:")
        if user_name:
            st.session_state.game_state['user_name'] = user_name
            st.rerun()
    
    if st.session_state.game_state['user_name']:
        st.write(f"¡Hola {st.session_state.game_state['user_name']}!")
        
        game_mode = st.selectbox(
            "Modo de juego:",
            ["text", "camera", "microphone"],
            format_func=lambda x: {
                "text": "📝 Texto",
                "camera": "📷 Cámara",
                "microphone": "🎤 Micrófono"
            }[x]
        )
        
        st.session_state.game_state['game_mode'] = game_mode
        
        st.write("### 📊 Estadísticas")
        st.write(f"**Puntuación:** {st.session_state.game_state['score']}")
        st.write(f"**Nivel:** {st.session_state.game_state['level']}")
        st.write(f"**Desafíos completados:** {st.session_state.game_state['challenges_completed']}")
        
        if st.button("🔄 Reiniciar juego"):
            for key in ['score', 'level', 'challenges_completed']:
                st.session_state.game_state[key] = 0 if key != 'level' else 1
            st.rerun()

# Juego principal
if st.session_state.game_state['user_name']:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generar nuevo desafío
        if st.button("🎯 Nuevo Desafío") or 'current_challenge' not in st.session_state:
            target_emotion, challenge_text = get_emotion_challenge()
            st.session_state.current_challenge = challenge_text
            st.session_state.target_emotion = target_emotion
        
        st.write("### 🎯 Desafío Actual")
        st.info(f"**Emoción objetivo:** {st.session_state.target_emotion.upper()}")
        st.write(st.session_state.current_challenge)
        
        # Modo de juego
        if st.session_state.game_state['game_mode'] == 'text':
            st.write("### 📝 Modo Texto")
            user_input = st.text_area("Escribe tu respuesta:", height=100)
            
            if st.button("🧠 Analizar con IA") and user_input:
                detected_emotion, confidence = analyze_emotion_from_text(user_input)
                
                st.write("### 🤖 Análisis de IA")
                st.write(f"**Emoción detectada:** {detected_emotion}")
                st.write(f"**Confianza:** {confidence:.2%}")
                
                # Verificar si coincide con el objetivo
                if detected_emotion.lower() in st.session_state.target_emotion.lower() or \
                   st.session_state.target_emotion.lower() in detected_emotion.lower():
                    st.success("¡Correcto! +10 puntos")
                    st.session_state.game_state['score'] += 10
                    st.session_state.game_state['challenges_completed'] += 1
                    st.balloons()
                else:
                    st.warning(f"Casi... Detecté '{detected_emotion}' pero necesitaba '{st.session_state.target_emotion}'")
                    st.session_state.game_state['score'] += 2
        
        elif st.session_state.game_state['game_mode'] == 'camera':
            st.write("### 📷 Modo Cámara")
            
            camera_input = st.camera_input("Toma una foto mostrando la emoción:")
            
            if camera_input:
                image = Image.open(camera_input)
                st.image(image, caption="Tu foto", width=300)
                
                # Detectar rostros
                faces_detected, faces = detect_faces_in_image(image)
                
                if faces_detected:
                    st.success(f"✅ Rostro detectado! Encontré {len(faces)} rostro(s)")
                    st.session_state.game_state['score'] += 5
                    st.write("*En una implementación completa, aquí analizaríamos la expresión facial*")
                else:
                    st.warning("No se detectó ningún rostro. Intenta de nuevo.")
        
        elif st.session_state.game_state['game_mode'] == 'microphone':
            st.write("### 🎤 Modo Micrófono")
            st.write("*Funcionalidad de micrófono requiere configuración adicional del navegador*")
            
            # Simulación de entrada de audio
            audio_text = st.text_input("Simula lo que dirías (en una implementación real se capturaría audio):")
            
            if st.button("🎵 Procesar Audio") and audio_text:
                detected_emotion, confidence = analyze_emotion_from_text(audio_text)
                
                st.write("### 🎧 Análisis de Audio (simulado)")
                st.write(f"**Texto procesado:** {audio_text}")
                st.write(f"**Emoción detectada:** {detected_emotion}")
                st.write(f"**Confianza:** {confidence:.2%}")
    
    with col2:
        st.write("### 🏆 Progreso")
        
        # Barra de progreso
        progress = min(st.session_state.game_state['challenges_completed'] / 10, 1.0)
        st.progress(progress)
        st.write(f"Progreso del nivel: {st.session_state.game_state['challenges_completed']}/10")
        
        # Nivel up
        if st.session_state.game_state['challenges_completed'] >= 10:
            st.session_state.game_state['level'] += 1
            st.session_state.game_state['challenges_completed'] = 0
            st.success(f"¡Nivel {st.session_state.game_state['level']}!")
        
        st.write("### 🧠 Modelo de IA")
        st.info("Usando modelo local de análisis de emociones")
        st.write("- Transformers (Hugging Face)")
        st.write("- OpenCV para detección facial")
        st.write("- Procesamiento local sin envío de datos")
        
        st.write("### 🎮 Instrucciones")
        st.write("""
        1. Elige tu modo preferido
        2. Lee el desafío emocional
        3. Responde según el modo:
           - **Texto:** Escribe expresando la emoción
           - **Cámara:** Muestra la expresión facial
           - **Micrófono:** Habla con la emoción
        4. La IA analizará tu respuesta
        5. ¡Gana puntos y sube de nivel!
        """)

else:
    st.write("### 👋 ¡Bienvenido!")
    st.write("Por favor, ingresa tu nombre en la barra lateral para comenzar a jugar.")

# Footer
st.write("---")
st.write("🤖 Creado con Streamlit y modelos de IA locales | 🔒 Procesamiento completamente local")