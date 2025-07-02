import streamlit as st
import numpy as np
from PIL import Image
import random
import time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cv2

# Configuración de la página
st.set_page_config(
    page_title="Juego de Emociones con IA",
    page_icon="😊",
    layout="wide"
)

# Inicialización de modelos de IA local
@st.cache_resource
def load_sentiment_analyzers():
    """Carga los analizadores de sentimientos"""
    vader_analyzer = SentimentIntensityAnalyzer()
    return vader_analyzer

@st.cache_resource  
def load_face_cascade():
    """Carga el clasificador de rostros de OpenCV"""
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except:
        return None

# Inicializar modelos
vader_analyzer = load_sentiment_analyzers()
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
    """Analiza la emoción del texto usando múltiples métodos de IA"""
    try:
        # VADER Sentiment Analysis
        vader_scores = vader_analyzer.polarity_scores(text)
        
        # TextBlob Analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Palabras clave para detectar emociones específicas
        text_lower = text.lower()
        
        # Palabras clave por emoción
        emotion_keywords = {
            'happy': ['feliz', 'alegre', 'contento', 'genial', 'fantástico', 'increíble', 'maravilloso', 'happy', 'joy', 'great', 'awesome', 'wonderful', 'amazing', 'fantastic', 'excellent', 'perfect', 'love', 'excited'],
            'sad': ['triste', 'melancólico', 'deprimido', 'sad', 'depressed', 'melancholy', 'sorrow', 'grief', 'unhappy', 'miserable', 'dejected', 'heartbroken', 'lonely', 'blue'],
            'angry': ['enojado', 'furioso', 'molesto', 'irritado', 'angry', 'furious', 'mad', 'irritated', 'annoyed', 'rage', 'hate', 'frustrated', 'livid', 'outraged'],
            'fear': ['miedo', 'terror', 'asustado', 'nervioso', 'fear', 'scared', 'afraid', 'terrified', 'anxious', 'worried', 'panic', 'frightened', 'nervous'],
            'surprise': ['sorprendido', 'asombrado', 'impresionado', 'surprise', 'shocked', 'amazed', 'astonished', 'stunned', 'wow', 'incredible', 'unbelievable'],
            'disgust': ['asco', 'repugnante', 'asqueroso', 'disgust', 'disgusting', 'gross', 'revolting', 'nauseating', 'repulsive', 'sick']
        }
        
        # Contar coincidencias de palabras clave
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Si hay palabras clave, usar la emoción con más coincidencias
        if emotion_scores:
            detected_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(emotion_scores[detected_emotion] * 0.3 + 0.4, 1.0)
            return detected_emotion, confidence
        
        # Si no hay palabras clave, usar análisis de sentimientos más sensible
        compound = vader_scores['compound']
        pos = vader_scores['pos']
        neg = vader_scores['neg']
        
        # Umbrales más bajos para detectar emociones
        if compound >= 0.1:  # Más sensible para positivo
            if pos > 0.3:
                emotion = 'happy'
            else:
                emotion = 'joy'
        elif compound <= -0.1:  # Más sensible para negativo
            if neg > 0.3:
                emotion = 'angry'
            else:
                emotion = 'sad'
        elif polarity > 0.05:  # TextBlob positivo
            emotion = 'happy'
        elif polarity < -0.05:  # TextBlob negativo
            emotion = 'sad'
        elif subjectivity > 0.5:  # Alto contenido emocional
            emotion = 'surprise'
        else:
            emotion = 'neutral'
            
        confidence = max(abs(compound), abs(polarity), 0.3)
        return emotion, confidence
        
    except Exception as e:
        st.error(f"Error en análisis: {e}")
        return "neutral", 0.5

def detect_faces_in_image(image):
    """Detecta rostros en la imagen"""
    try:
        if face_cascade is None:
            return False, []
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0, faces
    except Exception as e:
        st.error(f"Error en detección facial: {e}")
        return False, []

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
            
            # Ejemplos de texto por emoción
            st.write("💡 **Ejemplos de textos por emoción:**")
            examples = {
                'happy': "¡Estoy súper feliz! Hoy es un día maravilloso y me siento genial.",
                'sad': "Me siento muy triste y melancólico. Todo parece gris hoy.",
                'angry': "¡Estoy furioso! Esto me molesta muchísimo y me da rabia.",
                'fear': "Tengo mucho miedo. Esta situación me pone muy nervioso y ansioso.",
                'surprise': "¡Wow! ¡Qué sorpresa tan increíble! No puedo creer lo que veo.",
                'disgust': "Esto es asqueroso y repugnante. Me da mucho asco."
            }
            
            if st.session_state.target_emotion in examples:
                st.info(f"Ejemplo para '{st.session_state.target_emotion}': {examples[st.session_state.target_emotion]}")
            
            user_input = st.text_area("Escribe tu respuesta:", height=100)
            
            if st.button("🧠 Analizar con IA") and user_input:
                detected_emotion, confidence = analyze_emotion_from_text(user_input)
                
                # Debug info
                with st.expander("🔍 Debug - Ver análisis detallado"):
                    vader_scores = vader_analyzer.polarity_scores(user_input)
                    blob = TextBlob(user_input)
                    
                    st.write("**VADER Scores:**")
                    st.json(vader_scores)
                    st.write(f"**TextBlob Polarity:** {blob.sentiment.polarity}")
                    st.write(f"**TextBlob Subjectivity:** {blob.sentiment.subjectivity}")
                
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
        st.info("Usando análisis de sentimientos local")
        st.write("- VADER Sentiment Analysis")
        st.write("- TextBlob para análisis de texto")
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