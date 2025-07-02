# train_emotion_model.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import os

# Cargar datos de ejemplo (iris, solo como placeholder)
X, y = load_iris(return_X_y=True)

# Entrenar modelo simple
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Asegurarse de que el directorio exista
os.makedirs("model", exist_ok=True)

# Guardar el modelo como .pkl
joblib.dump(model, "model/emotion_model.pkl")

print("✅ Modelo guardado exitosamente en 'model/emotion_model.pkl'")
