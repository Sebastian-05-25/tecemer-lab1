import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# 1. Cargar parámetros de escalado guardados
datos = np.load("dataset_preparado.npz")
media = datos["media"]
desviacion = datos["desviacion"]

# 2. Cargar el modelo entrenado
modelo = tf.keras.models.load_model("modelo_lluvia.keras")


def predecir_lluvia(temp_max, temp_min):
    amplitud = temp_max - temp_min
    entrada_bruta = np.array([[temp_max, temp_min, amplitud]])

    # Normalizar con la misma media y desviación del entrenamiento
    entrada_escalada = (entrada_bruta - media) / desviacion

    probabilidad = modelo.predict(entrada_escalada, verbose=0)[0][0]
    prediccion_binaria = 1 if probabilidad >= 0.5 else 0

    return probabilidad, prediccion_binaria


# 3. Escenarios de prueba para Huancayo
casos = [
    {"desc": "Día cálido y húmedo", "tmax": 20.0, "tmin": 8.0},
    {"desc": "Día helado/seco típico de junio", "tmax": 18.0, "tmin": -1.0},
    {"desc": "Día templado", "tmax": 19.5, "tmin": 7.5},
]

print("=== PREDICCIONES DE LLUVIA EN HUANCAYO ===\n")
for c in casos:
    prob, pred = predecir_lluvia(c["tmax"], c["tmin"])
    resultado = "🌧️ Lloverá" if pred == 1 else "☀️ No lloverá"
    print(f"Escenario: {c['desc']}")
    print(f"  Temps: Max {c['tmax']}°C | Min {c['tmin']}°C")
    print(f"  Probabilidad: {prob * 100:.2f}% -> Resultado: {resultado}\n")
