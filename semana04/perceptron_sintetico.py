import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Fijar semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# 1. Generar dataset sintético (100 muestras, 4 características)
X_sintetico = np.random.randn(100, 4)
y_sintetico = (X_sintetico[:, 0] + X_sintetico[:, 1] > 0).astype(int)

# 2. Definir arquitectura del Perceptrón Multicapa (MLP)
modelo = Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# 3. Compilar el modelo
modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

# 4. Entrenar la red neuronal
historial = modelo.fit(
    X_sintetico,
    y_sintetico,
    epochs=30,
    batch_size=8,
    verbose=0
)

# 5. Graficar y guardar curvas de entrenamiento
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(historial.history["loss"], label="Pérdida (Loss)", color="crimson")
plt.title("Pérdida en Entrenamiento")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(historial.history["accuracy"], label="Precisión (Accuracy)", color="teal")
plt.title("Precisión en Entrenamiento")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("curvas_entrenamiento_sintetico.png", dpi=150)
print("\n✅ Entrenamiento sintético completado. Gráfica guardada en 'curvas_entrenamiento_sintetico.png'")
