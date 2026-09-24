import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
# Fijar semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# 1. Cargar el dataset preparado
datos = np.load("dataset_preparado.npz")
X_train = datos["X_train"]
X_test = datos["X_test"]
y_train = datos["y_train"]
y_test = datos["y_test"]

# 2. Construir la red neuronal con Regularización L2 y Dropout
modelo = Sequential(
    [
        layers.Input(shape=(3,)),
        layers.Dense(
            16,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),
        layers.Dropout(0.2),
        layers.Dense(
            8,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# 3. Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

modelo.summary()

# 4. Entrenar el modelo
historial = modelo.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=8,
    verbose=0,
)

# 5. Evaluar precisión en el conjunto de prueba
loss_test, accuracy_test = modelo.evaluate(X_test, y_test, verbose=0)
print(
    f"\n✅ Precisión en el conjunto de prueba (Test Accuracy): {accuracy_test * 100:.2f}%"
)

# 6. Guardar el modelo entrenado
modelo.save("modelo_lluvia.keras")
print("✅ Modelo guardado como 'modelo_lluvia.keras'")

# 7. Graficar curvas de entrenamiento y validación
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(
    historial.history["loss"], label="Pérdida Entrenamiento", color="crimson"
)
plt.plot(
    historial.history["val_loss"],
    label="Pérdida Validación",
    color="orange",
    linestyle="--",
)
plt.title("Curva de Pérdida (Loss)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(
    historial.history["accuracy"],
    label="Precisión Entrenamiento",
    color="teal",
)
plt.plot(
    historial.history["val_accuracy"],
    label="Precisión Validación",
    color="green",
    linestyle="--",
)
plt.title("Curva de Precisión (Accuracy)")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("curvas_entrenamiento_lluvia.png", dpi=150)
print("✅ Gráfica guardada como 'curvas_entrenamiento_lluvia.png'")
