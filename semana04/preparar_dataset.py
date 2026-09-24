import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar dataset ampliado
df = pd.read_csv("pronostico_huancayo.csv")


def calcular_dia_lluvioso(precipitacion):
    """Retorna 1 si hubo precipitación mayor a 0 mm, de lo contrario 0."""
    return 1 if precipitacion > 0 else 0


# Feature engineering
df["amplitud_termica"] = df["temp_max"] - df["temp_min"]
df["dia_lluvioso"] = df["precipitacion"].apply(calcular_dia_lluvioso)

# Definir características (X) y objetivo (y)
caracteristicas = ["temp_max", "temp_min", "amplitud_termica"]
X = df[caracteristicas].values
y = df["dia_lluvioso"].values

# Partición estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Escalado con StandardScaler
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# Guardar arreglos listos para el entrenamiento
np.savez(
    "dataset_preparado.npz",
    X_train=X_train_escalado,
    X_test=X_test_escalado,
    y_train=y_train,
    y_test=y_test,
    media=escalador.mean_,
    desviacion=escalador.scale_,
)

print("✅ Dataset preparado y exportado a 'dataset_preparado.npz'")
print(f"Dimensiones X_train: {X_train_escalado.shape}")
print(f"Dimensiones X_test:  {X_test_escalado.shape}")