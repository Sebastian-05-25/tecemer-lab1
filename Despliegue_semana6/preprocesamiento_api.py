import numpy as np

COLUMNAS_CARACTERISTICAS = ["temp_max", "temp_min", "viento_max"]


def extraer_caracteristicas(datos_json):
    # Valida que no falte ninguna columna
    for col in COLUMNAS_CARACTERISTICAS:
        if col not in datos_json:
            raise ValueError(f"Falta el parámetro requerido: {col}")

    # Devuelve un np.array de forma (1, 3) en el orden correcto
    valores = [datos_json[col] for col in COLUMNAS_CARACTERISTICAS]
    return np.array([valores], dtype=float)
