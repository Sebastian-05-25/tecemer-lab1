import pandas as pd

# Paso 3.1: Cargar CSV
df = pd.read_csv("pronostico_huancayo.csv")
df["fecha"] = pd.to_datetime(df["fecha"])

print("--- Primeras filas ---")
print(df.head())
print("\n--- Información del DataFrame ---")
print(df.info())

# Paso 3.2: Transformación de datos
df["amplitud_termica"] = df["temp_max"] - df["temp_min"]
df["dia_lluvioso"] = df["precipitacion"] > 0
df["categoria"] = df["temp_max"].apply(
    lambda t: "cálido" if t >= 20 else ("templado" if t >= 15 else "frío")
)

print("\n--- DataFrame Transformado ---")
print(df)
print("\n--- Descripción Estadística ---")
print(df.describe())

# Paso 3.3: Agregaciones
resumen = df.groupby("categoria").agg(
    dias=("categoria", "count"),
    temp_max_promedio=("temp_max", "mean"),
    precipitacion_total=("precipitacion", "sum"),
)

print("\n--- Resumen por Categoría ---")
print(resumen)

# Paso 3.4: Exportación de resultados
df.to_csv("pronostico_huancayo_procesado.csv", index=False)
resumen.to_csv("resumen_por_categoria.csv")