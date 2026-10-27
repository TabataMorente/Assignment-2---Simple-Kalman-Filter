import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('datos_imu_spc.csv')
df_spc = df.groupby('Subgrupo')['Accel_Z'].agg(X_bar='mean', S='std').reset_index()

# 1. Parámetros de la calibración (Subgrupos 1 al 25)
fase_base = df_spc[df_spc['Subgrupo'] <= 25]
media_global = fase_base['X_bar'].mean()

# Para n=10, la constante c4 es aprox 0.9727 para estimar sigma real
c4 = 0.9727
sigma_estimado = fase_base['S'].mean() / c4

# 2. Definir LSL y USL (Media +- 5 veces la desviación estándar base)
USL = media_global + (5 * sigma_estimado)
LSL = media_global - (5 * sigma_estimado)

print(f"Límites definidos: LSL = {LSL:.4f} | USL = {USL:.4f}\n")


def calcular_capacidad(datos, nombre_fase):
    media = datos['X_bar'].mean()
    sigma = datos['S'].mean() / c4

    Cp = (USL - LSL) / (6 * sigma)
    Cpk_superior = (USL - media) / (3 * sigma)
    Cpk_inferior = (media - LSL) / (3 * sigma)
    Cpk = min(Cpk_superior, Cpk_inferior)

    print(f"--- {nombre_fase} ---")
    print(f"Cp  = {Cp:.3f}")
    print(f"Cpk = {Cpk:.3f}\n")


# 3. Comparar las fases
calcular_capacidad(fase_base, "Fase Base (Bajo Control)")
# Separamos las perturbaciones (ajusta los números de subgrupo según los que hiciste)
calcular_capacidad(df_spc[(df_spc['Subgrupo'] >= 26) & (df_spc['Subgrupo'] <= 35)], "Perturbación: Inclinación")
calcular_capacidad(df_spc[df_spc['Subgrupo'] >= 36], "Perturbación: Vibración")