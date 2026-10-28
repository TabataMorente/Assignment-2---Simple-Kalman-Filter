import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('datos_imu_spc.csv')
df_spc = df.groupby('Subgrupo')['Accel_Z'].agg(X_bar='mean', S='std').reset_index()

# 1. Calibration parameters (Subgroups 1 to 25)
fase_base = df_spc[df_spc['Subgrupo'] <= 25]
media_global = fase_base['X_bar'].mean()

c4 = 0.9727
sigma_estimado = fase_base['S'].mean() / c4

# 2. Define LSL and USL (Mean +- 5 times base SD)
USL = media_global + (5 * sigma_estimado)
LSL = media_global - (5 * sigma_estimado)

print(f"Defined limits: LSL = {LSL:.4f} | USL = {USL:.4f}\n")


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


# 3. Compare phases
calcular_capacidad(fase_base, "Base phase (Under Control)")
# Separate perturbations
calcular_capacidad(df_spc[(df_spc['Subgrupo'] >= 26) & (df_spc['Subgrupo'] <= 35)], "Perturbation: Tilt")
calcular_capacidad(df_spc[df_spc['Subgrupo'] >= 36], "Perturbation: Vibration")