import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import anderson, shapiro

# Forzar a Pandas a mostrar todas las columnas sin puntos suspensivos
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Sensibilidad para convertir datos crudos a unidades del SI (m/s^2 y grados/s)
sensibilidad = {
    'Accel_X': -0.000613,
    'Accel_Y': -0.000613,
    'Accel_Z': -0.000605,
    'Gyro_X': 0.007633,
    'Gyro_Y': 0.007633,
    'Gyro_Z': 0.007633
}

archives = ['datos_base.csv', 'datos_X.csv', 'datos_Y.csv']

for archive in archives:
    print(f"\n=========================================")
    print(f"   RESULTS FOR: {archive}")
    print(f"=========================================")

    try:
        df = pd.read_csv(archive)
    except FileNotFoundError:
        print(f"{archive} not found")
        continue

    # CONVERSIÓN MATEMÁTICA A UNIDADES REALES
    df_si = df.copy()
    for col in sensibilidad:
        if col in df_si.columns:
            df_si[col] = df_si[col] * sensibilidad[col]

    print("\n--- MEAN AND STANDARD DEVIATION (Real Units) ---")
    print(df_si.agg(['mean', 'std']).T)

    print("\n--- COVARIANCE MATRIX (6x6) (Real Units) ---")
    print(df_si.cov())

    print("\n--- PEARSON CORRELATION ---")
    print(df_si.corr(method='pearson'))

    print("\n--- NORMALITY TEST (Anderson-Darling & Shapiro-Wilk) ---")
    for col in df_si.columns:
        # Se añade method='approx' para evitar el warning rojo de SciPy en tu terminal
        result = anderson(df_si[col])
        stat_sw, p_value = shapiro(df_si[col])
        print(
            f"[{col}] Anderson: {result.statistic:.3f} (Crit: {result.critical_values[2]:.3f}) | p-value: {p_value:.3e}")

    df_si.hist(bins=30, figsize=(10, 8), color='skyblue', edgecolor='black')
    plt.suptitle(f"Histogramas (Unidades Reales) - {archive}")
    plt.tight_layout()
    plt.show()