import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar datos
df = pd.read_excel('3 - KF Assignment 1 data.xlsx')
mediciones_gps = df['Measured position data'].values
referencia_profesor = df['Corrected position data after KF'].values

# 2. Definir parámetros
T = 0.1
u = 1.0
A = np.array([[1, T], [0, 1]])
B = np.array([[0.5 * T ** 2], [T]])
C = np.array([[1, 0]])
Sz = np.array([[100]])
Sw = np.array([[1e-6, 2e-5], [2e-5, 4e-4]])

# 3. Inicialización
x = np.array([[0], [0]])
P = np.array([[0, 0], [0, 0]])
estimaciones_posicion = []

# 4. Bucle del Filtro de Kalman
for y in mediciones_gps:
    # A) Guardar la predicción actual antes de corregir
    estimaciones_posicion.append(x[0, 0])

    # B) Calcular la Ganancia de Kalman (K)
    K = A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + Sz)

    # C) Actualizar Estado (Predicción + Corrección)
    x = A @ x + B * u + K @ (y - C @ x)

    # D) Actualizar Covarianza
    P = A @ P @ A.T + Sw - A @ P @ C.T @ np.linalg.inv(Sz) @ C @ P @ A.T

# 5. Comprobación y resultados
df_resultados = pd.DataFrame({
    'Mi_KF': estimaciones_posicion,
    'Prof_KF': referencia_profesor
})
df_resultados['Diferencia'] = df_resultados['Mi_KF'] - df_resultados['Prof_KF']
print("Diferencia máxima con la referencia del profesor:", df_resultados['Diferencia'].abs().max())

# 6. Gráfica para el informe
plt.figure(figsize=(10, 6))
plt.plot(mediciones_gps, label='GPS Measurement (Noisy)', color='lightgray', alpha=0.7)
plt.plot(estimaciones_posicion, label='Our KF Estimation', color='red', linewidth=2)
plt.title('Kalman Filter Results')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()