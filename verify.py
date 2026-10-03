import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar datos del Excel
df = pd.read_excel('3 - KF Assignment 1 data.xlsx')
gps_measurements = df['Measured position data'].values
prof_reference = df['Corrected position data after KF'].values

# 2. Configurar matrices del sistema
T, u = 0.1, 1.0
A = np.array([[1.0, T], [0.0, 1.0]])
B = np.array([[0.5 * T ** 2], [T]])
C = np.array([[1.0, 0.0]])
Sz = np.array([[100.0]])
Sw = np.array([[1e-6, 2e-5], [2e-5, 4e-4]])

# 3. Ejecutar el Filtro de Kalman exacto
x = np.array([[0.0], [0.0]])
P = np.array([[0.0, 0.0], [0.0, 0.0]])
our_estimates = [float(x[0, 0])]

for step in range(len(gps_measurements) - 1):
    y = gps_measurements[step + 1]
    K = A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + Sz)
    x = A @ x + B * u + K @ (y - C @ x)
    P = A @ P @ A.T + Sw - A @ P @ C.T @ np.linalg.inv(Sz) @ C @ P @ A.T
    our_estimates.append(float(x[0, 0]))

our_estimates = np.array(our_estimates)

# 4. Calcular la diferencia matemática
diferencia_absoluta = np.abs(our_estimates - prof_reference)
max_diff = np.max(diferencia_absoluta)
mean_diff = np.mean(diferencia_absoluta)

print("-" * 55)
print(" VERIFICACIÓN DE RESULTADOS: PYTHON vs EXCEL DEL PROFE")
print("-" * 55)
print(f"Diferencia Máxima: {max_diff:.6f} metros")
print(f"Diferencia Media:  {mean_diff:.6f} metros")
print("-" * 55)

# 5. Crear tabla comparativa para la consola
df_comparativa = pd.DataFrame({
    'Paso': range(len(prof_reference)),
    'Excel Profe': prof_reference,
    'Nuestro Python': our_estimates,
    'Diferencia': diferencia_absoluta
})

print("\n--- PRIMEROS 5 PASOS ---")
print(df_comparativa.head(5).to_string(index=False))

print("\n--- ÚLTIMOS 5 PASOS ---")
print(df_comparativa.tail(5).to_string(index=False))

# 6. Graficar el error de coma flotante
plt.figure(figsize=(10, 4))
plt.plot(diferencia_absoluta, color='purple', linewidth=1)
plt.title('Diferencia matemática entre nuestro Python y el Excel (Ruido de redondeo)')
plt.xlabel('Paso de tiempo')
plt.ylabel('Diferencia (m)')
plt.yscale('log') # Usamos escala logarítmica para visualizar valores minúsculos
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()