import numpy as np

# 1. Definir los parámetros nominales y desviaciones
vin = 100
n_simulaciones = 10000

# R1: media 200, std 3.333
r1_sim = np.random.normal(200, 3.3333, n_simulaciones)
# R2: media 50, std 0.833
r2_sim = np.random.normal(50, 0.8333, n_simulaciones)

# 2. Calcular Vout para los 10,000 casos
vout_sim = vin * (r2_sim / (r1_sim + r2_sim))

# 3. Mostrar resultados
print(f"Media de Vout: {np.mean(vout_sim):.4f} V")
print(f"Desviación estándar de Vout (Monte Carlo): {np.std(vout_sim):.4f} V")