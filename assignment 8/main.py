import pandas as pd
import numpy as np

# Fijamos la semilla para reproducibilidad
np.random.seed(42)

# Definimos los parámetros del experimento: 3 operadores x 5 IMUs x 3 intentos
operators = ['Student_1', 'Student_2', 'Student_3']
imus = ['IMU_1', 'IMU_2', 'IMU_3', 'IMU_4', 'IMU_5']
trials = [1, 2, 3]

# Valores base para las 5 IMUs (Part Variation - PV)
# Cada sensor tiene su propio sesgo de fábrica respecto al ángulo perfecto de 90º
imu_base = {
    'IMU_1': 90.1,
    'IMU_2': 89.4,
    'IMU_3': 90.8,
    'IMU_4': 89.1,
    'IMU_5': 90.5
}

# Sesgo de los 3 Operadores (Appraiser Variation - AV)
op_bias = {'Student_1': 0.05, 'Student_2': -0.15, 'Student_3': 0.10}

# Variación del Equipo (Equipment Variation - EV)
# Desviación estándar del ruido en cada lectura
noise_std = 0.2

# Generación de los datos
data = []
for op in operators:
    for imu in imus:
        for trial in trials:
            # Valor final = Base de la IMU + Sesgo humano + Ruido aleatorio
            val = imu_base[imu] + op_bias[op] + np.random.normal(0, noise_std)
            data.append([op, imu, trial, round(val, 2)])

# Creación de la tabla
df = pd.DataFrame(data, columns=['Operator', 'IMU', 'Try', 'Angle (º)'])

# Mostrar los resultados listos para copiar al Excel
print(df.to_string(index=False))