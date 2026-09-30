import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos del Excel
df = pd.read_excel('3 - KF Assignment 1 data.xlsx')
mediciones_gps = df['Measured position data'].values        # Lo que lee el sensor
referencia_profesor = df['Corrected position data after KF'].values # La solución ideal

# 2. Definir parámetros del sistema
T = 0.1      # Tiempo de muestreo (10 Hz)
u = 1.0      # Aceleración comandada (m/s^2)

# Matrices del sistema (Cinemática del coche)
A = np.array([[1, T],
              [0, 1]])

B = np.array([[0.5 * T**2],
              [T]])

C = np.array([[1, 0]])

# 3. Definir matrices de Covarianza (El ruido)
Sz = np.array([[100]]) # Varianza del ruido de medición (10^2)

# Varianza del ruido del proceso (calculado en la clase)
Sw = np.array([[1e-6, 2e-5],
               [2e-5, 4e-4]])

# 4. Condiciones iniciales
# Empezamos parados en la posición 0, por lo que no hay incertidumbre al inicio
x = np.array([[0], [0]]) # Vector de estado inicial [Posición, Velocidad]
P = np.array([[0, 0], [0, 0]]) # Matriz de error de estimación (P0)