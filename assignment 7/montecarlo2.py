import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de parámetros
N = 10000
# El enunciado dice: desviación estándar = precisión / 3
sigma_dist = 2.0 / 3.0  # cm por cada avance
sigma_ang = 2.0 / 3.0  # grados por cada giro

# Arrays para guardar dónde termina el coche en cada iteración
x_finals = np.zeros(N)
y_finals = np.zeros(N)

# 2. Simulación de las 10,000 rutas
for i in range(N):
    x, y = 0.0, 0.0

    # El vehículo empieza apuntando al Norte (90º en trigonometría), con un pequeño error inicial
    angulo_actual = 90.0 + np.random.normal(0, sigma_ang)

    # Ruta: (distancia_en_cm, giro_posterior_en_grados)
    # Giros: Positivo = Anti-horario (Izquierda), Negativo = Horario (Derecha)
    movimientos = [
        (200, -90),  # Sube 2m, gira a la derecha (Este)
        (200, 90),  # Avanza 2m, gira a la izquierda (Norte)
        (200, 90),  # Avanza 2m, gira a la izquierda (Oeste)
        (100, -90),  # Avanza 1m, gira a la derecha (Norte)
        (300, 90),  # Avanza 3m, gira a la izquierda (Oeste)
        (100, 0)  # Avanza 1m, fin de la ruta (sin giro)
    ]

    for dist_nom, giro_nom in movimientos:
        # Añadimos el error aleatorio de la medición de distancia
        dist_real = np.random.normal(dist_nom, sigma_dist)

        # Calculamos el avance trigonométrico
        angulo_rad = np.radians(angulo_actual)
        x += dist_real * np.cos(angulo_rad)
        y += dist_real * np.sin(angulo_rad)

        # Añadimos el error aleatorio del sensor de rotación (si toca girar)
        if giro_nom != 0:
            giro_real = np.random.normal(giro_nom, sigma_ang)
            angulo_actual += giro_real

    # Guardamos la posición final de esta iteración
    x_finals[i] = x
    y_finals[i] = y

# 3. Análisis respecto al objetivo perfecto
x_nom = 0.0
y_nom = 700.0
# Distancia geométrica desde donde quedó hasta donde debería haber quedado
distancias_error = np.sqrt((x_finals - x_nom) ** 2 + (y_finals - y_nom) ** 2)

# 4. Visualización
plt.figure(figsize=(14, 6))

# Gráfico 1: Nube de dispersión (Scatter plot)
plt.subplot(1, 2, 1)
plt.scatter(x_finals, y_finals, alpha=0.3, s=2, color='purple')
plt.plot(x_nom, y_nom, 'rX', markersize=10, label='Destino Nominal (0, 700)')
plt.title('Dispersión de las 10,000 Posiciones Finales')
plt.xlabel('Eje X (cm)')
plt.ylabel('Eje Y (cm)')
plt.legend()
plt.grid(True)

# Gráfico 2: Histograma del error
plt.subplot(1, 2, 2)
plt.hist(distancias_error, bins=50, color='skyblue', edgecolor='black')
plt.title('Histograma de Distancia al Objetivo Nominal')
plt.xlabel('Distancia de error (cm)')
plt.ylabel('Frecuencia')
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Resultados numéricos para el cálculo de Cp
print(f"Distancia media de error: {np.mean(distancias_error):.2f} cm")
print(f"Desviación estándar del error: {np.std(distancias_error):.2f} cm")