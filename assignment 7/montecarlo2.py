import numpy as np
import matplotlib.pyplot as plt

# 1. Parameter configuration
N = 10000
sigma_dist = 2.0 / 3.0
sigma_ang = 2.0 / 3.0

# Arrays to save where the car finishes in each iteration
x_finals = np.zeros(N)
y_finals = np.zeros(N)

# 2. 10,000 route simulation
for i in range(N):
    x, y = 0.0, 0.0

    # The vehicle starts pointing North (90º in trigonometry), with a small initial error
    angulo_actual = 90.0 + np.random.normal(0, sigma_ang)

    # Route: (distance_in_cm, subsequent_turn_in_degrees)
    # Turns: Positive = Counter-clockwise (Left), Negative = Clockwise (Right)
    movimientos = [
        (200, -90),  # Moves up 2m, turns right (East)
        (200, 90),  # Moves forward 2m, turns left (North)
        (200, 90),  # Moves forward 2m, turns left (West)
        (100, -90),  # Moves forward 1m, turns right (North)
        (300, 90),  # Moves forward 3m, turns left (West)
        (100, 0)  # Moves forward 1m, end of route (no turn)
    ]

    for dist_nom, giro_nom in movimientos:
        # Add the random error of the distance measurement
        dist_real = np.random.normal(dist_nom, sigma_dist)

        # Calculate the trigonometric advance
        angulo_rad = np.radians(angulo_actual)
        x += dist_real * np.cos(angulo_rad)
        y += dist_real * np.sin(angulo_rad)

        # Add the random error of the rotation sensor (if a turn is required)
        if giro_nom != 0:
            giro_real = np.random.normal(giro_nom, sigma_ang)
            angulo_actual += giro_real

    # Save the final position of this iteration
    x_finals[i] = x
    y_finals[i] = y

# 3. Analysis regarding the perfect target
x_nom = 0.0
y_nom = 700.0
# Geometric distance from where it ended up to where it should have ended up
distancias_error = np.sqrt((x_finals - x_nom) ** 2 + (y_finals - y_nom) ** 2)

# 4. Visualization
plt.figure(figsize=(14, 6))

# Graph 1: Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(x_finals, y_finals, alpha=0.3, s=2, color='purple')
plt.plot(x_nom, y_nom, 'rX', markersize=10, label='Nominal Target (0, 700)')
plt.title('Dispersion of the 10,000 Final Positions')
plt.xlabel('X Axis (cm)')
plt.ylabel('Y Axis (cm)')
plt.legend()
plt.grid(True)

# Graph 2: Error histogram
plt.subplot(1, 2, 2)
plt.hist(distancias_error, bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Distance to Nominal Target')
plt.xlabel('Error distance (cm)')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Numerical results for Cp calculation
print(f"Mean error distance: {np.mean(distancias_error):.2f} cm")
print(f"Standard deviation of error: {np.std(distancias_error):.2f} cm")