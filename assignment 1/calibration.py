import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Real Angles and archives
archivos = {
    'datos_0_grados.csv': 0.6,
    'datos_30_grados.csv': 30.2,
    'datos_45_grados.csv': 44.4,
    'datos_90_grados.csv': 93
}

sensores = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
gravity = 9.81

# Creating 6 spots for the graphics
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

print("==========================================================")
print("   FINAL RESULTS OF CALIBRATION   ")
print("==========================================================")

for i, sensor in enumerate(sensores):
    data_per_angle = []
    tags = []
    means = []
    theorical_accel = []

    print(f"\n--- STATISTICS FOR {sensor} ---")
    print(f"{'Archive':<20} | {'Min':<8} | {'Max':<8} | {'Mean':<8}")
    print("-" * 55)

    for archivo, angulo in archivos.items():
        try:
            df = pd.read_csv(archivo)
            values = df[sensor].values
            data_per_angle.append(values)

            # 1. Min, Max, Mean calculations
            vmin, vmax, vmean = np.min(values), np.max(values), np.mean(values)
            means.append(vmean)
            print(f"{archivo:<20} | {vmin:<8.1f} | {vmax:<8.1f} | {vmean:<8.1f}")

            # 2. Assigning the theorical real value according to the movements
            if sensor == 'Accel_X':
                teorico = gravity * np.sin(np.radians(angulo))
            elif sensor == 'Accel_Z':
                # Z axe El eje Z reaches its max at 0º and drops to zero at 90º (Cos)
                teorico = gravity * np.cos(np.radians(angulo))
            elif sensor == 'Accel_Y':
                # If tilted toward X, Y shouldn't receive gravity
                teorico = 0.0
            else:
                # Gyros were still (0 degrees per second)
                teorico = 0.0

            theorical_accel.append(teorico)
            tags.append(f"{angulo}º\n({teorico:.1f})")

        except FileNotFoundError:
            print(f"Error: {archivo} not found")
            continue

    # 3. Draw 6 Box-Whisker plots
    if data_per_angle:
        axes[i].boxplot(data_per_angle, tick_labels=tags)
        axes[i].set_title(f'Curve of {sensor}')
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # 4. Mathemathic formules assuming linearity
    if len(means) > 1:
        slope, intercept, r_value, _, _ = linregress(means, theorical_accel)
        print(f"\n> FORMULA {sensor}: Real_Value = ({slope:.6f} * Digital_Value) + ({intercept:.6f})")
        # R2 just for X y Z, where there is a dynamic change
        if sensor in ['Accel_X', 'Accel_Z']:
            print(f"> Linearity (R²): {r_value ** 2:.4f}")

plt.tight_layout()
plt.show()