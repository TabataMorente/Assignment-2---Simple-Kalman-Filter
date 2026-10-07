import math
import numpy as np
import requests
import time
import matplotlib

# ¡ESTA ES LA LÍNEA MÁGICA PARA ARREGLAR LAS DOS VENTANAS DE PYCHARM!
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 1. LA IP DE TU MÓVIL (Tu IP actual)
URL = "http://10.171.158.219:8080/get?accX&accY&accZ&gyrX&gyrY&gyrZ"

g = 9.81  # Gravedad en m/s^2

# 2. INICIALIZACIÓN DE MATRICES
Q = np.eye(4) * 4.444e-5
R = np.eye(4) * 0.088
H = np.eye(4)
x = np.array([[1.0], [0.0], [0.0], [0.0]])
P = np.eye(4)

# Configurar gráfica en tiempo real interactiva
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
line_roll, = ax.plot([], [], label='Roll (deg)', color='blue', linewidth=2)
line_pitch, = ax.plot([], [], label='Pitch (deg)', color='red', linewidth=2)

ax.set_title('Real-Time Attitude Estimation (Phyphox + Kalman Filter)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.legend()
ax.grid(True)
ax.set_ylim(-90, 90)

roll_data, pitch_data, time_data = [], [], []
start_time = time.time()
last_time = start_time

print("¡Conectando al móvil... Mueve la mano!")
print("Presiona Ctrl+C en la terminal para parar el script.")

# 3. EL BUCLE INFINITO
while True:
    try:
        response = requests.get(URL, timeout=1).json()

        fx = response['buffer']['accX']['buffer'][-1]
        fy = response['buffer']['accY']['buffer'][-1]
        p = response['buffer']['gyrX']['buffer'][-1]
        q_g = response['buffer']['gyrY']['buffer'][-1]
        r = response['buffer']['gyrZ']['buffer'][-1]

        current_time = time.time()
        dt = current_time - last_time

        if dt <= 0:
            continue

        last_time = current_time

        # B. FILTRO DE KALMAN
        Omega = np.array([
            [0, -p, -q_g, -r],
            [p, 0, r, -q_g],
            [q_g, -r, 0, p],
            [r, q_g, -p, 0]
        ])
        A = np.eye(4) + (dt / 2.0) * Omega

        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        val_theta = np.clip(fx / g, -1.0, 1.0)
        theta_acc = math.asin(val_theta)

        val_phi = np.clip(-fy / (g * math.cos(theta_acc)), -1.0, 1.0)
        phi_acc = math.asin(val_phi)

        q1_z = math.cos(phi_acc / 2) * math.cos(theta_acc / 2)
        q2_z = math.sin(phi_acc / 2) * math.cos(theta_acc / 2)
        q3_z = math.cos(phi_acc / 2) * math.sin(theta_acc / 2)
        q4_z = -math.sin(phi_acc / 2) * math.sin(theta_acc / 2)

        z = np.array([[q1_z], [q2_z], [q3_z], [q4_z]])

        S_mat = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S_mat)

        x = x_pred + K @ (z - H @ x_pred)
        x = x / np.linalg.norm(x)

        P = (np.eye(4) - K @ H) @ P_pred

        # C. CONVERTIR A EULER PARA LA GRÁFICA
        q1, q2, q3, q4 = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        roll_est = math.atan2(2 * (q1 * q2 + q3 * q4), 1 - 2 * (q2 ** 2 + q3 ** 2))
        pitch_est = math.asin(np.clip(2 * (q1 * q3 - q4 * q2), -1.0, 1.0))

        # D. ACTUALIZAR LA GRÁFICA EN DIRECTO
        t_elapsed = current_time - start_time
        roll_deg = math.degrees(roll_est)
        pitch_deg = math.degrees(pitch_est)

        time_data.append(t_elapsed)
        roll_data.append(roll_deg)
        pitch_data.append(pitch_deg)

        # ¡Imprimimos los datos en la terminal para confirmar que funciona!
        print(f"Tiempo: {t_elapsed:.1f}s | Roll: {roll_deg:.1f}° | Pitch: {pitch_deg:.1f}°")

        if len(time_data) > 150:
            time_data.pop(0)
            roll_data.pop(0)
            pitch_data.pop(0)

        line_roll.set_data(time_data, roll_data)
        line_pitch.set_data(time_data, pitch_data)

        ax.set_xlim(time_data[0], time_data[-1] + 0.1)

        # Forzamos el redibujado de la ventana
        fig.canvas.draw()
        fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("¡Programa detenido!")
        break
    except Exception as e:
        pass