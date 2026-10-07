import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

# --- 1. Load Data ---
# Nombres de columnas exactos del archivo "3 - KF Assignment 2 data.xlsx"
df = pd.read_excel('3 - KF Assignment 2 data.xlsx')

f_x = df['Accel X'].values
f_y = df['Accel Y'].values
f_z = df['Accel Z'].values

p_gyro = df['Gyro Roll'].values
q_gyro = df['Gyro Pitch'].values
r_gyro = df['Gyro Yaw'].values

dt = 0.01  # Sample time (100 Hz as per assignment)
g = 9.81  # Gravity constant

# --- 2. System Parameters & Matrices ---
# Process Noise Covariance (Q) - Gyroscope uncertainty based on professor's hints
Q = np.eye(4) * 4.444e-5

# Measurement Noise Covariance (R) - Accelerometer uncertainty
R = np.eye(4) * 0.088

# Measurement Matrix H (Identity because we measure quaternions directly after conversion)
H = np.eye(4)

# --- 3. Initialization ---
# Initial state [q1, q2, q3, q4]^T for a hovering/flat position
x = np.array([[1.0], [0.0], [0.0], [0.0]])
P = np.eye(4)  # Initial error covariance

roll_estimates = []
pitch_estimates = []

# --- 4. Kalman Filter Loop ---
for step in range(len(f_x)):
    # 1. READ GYRO & BUILD 'A' MATRIX (State Transition)
    p = p_gyro[step]
    q_g = q_gyro[step]
    r = r_gyro[step]

    Omega = np.array([
        [0, -p, -q_g, -r],
        [p, 0, r, -q_g],
        [q_g, -r, 0, p],
        [r, q_g, -p, 0]
    ])
    A = np.eye(4) + (dt / 2.0) * Omega

    # 2. PREDICT (State and Covariance)
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    # 3. READ ACCEL & CONVERT TO QUATERNIONS (Measurement z)
    fx = f_x[step]
    fy = f_y[step]

    # Convert Accel to Euler (clipping to [-1, 1] to avoid domain errors from noise spikes)
    val_theta = np.clip(fx / g, -1.0, 1.0)
    theta_acc = math.asin(val_theta)  # Pitch

    val_phi = np.clip(-fy / (g * math.cos(theta_acc)), -1.0, 1.0)
    phi_acc = math.asin(val_phi)  # Roll

    # Convert Euler to Quaternion measurements (z)
    q1_z = math.cos(phi_acc / 2) * math.cos(theta_acc / 2)
    q2_z = math.sin(phi_acc / 2) * math.cos(theta_acc / 2)
    q3_z = math.cos(phi_acc / 2) * math.sin(theta_acc / 2)
    q4_z = -math.sin(phi_acc / 2) * math.sin(theta_acc / 2)
    z = np.array([[q1_z], [q2_z], [q3_z], [q4_z]])

    # 4. COMPUTE KALMAN GAIN
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # 5. STATE & COVARIANCE UPDATE
    x = x_pred + K @ (z - H @ x_pred)

    # Normalize quaternion to prevent mathematical divergence
    x = x / np.linalg.norm(x)
    P = (np.eye(4) - K @ H) @ P_pred

    # 6. CONVERT CORRECTED QUATERNION TO EULER FOR PLOTTING
    q1, q2, q3, q4 = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
    roll_est = math.atan2(2 * (q1 * q2 + q3 * q4), 1 - 2 * (q2 ** 2 + q3 ** 2))
    pitch_est = math.asin(np.clip(2 * (q1 * q3 - q4 * q2), -1.0, 1.0))

    roll_estimates.append(math.degrees(roll_est))
    pitch_estimates.append(math.degrees(pitch_est))

# --- 5. Export & Plot ---
time_axis = np.arange(len(f_x)) * dt

plt.figure(figsize=(10, 6))
plt.plot(time_axis, roll_estimates, label='Estimated Roll (KF)', color='blue', linewidth=2)
plt.plot(time_axis, pitch_estimates, label='Estimated Pitch (KF)', color='red', linewidth=2)
plt.title('Attitude Estimation: Sensor Fusion (Gyro + Accel)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.grid(True)
plt.savefig('images/attitude_plot.png', bbox_inches='tight')
plt.show()