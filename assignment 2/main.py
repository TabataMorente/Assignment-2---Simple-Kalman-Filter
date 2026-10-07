import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

# --- 1. Load Data ---
df = pd.read_excel('3 - KF Assignment 1 data.xlsx')
gps_measurements = df['Measured position data'].values
prof_reference = df['Corrected position data after KF'].values

# --- 2. System Parameters & Matrices ---
T = 0.1  # Sample time (10 Hz)
u = 1.0  # Acceleration (m/s^2)

A = np.array([[1, T],
              [0, 1]])

B = np.array([[0.5 * T ** 2],
              [T]])

C = np.array([[1, 0]])

Sz = np.array([[100]])  # GPS measurement noise variance
Sw = np.array([[1e-6, 2e-5],
               [2e-5, 4e-4]])  # Process noise covariance

# --- 3. Initialization ---
x = np.array([[0], [0]])  # Initial state [pos, vel]
P = np.array([[0, 0], [0, 0]])  # Initial error covariance

kf_position_estimates = [float(x[0,0])]

# --- 4. Kalman Filter Loop ---
for step in range(len(gps_measurements)-1):
    y= gps_measurements[step+1]

    # Compute Kalman Gain
    K = A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + Sz)

    # State update
    x = A @ x + B * u + K @ (y - C @ x)

    # Covariance update
    P = A @ P @ A.T + Sw - A @ P @ C.T @ np.linalg.inv(Sz) @ C @ P @ A.T

    # Store corrected state rounded to 3 decimal places
    kf_position_estimates.append(float(x[0, 0]))

# --- 5. Verification & CSV Export ---
df_results = pd.DataFrame({
    'Measured position data': gps_measurements,
    'Corrected position data after KF': kf_position_estimates
})

# Verification of the maximum absolute difference against the professor's reference
diff_check = np.abs(df_results['Corrected position data after KF'] - prof_reference)
print("Max absolute difference:", diff_check.max())

df_results.to_csv('kalman_filter_results.csv', index=False)

# --- 6. General Plot ---
plt.figure(figsize=(10, 6))
plt.plot(gps_measurements, label='GPS (Noisy)', color='green', alpha=0.8, linewidth=1)
plt.plot(kf_position_estimates, label='Kalman Filter', color='red', linewidth=2.5)
plt.title('Kalman Filter Results')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('images/general_plot.png', bbox_inches='tight')
plt.show()

# --- 7. Simple Moving Average (SMA) Comparison ---
window_size = 10
sma_estimates = pd.Series(gps_measurements).rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(10, 6))
plt.plot(gps_measurements[800:1000], label='GPS (Noisy)', color='green', alpha=0.8, linewidth=1)
plt.plot(kf_position_estimates[800:1000], label='Kalman Filter', color='red', linewidth=2.5)
plt.plot(sma_estimates[800:1000].values, label=f'SMA (Window={window_size})', color='blue', linestyle='--', linewidth=2)
plt.title('KF vs SMA Comparison (Steps 800-1000)')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('images/zoom_plot.png', bbox_inches='tight')
plt.show()

# --- 8. Startup Super Zoom ---
plt.figure(figsize=(10, 6))
plt.plot(gps_measurements[:50], label='GPS (Noisy)', color='green', alpha=0.6, linewidth=1.5, marker='o', markersize=4)
plt.plot(kf_position_estimates[:50], label='Kalman Filter', color='red', linewidth=3)
plt.plot(sma_estimates[:50].values, label=f'SMA (Window={window_size})', color='blue', linestyle='--', linewidth=2)
plt.title('Startup Super Zoom (Steps 0-50)')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('images/superzoom_plot.png', bbox_inches='tight')
plt.show()


# --- 9. EXTRAS: PARAMETER SENSITIVITY ANALYSIS ---
def run_kf_with_custom_sz(sz_val):
    x_sens = np.array([[0], [0]])
    P_sens = np.array([[0, 0], [0, 0]])
    estimates = [float(x_sens[0, 0])]

    for step in range(len(gps_measurements) - 1):
        y_val = gps_measurements[step + 1]
        K_sens = A @ P_sens @ C.T @ np.linalg.inv(C @ P_sens @ C.T + sz_val)
        x_sens = A @ x_sens + B * u + K_sens @ (y_val - C @ x_sens)
        P_sens = A @ P_sens @ A.T + Sw - A @ P_sens @ C.T @ np.linalg.inv(sz_val) @ C @ P_sens @ A.T
        estimates.append(float(x_sens[0, 0]))  # Ponle el round() aquí si se lo pones al de arriba

    return estimates


est_sz_1 = run_kf_with_custom_sz(np.array([[1]]))
est_sz_10k = run_kf_with_custom_sz(np.array([[10000]]))

# Plot Sz = 1
plt.figure(figsize=(10, 6))
plt.plot(gps_measurements[:200], label='GPS (Noisy)', color='green', alpha=0.5, linewidth=1)
plt.plot(est_sz_1[:200], label='KF ($S_z = 1$: High Trust in GPS)', color='orange', linewidth=2)
plt.title('Parameter Sensitivity: Overconfident Sensor ($S_z = 1$)')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('images/sensitivity_sz1.png', bbox_inches='tight')
plt.show()

# Plot Sz = 10000
plt.figure(figsize=(10, 6))
plt.plot(gps_measurements[:200], label='GPS (Noisy)', color='green', alpha=0.5, linewidth=1)
plt.plot(est_sz_10k[:200], label='KF ($S_z = 10000$: Low Trust in GPS)', color='purple', linewidth=2)
plt.title('Parameter Sensitivity: Underconfident Sensor ($S_z = 10000$)')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('images/sensitivity_sz10000.png', bbox_inches='tight')
plt.show()