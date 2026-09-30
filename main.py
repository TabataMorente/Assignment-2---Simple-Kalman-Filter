import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA LOADING
# ==========================================
# Read the provided Excel file
df = pd.read_excel('3 - KF Assignment 1 data.xlsx')
gps_measurements = df['Measured position data'].values
prof_reference = df['Corrected position data after KF'].values

# ==========================================
# 2. SYSTEM MODELING (PHYSICS & NOISE)
# ==========================================
T = 0.1  # Sampling time in seconds (10 Hz)
u = 1.0  # Constant commanded acceleration (m/s^2)

# State Transition Matrix (Kinematics: x = x + v*T)
A = np.array([[1, T],
              [0, 1]])

# Control Input Matrix (Kinematics: x = x + 0.5*a*T^2, v = v + a*T)
B = np.array([[0.5 * T ** 2],
              [T]])

# Measurement Matrix (We only measure position, not velocity)
C = np.array([[1, 0]])

# Noise Covariance Matrices
Sz = np.array([[100]])  # Measurement noise (GPS standard deviation = 10m -> variance = 100)
Sw = np.array([[1e-6, 2e-5],
               [2e-5, 4e-4]])  # Process noise (external disturbances)

# ==========================================
# 3. INITIALIZATION
# ==========================================
# We assume the car starts at position 0 with velocity 0
x = np.array([[0], [0]])

# Initial uncertainty is 0 because we are absolutely sure of the starting point
P = np.array([[0, 0], [0, 0]])

kf_position_estimates = []

print("=== KALMAN FILTER INITIALIZATION ===")
print("Starting Position and Velocity:\n", x)
print("Starting Uncertainty (P):\n", P)
print("====================================\n")

# ==========================================
# 4. THE KALMAN FILTER LOOP
# ==========================================
for step, y in enumerate(gps_measurements):

    # A) Save the current prediction (To match the professor's Mathcad logic)
    current_position = x[0, 0]
    kf_position_estimates.append(current_position)

    # B) Calculate Kalman Gain (K)
    # K determines who to trust: the math model or the GPS measurement?
    K = A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + Sz)

    # C) Update State (Prediction + Correction)
    # New state = (Expected state based on physics) + K * (Difference between GPS and Expected GPS)
    x = A @ x + B * u + K @ (y - C @ x)

    # D) Update Covariance (Uncertainty)
    # We recalculate the error considering the new measurement and process noise (Sw)
    P = A @ P @ A.T + Sw - A @ P @ C.T @ np.linalg.inv(Sz) @ C @ P @ A.T

    # --- EDUCATIONAL PRINTS (Only for the first 3 steps to understand the process) ---
    if step < 3:
        print(f"--- STEP {step} (Time: {step * T:.1f}s) ---")
        print(f"1. GPS Measured Position: {y:.3f} m")
        print(f"2. Kalman Gain (K): \n{K}")
        print(f"3. Updated State (x) [Pos, Vel]: \n{x}")
        print(f"4. Updated Uncertainty (P): \n{P}\n")

# ==========================================
# 5. RESULTS AND VERIFICATION
# ==========================================
df_results = pd.DataFrame({
    'Our_KF': kf_position_estimates,
    'Prof_KF': prof_reference
})
df_results['Difference'] = df_results['Our_KF'] - df_results['Prof_KF']

print("=== FINAL VERIFICATION ===")
print("Max absolute difference with Professor's reference:", df_results['Difference'].abs().max())
print("==========================\n")

# ==========================================
# 6. PLOTTING: GENERAL RESULTS
# ==========================================
plt.figure(figsize=(10, 6))
# Cambiamos a dimgray y subimos el alpha para que el GPS se vea perfectamente
plt.plot(gps_measurements, label='GPS Measurement (Noisy)', color='green', alpha=0.8, linewidth=1)
plt.plot(kf_position_estimates, label='Our KF Estimation', color='red', linewidth=2.5)
plt.title('Kalman Filter Results')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 7. EXTRAS: SIMPLE MOVING AVERAGE
# ==========================================
# Apply a Simple Moving Average (SMA) filter with a window of 10
window_size = 10
sma_estimates = pd.Series(gps_measurements).rolling(window=window_size, min_periods=1).mean()

# Plot zoomed-in comparison (Last 200 steps)
plt.figure(figsize=(10, 6))
plt.plot(gps_measurements[800:1000], label='GPS (Noisy)', color='green', alpha=0.8, linewidth=1)
plt.plot(kf_position_estimates[800:1000], label='Kalman Filter (Ours)', color='red', linewidth=2.5)
plt.plot(sma_estimates[800:1000].values, label=f'Simple Moving Average (Window={window_size})', color='blue', linestyle='--', linewidth=2)

plt.title('Kalman Filter vs Simple Moving Average (Zoom: Steps 800-1000)')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 8. EXTRAS: SUPER ZOOM GRAPH
# ==========================================

# 3. "Super Zoom" plot at the beginning (Steps 0 to 50) to clearly expose the noise
plt.figure(figsize=(10, 6))

# Added small dots (marker='o') to visualize each individual GPS measurement point
plt.plot(gps_measurements[:50], label='GPS (Noisy)', color='green', alpha=0.6, linewidth=1.5, marker='o', markersize=4)
plt.plot(kf_position_estimates[:50], label='Kalman Filter (Ours)', color='red', linewidth=3)
plt.plot(sma_estimates[:50].values, label=f'SMA (Window={window_size})', color='blue', linestyle='--', linewidth=2)

plt.title('Super Zoom (Steps 0-50): GPS Noise Impact at Startup')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.show()