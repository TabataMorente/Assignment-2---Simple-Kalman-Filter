import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA AND CALCULATE METRICS
# ==========================================
# Cargar el CSV
df = pd.read_csv('datos_imu_spc.csv')

# Agrupar por subgrupo y calcular Media, Rango y Desviación Estándar
df_spc = df.groupby('Subgrupo')['Accel_Z'].agg(
    X_bar='mean',
    R=lambda x: x.max() - x.min(),
    S='std'
).reset_index()

# ==========================================
# 2. STATISTIC CONSTANTS FOR n = 10
# ==========================================
n = 10
A2 = 0.308  # For X-bar limits using R
A3 = 0.975  # For X-bar limits using S
D3 = 0.223  # For lower limit of R
D4 = 1.777  # For upper limit of R
B3 = 0.284  # For lower limit of S
B4 = 1.716  # For upper limit of S

# ==========================================
# 3. CALCULATE LIMITS WITH ONLY BASE LINE (1 to 25)
# ==========================================
fase_base = df_spc[df_spc['Subgrupo'] <= 25]

# Central lines (Means of base line metrics)
X_dbl_bar = fase_base['X_bar'].mean()
R_bar = fase_base['R'].mean()
S_bar = fase_base['S'].mean()

# X-bar Control limits (using range)
UCL_X = X_dbl_bar + (A2 * R_bar)
LCL_X = X_dbl_bar - (A2 * R_bar)

# Calculate the width of 1 sigma for drawing zones A, B and C of the rules
sigma_X = (UCL_X - X_dbl_bar) / 3

# R and S control limits
UCL_R = D4 * R_bar
LCL_R = D3 * R_bar
UCL_S = B4 * S_bar
LCL_S = B3 * S_bar


# ==========================================
# 4. FUNCTION FOR DRAWING GRAPHS
# ==========================================
def plot_chart(ax, data, CL, UCL, LCL, title, ylabel, sigma=None):
    ax.plot(df_spc['Subgrupo'], data, marker='o', color='blue', linestyle='-', markersize=6)

    # Central line y control limits
    ax.axhline(CL, color='green', linestyle='-', linewidth=2, label='Center (CL)')
    ax.axhline(UCL, color='red', linestyle='--', linewidth=2, label='Limits (UCL/LCL)')
    ax.axhline(LCL, color='red', linestyle='--', linewidth=2)

    # Draw zones (just for X-bar) in order to see the "Flagging"
    if sigma is not None:
        ax.axhspan(CL - sigma, CL + sigma, color='green', alpha=0.2, label='C Zone')
        ax.axhspan(CL + sigma, CL + 2 * sigma, color='yellow', alpha=0.2, label='B Zone')
        ax.axhspan(CL - 2 * sigma, CL - sigma, color='yellow', alpha=0.2)
        ax.axhspan(CL + 2 * sigma, UCL, color='red', alpha=0.2, label='A Zone')
        ax.axhspan(LCL, CL - 2 * sigma, color='red', alpha=0.2)

    # Other color for perturbation zone (Subgroups > 25)
    ax.axvspan(25.5, 40.5, color='gray', alpha=0.1, label='Perturbation zone')
    ax.axvline(25.5, color='black', linestyle=':', linewidth=2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9)


# ==========================================
# 5. CREATE COMPLETE FIGURE
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

# X-bar chart
plot_chart(ax1, df_spc['X_bar'], X_dbl_bar, UCL_X, LCL_X,
           'Mean chart (X-bar)', 'Accel_Z Mean', sigma=sigma_X)

# R chart
plot_chart(ax2, df_spc['R'], R_bar, UCL_R, LCL_R,
           'Range chart (R)', 'Accel_Z Range')

# S chart
plot_chart(ax3, df_spc['S'], S_bar, UCL_S, LCL_S,
           'Standard Deviation chart (S)', 'Accel_Z Standard Deviation')

ax3.set_xlabel('Subgroup number', fontsize=12)
plt.xticks(range(1, 41))
plt.tight_layout()

# Save img
plt.savefig('graficos_spc_imu.png', dpi=300)
plt.show()