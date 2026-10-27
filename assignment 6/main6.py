import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CARGAR DATOS Y CALCULAR MÉTRICAS
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
# 2. CONSTANTES ESTADÍSTICAS PARA n = 10
# ==========================================
n = 10
A2 = 0.308  # Para límites de X-bar usando R
A3 = 0.975  # Para límites de X-bar usando S
D3 = 0.223  # Para límite inferior de R
D4 = 1.777  # Para límite superior de R
B3 = 0.284  # Para límite inferior de S
B4 = 1.716  # Para límite superior de S

# ==========================================
# 3. CALCULAR LÍMITES SOLO CON LA FASE BASE (1 al 25)
# ==========================================
fase_base = df_spc[df_spc['Subgrupo'] <= 25]

# Líneas Centrales (Medias de las métricas de la fase base)
X_dbl_bar = fase_base['X_bar'].mean()
R_bar = fase_base['R'].mean()
S_bar = fase_base['S'].mean()

# Límites de Control X-bar (usando el rango)
UCL_X = X_dbl_bar + (A2 * R_bar)
LCL_X = X_dbl_bar - (A2 * R_bar)

# Calcular el ancho de 1 sigma para pintar las zonas (A, B, C) de las reglas
sigma_X = (UCL_X - X_dbl_bar) / 3

# Límites de Control R y S
UCL_R = D4 * R_bar
LCL_R = D3 * R_bar
UCL_S = B4 * S_bar
LCL_S = B3 * S_bar


# ==========================================
# 4. FUNCIÓN PARA DIBUJAR LOS GRÁFICOS
# ==========================================
def plot_chart(ax, data, CL, UCL, LCL, title, ylabel, sigma=None):
    ax.plot(df_spc['Subgrupo'], data, marker='o', color='blue', linestyle='-', markersize=6)

    # Línea central y límites de control
    ax.axhline(CL, color='green', linestyle='-', linewidth=2, label='Centro (CL)')
    ax.axhline(UCL, color='red', linestyle='--', linewidth=2, label='Límites (UCL/LCL)')
    ax.axhline(LCL, color='red', linestyle='--', linewidth=2)

    # Dibujar zonas (solo para X-bar) para ver el "Flagging"
    if sigma is not None:
        ax.axhspan(CL - sigma, CL + sigma, color='green', alpha=0.2, label='Zona C')
        ax.axhspan(CL + sigma, CL + 2 * sigma, color='yellow', alpha=0.2, label='Zona B')
        ax.axhspan(CL - 2 * sigma, CL - sigma, color='yellow', alpha=0.2)
        ax.axhspan(CL + 2 * sigma, UCL, color='red', alpha=0.2, label='Zona A')
        ax.axhspan(LCL, CL - 2 * sigma, color='red', alpha=0.2)

    # Sombrear la zona de perturbación (Subgrupos > 25)
    ax.axvspan(25.5, 40.5, color='gray', alpha=0.1, label='Fase Perturbación')
    ax.axvline(25.5, color='black', linestyle=':', linewidth=2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9)


# ==========================================
# 5. CREAR LA FIGURA COMPLETA
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

# Gráfico X-bar
plot_chart(ax1, df_spc['X_bar'], X_dbl_bar, UCL_X, LCL_X,
           'Gráfico de Medias (X-bar)', 'Media Accel_Z', sigma=sigma_X)

# Gráfico R
plot_chart(ax2, df_spc['R'], R_bar, UCL_R, LCL_R,
           'Gráfico de Rangos (R)', 'Rango Accel_Z')

# Gráfico S
plot_chart(ax3, df_spc['S'], S_bar, UCL_S, LCL_S,
           'Gráfico de Desviación Estándar (S)', 'Desviación Estándar Accel_Z')

ax3.set_xlabel('Número de Subgrupo', fontsize=12)
plt.xticks(range(1, 41))  # Mostrar números del 1 al 40 en el eje X
plt.tight_layout()

# Guardar la imagen para ponerla en tu Word
plt.savefig('graficos_spc_imu.png', dpi=300)
plt.show()