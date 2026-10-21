import csv
import math
import statistics

pitches_rad = []
rolls_rad = []

print("Procesando 1000 lecturas del CSV...")

# Abrir el archivo CSV
with open('sensor_data2.csv', 'r') as file:
    # Usar DictReader para leer por los nombres de las columnas
    reader = csv.DictReader(file)
    for row in reader:
        # Extraer las aceleraciones (asumiendo los nombres de columna que pusimos)
        ax = float(row['Accel_X_ms2'])
        ay = float(row['Accel_Y_ms2'])
        az = float(row['Accel_Z_ms2'])

        # Calcular ángulos en RADIANES
        pitch = math.atan2(abs(ax), math.sqrt(ay ** 2 + az ** 2))
        roll = math.atan2(abs(ay), abs(az))

        pitches_rad.append(pitch)
        rolls_rad.append(roll)

# Calcular estadísticas
pitch_mean = statistics.mean(pitches_rad)
pitch_std = statistics.stdev(pitches_rad)

roll_mean = statistics.mean(rolls_rad)
roll_std = statistics.stdev(rolls_rad)

print("\n--- RESULTADOS PARA EL INFORME (En Radianes) ---")
print(f"PITCH -> Media: {pitch_mean:.6f} rad  |  Desviación Estándar: {pitch_std:.6f} rad")
print(f"ROLL  -> Media: {roll_mean:.6f} rad  |  Desviación Estándar: {roll_std:.6f} rad")

print("\n(Para comprobar) En grados:")
print(f"Pitch Media: {math.degrees(pitch_mean):.1f}° | Roll Media: {math.degrees(roll_mean):.1f}°")