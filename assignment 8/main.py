import pandas as pd
import numpy as np

np.random.seed(42)

# Define the experiment parameters: 3 operators x 5 IMUs x 3 trials
operators = ['Student_1', 'Student_2', 'Student_3']
imus = ['IMU_1', 'IMU_2', 'IMU_3', 'IMU_4', 'IMU_5']
trials = [1, 2, 3]

# Base values for the 5 IMUs (Part Variation - PV)
# Each sensor has its own factory bias relative to the perfect 90º angle
imu_base = {
    'IMU_1': 90.1,
    'IMU_2': 89.4,
    'IMU_3': 90.8,
    'IMU_4': 89.1,
    'IMU_5': 90.5
}

# Bias of the 3 Operators (Appraiser Variation - AV)
op_bias = {'Student_1': 0.05, 'Student_2': -0.15, 'Student_3': 0.10}

# Equipment Variation (EV)
# Standard deviation of the noise in each reading
noise_std = 0.2

# Data generation
data = []
for op in operators:
    for imu in imus:
        for trial in trials:
            # Final value = IMU base + Human bias + Random noise
            val = imu_base[imu] + op_bias[op] + np.random.normal(0, noise_std)
            data.append([op, imu, trial, round(val, 2)])

# Create the dataframe
df = pd.DataFrame(data, columns=['Operator', 'IMU', 'Trial', 'Angle (º)'])

# Show the results
print(df.to_string(index=False))