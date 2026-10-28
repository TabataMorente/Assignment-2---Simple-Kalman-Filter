import numpy as np

# 1. Define the nominal parameters and deviations
vin = 100
n_simulations = 10000

# R1: mean 200, std 3.333
r1_sim = np.random.normal(200, 3.3333, n_simulations)
# R2: mean 50, std 0.833
r2_sim = np.random.normal(50, 0.8333, n_simulations)

# 2. Calculate Vout for the 10,000 cases
vout_sim = vin * (r2_sim / (r1_sim + r2_sim))

# 3. Show results
print(f"Vout mean: {np.mean(vout_sim):.4f} V")
print(f"Vout standard deviation (Monte Carlo): {np.std(vout_sim):.4f} V")