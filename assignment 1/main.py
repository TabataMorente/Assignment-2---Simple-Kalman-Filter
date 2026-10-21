import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import anderson, shapiro

# The 3 archives CSV with the data
archives = ['datos_base.csv', 'datos_X.csv', 'datos_Y.csv']

for archive in archives:
    print(f"\n=========================================")
    print(f"   RESULTS FOR: {archive}")
    print(f"=========================================")

    try:
        df = pd.read_csv(archive)
    except FileNotFoundError:
        print(f"{archive} not found")
        continue

    # Calculate mean and standard deviation from each row
    print("\n--- MEAN AND STANDARD DEVIATION ---")
    print(df.agg(['mean', 'std']).T)

    # Creates 6x6 covariance matrix
    print("\n--- COVARIANCE MATRIX (6x6) ---")
    print(df.cov())

    # Calculates Pearson correlation values
    print("\n--- PEARSON CORRELATION ---")
    print(df.corr(method='pearson'))

    # Anderson-Darling normality test
    print("\n--- NORMALITY TEST (Anderson-Darling (critical value) & Shapiro-Wilk (p-value)) ---")
    for col in df.columns:
        #Anderson-Darling -> Critical Values
        result = anderson(df[col])

        #Shapiro-Wilk -> p-values
        stat_sw, p_value = shapiro(df[col])
        print(f"[{col}] Anderson: {result.statistic:.3f} (Crit: {result.critical_values[2]:.3f}) | p-value: {p_value:.3e}")
    # Drawing the histogram
    df.hist(bins=30, figsize=(10, 8), color='skyblue', edgecolor='black')
    plt.suptitle(f"Histogramas - {archive}")
    plt.tight_layout()
    plt.show()