NB_TRADING_HOURS_PER_DAY = 7
NB_SAMPLE_PER_HOUR = 60 / 5  # 5mins ticks for vol estimates
NB_DAYS_PER_YEAR = 252

VIX_T = [1 / 52, 1 / 12, 3 / 12, 6 / 12]
VIX_K = [16, 20, 25, 30, 50, 70]
VIX_IVS = [
    [0.7, 0.75, 0.88, 1.15, 1.4, 1.6],
    [0.65, 0.7, 0.8, 1.02130222, 1.2, 1.3],
    [0.5, 0.55, 0.65, 0.79597744, 0.95, 1.15],
    [0.4, 0.45, 0.6, 0.71671509, 0.9, 1.05],
]

SPX_T = [
    1 / NB_DAYS_PER_YEAR,
    1 / 52,
    2 / 52,
    1 / 12,
    2 / 12,
    3 / 12,
    6 / 12,
    9 / 12,
    1.0,
]
SPX_K = [4000, 4500, 4800, 4900, 5000, 5100, 5200, 5500]
SPX_IVS = [
    [0.70, 0.40, 0.33, 0.28, 0.26, 0.24, 0.25, 0.29],
    [0.55, 0.37, 0.26, 0.24, 0.23, 0.22, 0.24, 0.25],
    [0.48, 0.34, 0.27, 0.24, 0.23, 0.21, 0.22, 0.26],
    [0.42, 0.32, 0.24, 0.22, 0.20, 0.22, 0.22, 0.25],
    [0.35, 0.30, 0.22, 0.20, 0.19, 0.18, 0.21, 0.24],
    [0.35, 0.29, 0.21, 0.18, 0.18, 0.19, 0.20, 0.23],
    [0.33, 0.27, 0.22, 0.20, 0.19, 0.19, 0.20, 0.24],
    [0.32, 0.28, 0.24, 0.20, 0.19, 0.18, 0.19, 0.23],
    [0.32, 0.25, 0.22, 0.17, 0.18, 0.18, 0.19, 0.23],
]

# To plot:
# 
# import numpy as np
# import matplotlib.pyplot as plt

# T = np.array(SPX_T)
# K = np.array(SPX_K)
# ivs = np.array(SPX_IVS)

# X, Y = np.meshgrid(K, T)
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, ivs, cmap='coolwarm')
# ax.set_xlabel('Strikes (K)')
# ax.set_ylabel('Maturities (T)')
# ax.set_zlabel('Implied Volatility')
# ax.set_title('Implied Volatility Surface')

# plt.show()

# import pandas as pd

# SPX_MOCK = pd.DataFrame(SPX_IVS, index=SPX_T, columns=SPX_K)
# SPX_MOCK = SPX_MOCK.stack().reset_index(name='iv').rename(columns={'level_0': 'T', 'level_1':'K'})