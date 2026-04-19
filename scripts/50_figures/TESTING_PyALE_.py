"""
ALE (Accumulated Local Effects) Visualization Test

ALE plots show the isolated effect of a feature on model predictions,
accounting for correlations with other features. Unlike partial dependence plots
(which show marginal effects), ALE plots show conditional effects by measuring
how predictions change as a feature varies within its local range.

This script demonstrates ALE on synthetic NEP data influenced by VPD and Temperature.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from PyALE import ale
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
ta = np.random.normal(20, 5, n)
vpd = 0.1 * ta + np.random.normal(1, 0.5, n)
nep = 20 - (2 * vpd) + (0.5 * ta) + np.random.normal(0, 1, n)

X = pd.DataFrame({'VPD': vpd, 'TA': ta})
y = nep

model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
print(f"Model R² score: {model.score(X, y):.4f}")

# Calculate ALE for VPD
plt.figure(figsize=(10, 6))
ale(X=X, model=model, feature=['VPD'], grid_size=20)
plt.title('ALE Plot: Effect of VPD on NEP')
plt.tight_layout()
plt.savefig('ale_vpd_plot.png', dpi=150)
print("Plot saved as 'ale_vpd_plot.png'")
plt.show()