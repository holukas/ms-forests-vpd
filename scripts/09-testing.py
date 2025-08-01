import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Let's create a hypothetical dataset for this example
np.random.seed(0)  # for reproducibility
data = {
    'vpd_zscore': np.concatenate([np.random.normal(loc=-1, scale=0.5, size=100), np.random.normal(loc=1.5, scale=0.5, size=100)]),
    'temp_zscore': np.concatenate([np.random.normal(loc=-1, scale=0.5, size=100), np.random.normal(loc=1.5, scale=0.5, size=100)]),
    'other_features': np.random.rand(200)
}
X = pd.DataFrame(data)
y = 10 / (1 + np.exp(-2 * (X['vpd_zscore'] - X['temp_zscore']))) + np.random.normal(0, 0.5, 200)

# Train an XGBoost model
model = xgb.XGBRegressor(n_estimators=100, random_state=0).fit(X, y)

# Define a robust prediction wrapper to handle numpy arrays from shap.KernelExplainer
def predict_wrapper(data_array):
    # Convert numpy array back to a DataFrame with the original column names
    data_df = pd.DataFrame(data_array, columns=X.columns)
    return model.predict(data_df)

# 1. Standard SHAP (TreeExplainer)
explainer_standard = shap.Explainer(model, X)
shap_values_standard = explainer_standard(X.iloc[150:151, :])

# 2. Conditional SHAP (KernelExplainer)
background = shap.kmeans(X, 10).data

# Pass our robust prediction wrapper to the explainer
explainer_kernel = shap.KernelExplainer(predict_wrapper, background)
shap_values_kernel = explainer_kernel.shap_values(X.iloc[150:151, :])

# Get the SHAP values and feature names for a single instance
instance_to_explain = X.iloc[150:151, :]
shap_kernel = shap_values_kernel[0]

# --- Matplotlib Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

feature_names = X.columns
index_to_plot = 0

# Get the SHAP values for the instance
shap_standard = shap_values_standard.values[index_to_plot]

# Create bar chart
bar_width = 0.35
index = np.arange(len(feature_names))

bar1 = ax.bar(index, shap_standard, bar_width, label='Standard SHAP')
bar2 = ax.bar(index + bar_width, shap_kernel, bar_width, label='Conditional SHAP')

# Customize plot
ax.set_xlabel('Features')
ax.set_ylabel('SHAP Value')
ax.set_title('Comparison of Standard vs. Conditional SHAP Values')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(feature_names)
ax.legend()
plt.tight_layout()
plt.show()