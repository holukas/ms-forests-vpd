import numpy as np
import scipy.stats as stats

import numpy as np


def calc_threshold(x_fit, y_fit, pi_lower, pi_upper):
    """
    Find the threshold and its 95% interval as zero crossings of the fit and its bands.

    Each crossing is linearly interpolated. With several crossings, the highest
    x value is returned. Returns (threshold, from pi_lower, from pi_upper).
    """

    def find_highest_zero_crossing(x, y):
        # Find all indices where the sign of y changes from positive to negative (or vice versa)
        sign_changes = np.where(np.diff(np.sign(y)))[0]

        if len(sign_changes) == 0:
            return np.nan  # Return NaN if the curve never crosses zero

        roots = []
        for i in sign_changes:
            # Perform a precise linear interpolation exactly between the two points crossing zero
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]

            # Interpolation formula: x = x0 - y0 * (x1 - x0) / (y1 - y0)
            root = x0 - y0 * (x1 - x0) / (y1 - y0)
            roots.append(root)

        # Select and return the maximum x-coordinate root
        return max(roots)

    # Apply the root-finding logic to the main fit and both confidence intervals
    threshold_main = find_highest_zero_crossing(x_fit, y_fit)
    threshold_lower = find_highest_zero_crossing(x_fit, pi_lower)
    threshold_upper = find_highest_zero_crossing(x_fit, pi_upper)

    # Print the results for your manuscript text
    print(f"VPD Threshold: {threshold_main:.2f} sigma (95% CI: [{threshold_lower:.2f}, {threshold_upper:.2f}])")

    return threshold_main, threshold_lower, threshold_upper


def fit_polynomial(X_data, Y_data):
    # Fit polynomial
    degree = 4
    # poly_coeffs = np.polyfit(X_data, Y_data, degree)
    poly_coeffs, residuals, _, _, _ = np.polyfit(X_data, Y_data, degree, full=True)
    poly_func = np.poly1d(poly_coeffs)



    # Calculate r2
    ss_res = residuals[0]  # The sum of squared residuals (SS_res) is the first element of the residuals array
    y_mean = np.mean(Y_data)
    ss_tot = np.sum((Y_data - y_mean) ** 2)  # Calculate the total sum of squares (SS_tot)
    r_squared = 1 - (ss_res / ss_tot)

    # Create x-range for plotting fitted curve and create fit values
    x_fit = np.linspace(X_data.min(), X_data.max(), 1000)
    y_fit = poly_func(x_fit)

    # Calculate the prediction interval
    # ----------------------------------------------------
    n = len(X_data)  # Number of data points
    p = degree + 1  # Number of parameters (coefficients)
    alpha = 0.05  # 95% prediction interval

    # Calculate AIC
    k = p  # number of parameters (degree + 1)
    aic = n * np.log(residuals[0] / n) + 2 * k

    # Calculate Mean Squared Error (MSE)
    mse = residuals[0] / (n - p)

    # Create Vandermonde matrix for the original data
    X_vander = np.vander(X_data, p)

    # Calculate the covariance matrix of the coefficients
    covariance_matrix = mse * np.linalg.inv(X_vander.T @ X_vander)

    # Create Vandermonde matrix for the fitted line
    x_fit_vander = np.vander(x_fit, p)

    # Calculate the standard error of the fitted line at each point
    se_fit = np.sqrt(np.diag(x_fit_vander @ covariance_matrix @ x_fit_vander.T))

    # Calculate the standard error for prediction intervals (adds the MSE)
    se_pred = np.sqrt(se_fit ** 2 + mse)

    # Get the t-statistic for a 95% confidence level
    t_value = stats.t.ppf(1 - alpha / 2, n - p)

    # Calculate the prediction interval bounds
    pi_upper = y_fit + t_value * se_pred
    pi_lower = y_fit - t_value * se_pred
    # ----------------------------------------------------

    return poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower
