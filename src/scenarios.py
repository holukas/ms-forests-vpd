import numpy as np
import pandas as pd


def calculate_scenario_stats(df_input, igbp, scenario_order, vars, shap_suffix_avg, shap_suffix_sd):
    scen_data = []

    for scen_id in scenario_order:
        df_scen = df_input[df_input['SCENARIO'] == scen_id].copy()
        if len(df_scen) == 0:
            raise ValueError(f"Scenario {scen_id} has no data.")

        scen_dict = {}

        # Individual drivers (calculate mean for bars)
        for var in vars:

            # Required cols
            col_mean = var + shap_suffix_avg
            col_sd = var + shap_suffix_sd
            req_cols = [col_mean, col_sd]
            # Check if ALL required columns are available
            if not set(req_cols).issubset(df_scen.columns):
                missing = list(set(req_cols) - set(df_scen.columns))
                raise ValueError(f"Required columns missing from dataframe: {missing}")

            # Site-level vectors
            sitemeans_vec = df_scen[col_mean]
            sitemeans_sd_vec = df_scen[col_sd]

            # Calculate stats for site-means (for the bar height)
            sitemeans_mean = sitemeans_vec.mean()
            sitemeans_count = sitemeans_vec.count()
            sitemeans_sem = sitemeans_vec.sem()  # SEM of site-means (standard error of the mean), sem = std / sqrt(n)
            sitemeans_min = sitemeans_vec.min()
            sitemeans_max = sitemeans_vec.max()

            # Calculate the SD of site-means
            # Law of total variance
            mean_of_variances = (sitemeans_sd_vec ** 2).mean()  # Average of within-site variances
            variance_of_means = sitemeans_vec.var(ddof=0)  # Variance of the site means
            total_sd = np.sqrt(mean_of_variances + variance_of_means)  # Global SD

            # Store in dict
            scen_dict[col_mean] = {
                'igbp': igbp,
                'scenario': scen_id,
                'mean': sitemeans_mean,
                'sem': sitemeans_sem,
                'total_sd': total_sd,
                'min': sitemeans_min,
                'max': sitemeans_max,
                'n_sites': sitemeans_count
            }

        # -------------------------------
        # Net effect (sum of SHAP values)
        # -------------------------------

        # Required net cols
        col_net_shapvals = 'NET_SHAPVALS_OVR_AVG'
        col_net_shapvals_sd = 'NET_SHAPVALS_OVR_SD'
        req_net_cols = [col_net_shapvals, col_net_shapvals_sd]
        # Check if ALL required columns are available
        if not set(req_net_cols).issubset(df_scen.columns):
            missing = list(set(req_net_cols) - set(df_scen.columns))
            raise ValueError(f"Required columns missing from dataframe: {missing}")

        # Site-level vectors
        net_shapvals_vec = df_scen[col_net_shapvals]
        net_shapvals_sd_vec = df_scen[col_net_shapvals_sd]

        # Calculate net mean stats
        net_shapvals_mean = net_shapvals_vec.mean()
        net_shapvals_count = net_shapvals_vec.count()
        net_shapvals_sem = net_shapvals_vec.sem()  # sem = std / sqrt(n)
        net_shapvals_min = net_shapvals_vec.min()
        net_shapvals_max = net_shapvals_vec.max()

        # Total SD
        # Calculate total SD using the law of total variance (sqrt(mean of variances + variance of means)).
        # This approach treats sites as equally representative (macro-average), normalizing
        # differences in sample counts between sites.
        net_mean_of_variances = (net_shapvals_sd_vec ** 2).mean()  # Average of within-site variances
        net_variance_of_means = net_shapvals_vec.var(ddof=0)  # Variance of the site means
        net_total_sd = np.sqrt(net_mean_of_variances + net_variance_of_means)

        # Store in dict
        scen_dict['NET_SHAPVALS'] = {
            'igbp': igbp,
            'scenario': scen_id,
            'mean': net_shapvals_mean,
            'sem': net_shapvals_sem,
            'total_sd': net_total_sd,
            'min': net_shapvals_min,
            'max': net_shapvals_max,
            'n_sites': net_shapvals_count
        }

        # Collect
        scen_data.append(scen_dict)

    rows = []
    for entry in scen_data:
        for var_name, stats in entry.items():
            # Create a new row dictionary from the stats
            row = stats.copy()
            # Add the variable name as a column
            row['Variable'] = var_name
            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns for clarity
    cols = ['Variable', 'igbp', 'scenario', 'mean', 'sem', 'total_sd', 'min', 'max', 'n_sites']
    df = df[cols]

    return df


def scenario_0(df):
    """Complete dataset, -1 = unrestricted"""
    condition = "all data"
    return df, -1, -1, -1, condition


def scenario_1(df, a: float = 0.6745):
    """Normal conditions, 0 = normal conditions"""
    # 50% of data (z-score = +/- 0.6745)
    mask_ta = (df['TA_ZSCORE'] >= -a) & (df['TA_ZSCORE'] <= a)
    mask_vpd = (df['VPD_ZSCORE'] >= -a) & (df['VPD_ZSCORE'] <= a)
    mask_swc = (df['SWC_ZSCORE'] >= -a) & (df['SWC_ZSCORE'] <= a)
    combined_mask = mask_ta & mask_vpd & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "normal conditions"
    return df, 0, 0, 0, condition


def scenario_2(df, a: float = 0.6745, c: float = 1.25):
    """Warmer conditions, soil moisture normal"""
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= c)
    mask_swc = (df['SWC_ZSCORE'] >= -a) & (df['SWC_ZSCORE'] <= a)
    combined_mask = mask_ta & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "warm"
    return df, 1, -1, 0, condition


def scenario_3(df, a: float = 0.6745, c: float = 1.25):
    """Warmer conditions, drier soil moisture"""
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= c)
    mask_swc = (df['SWC_ZSCORE'] >= -c) & (df['SWC_ZSCORE'] < -a)
    combined_mask = mask_ta & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "warm and dry"
    return df, 1, -1, 1, condition


def scenario_4(df, a: float = 0.6745, c: float = 1.25):
    """Hot conditions, drier soil moisture"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_swc = (df['SWC_ZSCORE'] >= -c) & (df['SWC_ZSCORE'] < -a)
    combined_mask = mask_ta & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "hot and dry"
    return df, 2, -1, 1, condition


def scenario_5(df, c: float = 1.25):
    """Compound extreme: extremely hot, extremely dry soil conditions"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_swc = df['SWC_ZSCORE'] < -c
    combined_mask = mask_ta & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "hot and driest"
    return df, 2, -1, 2, condition


def scenario_6(df, c: float = 1.25):
    """Compound extreme: extremely hot, extremely dry soil and atmosphere conditions"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_vpd = df['VPD_ZSCORE'] > c
    mask_swc = df['SWC_ZSCORE'] < -c
    combined_mask = mask_ta & mask_vpd & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "hot and 2 driest"
    return df, 2, 2, 2, condition

# def scenario_4(df, a: float = 0.6745, c: float = 1.25):
#     """Dry atmosphere conditions"""
#     mask_vpd = (df['VPD_ZSCORE'] > a) & (df['VPD_ZSCORE'] <= c)
#     df = df.loc[mask_vpd].copy()
#     condition = "dry atmosphere"
#     return df, 0, 1, 0, condition


# def scenario_4(df, a: float = 0.6745, c: float = 1.25):
#     """Compound conditions with hot air, dry soil and dry atmosphere conditions"""
#     mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= c)
#     mask_vpd = (df['VPD_ZSCORE'] > a) & (df['VPD_ZSCORE'] <= c)
#     mask_swc = (df['SWC_ZSCORE'] >= -c) & (df['SWC_ZSCORE'] < -a)
#     combined_mask = mask_ta & mask_vpd & mask_swc
#     df = df.loc[combined_mask].copy()
#     condition = "hot and dry"
#     return df, 1, 1, 1, condition


# def scenario_5(df, c: float = 1.25):
#     """Compound extreme: extremely hot, extremely dry conditions"""
#     mask_ta = df['TA_ZSCORE'] > c
#     mask_vpd = df['VPD_ZSCORE'] > c
#     mask_swc = df['SWC_ZSCORE'] < -c
#     combined_mask = mask_ta & mask_vpd & mask_swc
#     df = df.loc[combined_mask].copy()
#     condition = "extremely hot and dry"
#     return df, 2, 2, 2, condition
