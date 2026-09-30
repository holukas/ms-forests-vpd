import numpy as np
import pandas as pd


def calculate_stage_stats(df_input, igbp, stage_order, vars, shap_suffix_avg, shap_suffix_sd):
    scen_data = []

    for stage_id in stage_order:
        df_scen = df_input[df_input['SCENARIO'] == stage_id].copy()
        if len(df_scen) == 0:
            raise ValueError(f"Stage {stage_id} has no data.")

        scen_dict = {}

        # Individual drivers (calculate mean for bars)
        for var in vars:

            # Required cols
            col_shap_avg = var + shap_suffix_avg
            col_shap_sd = var + shap_suffix_sd
            req_cols = [col_shap_avg, col_shap_sd]
            # Check if ALL required columns are available
            if not set(req_cols).issubset(df_scen.columns):
                missing = list(set(req_cols) - set(df_scen.columns))
                raise ValueError(f"Required columns missing from dataframe: {missing}")

            # Site-level vectors
            sites_shap_avg = df_scen[col_shap_avg]
            sites_shap_sd = df_scen[col_shap_sd]

            # Calculate stats for site-means (for the bar height)
            stage_shap_avg = sites_shap_avg.mean()
            stage_shap_avg_counts = sites_shap_avg.count()
            stage_shap_avg_sem = sites_shap_avg.sem()  # SEM of site-means (standard error of the mean), sem = std / sqrt(n)
            stage_shap_avg_min = sites_shap_avg.min()
            stage_shap_avg_max = sites_shap_avg.max()

            # Calculate the SD of site-means
            # Law of total variance
            mean_of_variances = (sites_shap_sd ** 2).mean()  # Average of within-site variances
            variance_of_means = sites_shap_avg.var(ddof=0)  # Variance of the site means
            stage_shap_avg_total_sd = np.sqrt(mean_of_variances + variance_of_means)  # Global SD

            # Store in dict
            scen_dict[col_shap_avg] = {
                'igbp': igbp,
                'stage': stage_id,
                'mean': stage_shap_avg,
                'sem': stage_shap_avg_sem,
                'total_sd': stage_shap_avg_total_sd,
                'min': stage_shap_avg_min,
                'max': stage_shap_avg_max,
                'n_sites': stage_shap_avg_counts
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
            'stage': stage_id,
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
    cols = ['Variable', 'igbp', 'stage', 'mean', 'sem', 'total_sd', 'min', 'max', 'n_sites']
    df = df[cols]

    return df


def stage_0(df):
    """Complete dataset, -1 = unrestricted"""
    condition = "all data"
    return df, -1, -1, -1, condition


def stage_1(df, a: float = 0.31863936):
    """Stage 1, normal conditions: TA, VPD and SWC all within the middle 25% (|z| <= a)."""
    mask_ta = (df['TA_ZSCORE'] >= -a) & (df['TA_ZSCORE'] <= a)
    mask_vpd = (df['VPD_ZSCORE'] >= -a) & (df['VPD_ZSCORE'] <= a)
    mask_swc = (df['SWC_ZSCORE'] >= -a) & (df['SWC_ZSCORE'] <= a)
    combined_mask = mask_ta & mask_vpd & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "normal conditions"
    return df, 0, 0, 0, condition


def stage_2(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    # def stage_2(df, a: float = 0.31863936, b: float = 0.93458929, c: float = 1.2815515655446):
    """Warmer conditions, soil moisture normal, NO extreme VPD

    z-score +/- 0.31863936 = middle 25%
    z-score +/- 0.714367440280187 = next 13.75% above or below middle 25%
    z-score +/- 1.2815515655446 = upper 10% (90th percentile) or lower 10% (10th percentile)
    """
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= b)
    mask_swc = (df['SWC_ZSCORE'] >= -a) & (df['SWC_ZSCORE'] <= a)
    mask_vpd = df['VPD_ZSCORE'] <= c  # Prevents overlap with compound extremes
    combined_mask = mask_ta & mask_swc & mask_vpd
    df = df.loc[combined_mask].copy()
    condition = "warm"
    return df, 1, -1, 0, condition


def stage_3(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    # def stage_3(df, a: float = 0.31863936, b: float = 0.93458929, c: float = 1.2815515655446):
    """Warmer conditions, drier soil moisture, NO extreme VPD"""
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= b)
    mask_swc = (df['SWC_ZSCORE'] >= -b) & (df['SWC_ZSCORE'] < -a)
    mask_vpd = df['VPD_ZSCORE'] <= c
    combined_mask = mask_ta & mask_swc & mask_vpd
    df = df.loc[combined_mask].copy()
    condition = "warm and dry"
    return df, 1, -1, 1, condition


def stage_4(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    # def stage_4(df, a: float = 0.31863936, b: float = 0.93458929, c: float = 1.2815515655446):
    """Hot conditions, drier soil moisture, NO extreme VPD"""
    mask_ta = (df['TA_ZSCORE'] > b) & (df['TA_ZSCORE'] <= c)
    mask_swc = (df['SWC_ZSCORE'] >= -b) & (df['SWC_ZSCORE'] < -a)
    mask_vpd = df['VPD_ZSCORE'] <= c
    combined_mask = mask_ta & mask_swc & mask_vpd
    df = df.loc[combined_mask].copy()
    condition = "hot and dry"
    return df, 2, -1, 1, condition


def stage_5(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    # def stage_5(df, a: float = 0.31863936, b: float = 0.93458929, c: float = 1.2815515655446):
    """Hot conditions, very dry soil moisture, NO extreme VPD"""
    mask_ta = (df['TA_ZSCORE'] > b) & (df['TA_ZSCORE'] <= c)
    mask_swc = (df['SWC_ZSCORE'] >= -c) & (df['SWC_ZSCORE'] < -b)
    mask_vpd = df['VPD_ZSCORE'] <= c
    combined_mask = mask_ta & mask_swc & mask_vpd
    df = df.loc[combined_mask].copy()
    condition = "hot and driest"
    return df, 2, -1, 2, condition


def stage_6(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    # def stage_6(df, a: float = 0.31863936, b: float = 0.93458929, c: float = 1.2815515655446):
    """Extremely hot conditions, very dry soil moisture, NO extreme VPD"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_swc = (df['SWC_ZSCORE'] >= -c) & (df['SWC_ZSCORE'] < -b)
    mask_vpd = df['VPD_ZSCORE'] <= c
    combined_mask = mask_ta & mask_swc & mask_vpd
    df = df.loc[combined_mask].copy()
    condition = "hot and driest"
    return df, 3, -1, 2, condition


def stage_7(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    # def stage_7(df, a: float = 0.31863936, b: float = 0.93458929, c: float = 1.2815515655446):
    """Extremely hot conditions, extremely dry soil moisture, NO extreme VPD"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_swc = df['SWC_ZSCORE'] < -c
    mask_vpd = df['VPD_ZSCORE'] <= c
    combined_mask = mask_ta & mask_swc & mask_vpd
    df = df.loc[combined_mask].copy()
    condition = "hot and driest"
    return df, 3, -1, 3, condition


def stage_8(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    # def stage_8(df, a: float = 0.31863936, b: float = 0.93458929, c: float = 1.2815515655446):
    """Compound extreme: extremely hot, extremely dry soil and atmosphere conditions"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_swc = df['SWC_ZSCORE'] < -c
    mask_vpd = df['VPD_ZSCORE'] > c
    combined_mask = mask_ta & mask_vpd & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "hot and 2 driest"
    return df, 3, 3, 3, condition


# ---------------------------------------------------------------------------
# Mirrored stage sequence
# ---------------------------------------------------------------------------
# The main sequence lets temperature rise and soil water fall step by step
# while holding VPD below the extreme cut-off, and only adds extreme VPD at the
# last stage. Any driver held back to the last stage can look dramatic when it
# finally arrives, so that sequence may favor VPD.
#
# These stages swap the two roles. VPD escalates through the sequence and soil
# water is kept off its extreme until the last stage, where extreme dryness is
# added. Everything else, including the temperature ladder and the cut-offs, is
# unchanged, so the two sequences differ only in which driver comes last.
#
# Read them against the originals: mirror_2 is stage_2 with VPD and SWC swapped,
# and so on. A comparison of the two final stages shows whether the order matters.


def mirror_1(df, a: float = 0.31863936):
    """Normal conditions. Identical to stage_1, the sequences share a start."""
    return stage_1(df, a=a)


def mirror_2(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Warmer conditions, VPD normal, NO extremely dry soil"""
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= b)
    mask_vpd = (df['VPD_ZSCORE'] >= -a) & (df['VPD_ZSCORE'] <= a)
    mask_swc = df['SWC_ZSCORE'] >= -c  # Prevents overlap with compound extremes
    df = df.loc[mask_ta & mask_vpd & mask_swc].copy()
    return df, 1, 0, -1, "warm, VPD normal"


def mirror_3(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Warmer conditions, higher VPD, NO extremely dry soil"""
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= b)
    mask_vpd = (df['VPD_ZSCORE'] > a) & (df['VPD_ZSCORE'] <= b)
    mask_swc = df['SWC_ZSCORE'] >= -c
    df = df.loc[mask_ta & mask_vpd & mask_swc].copy()
    return df, 1, 1, -1, "warm and higher VPD"


def mirror_4(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Hot conditions, higher VPD, NO extremely dry soil"""
    mask_ta = (df['TA_ZSCORE'] > b) & (df['TA_ZSCORE'] <= c)
    mask_vpd = (df['VPD_ZSCORE'] > a) & (df['VPD_ZSCORE'] <= b)
    mask_swc = df['SWC_ZSCORE'] >= -c
    df = df.loc[mask_ta & mask_vpd & mask_swc].copy()
    return df, 2, 1, -1, "hot and higher VPD"


def mirror_5(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Hot conditions, high VPD, NO extremely dry soil"""
    mask_ta = (df['TA_ZSCORE'] > b) & (df['TA_ZSCORE'] <= c)
    mask_vpd = (df['VPD_ZSCORE'] > b) & (df['VPD_ZSCORE'] <= c)
    mask_swc = df['SWC_ZSCORE'] >= -c
    df = df.loc[mask_ta & mask_vpd & mask_swc].copy()
    return df, 2, 2, -1, "hot and high VPD"


def mirror_6(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Extremely hot conditions, high VPD, NO extremely dry soil"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_vpd = (df['VPD_ZSCORE'] > b) & (df['VPD_ZSCORE'] <= c)
    mask_swc = df['SWC_ZSCORE'] >= -c
    df = df.loc[mask_ta & mask_vpd & mask_swc].copy()
    return df, 3, 2, -1, "hottest and high VPD"


def mirror_7(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Extremely hot conditions, extreme VPD, NO extremely dry soil"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_vpd = df['VPD_ZSCORE'] > c
    mask_swc = df['SWC_ZSCORE'] >= -c
    df = df.loc[mask_ta & mask_vpd & mask_swc].copy()
    return df, 3, 3, -1, "hottest and extreme VPD"


def mirror_8(df, a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Compound extreme. Same records as stage_8, reached in the opposite order."""
    return stage_8(df, a=a, b=b, c=c)


# ---------------------------------------------------------------------------
# Factorial cells
# ---------------------------------------------------------------------------
# Neither stage sequence is a factorial design: each one escalates one driver while
# the other is only kept off its extreme. These cells cross four soil water classes
# with four VPD classes at the stage cut-offs and leave temperature free, so every
# combination is present and no driver comes first. The cells at central VPD hold
# VPD near its baseline while soil water falls, and the cells at central soil water
# do the same for VPD.
#
# Class 0 is the middle 25% (|z| <= a), classes 1 to 3 step away from it at b and c
# in the direction of stress: upward for VPD, downward for soil water. Records with
# wet soil or low VPD fall in no cell.

SWC_CLASS_NAMES = ('central', 'dry', 'very dry', 'extreme')
VPD_CLASS_NAMES = ('central', 'higher', 'high', 'extreme')


def _stress_class_mask(z, k, a, b, c):
    """Records in class k of a driver, with z already signed so that stress is positive."""
    if k == 0:
        return (z >= -a) & (z <= a)
    if k == 1:
        return (z > a) & (z <= b)
    if k == 2:
        return (z > b) & (z <= c)
    return z > c


def factorial_cell(swc_k: int, vpd_k: int,
                   a: float = 0.31863936, b: float = 0.714367440280187, c: float = 1.2815515655446):
    """Stage-like function for one cell: soil water class swc_k, VPD class vpd_k, TA free."""

    def cell(df):
        mask_swc = _stress_class_mask(-df['SWC_ZSCORE'], swc_k, a, b, c)
        mask_vpd = _stress_class_mask(df['VPD_ZSCORE'], vpd_k, a, b, c)
        df = df.loc[mask_swc & mask_vpd].copy()
        condition = f"SM {SWC_CLASS_NAMES[swc_k]}, VPD {VPD_CLASS_NAMES[vpd_k]}"
        return df, -1, vpd_k, swc_k, condition

    cell.__name__ = f"factorial_swc{swc_k}_vpd{vpd_k}"
    return cell


# Sixteen cells, soil water class first, so cells 0 to 3 are central soil water
FACTORIAL_CELLS = [factorial_cell(swc_k, vpd_k) for swc_k in range(4) for vpd_k in range(4)]
