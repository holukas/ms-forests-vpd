def scenario_0(df):
    # Complete dataset
    condition = "all data"
    return df, -1, -1, -1, condition


def scenario_1(df, a: float = 0.6745):
    # 50% of data (z-score = +/- 0.6745)
    mask_ta = (df['TA_ZSCORE'] >= -a) & (df['TA_ZSCORE'] <= a)
    mask_vpd = (df['VPD_ZSCORE'] >= -a) & (df['VPD_ZSCORE'] <= a)
    mask_swc = (df['SWC_ZSCORE'] >= -a) & (df['SWC_ZSCORE'] <= a)
    combined_mask = mask_ta & mask_vpd & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "normal condition"
    return df, 0, 0, 0, condition

def scenario_2(df, a: float = 0.6745, c: float = 1.25):
    """Dry soil conditions"""
    mask_swc = (df['SWC_ZSCORE'] >= -c) & (df['SWC_ZSCORE'] < -a)
    df = df.loc[mask_swc].copy()
    condition = "dry soil"
    return df, 0, 0, 1, condition


def scenario_3(df, a: float = 0.6745, c: float = 1.25):
    """Hot conditions"""
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= c)
    df = df.loc[mask_ta].copy()
    condition = "hot"
    return df, 1, 0, 0, condition

# def scenario_4(df, a: float = 0.6745, c: float = 1.25):
#     """Dry atmosphere conditions"""
#     mask_vpd = (df['VPD_ZSCORE'] > a) & (df['VPD_ZSCORE'] <= c)
#     df = df.loc[mask_vpd].copy()
#     condition = "dry atmosphere"
#     return df, 0, 1, 0, condition


def scenario_4(df, a: float = 0.6745, c: float = 1.25):
    """Compound conditions with hot air, dry soil and dry atmosphere conditions"""
    mask_ta = (df['TA_ZSCORE'] > a) & (df['TA_ZSCORE'] <= c)
    mask_vpd = (df['VPD_ZSCORE'] > a) & (df['VPD_ZSCORE'] <= c)
    mask_swc = (df['SWC_ZSCORE'] >= -c) & (df['SWC_ZSCORE'] < -a)
    combined_mask = mask_ta & mask_vpd & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "hot and dry"
    return df, 1, 1, 1, condition


def scenario_5(df, c: float = 1.25):
    """Compound extreme: extremely hot, extremely dry conditions"""
    mask_ta = df['TA_ZSCORE'] > c
    mask_vpd = df['VPD_ZSCORE'] > c
    mask_swc = df['SWC_ZSCORE'] < -c
    combined_mask = mask_ta & mask_vpd & mask_swc
    df = df.loc[combined_mask].copy()
    condition = "extremely hot and dry"
    return df, 2, 2, 2, condition
