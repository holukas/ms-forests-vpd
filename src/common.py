def get_variable_names(siteconfig):
    """
    Get variable names for this site.

    Args:
        siteconfig (dict): A dictionary containing site configuration.

    Returns:
        dict: A dictionary of variable names.
    """
    return {
        'nee_var': str(siteconfig['NEE_VAR']),
        'nee_qc_var': str(siteconfig['NEE_QC_VAR']),
        'le_var': str(siteconfig['LE_VAR']),
        'le_qc_var': str(siteconfig['LE_QC_VAR']),
        'gpp_var': str(siteconfig['GPP_VAR']),
        'reco_var': str(siteconfig['RECO_VAR']),
        'swin_var': str(siteconfig['SWIN_VAR']),
        'ta_var': str(siteconfig['TA_VAR']),
        'vpd_var': str(siteconfig['VPD_VAR']),
        'swc_var': str(siteconfig['SWC_VAR']),
        'swinpot_var': 'SW_IN_POT',
        'prec_var': 'P_F',
        'rh_var': str(siteconfig['RH_VAR']),
    }
