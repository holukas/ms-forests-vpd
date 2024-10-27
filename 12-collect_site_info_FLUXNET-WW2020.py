import funcs.files as fun

# Collect site stats
sites_fxn_df = fun.collect_sitestats(
    filepattern=r"FLX_*_FLUXNET2015_FULLSET_HH_*_beta-3.csv.parquet",
    searchdirs=r"L:\Sync\luhk_work\40 - DATA\Datasets\2022 - FLUXNET - Warm Winter 2020 - release 2022-1\2-FLUXNET_HH_PARQUET",
    source_info="FLUXNET_WW2020",
    testing=True
)

# # Add site info from ICOS
# sites_fxn_df = fun.collect_siteinfo(
#     sites_df=sites_fxn_df,
#     filepattern=r"ICOSETC_*_SITEINFO_L2.csv",
#     sourcedir=r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\3-SITEINFO",
#     testing=True
# )

print(sites_fxn_df)
sites_fxn_df.to_csv("OUT/12.1-site_info_FLUXNET-WW2020.csv")
