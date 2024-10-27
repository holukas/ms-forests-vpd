import funcs.files as fun

# Collect site stats from ICOS L2 2024
sites_icos_df = fun.collect_sitestats(
    filepattern=r"ICOSETC_*_FLUXNET_HH_L2.csv.parquet",
    searchdirs=r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\2-FLUXNET_HH_PARQUET",
    source_info="ICOS_L2",
    testing=True
)

# Add site info from ICOS L2 2024
sites_icos_df = fun.collect_siteinfo(
    sites_df=sites_icos_df,
    filepattern=r"ICOSETC_*_SITEINFO_L2.csv",
    sourcedir=r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\3-SITEINFO",
    testing=True
)

# # Collect site stats from FLUXNET WW2020
# sites_fxn_df = fun.collect_sitestats(
#     filepattern=r"FLX_*_FLUXNET2015_FULLSET_HH_*_beta-3.csv.parquet",
#     searchdirs=r"L:\Sync\luhk_work\40 - DATA\Datasets\2022 - FLUXNET - Warm Winter 2020 - release 2022-1\2-FLUXNET_HH_PARQUET",
#     source_info="FLUXNET_WW2020",
#     testing=False
# )
#
# a = sites_icos_df['SITE'].tolist()
# b = sites_fxn_df['SITE'].tolist()
#
# in_both = [c for c in a if c in b]
# only_icos = [c for c in a if c not in b]
# only_fxn = [c for c in b if c not in a]
#
# print(in_both)

print(sites_icos_df)
sites_icos_df.to_csv("OUT/11.1-site_info_ICOS.csv")
