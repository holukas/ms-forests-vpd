import src.files as files

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

# Create parquet files
print(siteinfo_df)

from collections import Counter
print(Counter(siteinfo_df['IGBP']))
siteinfo_df['ORIGIN+IGBP'] = siteinfo_df['ORIGIN'] + "_" + siteinfo_df['IGBP']
print(Counter(siteinfo_df['ORIGIN+IGBP']))
# set(siteinfo_df['SWC_VAR'].tolist())
# siteinfo_df[siteinfo_df['SWC_VAR'] != '-MISSING-']
# Counter(siteinfo_df['_DATE_LAST'].dt.year)
