import src.files as files
import src.stats as stats

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

# # Number of IGBPs
# # {'ENF': 72, 'DBF': 47, 'MF': 12, 'DNF': 2, 'EBF': 3, 'OSH': 1}
# from collections import Counter
# counts_igbps = Counter(siteinfo_df['IGBP'])
# print(dict(counts_igbps))

# Calculate basic stats
_siteinfo_df = siteinfo_df.copy()
for ix, siteconfig in _siteinfo_df.iterrows():
    siteinfo_df = stats.basic_stats(
        siteinfo_df=siteinfo_df,
        siteconfig=siteconfig,
        ix=ix)

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, settings=settings)
