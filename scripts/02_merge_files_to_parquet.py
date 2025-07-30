import src.files as files

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

# Create parquet files
data_nrows = None  # for testing
_siteinfo_df = siteinfo_df.copy()
for ix, siteconfig in _siteinfo_df.iterrows():
    siteinfo_df = files.create_parquet_files(
        siteinfo_df=siteinfo_df,
        data_nrows=data_nrows,
        settings=settings,
        siteconfig=siteconfig,
        ix=ix
    )

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, settings=settings)
