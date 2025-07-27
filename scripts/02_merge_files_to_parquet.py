import src.files as files

# Load site info
siteinfo_df = files.load_siteinfo(filename="01_siteinfo.csv")

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Create parquet files
data_nrows = None  # for testing
siteinfo_df = files.create_parquet_files(siteinfo_df=siteinfo_df, data_nrows=data_nrows, settings=settings)

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, filename="02_siteinfo.csv")
