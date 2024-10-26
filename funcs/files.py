from diive.core.io.filereader import search_files, ReadFileType
from diive.core.io.files import save_parquet


def convert_datafiles_to_parquet(filepattern: str, filetype: str, searchdirs: str, outpath: str):
    filelist = search_files(
        searchdirs=searchdirs,
        pattern=filepattern)

    for f in filelist:
        filename = f.name
        loaddatafile = ReadFileType(filetype=filetype, filepath=f, data_nrows=None)
        data_df, metadata_df = loaddatafile.get_filedata()
        filepath = save_parquet(filename=filename, data=data_df, outpath=outpath)
        print(f"Saved file {filename} to {filepath}.")
