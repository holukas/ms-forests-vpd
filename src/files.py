import fnmatch
import os
from pathlib import Path

import yaml


def read_settings_file(filepath_settings):
    """Read start values from settings file as strings into dict, with same variable names as in file"""
    with open(filepath_settings, 'r', encoding='utf-8') as f:
        settings_dict = yaml.safe_load(f)
    return settings_dict


def search_files(searchdirs: str or list, pattern: str) -> list:
    """ Search files and store their filename and the path to the file in dictionary. """
    # found_files_dict = {}
    foundfiles = []
    if isinstance(searchdirs, str):
        searchdirs = [searchdirs]  # Use str as list
    for searchdir in searchdirs:
        for root, dirs, files in os.walk(searchdir):
            for idx, settings_file_name in enumerate(files):
                if fnmatch.fnmatch(settings_file_name, pattern):
                    filepath = Path(root) / settings_file_name
                    # found_files_dict[settings_file_name] = filepath
                    foundfiles.append(filepath)
    foundfiles.sort()
    return foundfiles
