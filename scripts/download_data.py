# download_data.py

"""
This module provides functions to download a remote ZIP/CSV,
validate the HTTP response, and write extracted artifacts to disk with safe
defaults (no overwrite by default).
"""


import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_zip import read

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--data_path', type=str, help="Path to the data directory")

def main(url, data_path):
    """ 
    Download and extract the UCI Mushroom dataset, then verify its format.

    This function downloads a remote dataset (ZIP or CSV), extracts it to the
    specified directory using the `read` helper, and checks that at least one
    file with a valid extension (.csv or .data) exists afterward. If no such
    file is found, an informative error is raised.

    Parameters
    ----------
    url : str
        HTTP(S) URL pointing to a ZIP archive or a single CSV file.
    data_path : str
        Directory where files will be written. Created if it does not exist.

    Returns
    -------
    None
        This function is used for downloading and writing files.

     Raises
    ------
    ValueError
        If the provided URL is invalid (handled internally by `read`).
    FileNotFoundError
        If extraction completes but no valid file (.csv or .data) is found in
        `data_path`.
"""

    read(url, data_path)
    correct_exts = {".csv",".data"}
    found_file = False

    for filename in os.listdir(data_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in correct_exts:
            found_file = True
            print(f"Verification successful: Found valid file '{filename}'")
            break
    
    if not found_file:
        raise FileNotFoundError(
            f"Extraction failed to produce a file with extensions {sorted(correct_exts)} "
            f"in directory: '{data_path}'"
        )

if __name__ == '__main__':
    main()