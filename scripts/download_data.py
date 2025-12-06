# download_data.py
# author: Limor Winter
# date: 2025-12-02

import click
import requests
import zipfile
import os

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--data_path', type=str, help="Path to the data file to check")

def main(url, data_path):
    """Download and extract the UCI Mushroom dataset, then check its format."""
    response = requests.get(url)
    with open("data/raw/mushroom.zip", 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile("data/raw/mushroom.zip", 'r') as zip_ref:
        zip_ref.extractall("data/raw")

        
    correct_exts = {".csv",".data"}
    ext = os.path.splitext(data_path)[1].lower()
    assert ext in correct_exts, (
        f"[File format] '{data_path}' has extension '{ext}', "
        f"expected one of {sorted(correct_exts)}."
    )

if __name__ == '__main__':
    main()