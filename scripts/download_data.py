# download_data.py
# author: Limor Winter
# date: 2025-12-02

import click
import requests
import zipfile
import os

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--data_path', type=str, help="Path to the data directory")

def main(url, data_path):
    """Download and extract the UCI Mushroom dataset, then check its format."""
    response = requests.get(url)

    os.makedirs(data_path, exist_ok=True)

    zip_path = os.path.join(data_path, "mushroom.zip")

    with open(zip_path, 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)

        
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

    # ext = os.path.splitext(data_path)[1].lower()
    # assert ext in correct_exts, (
    #     f"[File format] '{data_path}' has extension '{ext}', "
    #     f"expected one of {sorted(correct_exts)}."
    # )

if __name__ == '__main__':
    main()