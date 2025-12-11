# read_zip.py
# author: Limor Winter
# date: 2025-12-10

import requests
import zipfile
import os


def read(url, directory):
    """
    Download a file from a given URL and save it locally.
    
    Parameters
    ----------
    url : str
        The URL of the zip file to be read.
    directory : str
        The directory where the contents of the zip file will be extracted.
    Returns
    -------
    None
    """
    
    response = requests.get(url)
    filename_from_url = os.path.basename(url)
    
    # check if URL exists, if not raise an error
    if response.status_code != 200:
        raise ValueError('The URL provided does not exist.')
    
    # check if the URL points to a zip file, if not raise an error  
    #if request.headers['content-type'] != 'application/zip':
    if filename_from_url[-4:] != '.zip':
        raise ValueError('The URL provided does not point to a zip file.')

    # create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # write the zip file to the directory
    zip_path = os.path.join(directory, "mushroom.zip")
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # get list of files/directories in the directory
    original_files = os.listdir(directory)
    original_timestamps = []
    for filename in original_files:
        filename = os.path.join(directory, filename)
        original_timestamp = os.path.getmtime(filename)
        original_timestamps.append(original_timestamp)


    # extract the zip file to the directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(directory)

         # check if any files were extracted, if not raise an error
    # get list of files/directories in the directory
    current_files = os.listdir(directory)
    current_timestamps = []
    for filename in current_files:
        filename = os.path.join(directory, filename)
        current_timestamp = os.path.getmtime(filename)
        current_timestamps.append(current_timestamp)
    if (len(current_files) == len(original_files)) & (original_timestamps == current_timestamps):
        raise ValueError('The ZIP file is empty.')

