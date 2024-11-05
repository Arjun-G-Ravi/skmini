import os
from pathlib import Path
import requests
import zipfile
import tarfile
import shutil
from warnings import filterwarnings
filterwarnings('ignore')
def download_to_cache(url, filename, force_download=False):
    def _get_custom_cache_dir():
        custom_cache_dir = Path.home() / ".cache"/ "skmini"/ "datasets"
        os.makedirs(custom_cache_dir, exist_ok=True)
        return custom_cache_dir

    file_path = _get_custom_cache_dir() / filename
    if os.path.isfile(file_path) or os.path.isdir(file_path):
        if not force_download:
            print('Warning: File Aldready exists. Use force_download=True to force download.') # I should maybe remove this line as it can be annoying
    else:
        force_download=True

    if force_download:
        try:
            if os.path.isfile(file_path) or os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and all its contents
                print(f"Successfully deleted the directory and its contents: {file_path}")
        except OSError as e:
            print(f"Error: {e.strerror}")

        response = requests.get(url, stream=True, verify=False)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded file to {file_path}")
        # now that it is downloaded, we might want to decompress the file
        if tarfile.is_tarfile(file_path):
            try:
                tar = tarfile.open(file_path)
                os.remove(file_path)
                tar.extractall(path = file_path)
                tar.close()
            except:
                print('Decompression Failed.')
        elif zipfile.is_zipfile(file_path): # change to endswith, which doesnt require us loading a library
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    os.remove(file_path)
                    os.mkdir(file_path)
                    zip_ref.extractall(file_path)
            except: print('Decompression Failed.')
    return file_path