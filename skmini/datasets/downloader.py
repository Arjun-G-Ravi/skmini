import os
from pathlib import Path
import requests
import zipfile
import tarfile
import shutil
import gzip
from warnings import filterwarnings
import sys
filterwarnings('ignore')

def download_to_cache(url, filename,format=None, force_download=False):
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
                try:
                    shutil.rmtree(file_path)  # Remove the directory and all its contents
                    print(f"Successfully deleted the directory and its contents: {file_path}")
                except:
                    try:
                        os.remove(file_path)
                        print('Successfully deleted the file')
                    except: print('Couldn\'t delete existing file')
        except OSError as e:
            print(f"Error: {e.strerror}")

        response = requests.get(url, stream=True, verify=False)
        print('Downloading dataset...')
        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, "wb") as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    from .helper import show_progress_bar
                    show_progress_bar(downloaded, total_size)
        print(f"\nDownloaded file to {file_path}")

        def is_gzip(file_path):
            with open(file_path, 'rb') as f:
                return f.read(2) == b'\x1f\x8b'
        # now that it is downloaded, we might want to decompress the file
        if tarfile.is_tarfile(file_path):
            try:
                tar = tarfile.open(file_path)
                os.remove(file_path)
                tar.extractall(path = file_path)
                tar.close()
            except:
                print('Decompression Failed.')

        elif zipfile.is_zipfile(file_path): # TODO: change to endswith in the future, which doesnt require us loading a library
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    os.remove(file_path)
                    os.mkdir(file_path)
                    zip_ref.extractall(file_path)
            except: print('Decompression Failed.')


        elif is_gzip(file_path):
            zip_path =  file_path.parent / (f'{filename}' + format)
            try:
                with gzip.open(file_path, 'rb') as f_in:
                    with open(zip_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(file_path)  # Uncomment to remove the original .gz file if needed
                print("Decompression successful.")
            except Exception as e:
                print(f"Decompression Failed: {e}")
    return file_path