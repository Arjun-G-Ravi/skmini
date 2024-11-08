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
        elif zipfile.is_zipfile(file_path): # TODO: change to endswith in the future, which doesnt require us loading a library
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    os.remove(file_path)
                    os.mkdir(file_path)
                    zip_ref.extractall(file_path)
            except: print('Decompression Failed.')
        print(file_path)

        import gzip
        import shutil

        def is_gzip(file_path):
            with open(file_path, 'rb') as f:
                return f.read(2) == b'\x1f\x8b'


        # file_path = str(file_path)
        import gzip
        import shutil
        import os

        def is_gzip(file_path):
            with open(file_path, 'rb') as f:
                return f.read(2) == b'\x1f\x8b'

        if is_gzip(file_path):
            print('is gzip')
            try:
                decompressed_path = file_path[:-3]  # Remove '.gz' from the filename
                with gzip.open(file_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                # os.remove(file_path)  # Uncomment to remove the original .gz file if needed
                print("Decompression successful.")
            except Exception as e:
                print(f"Decompression Failed: {e}")
        else:
            print("The file is not gzip encoded.")


        # if is_gzip(file_path):
        #     try:
        #         print('gzip')
        #         with gzip.open(file_path, 'rb') as f_in:
        #                 with open(file_path[:-3], 'wb') as f_out:
        #                     shutil.copyfileobj(f_in, f_out)
        #         os.remove(file_path)

        #     except:
        #         print('Decompression Failed.')
        # else:
        #     print('other format')

        # elif file_path.endswith('.gz'):
        #     print('here')
        #     import gzip
        #     import shutil

        #     if file_path.endswith('.gz'):
        #         try:
        #             with gzip.open(file_path, 'rb') as f_in:
        #                 with open(file_path[:-3], 'wb') as f_out:
        #                     shutil.copyfileobj(f_in, f_out)
        #             os.remove(file_path)
        #         except:
        #             print('Decompression Failed.')
        #     print('other format')
    return file_path