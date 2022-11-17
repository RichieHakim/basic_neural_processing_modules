
import pickle
import json
import yaml
from pathlib import Path
import math

from tqdm import tqdm

def pickle_save(obj, path_save, mode='wb', mkdir=False, allow_overwrite=True):
    Path(path_save).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path_save).exists(), f'{path_save} already exists.'
    assert Path(path_save).parent.exists(), f'{Path(path_save).parent} does not exist.'
    assert Path(path_save).parent.is_dir(), f'{Path(path_save).parent} is not a directory.'
    with open(path_save, mode) as f:
        pickle.dump(obj, f,)
def pickle_load(filename, mode='rb'):
    with open(filename, mode) as f:
        return pickle.load(f)

def json_save(obj, path_save, mode='w', mkdir=False, allow_overwrite=True):
    Path(path_save).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path_save).exists(), f'{path_save} already exists.'
    assert Path(path_save).parent.exists(), f'{Path(path_save).parent} does not exist.'
    assert Path(path_save).parent.is_dir(), f'{Path(path_save).parent} is not a directory.'
    with open(path_save, mode) as f:
        json.dump(obj, f, indent=4)
def json_load(filename, mode='r'):
    with open(filename, mode) as f:
        return json.load(f)

def yaml_save(obj, path_save, mode='w', mkdir=False, allow_overwrite=True):
    Path(path_save).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path_save).exists(), f'{path_save} already exists.'
    assert Path(path_save).parent.exists(), f'{Path(path_save).parent} does not exist.'
    assert Path(path_save).parent.is_dir(), f'{Path(path_save).parent} is not a directory.'
    with open(path_save, mode) as f:
        yaml.dump(obj, f, indent=4)
def yaml_load(filename, mode='r'):
    with open(filename, mode) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
        


def hash_file(path, type_hash='MD5', buffer_size=65536):
    """
    Gets hash of a file.
    Based on: https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    RH 2022

    Args:
        path (str):
            Path to file to be hashed.
        type_hash (str):
            Type of hash to use. Can be:
                'MD5'
                'SHA1'
                'SHA256'
                'SHA512'
        buffer_size (int):
            Buffer size for reading file.
            65536 corresponds to 64KB.

    Returns:
        hash_val (str):
            Hash of file.
    """
    import hashlib

    if type_hash == 'MD5':
        hasher = hashlib.md5()
    elif type_hash == 'SHA1':
        hasher = hashlib.sha1()
    elif type_hash == 'SHA256':
        hasher = hashlib.sha256()
    elif type_hash == 'SHA512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f'{type_hash} is not a valid hash type.')

    with open(path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hasher.update(data)

    hash_val = hasher.hexdigest()
        
    return hash_val


def download_file(
    url, 
    path_save, 
    check_local_first=True, 
    check_hash=False, 
    hash_type='MD5', 
    hash_hex=None,
    mkdir=False,
    write_mode='wb',
    verbose=True,
    chunk_size=1024,
):
    """
    Download a file from a URL to a local path using requests.
    Allows for checking if file already exists locally and
    checking the hash of the downloaded file against a provided hash.
    RH 2022

    Args:
        url (str):
            URL of file to download.
        path_save (str):
            Path to save file to.
        check_local_first (bool):
            If True, checks if file already exists locally.
            If True and file exists locally, skips download.
            If True and check_hash is True, checks hash of local file.
        check_hash (bool):
            If True, checks hash of local or downloaded file against
             hash_hex.
        hash_type (str):
            Type of hash to use. Can be:
                'MD5', 'SHA1', 'SHA256', 'SHA512'
        hash_hex (str):
            Hash to compare to. In hex format (e.g. 'a1b2c3d4e5f6...').
            Can be generated using hash_file() or hashlib and .hexdigest().
            If check_hash is True, hash_hex must be provided.
        mkdir (bool):
            If True, creates parent directory of path_save if it does not exist.
        write_mode (str):
            Write mode for saving file. Should be one of:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (write binary, fail if file exists)
        verbose (bool):
            If True, prints status messages.
        chunk_size (int):
            Size of chunks to download file in.
    """
    import os
    import requests

    # Check if file already exists locally
    if check_local_first:
        if os.path.isfile(path_save):
            print(f'File already exists locally: {path_save}') if verbose else None
            # Check hash of local file
            if check_hash:
                hash_local = hash_file(path_save, type_hash=hash_type)
                if hash_local == hash_hex:
                    print('Hash of local file matches provided hash_hex.') if verbose else None
                    return True
                else:
                    print('Hash of local file does not match provided hash_hex.') if verbose else None
                    print(f'Hash of local file: {hash_local}') if verbose else None
                    print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
                    print('Downloading file...') if verbose else None
            else:
                return True
        else:
            print(f'File does not exist locally: {path_save}. Will attempt download from {url}') if verbose else None

    # Download file
    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        print(e)
        return False
    # Check response
    if response.status_code != 200:
        print(f'Response status code: {response.status_code}')
        return False
    # Create parent directory if it does not exist
    if mkdir:
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
    # Download file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    wrote = 0
    with open(path_save, write_mode) as f:
        with tqdm(total=total_size, disable=(verbose==False), unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size):
                wrote = wrote + len(data)
                f.write(data)
                pbar.update(len(data))
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        return False
    # Check hash
    if check_hash:
        hash_local = hash_file(path_save, type_hash=hash_type)
        if hash_local == hash_hex:
            print('Hash of downloaded file matches hash_hex.') if verbose else None
            return True
        else:
            print('Hash of downloaded file does not match hash_hex.') if verbose else None
            print(f'Hash of downloaded file: {hash_local}') if verbose else None
            print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
            return False
    else:
        return True

def is_valid_hash(hash_hex, hash_type='MD5'):
    """
    Checks if a hash is valid.
    RH 2022

    Args:
        hash_hex (str):
            Hash to check. In hex format (e.g. 'a1b2c3d4e5f6...').
        hash_type (str):
            Type of hash to use. Can be:
                'MD5', 'SHA1', 'SHA256', 'SHA512'

    Returns:
        bool:
            True if hash is valid, False if not.
    """
    import hashlib

    if hash_type == 'MD5':
        hasher = hashlib.md5()
    elif hash_type == 'SHA1':
        hasher = hashlib.sha1()
    elif hash_type == 'SHA256':
        hasher = hashlib.sha256()
    elif hash_type == 'SHA512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f'{hash_type} is not a valid hash type.')

    try:
        hasher.update(bytes.fromhex(hash_hex))
    except ValueError:
        return False
    return True


def extract_zip(path_zip, path_extract, verbose=True):
    """
    Extracts a zip file to a specified path.
    RH 2022

    Args:
        path_zip (str):
            Path to zip file.
        path_extract (str):
            Path to extract zip file to.
        verbose (bool):
            If True, prints status messages.
    """
    import zipfile

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        print(f'Extracting {path_zip} to {path_extract}') if verbose else None
        zip_ref.extractall(path_extract)
