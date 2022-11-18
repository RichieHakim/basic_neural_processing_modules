
import pickle
import json
import yaml
from pathlib import Path
import zipfile

from tqdm import tqdm


def prepare_filepath_for_saving(path, mkdir=False, allow_overwrite=True):
    """
    Checks if a file path is valid.
    RH 2022

    Args:
        path (str):
            Path to check.
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path).exists(), f'{path} already exists.'
    assert Path(path).parent.exists(), f'{Path(path).parent} does not exist.'
    assert Path(path).parent.is_dir(), f'{Path(path).parent} is not a directory.'


def pickle_save(
    obj, 
    path_save, 
    mode='wb', 
    zipCompress=False, 
    kwargs_zipfile=None,
    mkdir=False, 
    allow_overwrite=True
):
    """
    Saves an object to a pickle file.
    Uses pickle.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        path_save (str):
            Path to save object to.
        mode (str):
            Mode to open file in.
            Can be:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (exclusive write binary. Raises FileExistsError if file already exists.)
        zipCompress (bool):
            If True, compresses pickle file using zipfileCompressionMethod.
            This is similar to savez_compressed in numpy (with zipfile.ZIP_DEFLATED),
             and is useful for saving redundant and/or sparse arrays objects.
        kwargs_zipfile (dict):
            Keyword arguments to pass to zipfile.ZipFile.
            See https://docs.python.org/3/library/zipfile.html#zipfile-objects.
            By default, uses compression=zipfile.ZIP_DEFLATED.
            Other options for compression are:
                zipfile.ZIP_STORED (no compression)
                zipfile.ZIP_DEFLATED (usual zip compression)
                zipfile.ZIP_BZIP2 (bzip2 compression) (usually not as good as ZIP_DEFLATED)
                zipfile.ZIP_LZMA (lzma compression) (usually better than ZIP_DEFLATED but slower)
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.        
    """
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)

    if kwargs_zipfile is None:
        kwargs_zipfile = {
            'compression': zipfile.ZIP_DEFLATED,
        }

    if zipCompress:
        with zipfile.ZipFile(path_save, 'w', **kwargs_zipfile) as f:
            f.writestr('data', pickle.dumps(obj))
    else:
        with open(path_save, mode) as f:
            pickle.dump(obj, f)

def pickle_load(
    filename, 
    zipCompressed=False,
    mode='rb'
):
    """
    Loads a pickle file.
    RH 2022

    Args:
        filename (str):
            Path to pickle file.
        zipCompressed (bool):
            If True, then file is assumed to be a .zip file.
            This function will first unzip the file, then
             load the object from the unzipped file.
        mode (str):
            Mode to open file in.

    Returns:
        obj (object):
            Object loaded from pickle file.
    """
    if zipCompressed:
        with zipfile.ZipFile(filename, 'r') as f:
            return pickle.loads(f.read('data'))
    else:
        with open(filename, mode) as f:
            return pickle.load(f)


def json_save(obj, path_save, indent=4, mode='w', mkdir=False, allow_overwrite=True):
    """
    Saves an object to a json file.
    Uses json.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        path_save (str):
            Path to save object to.
        mode (str):
            Mode to open file in.
            Can be:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (exclusive write binary. Raises FileExistsError if file already exists.)
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.        
    """
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path_save, mode) as f:
        json.dump(obj, f, indent=indent)

def json_load(filename, mode='r'):
    """
    Loads a json file.
    RH 2022

    Args:
        filename (str):
            Path to pickle file.
        mode (str):
            Mode to open file in.

    Returns:
        obj (object):
            Object loaded from pickle file.
    """
    with open(filename, mode) as f:
        return json.load(f)


def yaml_save(obj, path_save, indent=4, mode='w', mkdir=False, allow_overwrite=True):
    """
    Saves an object to a yaml file.
    Uses yaml.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        path_save (str):
            Path to save object to.
        mode (str):
            Mode to open file in.
            Can be:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (exclusive write binary. Raises FileExistsError if file already exists.)
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.        
    """
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path_save, mode) as f:
        yaml.dump(obj, f, indent=indent)

def yaml_load(filename, mode='r', loader=yaml.FullLoader):
    """
    Loads a yaml file.
    RH 2022

    Args:
        filename (str):
            Path to pickle file.
        mode (str):
            Mode to open file in.
        loader (yaml.Loader):
            Loader to use.
            Can be:
                yaml.FullLoader: Loads the full YAML language. Avoids arbitrary code execution. Default for PyYAML 5.1+.
                yaml.SafeLoader: Loads a subset of the YAML language, safely. This is recommended for loading untrusted input.
                yaml.UnsafeLoader: The original Loader code that could be easily exploitable by untrusted data input.
                yaml.BaseLoader: Only loads the most basic YAML. All scalars are loaded as strings.

    Returns:
        obj (object):
            Object loaded from pickle file.
    """
    with open(filename, mode) as f:
        return yaml.load(f, Loader=loader)
        

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
    allow_overwrite=True,
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
            If True and file exists locally, plans to skip download.
            If True and check_hash is True, checks hash of local file.
             If hash matches, skips download. If hash does not match, 
             downloads file.
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
        print(f'Error downloading file: {e}') if verbose else None
        return False
    # Check response
    if response.status_code != 200:
        print(f'Error downloading file. Response status code: {response.status_code}') if verbose else None
        return False
    # Create parent directory if it does not exist
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)
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