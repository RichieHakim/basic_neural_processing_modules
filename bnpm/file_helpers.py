from typing import Any, Union, Callable
import pickle
import json
import yaml
from pathlib import Path
import zipfile

from tqdm.auto import tqdm

from . import path_helpers


def prepare_path(path, mkdir=False, exist_ok=True):
    """
    Checks if a directory or filepath for validity for different
     purposes: saving, loading, etc.
    If exists:
        If exist_ok=True: all good
        If exist_ok=False: raises error
    If doesn't exist:
        If file:
            If parent directory exists:
                All good
            If parent directory doesn't exist:
                If mkdir=True: creates parent directory
                If mkdir=False: raises error
        If directory:
            If mkdir=True: creates directory
            If mkdir=False: raises error
   
    Returns a resolved path.
    RH 2023

    Args:
        path (str):
            Path to check.
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        exist_ok (bool):
            If True, allows overwriting of existing file.

    Returns:
        path (str):
            Resolved path.
    """
    ## check if path is valid
    try:
        path_obj = Path(path).resolve()
    except FileNotFoundError as e:
        print(f'Invalid path: {path}')
        raise e
    
    ## check if path object exists
    flag_exists = path_obj.exists()

    ## determine if path is a directory or file
    if flag_exists:
        flag_dirFileNeither = 'dir' if path_obj.is_dir() else 'file' if path_obj.is_file() else 'neither'  ## 'neither' should never happen since path.is_file() or path.is_dir() should be True if path.exists()
        assert flag_dirFileNeither != 'neither', f'Path: {path} is neither a file nor a directory.'
        assert exist_ok, f'{path} already exists and exist_ok=False.'
    else:
        flag_dirFileNeither = 'dir' if path_obj.suffix == '' else 'file'  ## rely on suffix to determine if path is a file or directory

    ## if path exists and is a file or directory
    # all good. If exist_ok=False, then this should have already been caught above.
    
    ## if path doesn't exist and is a file
    ### if parent directory exists        
    # all good
    ### if parent directory doesn't exist
    #### mkdir if mkdir=True and raise error if mkdir=False
    if not flag_exists and flag_dirFileNeither == 'file':
        if Path(path).parent.exists():
            pass ## all good
        elif mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        else:
            assert False, f'File: {path} does not exist, Parent directory: {Path(path).parent} does not exist, and mkdir=False.'
        
    ## if path doesn't exist and is a directory
    ### mkdir if mkdir=True and raise error if mkdir=False
    if not flag_exists and flag_dirFileNeither == 'dir':
        if mkdir:
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            assert False, f'{path} does not exist and mkdir=False.'

    ## if path is neither a file nor a directory
    ### raise error
    if flag_dirFileNeither == 'neither':
        assert False, f'{path} is neither a file nor a directory. This should never happen. Check this function for bugs.'

    return str(path_obj)

### Custom functions for preparing paths for saving and loading files and directories
def prepare_filepath_for_saving(filepath, mkdir=False, allow_overwrite=False):
    return prepare_path(filepath, mkdir=mkdir, exist_ok=allow_overwrite)
def prepare_filepath_for_loading(filepath, must_exist=True):
    path = prepare_path(filepath, mkdir=False, exist_ok=must_exist)
    if must_exist:
        assert Path(path).is_file(), f'{path} is not a file.'
    return path
def prepare_directory_for_saving(directory, mkdir=False, exist_ok=True):
    """Rarely used."""
    return prepare_path(directory, mkdir=mkdir, exist_ok=exist_ok)
def prepare_directory_for_loading(directory, must_exist=True):
    """Rarely used."""
    path = prepare_path(directory, mkdir=False, exist_ok=must_exist)
    if must_exist:
        assert Path(path).is_dir(), f'{path} is not a directory.'
    return path


def pickle_save(
    obj, 
    filepath, 
    mode='wb', 
    zipCompress=False, 
    mkdir=False, 
    allow_overwrite=False,
    library='pickle',
    **kwargs_zipfile,
):
    """
    Saves an object to a pickle file.
    Allows for zipping of file.
    Uses pickle.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        filepath (str):
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
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.  
        library (str):
            Library to use for pickling. Can be:\n
                * 'pickle': Uses the built-in pickle library.\n
                * 'dill': Uses the dill library. Useful for pickling objects
                  that cannot be pickled with the built-in pickle library.\n
                * 'cloudpickle': Uses the cloudpickle library. Also very
                  flexible, but requires that the file is loaded on the same
                  version of Python that it was saved on.
        kwargs_zipfile (dict):
            Keyword arguments that will be passed into zipfile.ZipFile.
            compression=zipfile.ZIP_DEFLATED by default.
            See https://docs.python.org/3/library/zipfile.html#zipfile-objects.
            Other options for 'compression' are (input can be either int or object):
                0:  zipfile.ZIP_STORED (no compression)
                8:  zipfile.ZIP_DEFLATED (usual zip compression)
                12: zipfile.ZIP_BZIP2 (bzip2 compression) (usually not as good as ZIP_DEFLATED)
                14: zipfile.ZIP_LZMA (lzma compression) (usually better than ZIP_DEFLATED but slower)
    """
    pickle = __import__(library)
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)

    if len(kwargs_zipfile)==0:
        kwargs_zipfile = {
            'compression': zipfile.ZIP_DEFLATED,
        }

    if zipCompress:
        with zipfile.ZipFile(path, 'w', **kwargs_zipfile) as f:
            f.writestr('data', pickle.dumps(obj))
    else:
        with open(path, mode) as f:
            pickle.dump(obj, f)

def pickle_load(
    filepath, 
    zipCompressed=False,
    mode='rb',
    library='pickle',
):
    """
    Loads a pickle file.
    Allows for loading of zipped pickle files.
    RH 2022

    Args:
        filepath (str):
            Path to pickle file.
        zipCompressed (bool):
            If True, then file is assumed to be a .zip file.
            This function will first unzip the file, then
             load the object from the unzipped file.
        mode (str):
            Mode to open file in.
        library (str):
            Library to use for pickling. Can be:\n
                * 'pickle': Uses the built-in pickle library.\n
                * 'dill': Uses the dill library. Useful for pickling objects
                  that cannot be pickled with the built-in pickle library.\n
                * 'cloudpickle': Uses the cloudpickle library. Also very
                  flexible, but requires that the file is loaded on the same
                  version of Python that it was saved on.

    Returns:
        obj (object):
            Object loaded from pickle file.
    """
    pickle = __import__(library)
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    if zipCompressed:
        with zipfile.ZipFile(path, 'r') as f:
            return pickle.loads(f.read('data'))
    else:
        with open(path, mode) as f:
            return pickle.load(f)


def json_save(obj, filepath, indent=4, mode='w', mkdir=False, allow_overwrite=False):
    """
    Saves an object to a json file.
    Uses json.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        filepath (str):
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
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path, mode) as f:
        json.dump(obj, f, indent=indent)

def json_load(filepath, mode='r'):
    """
    Loads a json file.
    RH 2022

    Args:
        filepath (str):
            Path to json file.
        mode (str):
            Mode to open file in.

    Returns:
        obj (object):
            Object loaded from json file.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    with open(path, mode) as f:
        return json.load(f)


def yaml_save(obj, filepath, indent=4, mode='w', mkdir=False, allow_overwrite=False):
    """
    Saves an object to a yaml file.
    Uses yaml.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        filepath (str):
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
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path, mode) as f:
        yaml.dump(obj, f, indent=indent, sort_keys=False)

def yaml_load(filepath, mode='r', loader=yaml.FullLoader):
    """
    Loads a yaml file.
    RH 2022

    Args:
        filepath (str):
            Path to yaml file.
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
            Object loaded from yaml file.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    with open(path, mode) as f:
        return yaml.load(f, Loader=loader)
    

def matlab_load(
    filepath, 
    simplify_cells=True, 
    kwargs_scipy={}, 
    kwargs_mat73={}, 
    verbose=False
):
    """
    Loads a matlab file.
    Uses scipy.io.loadmat if .mat file is not version 7.3.
    Uses mat73.loadmat if .mat file is version 7.3.
    RH 2023

    Args:
        filepath (str):
            Path to matlab file.
        simplify_cells (bool):
            If True and file is not version 7.3, then
             simplifies cells to numpy arrays.
        kwargs_scipy (dict):
            Keyword arguments to pass to scipy.io.loadmat.
        kwargs_mat73 (dict):
            Keyword arguments to pass to mat73.loadmat.
        verbose (bool):
            If True, prints information about file.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    assert path.endswith('.mat'), 'File must be .mat file.'

    try:
        import scipy.io
        out = scipy.io.loadmat(path, simplify_cells=simplify_cells, **kwargs_scipy)
    except NotImplementedError as e:
        print(f'File {path} is version 7.3. Loading with mat73.') if verbose else None
        import mat73
        out = mat73.loadmat(path, **kwargs_mat73)
        print(f'Loaded {path} with mat73.') if verbose else None
    return out

def matlab_save(
    obj,
    filepath,
    mkdir=False, 
    allow_overwrite=False,
    clean_string=True,
    list_to_objArray=True,
    none_to_nan=True,
    kwargs_scipy_savemat={
        'appendmat': True,
        'format': '5',
        'long_field_names': False,
        'do_compression': False,
        'oned_as': 'row',
    }
):
    """
    Saves data to a matlab file.
    Uses scipy.io.savemat.
    Provides additional functionality by cleaning strings,
     converting lists to object arrays, and converting None to
     np.nan.
    RH 2023

    Args:
        obj (dict):
            Data to save.
        filepath (str):
            Path to save file to.
        clean_string (bool):
            If True, converts strings to bytes.
        list_to_objArray (bool):
            If True, converts lists to object arrays.
        none_to_nan (bool):
            If True, converts None to np.nan.
        kwargs_scipy_savemat (dict):
            Keyword arguments to pass to scipy.io.savemat.
    """
    import numpy as np

    prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)

    def walk(d, fn):
        return {key: fn(val) if isinstance(val, dict)==False else walk(val, fn) for key, val in d.items()}
    
    fn_clean_string = (lambda x: x.encode('utf-8') if isinstance(x, str) and clean_string else x) if clean_string else (lambda x: x)
    fn_list_to_objArray = (lambda x: np.array(x, dtype=object) if isinstance(x, list) and list_to_objArray else x) if list_to_objArray else (lambda x: x)
    fn_none_to_nan = (lambda x: np.nan if x is None and none_to_nan else x) if none_to_nan else (lambda x: x)

    data_cleaned = walk(walk(walk(obj, fn_clean_string), fn_list_to_objArray), fn_none_to_nan)

    import scipy.io
    scipy.io.savemat(filepath, data_cleaned, **kwargs_scipy_savemat)


# def zarr_save(
#     obj,
#     filepath,
#     mode='w',
#     mkdir=False,
#     allow_overwrite=False,
#     function_unsupported: Callable = lambda data: repr(data),
#     **kwargs_zarr,
# ):
#     """
#     Saves an object to a zarr file. Uses recursive approach to save
#     hierarchical objects.
#     RH 2024

#     Args:
#         obj (object):
#             Object to save. Can be any array or hierarchical object.
#         filepath (str):
#             Path to save object to.
#         mode (str):
#             Mode to open file in.
#             Can be:
#                 'wb' (write binary)
#                 'ab' (append binary)
#                 'xb' (exclusive write binary. Raises FileExistsError if file already exists.)
#         mkdir (bool):
#             If True, creates parent directory if it does not exist.
#         allow_overwrite (bool):
#             If True, allows overwriting of existing file.        
#         kwargs_zarr (dict):
#             Keyword arguments to pass to zarr.save.
#     """
#     import zarr

#     path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)
    
#     import numpy as np
#     import scipy.sparse

#     def save_data_to_zarr(data, group, name=None):
#         """
#         Recursively saves complex nested data structures into a Zarr group.

#         Parameters:
#         - data: The data to save (dict, list, tuple, np.ndarray, int, float, str, bool, or None).
#         - group: The Zarr group to save data into.
#         - name: The name of the dataset or subgroup (used in recursive calls).
#         """
#         if isinstance(data, dict):
#             # Use the given name or the current group
#             sub_group = group.require_group(name) if name else group
#             for key, value in data.items():
#                 # Ensure keys are strings
#                 key_str = str(key) if not isinstance(key, str) else key
#                 # Recursively save data
#                 save_data_to_zarr(value, sub_group, name=key_str)
#         elif isinstance(data, (list, tuple)):
#             # Create a subgroup for lists and tuples
#             sub_group = group.require_group(name) if name else group
#             for idx, item in enumerate(data):
#                 key_str = str(idx)
#                 save_data_to_zarr(item, sub_group, name=key_str)
#         elif isinstance(data, np.ndarray):
#             if name is None:
#                 raise ValueError("Name must be provided for dataset")
#             group.create_dataset(name, data=data)
#         elif isinstance(data, scipy.sparse.spmatrix):
#             if name is None:
#                 raise ValueError("Name must be provided for dataset")
#             group.create_dataset(name, data=data)
#         elif isinstance(data, (int, float, str, bool)):
#             if name is None:
#                 raise ValueError("Name must be provided for dataset")
#             group.create_dataset(name, data=data)
#         elif data is None:
#             if name is None:
#                 raise ValueError("Name must be provided for dataset")
#             # Store None as a special attribute
#             group.attrs[name] = 'None'
#         else:
#             # For unsupported types, store the string representation
#             if name is None:
#                 raise ValueError("Name must be provided for attribute")
#             group.attrs[name] = repr(data)

#     zarr_group = zarr.open(path, mode=mode, **kwargs_zarr)
#     save_data_to_zarr(obj, zarr_group, name=None)


# def zarr_load(
#     filepath,
#     mode='r',
#     **kwargs_zarr,
# ):
#     """
#     Loads a zarr file. Uses recursive approach to load hierarchical
#     objects.
#     RH 2024

#     Args:
#         filepath (str):
#             Path to zarr file.
#         mode (str):
#             Mode to open file in.
#         kwargs_zarr (dict):
#             Keyword arguments to pass to zarr.load.
#     """
#     import zarr

#     # path = prepare_filepath_for_loading(filepath, must_exist=True)
#     path = filepath
    
#     def load_data_from_zarr(group):
#         """
#         Recursively loads complex nested data structures from a Zarr group.

#         Parameters:
#         - group: The Zarr group to load data from.

#         Returns:
#         - The loaded data (dict, list, tuple, np.ndarray, int, float, str, bool, or None).
#         """
#         data = {}
#         for key in group.array_keys():
#             data[key] = group[key][...]
#         for key in group.group_keys():
#             data[key] = load_data_from_zarr(group[key])
#         for key, value in group.attrs.items():
#             if value == 'None':
#                 data[key] = None
#             else:
#                 try:
#                     data[key] = eval(value)
#                 except Exception:
#                     data[key] = value
#         return data

#     zarr_group = zarr.open(path, mode=mode, **kwargs_zarr)
#     return load_data_from_zarr(zarr_group)
        

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
    allow_overwrite=False,
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
            If url is None, then no download is attempted.
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

    path_save = prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)

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
    if url is None:
        print('No URL provided. No download attempted.') if verbose else None
        return None
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
        print("Error downloading file. Downloaded file is not the same size as the file on the server.")
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


def compare_file_hashes(
    hash_dict_true,
    dir_files_test=None,
    paths_files_test=None,
    verbose=True,
):
    """
    Compares hashes of files in a directory or list of paths
     to user provided hashes.
    RH 2022

    Args:
        hash_dict_true (dict):
            Dictionary of hashes to compare to.
            Each entry should be:
                {'key': ('filename', 'hash')}
        dir_files_test (str):
            Path to directory to compare hashes of files in.
            Unused if paths_files_test is not None.
        paths_files_test (list of str):
            List of paths to files to compare hashes of.
            Optional. dir_files_test is used if None.
        verbose (bool):
            Whether or not to print out failed comparisons.

    Returns:
        total_result (bool):
            Whether or not all hashes were matched.
        individual_results (list of bool):
            Whether or not each hash was matched.
        paths_matching (dict):
            Dictionary of paths that matched.
            Each entry is:
                {'key': 'path'}
    """
    if paths_files_test is None:
        if dir_files_test is None:
            raise ValueError('Must provide either dir_files_test or path_files_test.')
        
        ## make a dict of {filename: path} for each file in dir_files_test
        files_test = {filename: (Path(dir_files_test).resolve() / filename).as_posix() for filename in path_helpers.get_dir_contents(dir_files_test)[1]} 
    
    paths_matching = {}
    results_matching = {}
    for key, (filename, hash) in hash_dict_true.items():
        match = True
        if filename not in files_test:
            print(f'{filename} not found in test directory: {dir_files_test}.') if verbose else None
            match = False
        elif hash != hash_file(files_test[filename]):
            print(f'{filename} hash mismatch with {key, filename}.') if verbose else None
            match = False
        if match:
            paths_matching[key] = files_test[filename]
        results_matching[key] = match

    return all(results_matching.values()), results_matching, paths_matching


def extract_zip(
    path_zipFile,
    path_extract=None,
    mkdir=True,
    verbose=True,
):
    """
    Extracts a zip file.
    RH 2022

    Args:
        path_zipFile (str):
            Path to zip file.
        path_extract (str):
            Path to extract zip file to.
            If None, extracts to the same directory as the zip file.
        verbose (int):
            Whether to print progress.
    """
    path_zipFile = prepare_filepath_for_loading(path_zipFile, must_exist=True)
    import zipfile
    if path_extract is None:
        path_extract = str(Path(path_zipFile).parent)
    path_extract = prepare_directory_for_saving(path_extract, mkdir=mkdir, exist_ok=True)

    print(f'Extracting {path_zipFile} to {path_extract}.') if verbose else None

    with zipfile.ZipFile(path_zipFile, 'r') as zip_ref:
        zip_ref.extractall(path_extract)

    print('Completed zip extraction.') if verbose else None
