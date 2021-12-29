"""Functions for path manipulation and retrieval of files."""
'Currently most functions are stolen from https://github.com/MartinThoma/mpu/blob/master/mpu/path.py'

# Core Library
import os
from typing import List
import numpy as np
from pathlib import Path

# Third party
import pkg_resources

def mkdir(dir, parents=True, exist_ok=True):
    '''
    Create a directory if it doesn't exist.
    RH 2021

    Args:
        dir (str):
            path to directory
        parents (bool):
            whether to create parent directories
        exist_ok (bool):
            whether to raise an exception if the 
             directory already exists
    '''
    Path(dir).mkdir(parents=parents, exist_ok=exist_ok)


def get_dir_contents(dir):
    '''
    Get the contents of a directory (does not
     include subdirectories).
    RH 2021

    Args:
        dir (str):
            path to directory
    
    Returns:
        folders (List):
            list of folder names
        files (List):
            list of file names
    '''
    walk = os.walk(dir, followlinks=False)
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files


def get_all_files(root: str, followlinks: bool = False) -> List:
    """
    Get all files within the given root directory.
    Note that this list is not ordered.
    Parameters
    ----------
    root : str
        Path to a directory
    followlinks : bool, optional (default: False)
    Returns
    -------
    filepaths : List
        List of absolute paths to files
    """
    filepaths = []
    for path, _, files in os.walk(root, followlinks=followlinks):
        for name in files:
            filepaths.append(os.path.abspath(os.path.join(path, name)))
    return filepaths


def get_from_package(package_name: str, path: str) -> str:
    """
    Get the absolute path to a file in a package.
    Parameters
    ----------
    package_name : str
        e.g. 'mpu'
    path : str
        Path within a package
    Returns
    -------
    filepath : str
    """
    filepath = pkg_resources.resource_filename(package_name, path)
    return os.path.abspath(filepath)


def get_nums_from_string(string_with_nums):
    """
    Return the numbers from a string as an int
    RH 2021

    Args:
        string_with_nums (str):
            string with numbers in it
    
    Returns:
        nums (int):
            the numbers from the string            
    """
    idx_nums = [ii in str(np.arange(10)) for ii in string_with_nums]
    
    nums = []
    for jj, val in enumerate(idx_nums):
        if val:
            nums.append(string_with_nums[jj])
    nums = int(''.join(nums))
    return nums