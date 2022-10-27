"""
Functions for path manipulation and retrieval of files.
"""

import os
import numpy as np
from pathlib import Path
import re

def mkdir(directory, parents=True, exist_ok=True):
    '''
    Create a directory if it doesn't exist.
    RH 2021

    Args:
        directory (str):
            path to directory
        parents (bool):
            whether to create parent directories
        exist_ok (bool):
            whether to raise an exception if the 
             directory already exists
    '''
    Path(directory).mkdir(parents=parents, exist_ok=exist_ok)


def get_dir_contents(directory):
    '''
    Get the contents of a directory (does not
     include subdirectories).
    RH 2021

    Args:
        directory (str):
            path to directory
    
    Returns:
        folders (List):
            list of folder names
        files (List):
            list of file names
    '''
    walk = os.walk(directory, followlinks=False)
    folders = []
    files = []
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files


def get_numeric_contents(directory, sort=True, contains_string=None):
    """
    Get the contents of a directory that have
     numeric names (files and/or folders).
    RH 2022

    Args:
        directory (str):
            path to directory
        sort (bool):
            whether to sort the contents

    Returns:
        paths_output (List of str):
            Paths with numeric contents
        contents_output (List of str):
            Contents of the paths
        numerics_output (np.float64):
            Numeric contents of the paths.
            If there are no numeric contents, 
             return np.nan.
    """
    contents = np.concatenate(get_dir_contents(directory))
    numerics = [ get_nums_from_string(Path(path).name) for path in contents ]
    for ii, num in enumerate(numerics):
        if num is None:
            numerics[ii] = np.nan
    # numerics = np.array(numerics, dtype=np.uint64)
    numerics = np.array(numerics)
    if sort:
        numerics_argsort = np.argsort(numerics)
        numerics_output = numerics[numerics_argsort]
        isnan = np.array([np.isnan(np.float64(val)) for val in numerics_output])
        contents_output = contents[numerics_argsort[~isnan]]
    else:
        numerics_output = numerics
        contents_output = contents[np.isnan(numerics_output)==False]

    paths_output = [str(Path(directory).resolve() / str(ii)) for ii in contents_output]

    if contains_string is not None:
        paths_output = [path for path in paths_output if contains_string in path]
        contents_output = [contents_output[ii] for ii, path in enumerate(paths_output) if contains_string in path]
        numerics_output = [numerics_output[ii] for ii, path in enumerate(paths_output) if contains_string in path]

    return paths_output, contents_output, numerics_output


def find_paths(
    dir_outer, 
    reMatch='filename', 
    find_files=True, 
    find_folders=False, 
    depth=0, 
    natsorted=True, 
    alg_ns=None, 
):
    """
    Search for files and/or folders recursively in a directory.
    RH 2022

    Args:
        dir_outer (str):
            Path to directory to search
        reMatch (str):
            Regular expression to match
            Each path name encountered will be compared using
             re.search(reMatch, filename). If the output is not None,
             the file will be included in the output.
        find_files (bool):
            Whether to find files
        find_folders (bool):
            Whether to find folders
        depth (int):
            Maximum folder depth to search.
            depth=0 means only search the outer directory.
            depth=2 means search the outer directory and two levels
             of subdirectories below it.
        natsorted (bool):
            Whether to sort the output using natural sorting
             with the natsort package.
        alg_ns (str):
            Algorithm to use for natural sorting.
            See natsort.ns or
             https://natsort.readthedocs.io/en/4.0.4/ns_class.html
             for options.
            Default is PATH.
            Other commons are INT, FLOAT, VERSION.

    Returns:
        paths (List of str):
            Paths to matched files and/or folders in the directory
    """
    import natsort
    if alg_ns is None:
        alg_ns = natsort.ns.PATH

    def get_paths_recursive_inner(dir_inner, depth_end, depth=0):
        paths = []
        for path in os.listdir(dir_inner):
            path = os.path.join(dir_inner, path)
            if os.path.isdir(path):
                if find_folders:
                    if re.search(reMatch, path) is not None:
                        paths.append(path)
                if depth < depth_end:
                    paths += get_paths_recursive_inner(path, depth_end, depth=depth+1)
            else:
                if find_files:
                    if re.search(reMatch, path) is not None:
                        paths.append(path)
        return paths

    paths = get_paths_recursive_inner(dir_outer, depth, depth=0)
    if natsorted:
        paths = natsort.natsorted(paths, alg=alg_ns)
    return paths
        

def get_nums_from_string(string_with_nums):
    """
    Return the numbers from a string as an int
    RH 2021-2022

    Args:
        string_with_nums (str):
            String with numbers in it
    
    Returns:
        nums (int):
            The numbers from the string    
            If there are no numbers, return None.        
    """
    idx_nums = [ii in str(np.arange(10)) for ii in string_with_nums]
    
    nums = []
    for jj, val in enumerate(idx_nums):
        if val:
            nums.append(string_with_nums[jj])
    if not nums:
        return None
    nums = int(''.join(nums))
    return nums

def fix_spaces_in_unix_path(path):
    """
    Fix spaces in a unix path.
    Spaces in unix paths are represented by '\ ', but
     this is not recognized by python. This function
     replaces ' ' with r'\ ' in a path.
    RH 2022

    Args:
        path (str):
            Unix path with spaces

    Returns:
        path (str):
            Unix path with spaces replaced by r'\ '
    """
    from pathlib import Path
    return Path(path).as_posix().replace(' ', r'\ ')