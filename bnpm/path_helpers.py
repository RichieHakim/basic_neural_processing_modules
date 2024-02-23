"""
Functions for path manipulation and retrieval of files.
"""
from typing import Union, Optional, List

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
    dir_outer: Union[str, List[str]],
    reMatch: str = 'filename', 
    reMatch_in_path: Optional[str] = None,
    find_files: bool = True, 
    find_folders: bool = False, 
    depth: int = 0, 
    natsorted: bool = True, 
    alg_ns: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """
    Searches for files and/or folders recursively in a directory using a regex
    match. 
    RH 2022-2023

    Args:
        dir_outer (Union[str, List[str]]):
            Path(s) to the directory(ies) to search. If a list of directories,
            then all directories will be searched.
        reMatch (str): 
            Regular expression to match. Each file or folder name encountered
            will be compared using ``re.search(reMatch, filename)``. If the
            output is not ``None``, the file will be included in the output.
        reMatch_in_path (Optional[str]):
            Additional regular expression to match anywhere in the upper path.
            Useful for finding files/folders in specific subdirectories. If
            ``None``, then no additional matching is done. \n
            (Default is ``None``)
        find_files (bool): 
            Whether to find files. (Default is ``True``)
        find_folders (bool): 
            Whether to find folders. (Default is ``False``)
        depth (int): 
            Maximum folder depth to search. (Default is *0*). \n
            * depth=0 means only search the outer directory. 
            * depth=2 means search the outer directory and two levels of
              subdirectories below it
        natsorted (bool): 
            Whether to sort the output using natural sorting with the natsort
            package. (Default is ``True``)
        alg_ns (str): 
            Algorithm to use for natural sorting. See ``natsort.ns`` or
            https://natsort.readthedocs.io/en/4.0.4/ns_class.html/ for options.
            Default is PATH. Other commons are INT, FLOAT, VERSION. (Default is
            ``None``)
        verbose (bool):
            Whether to print the paths found. (Default is ``False``)

    Returns:
        (List[str]): 
            paths (List[str]): 
                Paths to matched files and/or folders in the directory.
    """
    import natsort
    if alg_ns is None:
        alg_ns = natsort.ns.PATH

    def fn_match(path, reMatch, reMatch_in_path):
        # returns true if reMatch is basename and reMatch_in_path in full dirname
        if reMatch is not None:
            if re.search(reMatch, os.path.basename(path)) is None:
                return False
        if reMatch_in_path is not None:
            if re.search(reMatch_in_path, os.path.dirname(path)) is None:
                return False
        return True

    def get_paths_recursive_inner(dir_inner, depth_end, depth=0):
        paths = []
        for path in os.listdir(dir_inner):
            path = os.path.join(dir_inner, path)
            if os.path.isdir(path):
                if find_folders:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found folder: {path}') if verbose else None
                        paths.append(path)
                if depth < depth_end:
                    paths += get_paths_recursive_inner(path, depth_end, depth=depth+1)
            else:
                if find_files:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found file: {path}') if verbose else None
                        paths.append(path)
        return paths

    def fn_check_pathLike(obj):
        if isinstance(obj, (
            str,
            Path,
            os.PathLike,
            np.str_,
            bytes,
            memoryview,
            np.bytes_,
            np.unicode_,
            re.Pattern,
            re.Match,
        )):
            return True
        else:
            return False            

    dir_outer = [dir_outer] if fn_check_pathLike(dir_outer) else dir_outer
    paths = list(set(sum([get_paths_recursive_inner(str(d), depth, depth=0) for d in dir_outer], start=[])))
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


def find_date_in_path(
    path, 
    regex_date_formats=[
        r'\d{4}\d{2}\d{2}',     # 20220203
        r'\d{4}\D\d{2}\D\d{2}', # 2022_02_03
        r'\d{2}\D\d{2}\D\d{4}', # 02_03_2022
        r'\d{1}\D\d{1}\D\d{4}', # 2_3_2022
        r'\d{1}\D\d{2}\D\d{4}', # 2_03_2022
        r'\d{2}\D\d{1}\D\d{4}', # 02_3_2022
        r'\d{2}\D\d{2}\D\d{2}', # 02_03_22
        r'\d{1}\D\d{1}\D\d{2}', # 2_3_22
        r'\d{1}\D\d{2}\D\d{2}', # 2_03_22
        r'\d{2}\D\d{1}\D\d{2}', # 02_3_22
    ],
    reverse_path_order=True,
):
    """
    Searches a file or directory path for a date string matching one of several
    regex patterns and returns the first match.
    
    RH 2024

    Args:
        path (str):
            The file or directory path in which to search for a date.
        regex_date_formats (List[str]):
            A list of regex patterns to match against parts of the path.\n
            Search goes in order of the list and stops at the first match.\n
            (Default is a list of common date formats)
        reverse_path_order (bool):
            If True, search from the end of the path backwards.
    
    Returns:
        str or None:
            The first matching date string found, or None if no match is found.
    """
    ## make a list of strings
    regex_date_formats = [regex_date_formats] if isinstance(regex_date_formats, str) else regex_date_formats

    ## Dictionary to modify regex based on the presence of separators at start/end of the date.
    modifiers = {
        (0, 0): [r''  , r''  ],
        (1, 0): [r'\D', r''  ],
        (0, 1): [r''  , r'\D'],
        (1, 1): [r'\D', r'\D'],
    }

    ## Split the path into components and optionally reverse the order of search.
    parts = Path(path).parts
    parts = parts[::-1] if reverse_path_order else parts

    def _finder(regexs, parts):
        """Inner function to find the first date in the path parts based on
        provided regex patterns."""
        date = []
        for part in parts:
            for regex in regexs:
                date = re.findall(regex, part)
                if len(date) > 0:
                    ## Return the last match found in the current part.
                    date = date[-1]
                    break
            if isinstance(date, str):
                break
        date = None if isinstance(date, list) else date
        return date
    
    ## Run the finder with each modifier and stop at the first match.
    date = None
    for num, mod in modifiers.items():
        ## Apply modifiers to each regex pattern and search the path parts.
        date = _finder(
            regexs=[mod[0] + regex + mod[1] for regex in regex_date_formats], 
            parts=parts,
        )
        if date is not None:
            ## Remove the modifiers from the date string.
            date = date[num[0]:-num[1] if num[1]>0 else None]
            break

    return date
