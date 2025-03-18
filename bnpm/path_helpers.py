"""
Functions for path manipulation and retrieval of files.
"""
from typing import Union, Optional, List

import os
import numpy as np
from pathlib import Path
import re
import warnings
import datetime


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
        verbose (Union[bool, int]):
            Whether to print the paths found. (Default is ``False``) \n
                * If False/0, then no printout.
                * If True/1, then printout.
                * If 2, then printout with additional details.

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
            print(f'Checking path: {path}') if verbose > 1 else None
            if os.path.isdir(path):
                if find_folders:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found folder: {path}') if verbose > 0 else None
                        paths.append(path)
                    else:
                        print(f'Not matched: {path}') if verbose > 1 else None
                if depth < depth_end:
                    print(f'Entering folder: {path}') if verbose > 1 else None
                    paths += get_paths_recursive_inner(path, depth_end, depth=depth+1)
            else:
                if find_files:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found file: {path}') if verbose > 0 else None
                        paths.append(path)
                    else:
                        print(f'Not matched: {path}') if verbose > 1 else None
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


def check_files_openable(dir_outer, time_limit_per_file=1, verbose=False):
    """
    Check if files within an outer directory are able to be opened. \n
    RH 2024

    Args:
        dir_outer (str):
            Path to the outer directory
        depth (int):
            Maximum depth of subdirectories to search. \n
            Depth=0 means only the outer directory. \n
            Default is 2.
        time_limit_per_file (int):
            Time limit in seconds for checking if a file can be opened.
        verbose (bool):
            Whether to print the files that can't be opened. \n
            Default is False.

    Returns:
        (dict):
            Dictionary with keys as the file paths and values as booleans
            indicating whether the file can be opened.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from contextlib import contextmanager

    def check_file_openable(file_path):
        """
        Check if a file can be opened.
        """
        try:
            with open(file_path, 'rb') as f:
                ## Read a maximum of 1024 bytes. If file is smaller, read the whole file.
                f.read(1024)
            print(f"File {file_path} can be opened.") if verbose > 1 else None
            return True
        except Exception as e:
            print(f"FAILURE: File {file_path} could not be opened: {e}") if verbose > 0 else None
            return False
        
    def check_with_timeout(file_path, time_limit):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(check_file_openable, file_path)
            try:
                return future.result(timeout=time_limit)
            except TimeoutError:
                print(f"FAILURE: File {file_path} took too long to open.") if verbose > 0 else None
                return False
        
    def walk_files(dir_outer):
        """
        Walk through files in a directory.
        """
        files = []
        for root, dirs, filenames in os.walk(dir_outer):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files
    
    files = walk_files(dir_outer)
    file_openable = {file: check_with_timeout(file, time_limit=time_limit_per_file) for file in files}
    return file_openable


def touch_path(
    path: Union[str, Path],
    recursive: bool = False,
    dt: Optional[datetime.datetime] = None,
    files: bool = True,
    directories: bool = True,
    verbose: Union[bool, int] = False,
) -> None:
    """
    Update the last modified datetime of specified files and/or directories
    without creating new ones. Symbolic links are skipped. \n
    RH 2024

    Args:
        path (Union[str, Path]): 
            The directory or file path to modify.
        recursive (bool): 
            Whether to recursively apply changes to all subfiles and subfolders.
        dt (Optional[datetime]): 
            The datetime to set as the last modified time. If not specified, the
            current datetime is used.
        files (bool): 
            If True, update modification times of files.
        directories (bool): 
            If True, update modification times of directories.
        verbose (Union[bool, int]):
            Whether to print the paths that are modified / Level of verbosity.
            \n
                * If False/0, then no printout.
                * If True/1, then print changed paths.
                * If 2, also print skipped paths.

    Returns:
        None

    Raises:
        FileNotFoundError: If the path does not exist or would be created by the touch command.

    Demo:
        .. code-block:: python
            touch_path('/tmp/example.txt', dt=datetime(2024, 4, 8, 23, 30), files=True, directories=False)
    """
    ## Get the timestamp.
    if dt:
        timestamp = dt.timestamp()
    else:
        timestamp = datetime.datetime.now().timestamp()

    def update_mod_time(target_path: Path) -> None:
        """Updates the modification time of a file or directory."""
        ## Pre-modification time.
        try:
            t_pre = datetime.datetime.fromtimestamp(target_path.stat().st_mtime)
        except Exception as e:
            t_pre = None
            warnings.warn(f"Could not get the modification time of {target_path}: {e}")

        ## Update the modification time.
        ### utime inputs are (path, (atime, mtime)). atime: access time, mtime: modification time.
        os.utime(target_path, (timestamp, timestamp))
        print(f"Modified: {target_path}       from {t_pre} to {datetime.datetime.fromtimestamp(timestamp)}") if verbose > 0 else None

    ## Convert the path to a Path object.
    path = Path(path)

    ## Check if the path exists.
    if not path.exists():
        raise FileNotFoundError(f"The path {path} does not exist.")

    ## Update the modification time of the path.
    if path.is_file():
        if files:
            update_mod_time(path)
        else:
            print(f"Skipping: {path}") if verbose > 1 else None
    elif path.is_dir():
        if directories:
            update_mod_time(path)
        else:
            print(f"Skipping: {path}") if verbose > 1 else None
        if recursive:
            for sub_path in path.rglob('*'):  ## rglob('*') fetches all files and directories recursively.
                if sub_path.is_file() and files:
                    update_mod_time(sub_path)
                elif sub_path.is_dir() and directories:
                    update_mod_time(sub_path)
                else:
                    print(f"Skipping: {sub_path}") if verbose > 1 else None
    elif path.is_symlink():  ## Skip symbolic links.
        print(f"Skipping: {path} (symlink)") if verbose > 1 else None
    else:
        raise FileNotFoundError(f"The path {path} returned neither .is_file() nor .is_dir(). It may be a symlink, missing, or otherwise inaccessible.")


def generate_date_range_regex(start, end):
    """
    Generate a regex string that matches any 8-digit ISO date (YYYYMMDD)
    between start and end (inclusive). The returned regex is a plain pattern
    (without start/end anchors) that can be embedded into a larger regex.
    
    This implementation works by recursing digit by digit. At each position,
    if the digits so far still match the start (or end) boundary, then the
    digit at that position is forced to be no less than the corresponding digit
    in start (or no greater than that in end). Once a digit “falls inside” the
    allowed range, subsequent digits can be any digit.
    
    Note:
      For very wide ranges this method will generate a very large regex.
      An alternative approach is to use lookahead assertions based on the fact
      that ISO dates (with zero-padding) sort lexicographically—but such a solution
      may rely on non-standard regex features.
    
    Parameters:
      start (str or int): The start date in YYYYMMDD format.
      end   (str or int): The end date in YYYYMMDD format.
    
    Returns:
      A string containing a regex that matches any 8-digit number between start and end.
    """
    start_str = str(start)
    end_str = str(end)
    if len(start_str) != 8 or len(end_str) != 8:
        raise ValueError("Both start and end must be in YYYYMMDD format (8 digits)")
    
    def rec(pos, lower_bound_enforced, upper_bound_enforced):
        # Base case: when we've processed all 8 digits, return empty string.
        if pos == 8:
            return ""
        # Determine the allowed range for the current digit.
        lower_digit = int(start_str[pos]) if lower_bound_enforced else 0
        upper_digit = int(end_str[pos]) if upper_bound_enforced else 9
        parts = []
        # Loop through all allowed digits for the current position.
        for d in range(lower_digit, upper_digit + 1):
            # If we use the lower bound digit, we must continue enforcing
            new_lower = lower_bound_enforced and (d == lower_digit)
            # Similarly, if we use the upper bound digit.
            new_upper = upper_bound_enforced and (d == upper_digit)
            tail = rec(pos + 1, new_lower, new_upper)
            parts.append(str(d) + tail)
        # If there’s only one possibility, no need for a grouping alternation.
        if len(parts) == 1:
            return parts[0]
        else:
            return "(?:" + "|".join(parts) + ")"
    
    return rec(0, True, True)