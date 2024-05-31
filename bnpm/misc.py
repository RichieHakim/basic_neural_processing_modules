from typing import Callable, List, Any, Dict, Union
import sys
import re
import hashlib
from pathlib import Path
import warnings
from contextlib import contextmanager, ExitStack

import numpy as np
import scipy.sparse


def estimate_array_size(
    array=None, 
    numel=None, 
    input_shape=None, 
    bitsize=64, 
    units='GB',
):
    '''
    Estimates the size of a hypothetical array based on shape or number of 
    elements and the bitsize
    RH 2021

    Args:
        array (np.ndarray or torch.Tensor or scipy.sparse):
            array to estimate size of. If supplied, then 'numel' and 
            'input_shape' are ignored
        numel (int): 
            number of elements in the array. If None, then 'input_shape'
            is used instead
        input_shape (tuple of ints):
            shape of array. Output of array.shape . Used if numel is None
        bitsize (int):
            bit size / width of the hypothetical data. eg:
                'float64'=64
                'float32'=32
                'uint8'=8
        units (str):
            units of the output. eg:
                'GB'=gigabytes
                'MB'=megabytes
                'KB'=kilobytes
                'B'=bytes
    
    Returns:
        size_estimate_in_bytes (int):
            size, in bytes, of hypothetical array. Doesn't include metadata,
            but for numpy arrays, this is usually very small (~128 bytes)

    '''
    ## Either array supplied or numel or input_shape
    assert sum([(array is not None), (numel is not None), (input_shape is not None)]) == 1, 'Exactly one of array, numel, or input_shape must be supplied'

    if scipy.sparse.issparse(array):
        assert hasattr(array, 'shape'), f'array must have a shape attribute'
        numel = array.nnz
        bitsize = array.dtype.itemsize * 8
    elif array is not None:
        assert hasattr(array, 'shape'), f'array must have a shape attribute'
        input_shape = array.shape

        if isinstance(array, np.ndarray):
            bitsize = array.dtype.itemsize * 8
        else:
            import torch
            if isinstance(array, torch.Tensor):
                bitsize = array.element_size() * 8
            else:
                raise TypeError(f'array must be a numpy or torch array. Got {type(array)}')

    if numel is None:
        numel = np.prod(np.array(input_shape, dtype=np.float64))
    
    bytes_per_element = float(bitsize / 8)
    
    size_estimate_in_bytes = numel * bytes_per_element
    size_out = convert_size(size_estimate_in_bytes, units)
    return size_out


def get_vars(globals, size_thresh=0, var_type=None, return_vars_pref=False):
    '''
    Returns variable info that matches defined criteria.
    RH 2021

    Args:
        globals:
            `globals()` must be passed here
        size_thresh (scalar):
            Minimum size, in MB of variables you'd like returned
        var_type (type obj):
            Class type you'd like returned
        return_vars_pref (bool):
            Whether or not you'd like the outputs returned

    Returns:
        var_names (np.array of str):
            Names of variables sorted by size
        var_sizes (np.array of float64):
            Sizes of variables sorted by size
        var_types (np.array of type objects):
            Types of variables sorted by size

    Demo:
        var_names, var_sizes, var_types = get_vars(globals(), size_thresh=0.1, var_type=np.ndarray, return_vars_pref=True)
    '''
    var_info = []
    for ii, (name, var) in enumerate(globals.items()):
        var_info.append((name, sys.getsizeof(var), type(var)))

    var_names = np.array(var_info)[:,0]
    var_sizes = np.float64(np.array(var_info)[:,1])
    var_types = np.array(var_info)[:,2]

    sort_idx = np.flip(np.argsort(var_sizes))
    var_types = var_types[sort_idx]
    var_names = var_names[sort_idx]
    var_sizes = var_sizes[sort_idx]

    idx_toInclude = []
    for ii, (name, size, val_type) in enumerate(zip(var_names, var_sizes, var_types)):
        if var_type is not None:
            if size > size_thresh*1000000 and (val_type==var_type):
                idx_toInclude.append(ii)
                print(f'{name}, {size/1000000} MB, type: {val_type}')
        else:
            if size > size_thresh*1000000:
                idx_toInclude.append(ii)
                print(f'{name}, {size/1000000} MB, type: {val_type}')
    
    if return_vars_pref:
        return var_names[idx_toInclude], var_sizes[idx_toInclude], var_types[idx_toInclude]


def get_nums_from_str(str_in, dtype_out=np.float64):
    """
    Returns a list of numbers from a string.
    Numbers can be negative and decimals.
    RH 2022

    Args:
        str_in (str):
            String to be parsed.
            Should contain numbers separated by spaces, commas,
             letters, or most other characters.
        dtype_out (type obj):
            dtype of output.

    Returns:
        nums (np.array):
            List of numbers found in the string.

    """
    return np.array([float(i) for i in re.findall(r'\-?\d+\.?\d*', str_in)], dtype=dtype_out)


def write_to_log(
    *text, 
    path=None, 
    mode='a', 
    start_on_new_line=True, 
    pref_print=True, 
):
    """
    Writes text to a log file.
    RH 2022

    Args:
        text (str):
            Text to write to log file.
        path (str):
            Path to log file.\n
            Use suffix '.log' or '.txt' for best results.\n
            If None, text is not written to a file.
        mode (str):
            Mode to open log file in.
            'a' for append, 'w' for write.
        start_on_new_line (bool):
            Whether or not to start on a new line.
        pref_print (bool):
            Whether or not to print text to console.

    Returns:
        None
    """
    if pref_print:
        print(text)
    if path is not None:
        with open(path, mode=mode) as log:
            if start_on_new_line==True:
                log.write('\n')
            log.write(str(text))


def format_text(
    text, 
    color=None, 
    bold=False, 
    italic=False, 
    background=None, 
    underline=False, 
    blink=False, 
    dim=False,
):
    """
    Formats text with ANSI escape sequences.
    RH 2023

    Args:
        text (str):
            Text to format.
        color (tuple of ints):
            RGB color of text.
            Range: 0-255
        bold (bool):
            Whether or not to bold text.
        italic (bool):
            Whether or not to italicize text.
            Doesn't work in ipynb
        background (tuple of ints):
            RGB color of text background.
            Range: 0-255
        underline (bool):
            Whether or not to underline text.
        blink (bool):
            Whether or not to blink text.
            Doesn't work in ipynb
        dim (bool):
            Whether or not to dim text.
            Doesn't work in ipynb

    Returns:
        formatted_text (str):
            Formatted text.
    """
    # ANSI escape sequences for text formatting
    modifiers = []
    
    # Color
    if color:
        r, g, b = color
        modifiers.append(f'\x1b[38;2;{r};{g};{b}m')

    # Background color
    if background:
        r, g, b = background
        modifiers.append(f'\x1b[48;2;{r};{g};{b}m')
    
    # Other modifiers
    if bold:
        modifiers.append('\x1b[1m')
    if italic:
        modifiers.append('\x1b[3m')
    if underline:
        modifiers.append('\x1b[4m')
    if blink:
        modifiers.append('\x1b[5m')
    if dim:
        modifiers.append('\x1b[2m')

    # Reset all formatting
    reset = '\x1b[0m'
    
    return ''.join(modifiers) + text + reset


def system_info(verbose: bool = False,) -> Dict:
    """
    Checks and prints the versions of various important software packages.
    RH 2022

    Args:
        verbose (bool): 
            Whether to print the software versions. 
            (Default is ``False``)

    Returns:
        (Dict): 
            versions (Dict):
                Dictionary containing the versions of various software packages.
    """
    ## Operating system and version
    import platform
    def try_fns(fn):
        try:
            return fn()
        except:
            return None
    fns = {key: val for key, val in platform.__dict__.items() if (callable(val) and key[0] != '_')}
    operating_system = {key: try_fns(val) for key, val in fns.items() if (callable(val) and key[0] != '_')}
    print(f'== Operating System ==: {operating_system["uname"]}') if verbose else None

    ## CPU info
    try:
        import cpuinfo
        import multiprocessing as mp
        # cpu_info = cpuinfo.get_cpu_info()
        cpu_n_cores = mp.cpu_count()
        cpu_brand = cpuinfo.cpuinfo.CPUID().get_processor_brand(cpuinfo.cpuinfo.CPUID().get_max_extension_support())
        cpu_info = {'n_cores': cpu_n_cores, 'brand': cpu_brand}
        if 'flags' in cpu_info:
            cpu_info['flags'] = 'omitted'
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get cpu info. Got error: {e}')
        cpu_info = 'Error: Failed to get'
    print(f'== CPU Info ==: {cpu_info}') if verbose else None

    ## RAM
    import psutil
    ram = psutil.virtual_memory()
    print(f'== RAM ==: {ram}') if verbose else None

    ## User
    import getpass
    user = getpass.getuser()

    ## GPU
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_info = {gpu.id: gpu.__dict__ for gpu in gpus}
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get gpu info. Got error: {e}')
        gpu_info = 'Error: Failed to get'
    print(f'== GPU Info ==: {gpu_info}') if verbose else None
    
    ## Conda Environment
    import os
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        conda_env = 'None'
    else:
        conda_env = os.environ['CONDA_DEFAULT_ENV']
    print(f'== Conda Environment ==: {conda_env}') if verbose else None

    ## Python
    import sys
    python_version = sys.version.split(' ')[0]
    print(f'== Python Version ==: {python_version}') if verbose else None

    ## GCC
    import subprocess
    try:
        gcc_version = subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0].split(' ')[-1]
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get gcc version. Got error: {e}')
        gcc_version = 'Faled to get'
    print(f'== GCC Version ==: {gcc_version}') if verbose else None
    
    ## PyTorch
    try:
        import torch
        torch_version = str(torch.__version__)
        print(f'== PyTorch Version ==: {torch_version}') if verbose else None
        ## CUDA
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            torch_devices = [f'device {i}: Name={torch.cuda.get_device_name(i)}, Memory={torch.cuda.get_device_properties(i).total_memory / 1e9} GB' for i in range(torch.cuda.device_count())]
            print(f"== CUDA Version ==: {cuda_version}, CUDNN Version: {cudnn_version}, Number of Devices: {torch.cuda.device_count()}, Devices: {torch_devices}, ") if verbose else None
        else:
            cuda_version = 'cuda not available'
            cudnn_version = 'cuda not available'
            torch_devices = 'cuda not available'
            print('== CUDA is not available ==') if verbose else None
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get torch info. Got error: {e}')
        torch_version = 'torch not found'
        cuda_version = 'torch not found'
        cudnn_version = 'torch not found'
        torch_devices = 'torch not found'

    ## all packages in environment
    import importlib.metadata
    pkgs_dict = {dist.metadata['Name']: dist.metadata['Version'] for dist in importlib.metadata.distributions()}

    ## get datetime
    from datetime import datetime
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    versions = {
        'datetime': dt,
        'operating_system': operating_system,
        'cpu_info': cpu_info,  ## This is the slow one.
        'user': user,
        'ram': ram,
        'gpu_info': gpu_info,
        'conda_env': conda_env,
        'python': python_version,
        'gcc': gcc_version,
        'torch': torch_version,
        'cuda': cuda_version,
        'cudnn': cudnn_version,
        'torch_devices': torch_devices,
        'pkgs': pkgs_dict,
    }

    return versions


def array_hasher():
    """
    Returns a function that hashes an array.
    """
    from functools import partial
    import xxhash
    return partial(xxhash.xxh64_hexdigest, seed=0)


def reset_warnings():
    """
    Resets warnings to default settings.
    """
    import warnings
    warnings.simplefilter('default')

def reset_numpy_warnings():
    """
    Resets numpy warnings to default settings.
    """
    import numpy as np
    np.seterr(all='warn')
    


#####################################################################
################# CONTEXT MANAGERS AND DECORATORS ###################
#####################################################################


@contextmanager
def temp_set_attr(obj, attr_name, new_value):
    """
    Temporarily set an attribute of an object to a new value within a context
    manager / closure.
    RH 2024

    Args:
        obj (object):
            Object to toggle attribute for.
        attr_name (str):
            Attribute to toggle.
        new_value (Any):
            New value to set attribute to.

    Demo:
        .. code-block:: python
            
                with temp_set_attr(obj, attr, new_val):
                    # do something
    """
    original_value = getattr(obj, attr_name)
    setattr(obj, attr_name, new_value)
    try:
        yield
    finally:
        setattr(obj, attr_name, original_value)


def wrapper_flexible_args(names_kwargs: List[str]) -> Callable:
    """
    Make a wrapper function that allows for flexible argument names.\n
    Useful for things like `dim` vs. `axis` or `keepdim` vs. `keepdims`.\n
    RH 2024

    Args:
        names_kwargs (List[str]):
            List of argument names that the wrapper function should accept.

    Returns:
        (Callable):
            Wrapper function.
    """
    import inspect
    from functools import wraps

    def wrapper(func: Callable) -> Callable:
        try:
            sig = inspect.signature(func)
        except ValueError:
            raise ValueError(f'Function {func.__name__} must have a signature to use this wrapper')
        
        @wraps(func)
        def wrapped(*args, **kwargs):
            ## Check for intersection of names_kwargs with sig.parameters and kwargs
            sp = set(sig.parameters.keys())
            nk = set(names_kwargs)
            k = set(kwargs.keys())
            ix_nk_sp = nk.intersection(sp)
            ix_nk_k = nk.intersection(k)
            ix_nk_k_sp = ix_nk_k.intersection(sp)
            ## Logic steps:
            ### If none of the kwargs are in the signature, and there is at least one named_kwarg in both the kwargs and signature, then rename the kwargs
            if (len(ix_nk_k_sp) == 0) and (len(ix_nk_k) > 0) and (len(ix_nk_sp) > 0):
                k_toUse = ix_nk_sp.pop()
                k_toRename = ix_nk_k.pop()
                kwargs[k_toUse] = kwargs.pop(k_toRename)
            return func(*args, **kwargs)
        return wrapped
    return wrapper


class MultiContextManager:
    """
    A context manager that allows multiple context managers to be used at once.
    RH 2024

    Args:
        *managers (context managers):
            Multiple context managers to be used at once.

    Demo:
        .. code-block:: python

            with MultiContextManager(
                torch.no_grad(), 
                temp_set_attr(obj, attr, new_val), 
                open('file.txt', 'w') as f,
            ):
                # do something
    """
    def __init__(self, *managers):
        self.managers = managers
        self.stack = ExitStack()

    def __enter__(self):
        for manager in self.managers:
            self.stack.enter_context(manager)

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.__exit__(exc_type, exc_value, traceback)
        
        
class TimeoutException(Exception):
    pass
@contextmanager
def time_limit(seconds):
    """
    Wrapper to set a time limit for a block of code, after which a
    TimeoutException is raised.
    """
    import signal
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@contextmanager
def nullcontext():
    """
    A context manager that does nothing.
    """
    yield


#########################################################
############ INTRA-MODULE HELPER FUNCTIONS ##############
#########################################################


def convert_size(size, return_size='GB'):
    """
    Convert size to GB, MB, KB, from B.
    RH 2021

    Args:
        size (int or float):
            Size in bytes.
        return_size (str):
            Size unit to return.
            Options: 'TB', 'GB', 'MB', or 'KB'
        
    Returns:
        out_size (float):
            Size in specified unit.      
    """

    if return_size == 'TB':
        out_size = size / 1000000000000
    elif return_size == 'GB':
        out_size = size / 1000000000
    elif return_size == 'MB':
        out_size = size / 1000000
    elif return_size == 'KB':
        out_size = size / 1000
    elif return_size == 'B':
        out_size = size / 1

    return out_size
