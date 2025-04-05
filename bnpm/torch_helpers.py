import sys
import gc
import copy
from typing import Union, List, Tuple, Dict, Callable, Optional, Any, Iterable, Iterator, Generator
import contextlib
from contextlib import contextmanager
import warnings
import math

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm

from . import indexing
from . import misc


############################################
############ VARIABLE HELPERS ##############
############################################

def show_all_tensors(max_depth=1, verbose=0) -> None:
    """
    Displays all tensors present in the provided dictionary.
    From: https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
    RH 2024
    """
    # prints currently alive Tensors and Variables
    import torch
    import gc
    def get_info(obj, depth=0):
        if depth > max_depth:
            return
        
        if check_conditions(obj):
            objs[id(obj)] = {'type': type(obj), 'size': obj.size(), 'mem': misc.estimate_array_size(obj)}
            print(type(obj), obj.size(), misc.estimate_array_size(obj)) if verbose else None
        elif isinstance(obj, dict):
            for key, value in obj.items():
                get_info(value, depth+1)
        elif hasattr(obj, '__dict__'):
            get_info(obj.__dict__, depth+1)
        elif hasattr(obj, '__slots__'):
            for item in obj.__slots__:
                get_info(getattr(obj, item), depth+1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                get_info(item, depth+1)
        
    def check_conditions(obj):
        return torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))

    objs = {}
    for obj in gc.get_objects():
        try:
            get_info(obj)
        except:
            pass

    ## Sort by size
    objs = {k: v for k, v in sorted(objs.items(), key=lambda item: item[1]['mem'], reverse=True)}
    return objs



@contextmanager
def temp_eval(module):
    """
    Temporarily sets the network to evaluation mode within a context manager.
    RH 2024

    Args:
        module (torch.nn.Module):
            The network to temporarily set to evaluation mode.

    Yields:
        (torch.nn.Module):
            The network temporarily set to evaluation mode.

    Demo:
        .. highlight:: python
        .. code-block:: python

            with temp_eval(model):
                y = model(x)
    """
    state_train = module.training
    module.eval()
    try:
        yield module
    finally:
        if state_train:
            module.train()


######################################
############ CUDA STUFF ##############
######################################

def show_torch_cuda_info() -> None:
    """
    Displays PyTorch, CUDA, and device information.

    This function prints Python, PyTorch, CUDA, cuDNN version details and lists
    available CUDA devices. Note that the CUDA version cannot be directly
    fetched using Python, you should use the command `nvcc --version` in your
    terminal or `!nvcc --version` in a Jupyter notebook.

    The function also utilizes the `nvidia-smi` command to retrieve detailed
    information about the CUDA devices, including the GPU index, name, driver
    version, total memory, used memory, and free memory.
    RH 2021
    """
    print('Python version:', sys.version)
    print('PyTorch version:', torch.__version__)
    print('CUDA version: cannot be directly found with python function. Use `nvcc --version` in terminal or `! nvcc --version in notebook')
    from subprocess import call
    # ! nvcc --version
    print('CUDNN version:', torch.backends.cudnn.version())
    print('Devices:')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print ('Available torch cuda devices ', torch.cuda.device_count())
    print ('Current torch cuda device ', torch.cuda.current_device())


def show_cuda_devices(
    verbose: bool = True
) -> None:
    """
    Displays available CUDA devices.
    RH 2023

    Args:
        verbose (bool): 
            If ``True``, the function prints detailed information about each 
            device.
    """
    devices = [torch.cuda.get_device_properties(ii) for ii in range(torch.cuda.device_count())]
    if verbose:
        for ii, device in enumerate(devices):
            print(f'GPU {ii}: {device.name}, {device.total_memory/1000000000} GB')

def delete_all_cuda_tensors(
    globals: dict,
) -> None:
    """
    Deletes all CUDA tensors from the provided global variable dictionary.

    Call this function with: delete_all_cuda_tensors(globals())
    RH 2021

    Args:
        globals (dict):
            Dictionary of global variables.
            To obtain this, call the built-in ``globals()`` function.
    """
    types = [type(ii[1]) for ii in globals.items()]
    keys = list(globals.keys())
    for ii, (i_type, i_key) in enumerate(zip(types, keys)):
        if i_type is torch.Tensor:
            if globals[i_key].device.type == 'cuda':
                print(f'deleting: {i_key}, size: {globals[i_key].element_size() * globals[i_key].nelement()/1000000} MB')
                del(globals[i_key])
    
    clear_cuda_cache()

def clear_cuda_cache() -> None:
    """
    Clears the CUDA cache to free up memory resources.

    This function forces the garbage collector to release unreferenced memory and
    then clears the CUDA cache. This process is repeated four times in an attempt
    to maximize memory release.

    Note that this function will only have an effect if your machine has a GPU and
    if CUDA is enabled.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def wrap_clear_cuda_cache(func: Callable, before=True, after=True) -> Callable:
    """
    Wraps a function with a call to clear the CUDA cache before and/or after the
    function call.

    Args:
        func (Callable):
            The function to wrap.
        before (bool):
            If ``True``, clears the CUDA cache before calling the function.
            (Default is ``True``)
        after (bool):
            If ``True``, clears the CUDA cache after calling the function.
            (Default is ``True``)

    Returns:
        (Callable):
            The wrapped function.

    Demo:
        .. code-block:: python
            
                @wrap_clear_cuda_cache
                def my_function():
                    pass
    """
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if before:
            clear_cuda_cache()
        result = func(*args, **kwargs)
        if after:
            clear_cuda_cache()
        return result
    return wrapper


def set_device(
    use_GPU: bool = True, 
    device_num: int = 0, 
    device_types: List[str] = ['cuda', 'mps', 'xpu', 'cpu'],
    verbose: bool = True
) -> str:
    """
    Sets the device for PyTorch. If a GPU is available and **use_GPU** is
    ``True``, it will be set as the device. Otherwise, the CPU will be set as
    the device. 
    RH 2022

    Args:
        use_GPU (bool): 
            Determines if the GPU should be utilized: \n
            * ``True``: the function will attempt to use the GPU if a GPU is
              not available.
            * ``False``: the function will use the CPU. \n
            (Default is ``True``)
        device_num (int): 
            Specifies the index of the GPU to use. (Default is ``0``)
        device_types (List[str]):
            The types and order of devices to attempt to use. The first device
            type that is available will be used. Options are ``'cuda'``,
            ``'mps'``, ``'xpu'``, and ``'cpu'``.
        verbose (bool): 
            Determines whether to print the device information. \n
            * ``True``: the function will print out the device information.
            \n
            (Default is ``True``)

    Returns:
        (str): 
            device (str): 
                A string specifying the device, either *"cpu"* or
                *"cuda:<device_num>"*.
    """
    devices = list_available_devices()

    if not use_GPU:
        device = 'cpu'
    else:
        device = None
        for device_type in device_types:
            if len(devices[device_type]) > 0:
                device = devices[device_type][device_num]
                break

    if verbose:
        print(f'Using device: {device}')

    return device
    

def list_available_devices() -> dict:
    """
    Lists all available PyTorch devices on the system.
    RH 2024

    Returns:
        (dict): 
            A dictionary with device types as keys and lists of available devices as values.
    """
    devices = {}

    # Check for CPU devices
    if torch.cpu.is_available():
        devices['cpu'] = ['cpu']
    else:
        devices['cpu'] = []

    # Check for CUDA devices
    if torch.cuda.is_available():
        devices['cuda'] = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices['cuda'] = []

    # Check for MPS devices
    if torch.backends.mps.is_available():
        devices['mps'] = ['mps:0']
    else:
        devices['mps'] = []

    # Check for XPU devices
    if hasattr(torch, 'xpu'):
        if torch.xpu.is_available():
            devices['xpu'] = [f'xpu:{i}' for i in range(torch.xpu.device_count())]
        else:
            devices['xpu'] = []
    else:
        devices['xpu'] = []

    return devices


def initialize_torch_settings(
    benchmark: Optional[bool] = None,
    enable_cudnn: Optional[bool] = None,
    deterministic_cudnn: Optional[bool] = None,
    deterministic_torch: Optional[bool] = None,
    set_global_device: Optional[Union[str, torch.device]] = None,
    init_linalg_device: Optional[Union[str, torch.device]] = None,
) -> None:
    """
    Initalizes some CUDA libraries and sets some environment variables. \n
    RH 2024

    Args:
        benchmark (Optional[bool]):
            If ``True``, sets torch.backends.cudnn.benchmark to ``True``.\n 
            This results in the built-in cudnn auto-tuner to find the best
            algorithm for the hardware. Good for when input sizes are the same
            for each batch.
        enable_cudnn (Optional[bool]):
            If ``True``, sets torch.backends.cudnn.enabled to ``True``.\n
            This enables the cudnn library.
        deterministic_cudnn (Optional[bool]):
            If ``True``, sets torch.backends.cudnn.deterministic to ``True``.\n
            This makes cudnn deterministic. It may slow down operations.
        deterministic_torch (Optional[bool]):
            If ``True``, sets torch.set_deterministic to ``True``.\n
            This makes torch deterministic. It may slow down operations.
        set_global_device (bool):
            If ``False``, does not set the global device. If a string or torch.device,
            sets the global device to the specified device.
        init_linalg_device (str):
            The device to use for initializing the linalg library. Either a
            string or a torch.device. This is necessary to avoid a bug. Often
            solves the error: "RuntimeError: lazy wrapper should be called at
            most once". (Default is ``None``)
    """
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark
    if enable_cudnn:
        torch.backends.cudnn.enabled = enable_cudnn
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = False
    if deterministic_torch:
        torch.set_deterministic(False)
    if set_global_device is not None:
        torch.cuda.set_device(set_global_device)
    
    ## Initialize linalg libarary
    ## https://github.com/pytorch/pytorch/issues/90613
    if init_linalg_device is not None:
        if type(init_linalg_device) is str:
            init_linalg_device = torch.device(init_linalg_device)
        torch.inverse(torch.ones((1, 1), device=init_linalg_device))
        torch.linalg.qr(torch.as_tensor([[1.0, 2.0], [3.0, 4.0]], device=init_linalg_device))


def profiler_simple(
    path_save: str = 'trace.json',
    activities: List[str] = ['CPU', 'CUDA'],
    with_stack: bool = False,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_flops: bool = False,
    with_modules: bool = False,
):
    """
    Simple profiler for PyTorch. \n
    Makes a context manager that can be used to profile code. \n
    Upon exit, will save the trace to the specified path. \n
    Use Chrome's chrome://tracing/ to view the trace. \n
    RH 2024

    Args:
        path_save (str):
            The path to save the trace.
        activities (List[str]):
            The activities to profile. Options are ``'CPU'`` and ``'CUDA'``.
        with_stack (bool):
            If ``True``, records the stack, which are all the functions that
            were called. 
        record_shapes (bool):
            If ``True``, records shapes of the tensors.
        profile_memory (bool):
            If ``True``, profiles memory usage of the tensors.
        with_flops (bool):
            If ``True``, records flops (floating point operations).
        with_modules (bool):
            If ``True``, records modules that were called.

    Returns:
        (contextmanager):
            A context manager that can be used to profile code.

    Demo:
        .. highlight:: python
        .. code-block:: python

            with profiler_simple():
                y = model(x)

            >> Then open google chrome and go to the url chrome://tracing/ and
            load the trace file.
    """
    from torch.profiler import profile, record_function, ProfilerActivity
    from contextlib import contextmanager
    
    activities_dict = {
        'CPU': ProfilerActivity.CPU,
        'CUDA': ProfilerActivity.CUDA,
    }
    activities = [activities_dict[act.upper()] for act in activities]

    @contextmanager
    def simple_profiler(path_save: str = 'trace.json'):
        with profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        ) as p:
            with record_function("model_inference"):
                yield
        p.export_chrome_trace(path_save)

    return simple_profiler(path_save=path_save)



######################################
############ DATA HELPERS ############
######################################


class Basic_dataset(Dataset):
    """
    Creates a basic PyTorch dataset. 
    RH 2021

    Args:
        X (torch.Tensor or np.ndarray): 
            The data from which to create the dataset. 
        device (str): 
            The device where the tensors will be stored. 
            (Default is ``'cpu'``)
        dtype (torch.dtype): 
            The data type to use for the tensor. 
            (Default is ``torch.float32``)

    Attributes:
        X (torch.Tensor):
            The data from which the dataset is created. The first (0th)
            dimension will be subsampled from.
        n_samples (int):
            The number of samples in the dataset.

    Example:
        .. highlight:: python
        .. code-block:: python

            ds = Basic_dataset(X, device='cuda:0')
            dl = DataLoader(ds, batch_size=32, shuffle=True)
    """
    def __init__(
        self,
        X: Union[torch.Tensor, np.ndarray],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the Basic_dataset with the provided data, device, and dtype.
        """
        self.X = torch.as_tensor(X, dtype=dtype, device=device) # first (0th) dim will be subsampled from
        self.n_samples = self.X.shape[0]
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            (int):
                n_samples (int):
                    The number of samples in the dataset.
        """
        return self.n_samples
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns a single sample and its index from the dataset.

        Args:
            idx (int):
                The index of the sample to return.

        Returns:
            (tuple): tuple containing:
                sample (torch.Tensor):
                    The requested sample from the dataset.
                idx (int):
                    The index of the requested sample.
        """
        return self.X[idx]
    

class Dataset_numpy(Dataset):
    """
    Creates a PyTorch dataset from a numpy array. 
    RH 2024

    Args:
        X (np.ndarray):
            The data from which to create the dataset.
        axis (int):
            The dimension along which to sample the data.
        device (str): 
            The device where the tensors will be stored. 
        dtype (torch.dtype): 
            The data type to use for the tensor.

    Attributes:
        X (np.ndarray or np.memmap):
            The data from the numpy file.
        n_samples (int):
            The number of samples in the dataset.

    Returns:
        (torch.utils.data.Dataset):
            A PyTorch dataset.
    """
    def __init__(
        self,
        X: Union[np.ndarray, np.memmap],
        axis: int = 0,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the Dataset_NumpyFile with the provided parameters.
        """
        assert isinstance(X, (np.ndarray, np.memmap)), 'X must be a numpy array or memmap.'
        self.X = X
        self.n_samples = self.X.shape[axis]
        self.is_memmap = isinstance(self.X, np.memmap) 
        self.axis = axis
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            (int):
                n_samples (int):
                    The number of samples in the dataset.
        """
        return self.n_samples
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns a single sample and its index from the dataset.

        Args:
            idx (int):
                The index of the sample to return.

        Returns:
            sample (torch.Tensor):
                The requested sample from the dataset.
        """
        arr = np.take(self.X, idx, axis=self.axis)
        if self.is_memmap:
            arr = np.array(arr)
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device)
    
    def close(self):
        """
        Closes the numpy file.
        """
        if self.is_memmap:
            self.X.close()


class Dataset_numpy_concatenated(torch.utils.data.Dataset):
    """
    Dataset class for loading slices from arrays within a multiple numpy
    arrays.\n
    Input is a list of numpy arrays with similar shapes.\n
    Output is a Dataset where the queried index pulls slices from the
    concatenated first dimension indices of all the input arrays.\n
    RH 2024

    Args:
        arrays (list):
            A list of numpy arrays with the following organization: \n
                * Each array may have different first dimension sizes, but all
                  other dimensions must be the same.
    """
    def __init__(self, arrays: List[np.ndarray], verbose: bool = True):
        super(Dataset_numpy_concatenated, self).__init__()

        self.arrays = arrays

        ## Check that all arrays have the same shape except for the first dimension
        shapes = [arr.shape for arr in arrays]
        check_shape = lambda shape1, shape2: shape1[1:] == shape2[1:] if len(shape1) > 1 else shape1[0] == shape2[0]
        assert all([check_shape(shape, shapes[0]) for shape in shapes]), "All arrays must have the same shape except for the first dimension."
        self.n_samples = sum([shape[0] for shape in shapes])
        self.shape = [self.n_samples] + list(shapes[0][1:])

        ## Create an index to field mapping
        ### Use a binary search to find the field for a given index using the cumsum of the first dimensions
        self.cumsum = np.cumsum([0] + [shape[0] for shape in shapes], dtype=np.int64)
        self.idx_to_idxArray = lambda idx: np.searchsorted(self.cumsum, idx, side='right') - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if (idx < 0):
            idx = self.n_samples + idx
        elif (idx >= self.n_samples):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.n_samples}.")
        
        idx_array = self.idx_to_idxArray(idx)
        idx_withinArray = idx - self.cumsum[idx_array]
        sample = self.arrays[idx_array][idx_withinArray]

        ## Return as a cloned tensor
        # return torch.as_tensor(sample.copy())
        return torch.as_tensor(sample)


class BatchRandomSampler(torch.utils.data.Sampler):
    """
    Creates a sampler similar to torch.utils.data.BatchSampler, but allows for
    randomized sampling of the batches. Wraps indexing.make_batches.
    RH 2024

    Args:
        len_dataset(int):
            The number of samples in the dataset.
        batch_size (int):
            Size of each batch.\n
            if None, then ``batch_size`` based on ``num_batches``.
        num_batches (int):
            Number of batches to make.\n
            if None, then ``num_batches`` based on ``batch_size``.
        min_batch_size (int):
            Minimum size of each batch. Set to ``-1`` to equal ``batch_size``.
        randomize_batch_indices
            Whether to randomize the transition indices between batches.\n
            So ``[(1,2,3), (4,5,6),]`` can become ``[(1,), (2,3,4), (5,6),]``.
        shuffle_batch_order (bool):
            Whether to shuffle the order in which batches are yielded.
        shuffle_iterable_order (bool):
            Whether to shuffle the order of the contents of the iterable before
            batching.

    Returns:
        (torch.utils.data.Sampler):
            A sampler for the dataset.
    """
    def __init__(
        self,
        len_dataset: int,
        batch_size: Optional[int] = None,
        num_batches: Optional[int] = None,
        min_batch_size: int = 1,
        randomize_batch_indices: bool = False,
        shuffle_batch_order: bool = False,
        shuffle_iterable_order: bool = False,
    ):
        """
        Initializes the BatchRandomSampler with the provided parameters.
        """
        assert isinstance(len_dataset, int), 'len_dataset must be an integer.'
        self.len_dataset = len_dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.min_batch_size = min_batch_size
        self.randomize_batch_indices = randomize_batch_indices
        self.shuffle_batch_order = shuffle_batch_order
        self.shuffle_iterable_order = shuffle_iterable_order

    def __iter__(self):
        """
        Returns an iterator over the batches.
        """
        self.batches = indexing.make_batches(
            iterable=torch.arange(self.len_dataset),
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            min_batch_size=self.min_batch_size,
            randomize_batch_indices=self.randomize_batch_indices,
            shuffle_batch_order=self.shuffle_batch_order,
            shuffle_iterable_order=self.shuffle_iterable_order,
        )
        return iter(self.batches)
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.len_dataset


def map_batches_cuda(
    batches: Union[Iterable, Iterator, Generator],
    func: Callable,
    device_func: str = 'cuda:0',
    device_return: str = 'cpu',
    pin_memory: bool = True,
    non_blocking: bool = True,
    progress_bar: bool = False,
    len_batches: Optional[int] = None,
    verbose: bool = False,
):
    """
    Run batches through a user specified function on a cuda device using CUDA
    streams. This function is useful for running batches through a model in
    serial, but with asynchronous data transfers. This is similar to
    torch.utils.data.DataLoader, but is simpler and doesn't implement
    multiprocessing.
    RH 2024

    Args:
        batches (Union[Iterable, Iterator, Generator]):
            The batches to process. Each batch can either be a single tensor, a
            tuple containing tensors, or other.
        func (Callable):
            The function to run on the batches. The function must accept the
            items in the batches as arguments.
        device_func (str):
            The device to run the function on. Should typically be a CUDA device
            (e.g., 'cuda:0'). Can also be 'cpu' or other, but this will not
            utilize CUDA streams, memory pinning, or non-blocking transfers.
            (Default is 'cuda:0')
        device_return (str):
            The device to return the results on. For this function, it will
            typically be 'cpu'. (Default is 'cpu')
        pin_memory (bool):
            If ``True``, the function will pin the memory for the input tensors.
            Ideally, the original, non-batched, tensors should be pinned.
            However, this option will pin the memory for each batch. Warning,
            pinning memory can cause excess and persistent memory usage.
            (Default is ``True``)
        non_blocking (bool):
            If ``True``, the function will use non-blocking transfers to and
            from the device. Warning, pinning memory can cause excess and
            persistent memory usage. (Default is ``True``)
        progress_bar (bool):
            If ``True``, displays a progress bar. (Default is ``False``)
        len_batches Optional[int]:
            The length of the batches to be used as the total length of the
            progress bar. If not provided, will attempt to find len(batches).
            (Default is ``None``)
        verbose (bool):
            If ``True``, displays warnings. (Default is ``False``)

    Returns:
        (List):
            results (List):
                The results of the function run on the batches.

    Example:
        .. highlight:: python
        .. code-block:: python

            def func(x, y):
                a, b = model.forward(x, y)
                return a, b
            
            batches = ((x_i, y_i) for x_i, y_i in zip(X.pin_memory(), Y.pin_memory()))
        
            results = process_batches_cuda(
                batches,
                func,
                device_func='cuda:0',
                device_return='cpu',
                pin_memory=False,
                non_blocking=True,
                progress_bar=True,
                len_batches=int(math.ceil(len(X)/batch_size),
                verbose=False,
            )
    """
    results = []
    device_func = torch.device(device_func)

    assert device_func.type in ['cuda', 'cpu'], 'device must be either "cuda" or "cpu" type.'

    if len_batches is None:
        len_batches = len(batches) if hasattr(batches, '__len__') else None
    
    def send_to_device(batch, device, non_blocking, pin_memory):
        """
        Send a batch (either a single tensor, tuple containing tensors, or
        other) to a device.
        """
        if isinstance(batch, torch.Tensor):
            if pin_memory and batch.device.type != 'cuda':
                batch = batch.pin_memory()

            if batch.device != device:
                if non_blocking:
                    batch = batch.to(device, non_blocking=non_blocking)
                else:
                    batch = batch.to(device)

        elif isinstance(batch, tuple):
            batch = tuple([send_to_device(b, device, non_blocking, pin_memory) for b in batch])
        
        return batch

    if device_func.type == 'cuda':
        stream = torch.cuda.Stream(device=device_func)  ## create a stream
        for ii, batch in tqdm(enumerate(batches), total=len_batches, disable=not progress_bar):
            if not isinstance(batch, tuple):  ## Make batch a tuple
                batch = (batch,)
            with torch.cuda.stream(stream):  ## run on new stream
                batch = send_to_device(batch=batch, device=device_func, non_blocking=non_blocking, pin_memory=pin_memory)
                outs = func(*batch)  ## run func on batch
                outs = send_to_device(batch=outs, device=device_return, non_blocking=non_blocking, pin_memory=False)
                results.append(outs)  ## append to results
        torch.cuda.synchronize()  ## wait for all streams to finish
    else:
        warnings.warn('device_func is not cuda. No streams will be used.') if verbose else None
        for ii, batch in tqdm(enumerate(batches), total=len_batches, disable=not progress_bar):
            ## Make batch a tuple
            if not isinstance(batch, tuple):
                batch = (batch,)
            batch = send_to_device(batch=batch, device=device_func, non_blocking=non_blocking, pin_memory=pin_memory)
            outs = func(*batch)
            outs = send_to_device(batch=outs, device=device_return, non_blocking=non_blocking, pin_memory=False)
            results.append(outs)

    return results


##################################################################
############# STUFF PYTORCH SHOULD ALREADY HAVE ##################
##################################################################


numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}


@misc.wrapper_flexible_args(['dim', 'axis'])
@misc.wrapper_flexible_args(['keepdim', 'keepdims'])
def nanvar(
    arr: torch.Tensor, 
    dim: Optional[int] = None, 
    keepdim: bool = False, 
    ddof: int = 0, 
    ret_mean: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes the variance of an array ignoring any NaN values.
    RH 2022

    Args:
        arr (torch.Tensor): 
            The input tensor from which to calculate the variance. 
        dim (Optional[int]): 
            The dimension along which to compute the variance. 
            If ``None``, computes variance over all dimensions. (Default is ``None``)
        keepdim (bool): 
            If ``True``, the output tensor is of the same size as input 
            except in the dimension(s) dim where it is of size 1. 
            Otherwise, dim is squeezed. (Default is ``False``)
        ddof (int): 
            Delta degrees of freedom. (Default is *0*)
        ret_mean (bool): 
            If ``True``, return mean along with variance. (Default is ``False``)

    Returns:
        (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
            var (torch.Tensor):
                Variance of the input tensor ignoring any NaN values.
            avg (torch.Tensor):
                Mean of the input tensor ignoring any NaN values. This is only returned if ret_mean is ``True``.
    """
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, 0)
    cnt = torch.sum(torch.logical_not(nan_mask), dim=dim, keepdim=True)
    avg = torch.sum(arr_no_nan, dim=dim, keepdim=True) / cnt
    dev_sqrd = (arr_no_nan - avg)**2
    var = torch.sum(dev_sqrd, dim=dim, keepdim=True) / cnt
    dof = cnt - ddof
    out = var * cnt / dof

    out = squeeze_multiple_dims(out, dims=dim) if keepdim==False else out
    return out if ret_mean==False else (out, avg)

@misc.wrapper_flexible_args(['dim', 'axis'])
@misc.wrapper_flexible_args(['keepdim', 'keepdims'])
def nanstd(
    arr: torch.Tensor, 
    dim: Optional[int] = None, 
    keepdim: bool = False, 
    ddof: int = 0, 
    ret_mean: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes the standard deviation of an array ignoring any NaN values. 
    RH 2022

    Args:
        arr (torch.Tensor): 
            The input tensor from which to calculate the standard deviation. 
        dim (Optional[int]): 
            The dimension along which to compute the standard deviation. If
            ``None``, computes standard deviation over all dimensions. (Default
            is ``None``)
        keepdim (bool): 
            If ``True``, the output tensor is of the same size as input except
            in the dimension(s) dim where it is of size 1. Otherwise, dim is
            squeezed. (Default is ``False``)
        ddof (int): 
            Delta degrees of freedom. (Default is *0*)
        ret_mean (bool): 
            If ``True``, return mean along with standard deviation. (Default is
            ``False``)

    Returns:
        (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
            std (torch.Tensor):
                Standard deviation of the input tensor ignoring any NaN values.
            avg (torch.Tensor):
                Mean of the input tensor ignoring any NaN values. This is only
                returned if ret_mean is ``True``.
    """
    var = nanvar(arr, dim=dim, keepdim=keepdim, ddof=ddof, ret_mean=ret_mean)
    if ret_mean:
        var, avg = var
    std = torch.sqrt(var)
    return std if ret_mean==False else (std, avg)
        
@misc.wrapper_flexible_args(['dim', 'axis'])
@misc.wrapper_flexible_args(['keepdim', 'keepdims'])
def nanmax(
    arr: torch.Tensor, 
    dim: Optional[int] = None, 
    keepdim: bool = False
) -> torch.Tensor:
    """
    Computes the max of an array ignoring any NaN values.
    RH 2022

    Args:
        arr (torch.Tensor): 
            The input tensor from which to calculate the max value. 
        dim (Optional[int]): 
            The dimension along which to compute the max. If ``None``, computes
            max over all dimensions. (Default is ``None``)
        keepdim (bool): 
            If ``True``, the output tensor is of the same size as input except
            in the dimension(s) dim where it is of size 1. Otherwise, dim is
            squeezed. (Default is ``False``)

    Returns:
        (torch.Tensor):
            max_val (torch.Tensor):
                Maximum value of the input tensor ignoring any NaN values.
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    ## Get min value for the dtype. Use -inf if dtype is float, otherwise use min value for the dtype
    val_min = torch.as_tensor(float('-inf'), dtype=arr.dtype, device=arr.device) if arr.dtype in [torch.float32, torch.float64] else torch.iinfo(arr.dtype).min
    arr_no_nan = arr.masked_fill(nan_mask, val_min)
    return torch.max(arr_no_nan, **kwargs)

@misc.wrapper_flexible_args(['dim', 'axis'])
@misc.wrapper_flexible_args(['keepdim', 'keepdims'])
def nanmin(
    arr: torch.Tensor, 
    dim: Optional[int] = None, 
    keepdim: bool = False
) -> torch.Tensor:
    """
    Computes the min of an array ignoring any NaN values.
    RH 2022

    Args:
        arr (torch.Tensor): 
            The input tensor from which to calculate the min value. 
        dim (Optional[int]): 
            The dimension along which to compute the min. If ``None``, computes
            min over all dimensions. (Default is ``None``)
        keepdim (bool): 
            If ``True``, the output tensor is of the same size as input except
            in the dimension(s) dim where it is of size 1. Otherwise, dim is
            squeezed. (Default is ``False``)

    Returns:
        (torch.Tensor):
            min_val (torch.Tensor):
                Minimum value of the input tensor ignoring any NaN values.
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    ## Get max value for the dtype. Use inf if dtype is float, otherwise use max value for the dtype
    val_max = torch.as_tensor(float('inf'), dtype=arr.dtype, device=arr.device) if arr.dtype in [torch.float32, torch.float64] else torch.iinfo(arr.dtype).max
    arr_no_nan = arr.masked_fill(nan_mask, val_max)
    return torch.min(arr_no_nan, **kwargs)


def rolling_mean(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the running mean along a specified dimension using a rolling accumulation method
    (Welford's update for the mean).

    RH 2025

    Args:
        tensor (torch.Tensor):
            The input tensor on which the running mean is computed.
        dim (int):
            The dimension along which to compute the running mean.

    Returns:
        torch.Tensor:
            A tensor of the same shape as `tensor`, where each element along the specified dimension
            is the running mean of the elements from the start up to that index.
    """
    # Ensure the dimension is non-negative and valid.
    if dim < 0:
        dim += tensor.dim()
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Invalid dimension {dim} for tensor with {tensor.dim()} dimensions.")
    
    # Unbind the tensor along the given dimension to get a list of slices.
    dims_permute = list(range(tensor.dim()))
    ## remove dim from the list
    dims_permute.remove(dim)
    dims_permute = [dim] + dims_permute
        
    # Initialize an empty list to store the running means.
    current_mean = None
    
    # Iterate through each slice along the given dimension.
    # Use a counter starting at 1 since we divide by the count.
    for i, slice in enumerate(tensor.permute(dims_permute)):
        if current_mean is None:
            # For the first element, the running mean is the element itself.
            current_mean = slice
        else:
            # Update the running mean using:
            current_mean = current_mean + (slice - current_mean) / (i + 1)
    
    # Stack the list of running means back into a tensor along the specified dimension.
    return current_mean


def unravel_index(
    index: int, 
    shape: Tuple[int]
) -> List[int]:
    """
    Converts a flat index into a coordinate in a tensor of certain shape.
    RH 2022

    Args:
        index (int): 
            The flat index to be converted into a coordinate.
        shape (Tuple[int]): 
            The shape of the tensor in which the coordinate is calculated.

    Returns:
        (List[int]): 
            coord (List[int]): 
                The coordinate in the tensor.
    """
    out = []
    for dim in shape[::-1]:
        out.append(index % dim)
        index = index // dim
    return tuple(out[::-1])

@misc.wrapper_flexible_args(['dim', 'axis'])
def squeeze_multiple_dims(
    arr: torch.Tensor, 
    dims: Tuple[int, int] = (0, 1)
) -> torch.Tensor:
    """
    Squeezes multiple dimensions of a tensor.
    RH 2022

    Args:
        arr (torch.Tensor): 
            The input tensor to squeeze.
        dims (Tuple[int, int]): 
            The dimensions to squeeze. \n
            * dims[0]: Dimension to squeeze.
            * dims[1]: Dimension to squeeze. \n
            (Default is ``(0, 1)``)

    Returns:
        (torch.Tensor): 
            arr_squeezed (torch.Tensor):
                The squeezed tensor.
    """
    assert all([arr.shape[d] == 1 for d in dims])
    ## make custom slices to squeeze out the dims
    slices = [slice(None)] * arr.ndim
    for d in dims:
        slices[d] = 0
    return arr[tuple(slices)]


def multiply_elementwise_sparse_dense(
    s: torch.sparse.FloatTensor, 
    d: torch.Tensor
) -> torch.Tensor:
    """
    Multiplies a sparse tensor (s) with a dense tensor (d) elementwise.
    **Warning:** This creates an intermediate dense tensor with the same shape as s.
    RH 2022

    Args:
        s (torch.sparse.FloatTensor): 
            Sparse tensor to multiply.
        d (torch.Tensor): 
            Dense tensor to multiply.

    Returns:
        (torch.Tensor): 
            multiplied_tensor (torch.Tensor):
                Tensor resulting from the elementwise multiplication of `s` and
                `d`.
    """
    return torch.mul(d.expand(s.shape).sparse_mask(s.coalesce()), s)

def permute_sparse(
    input: torch.sparse.FloatTensor, 
    dims: List[int]
) -> torch.sparse.FloatTensor:
    """
    Permutes the dimensions of a sparse tensor.
    Adapted from https://github.com/pytorch/pytorch/issues/78422

    Args:
        input (torch.sparse.FloatTensor): 
            The input sparse tensor to permute.
        dims (List[int]): 
            The desired ordering of dimensions.

    Returns:
        (torch.sparse.FloatTensor): 
            permuted_tensor (torch.sparse.FloatTensor):
                The permuted sparse tensor.
    """
    dims = torch.LongTensor(dims)
    return torch.sparse_coo_tensor(indices=input._indices()[dims], values=input._values(), size=torch.Size(torch.tensor(input.size())[dims]))

def roll_sparse(
    X: torch.sparse.FloatTensor, 
    shifts: Union[int, Tuple[int]], 
    dims: Union[int, Tuple[int]]
) -> torch.sparse.FloatTensor:
    """
    Rolls a sparse tensor along the specified dimensions.

    Args:
        X (torch.sparse.FloatTensor): 
            Pytorch sparse tensor to roll.
        shifts (Union[int, Tuple[int]]): 
            Number of places by which elements are shifted.
        dims (Union[int, Tuple[int]]): 
            The dimensions along which the tensor is rolled.

    Returns:
        (torch.sparse.FloatTensor): 
            X_out (torch.sparse.FloatTensor): 
                The rolled sparse tensor.

    RH 2022
    """
    if type(shifts) is not tuple:
        shifts = (shifts,)
    if type(dims) is not tuple:
        dims = (dims,)

    X_out = copy.copy(X)
    for shift, dim in zip(shifts, dims):
        idx = X_out._indices()
        idx[dim] = (idx[dim] + shift) % X_out.shape[dim]
        X_out = torch.sparse_coo_tensor(indices=idx, values=X.coalesce().values(), size=X_out.shape)
    return X_out

def diag_sparse(
    x: torch.sparse.FloatTensor, 
    return_sparse_vals: bool = True
) -> torch.Tensor:
    """
    Gets the diagonal of a sparse tensor.

    Args:
        x (torch.sparse.FloatTensor): 
            Pytorch sparse tensor to extract the diagonal from.
        return_sparse_vals (bool): \n
            * If ``True``, returns 'sparse' (zeroed) values as well. 
            * If ``False``, returns only specified (non-sparsed out) values. 

    Returns:
        (torch.Tensor): 
            Diagonal values from the input sparse tensor.

    RH 2022
    """
    if return_sparse_vals is False:
        row, col = x.indices()
        values = x.values()
        return values[row == col]
    else:
        x_ts = indexing.torch_to_torchSparse(x)
        return x_ts.get_diag()

class Diag_sparse:
    """
    Gets the diagonal of a sparse tensor.
    The indices are kept as a property of the class for faster access.

    RH 2022

    Attributes:
        idx (torch.Tensor): 
            Indices of the diagonal elements in the sparse tensor.

    Args:
        x (torch.sparse.FloatTensor): 
            Sparse tensor to get diagonal of. For the __init__, 
            only the indices of the diagonal are kept as a property. 
            Subsequent calls of the class can use any x with the same indices.
    """
    def __init__(self, x):
        """
        Initializes the Diag_sparse class and stores the indices of the
        diagonal.
        """
        row, col = x.indices()
        self.idx = (row == col)
    def __call__(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Returns the diagonal elements from the sparse tensor using stored
        indices.

        Args:
            x (torch.sparse.FloatTensor): 
                Sparse tensor to get diagonal of.

        Returns:
            (torch.Tensor): 
                Diagonal values from the input sparse tensor.
        """
        values = x.values()
        return values[self.idx]


@misc.wrapper_flexible_args(['dim', 'axis'])
def zscore(
    X: torch.Tensor, 
    dim: Optional[int] = None, 
    ddof: int = 0, 
    nan_policy: str = 'propagate', 
) -> torch.Tensor:
    """
    Computes the z-score of a tensor.

    Args:
        X (torch.Tensor): 
            Tensor to compute z-score of.
        dim (Optional[int]): 
            Dimension to compute z-score over. 
            (Default is ``None``)
        ddof (int): 
            Means Delta Degrees of Freedom. 
            The divisor used in calculations is ``N - ddof``, 
            where ``N`` represents the number of elements. 
            (Default is *0*)
        nan_policy (str): 
            Defines how to handle when input contains nan. 
            The following options are available (default is ``'propagate'``): \n
            * ``'propagate'``: returns nan
            * ``'raise'``: throws an error
            * ``'omit'``: performs the calculations ignoring nan values \n
            (Default is ``'propagate'``)

    Returns:
        (torch.Tensor): 
            Z-scored tensor.

    RH 2022
    """
    assert dim is not None, 'Must specify dimension to compute z-score over.'

    if nan_policy == 'omit':
        std, mean = nanstd(X, dim=dim, ddof=ddof, keepdim=True, ret_mean=True)
    elif nan_policy == 'propagate':
        std, mean = torch.std_mean(X, dim=dim, keepdim=True, unbiased=ddof==1)
    else:
        raise ValueError('nan_policy must be "omit" or "propagate".')
    return (X - mean) / std


@misc.wrapper_flexible_args(['dim', 'axis'])
def slice_along_dim(
    X: torch.Tensor, 
    dim: int, 
    idx: Union[int, slice, List[int], torch.Tensor]
) -> torch.Tensor:
    """
    Slices a tensor along a specified dimension.
    RH 2022

    Args:
        X (torch.Tensor): 
            Tensor to slice.
        dim (int): 
            Dimension to slice along.
        idx Union[int, slice, List[int], torch.Tensor]:
            Index / slice / list of indices / tensor of indices to slice with.

    Returns:
        (torch.Tensor): 
            sliced_tensor (torch.Tensor):
                Sliced tensor.
    """
    slices = [slice(None)] * X.ndim
    slices[dim] = idx
    return X[tuple(slices)]


def orthogonal_procrustes(
    A: torch.Tensor,
    B: torch.Tensor,
    check_finite: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Port of the scipy.linalg.orthogonal_procrustes function:
    https://github.com/scipy/scipy/blob/v1.13.0/scipy/linalg/_procrustes.py

    Computes the matrix solution of the orthogonal Procrustes problem.
    Given two matrices, A and B, find the orthogonal matrix that most closely
    maps A to B using the algorithm in [1].

    Args:
        A (torch.Tensor): 
            The input matrix.
        B (torch.Tensor): 
            The target matrix.
        check_finite (bool): 
            Whether to check that the input matrices contain only finite
            numbers. Disabling may give a performance gain, but may result in
            problems (crashes, non-termination) if the inputs do contain infinities
            or NaNs. (Default is ``True``)

    Returns:
        (Tuple[torch.Tensor, torch.Tensor]): 
            (R, scale):
                R (torch.Tensor):
                    The matrix solution of the orthogonal Procrustes problem.
                    Minimizes the Frobenius norm of ``(A @ R) - B``, subject to
                    ``R.T @ R = I``.
                scale (torch.Tensor):
                    Sum of the singular values of ``A.T @ B``.

    References:
        [1] Peter H. Schonemann, "A generalized solution of the orthogonal
        Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1966.
        :doi:`10.1007/BF02289451`
    """
    if check_finite:
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError("Input contains non-finite values.")
    assert A.shape == B.shape, 'Input matrices must have the same shape.'
    assert A.ndim == 2, 'Input matrices must be 2D.'

    U, S, V = torch.linalg.svd((B.T @ A).T, full_matrices=False)
    R = U @ V
    scale = S.sum()
    return R, scale


def dtype_to_complex(dtype: torch.dtype) -> torch.dtype:
    """
    Converts a real torch dtype to a complex dtype. \n
    Pytorch 2.2 has this, but it crashes after calling it ~100 times. \n
    RH 2024

    Args:
        dtype (torch.dtype): 
            Real dtype to convert to complex dtype.

    Returns:
        (torch.dtype): 
            complex_dtype (torch.dtype):
                Complex dtype.
    """
    map = {
        torch.float16: torch.complex32,
        torch.bfloat16: torch.complex64,
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }
    if dtype not in map:
        raise ValueError(f'{dtype} does not have a complex equivalent in map.')
    return map[dtype]
def dtype_to_real(dtype: torch.dtype) -> torch.dtype:
    """
    Converts a complex torch dtype to a real dtype. \n
    Pytorch 2.2 has this, but it crashes after calling it ~100 times. \n
    RH 2024

    Args:
        dtype (torch.dtype): 
            Complex dtype to convert to real dtype.

    Returns:
        (torch.dtype): 
            real_dtype (torch.dtype):
                Real dtype.
    """
    map = {
        torch.complex32: torch.float16,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }
    if dtype not in map:
        raise ValueError(f'{dtype} does not have a real equivalent in map.')
    return map[dtype]


def geomspace(
    start: Union[int, float], 
    stop: Union[int, float], 
    num: int, 
    endpoint: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns numbers spaced evenly on a log scale (a geometric progression).
    Matches numpy's geomspace.

    Args:
        start (Union[int, float]): 
            The starting value of the sequence.
        stop (Union[int, float]): 
            The final value of the sequence, unless `endpoint` is False.
        num (int): 
            Number of samples to generate.
        endpoint (bool): 
            If ``True``, `stop` is the last sample. Otherwise, it is not
            included. 
        dtype (torch.dtype):
            The data type to use for the tensor. 

    Returns:
        (torch.Tensor): 
            samples (torch.Tensor): 
                The samples on a log scale.
    """
    if start <= 0:
        raise ValueError('start must be greater than 0.')
    if stop <= 0:
        raise ValueError('stop must be greater than 0.')
    if endpoint:
        return torch.logspace(
            math.log10(start),
            math.log10(stop),
            int(num),
            dtype=dtype,
            base=10.0,
        )
    else:
        gain = 10 ** (math.log10(stop / start) / (num))
        return torch.logspace(
            math.log10(start),
            math.log10(start) + math.log10(gain) * (num - 1),
            int(num),
            dtype=dtype,
            base=10.0,
        )
    

class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, xnew, out=None):
        """
        Copy pasted from: https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py
        
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes 
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)
interp1d = Interp1d.apply
    

################ Circular Statistics ################


def _circfuncs_common(samples: torch.Tensor, high: float, low: float):
    """
    Helper function for circular statistics. \n 
    This function is used to ensure that the output of a circular operation is
    always within the range [low, high). \n 
    RH 2024

    Args:
        samples (np.ndarray or torch.Tensor):
            Input values
        high (float or np.ndarray or torch.Tensor):
            High value
        low (float or np.ndarray or torch.Tensor):
            Low value

    Returns:
        output (np.ndarray or torch.Tensor):
            Output values
    """
    nan, pi = (torch.tensor(v, dtype=samples.dtype, device=samples.device) for v in [torch.nan, torch.pi])

    ## If number of elements is 0, return nan
    if samples.numel() == 0:
        return nan, nan
    
    ## sin and cos
    samples = (samples - low) * 2.0 * pi / (high - low)
    sin_samp = torch.sin(samples)
    cos_samp = torch.cos(samples)

    return sin_samp, cos_samp


def circmean(samples: torch.Tensor, high: float = 2*np.pi, low: float = 0.0, axis: Optional[List[int]] = None, nan_policy: str = 'propagate'):
    """
    Circular mean of samples. Equivalent results to scipy.stats.circmean. \n
    RH 2024

    Args:
        samples (np.ndarray or torch.Tensor):
            Input values
        high (float or np.ndarray or torch.Tensor):
            High value
        low (float or np.ndarray or torch.Tensor):
            Low value
        axis (int):
            Axis along which to take the mean
        nan_policy (str):
            Policy for handling NaN values: \n
                * 'propagate' - Propagate NaN values.
                * 'omit' - Ignore NaN values.
                * 'raise' - Raise an error if NaN values are present.

    Returns:
        mean (np.ndarray or torch.Tensor):
            Mean values
    """

    if nan_policy == 'raise':
        if torch.any(torch.isnan(samples)):
            raise ValueError("Input contains NaN values")
    
    pi = torch.tensor(torch.pi, dtype=samples.dtype, device=samples.device)
    sin_samp, cos_samp = _circfuncs_common(samples, high, low)

    if nan_policy == 'omit':
        sin_sum = torch.nansum(sin_samp, dim=axis)
        cos_sum = torch.nansum(cos_samp, dim=axis)
    elif nan_policy == 'propagate':
        sin_sum = torch.sum(sin_samp, dim=axis)
        cos_sum = torch.sum(cos_samp, dim=axis)
    elif nan_policy == 'raise':
        sin_sum = torch.sum(sin_samp, dim=axis)
        cos_sum = torch.sum(cos_samp, dim=axis)
    else:
        raise ValueError("Invalid nan_policy")
    
    res = torch.arctan2(sin_sum, cos_sum)

    if res.ndim == 0:
        res = res + 2 * pi if res < 0 else res
    else:
        res[res < 0] += 2 * pi
    
    # res = res[()]

    return res*(high - low)/2.0/pi + low


def cirvar(samples: torch.Tensor, high=2*np.pi, low=0, axis=None, nan_policy='propagate'):
    """
    Circular variance of samples. Equivalent results to scipy.stats.circvar. \n
    RH 2024

    Args:
        samples (np.ndarray or torch.Tensor):
            Input values
        high (float or np.ndarray or torch.Tensor):
            High value
        low (float or np.ndarray or torch.Tensor):
            Low value
        axis (int):
            Axis along which to take the variance
        nan_policy (str):
            Policy for handling NaN values. Can only be 'propagate' for now.

    Returns:
        variance (np.ndarray or torch.Tensor):
            Variance values
    """

    if nan_policy != 'propagate':
        raise NotImplementedError("Only 'propagate' nan_policy is supported")
        
    sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_mean = sin_samp.mean(axis)
    cos_mean = cos_samp.mean(axis)
    
    R = torch.sqrt(sin_mean**2 + cos_mean**2)

    return 1 - R


def circstd(samples: torch.Tensor, high: float = 2*np.pi, low: float = 0.0, axis: Optional[List[int]] = None, nan_policy: str = 'propagate', normalize: bool = False):
    """
    Circular standard deviation of samples. Equivalent results to
    scipy.stats.circstd. \n 
    RH 2024

    Args:
        samples (np.ndarray or torch.Tensor):
            Input values
        high (float or np.ndarray or torch.Tensor):
            High value
        low (float or np.ndarray or torch.Tensor):
            Low value
        axis (int):
            Axis along which to take the standard deviation
        nan_policy (str):
            Policy for handling NaN values. Can only be 'propagate' for now.
        normalize (bool):
            Whether to normalize the standard deviation. If True, the result is
            equal to ``sqrt(-2*log(R))`` and does not depend on the variable
            units. If False (default), the returned value is scaled by
            ``((high-low)/(2*pi))``.


    Returns:
        std (np.ndarray or torch.Tensor):
            Standard deviation values
    """

    if nan_policy != 'propagate':
        raise NotImplementedError("Only 'propagate' nan_policy is supported")
    
    pi = torch.tensor(np.pi, dtype=samples.dtype, device=samples.device)

    sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_mean = sin_samp.mean(axis)
    cos_mean = cos_samp.mean(axis)
    R = torch.sqrt(sin_mean**2 + cos_mean**2)

    res = torch.sqrt(-2*torch.log(R))
    if not normalize:
        res *= (high-low)/(2.*pi)
    return res


##################################################################
######################### TENSORLY ###############################
##################################################################

def tensorly_cp_to_device(cp, device='cpu'):
    """
    Moves the factors and weights of a tensorly cp object to a particular
    device.

    RH 2024

    Args:
        cp (tensorly.cp_tensor.CP):
            The tensorly CP object to move to a device.
        device (str):
            The device to move the factors and weights to.

    Returns:
        cp (tensorly.cp_tensor.CP):
            The tensorly CP object with factors and weights moved to the device.
    """
    for ii in range(len(cp.factors)):
        cp.factors[ii] = cp.factors[ii].to(device)
    cp.weights = cp.weights.to(device)
    return cp