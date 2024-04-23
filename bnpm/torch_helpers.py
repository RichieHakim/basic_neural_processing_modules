import sys
import gc
import copy
from typing import Union, List, Tuple, Dict, Callable, Optional, Any, Iterable, Iterator, Generator
from contextlib import contextmanager
import warnings

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from . import indexing
from . import misc


############################################
############ VARIABLE HELPERS ##############
############################################

def show_all_tensors() -> None:
    """
    Displays all tensors present in the provided dictionary.
    From: https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
    RH 2024
    """
    # prints currently alive Tensors and Variables
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), misc.estimate_array_size(obj))
        except:
            pass



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
    if use_GPU:
        print(f'devices available: {[torch.cuda.get_device_properties(ii) for ii in range(torch.cuda.device_count())]}') if verbose else None
        device = torch.device(device_num) if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print(f"Using device: '{device}': {torch.cuda.get_device_properties(device_num)}") if verbose else None
    else:
        device = "cpu"
        print(f"device: '{device}'") if verbose else None

    return device
    

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
    """
    from torch.profiler import profile, record_function, ProfilerActivity
    from contextlib import contextmanager
    
    activities_dict = {
        'CPU': ProfilerActivity.CPU,
        'CUDA': ProfilerActivity.CUDA,
    }
    activities = [activities_dict[act] for act in activities]

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
        return self.X[idx], idx


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


def process_batches_cuda(
    batches: Union[Iterable, Iterator, Generator],
    func: Callable,
    device_func: str = 'cuda:0',
    device_return: str = 'cpu',
    pin_memory: bool = False,
    non_blocking: bool = True,
    progress_bar: bool = False,
    len_batches: Optional[int] = None,
    verbose: bool = False,
):
    """
    Run batches through func on a specified device using CUDA streams.
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
            However, this option will pin the memory for each batch. (Default is
            ``False``)
        non_blocking (bool):
            If ``True``, the function will use non-blocking transfers to and
            from the device. (Default is ``True``)
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
        Send a batch (either a single tensor, tuple containing tensors, or other) to a device.
        """
        if isinstance(batch, torch.Tensor):
            if pin_memory and batch.device.type != 'cuda':
                batch = batch.pin_memory()

            if non_blocking:
                batch = batch.to(device, non_blocking=non_blocking)
            else:
                batch = batch.to(device)

        elif isinstance(batch, tuple):
            batch = tuple([send_to_device(b, device, non_blocking, pin_memory) for b in batch])
        
        return batch

    if device_func.type == 'cuda':
        stream = torch.cuda.Stream(device=device_func)
        for ii, batch in tqdm(enumerate(batches), total=len_batches, disable=not progress_bar):
            ## Make batch a tuple
            if not isinstance(batch, tuple):
                batch = (batch,)
            with torch.cuda.stream(stream):
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


#########################################################
############ INTRA-MODULE HELPER FUNCTIONS ##############
#########################################################

def _convert_size(
    size: Union[int, float], 
    return_size: str = 'GB'
) -> float:
    """
    Converts the size in bytes to another unit (TB, GB, MB, KB, or B).
    RH 2021

    Args:
        size (Union[int, float]): 
            Size in bytes.
        return_size (str): 
            Unit to which the size should be converted. Either \n
            * ``'TB'``: Terabytes
            * ``'GB'``: Gigabytes
            * ``'MB'``: Megabytes
            * ``'KB'``: Kilobytes
            * ``'B'``: Bytes \n
            (Default is ``'GB'``)

    Returns:
        (float): 
            out_size (float):
                Size converted to the specified unit.

    Example:
        .. highlight:: python
        .. code-block:: python

            size_in_gb = _convert_size(1000000000, 'GB')
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

