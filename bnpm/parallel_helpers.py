from typing import Callable, List, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import numpy as np
from tqdm.auto import tqdm

class ParallelExecutionError(Exception):
    """
    Exception class for errors that occur during parallel execution.
    Intended to be used with the ``map_parallel`` function.
    RH 2023

    Attributes:
        index (int):
            Index of the job that failed.
        original_exception (Exception):
            The original exception that was raised.
    """
    def __init__(self, index, original_exception):
        self.index = index
        self.original_exception = original_exception

    def __str__(self):
        return f"Job {self.index} raised an exception: {self.original_exception}"

def map_parallel(
    func: Callable, 
    args: List[Any], 
    method: str = 'multithreading', 
    n_workers: int = -1, 
    n_args: int = None,
    prog_bar: bool = True,
) -> List[Any]:
    """
    Maps a function to a list of arguments in parallel. Includes a progress bar
    and error handling that points to the job that failed.
    RH 2022

    Args:
        func (Callable): 
            The function to be mapped.
        args (List[Any]): 
            List of arguments to which the function should be mapped.
            Length of list should be equal to the number of arguments.
            Each element should then be an iterable for each job that is run.
        method (str): 
            Method to use for parallelization. Either \n
            * ``'multithreading'``: Use multithreading from concurrent.futures.
            * ``'multiprocessing'``: Use multiprocessing from concurrent.futures.
            * ``'mpire'``: Use mpire.
            * ``'pathos'``: Use pathos.multiprocessing.ProcessingPool (similar
            to multiprocessing, but with better serialization support).
            * ``'pytorch'``: Use PyTorch multiprocessing.
            * ``'serial'``: Use list comprehension. \n
            (Default is ``'multithreading'``)
        workers (int): 
            Number of workers to use. If set to -1, all available workers are
            used. (Default is ``-1``)
        n_args (int):
            Number of arguments to expect. If not provided, the length of the
            first element of args is used.
        prog_bar (bool): 
            Whether to display a progress bar using tqdm. (Default is ``True``)

    Returns:
        (List[Any]): 
            output (List[Any]): 
                List of results from mapping the function to the arguments.
                
    Example:
        .. highlight::python
        .. code-block::python

            result = map_parallel(max, [[1,2,3,4],[5,6,7,8]], method='multiprocessing', n_workers=3)
    """
    if n_workers == -1:
        n_workers = mp.cpu_count()

    ## Assert that args is a list
    assert isinstance(args, list), "args must be a list"
    ## Assert that each element of args is an iterable
    assert all([hasattr(arg, '__iter__') for arg in args]), "All elements of args must be iterable"

    ## Get number of arguments. If args is a generator, make None.
    if n_args is None:
        assert hasattr(args[0], '__len__'), f"If n_args is not provided, the first element of args must have a length."
        n_args = len(args[0])
    ## Assert that all args are the same length
    assert all([len(arg) == n_args for arg in args if hasattr(arg, '__len__')]), "All elements of args must have the same length"

    ## Make indices
    indices = np.arange(n_args)

    ## Prepare args_map (input to map function)
    args_map = [[func] * n_args, *args, indices]
        
    if method == 'multithreading':
        executor = ThreadPoolExecutor
        star_args = True
    elif method == 'multiprocessing':
        executor = ProcessPoolExecutor
        star_args = True
    elif method == 'mpire':
        import mpire
        executor = mpire.WorkerPool
        star_args = False
    elif method == 'pathos':
        from pathos.multiprocessing import ProcessingPool
        executor = ProcessingPool
        star_args = True
    elif method == 'pytorch':
        from torch.multiprocessing import Pool
        executor = Pool
        star_args = False
    elif method == 'serial':
        return    list(tqdm(map(_func_wrapper_helper, *args_map), total=n_args, disable=prog_bar!=True))
    else:
        raise ValueError(f"method {method} not recognized")

    with executor(n_workers) as ex:
        if star_args:
            return list(tqdm(ex.map(_func_wrapper_helper, *args_map), total=n_args, disable=prog_bar!=True))
        else:
            return list(tqdm(ex.map(_func_wrapper_helper, args_map), total=n_args, disable=prog_bar!=True))
def _func_wrapper_helper(*func_args_index):
    """
    Wrapper function to catch exceptions.
    
    Args:
    *func_args_index (tuple):
        Tuple of arguments to be passed to the function.
        Should take the form of (func, arg1, arg2, ..., argN, index)
        The last element is the index of the job.
    """
    func = func_args_index[0]
    args = func_args_index[1:-1]
    index = func_args_index[-1]

    try:
        return func(*args)
    except Exception as e:
        raise ParallelExecutionError(index, e)
    
    

def multiprocessing_pool_along_axis(x_in, function, n_workers=None, axis=0, **kwargs):
    pool = mp.Pool(processes=n_workers)
    if axis==0:
        results = pool.map(partial(function , **kwargs), [x_in[ii] for ii in range(x_in.shape[0])])
        pool.close()
        pool.join()
        return np.row_stack(results)
    elif axis==1:
        results = pool.map(partial(function , **kwargs), [x_in[:,ii] for ii in range(x_in.shape[1])])
        pool.close()
        pool.join()
        return np.column_stack(results)

def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)
def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    Taken from here: https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, mp.cpu_count())]

    pool = mp.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)
