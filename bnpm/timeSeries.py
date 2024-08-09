from typing import List, Tuple, Union, Optional, Dict, Any
import functools
import time
import copy
import math

import numpy as np
from matplotlib import pyplot as plt
from numba import njit, jit, prange
import pandas as pd
import torch

from . import parallel_helpers
from .parallel_helpers import multiprocessing_pool_along_axis
from . import cross_validation
from .math_functions import polar2cartesian, cartesian2polar, gaussian
from . import indexing
from . import torch_helpers


def convolve_along_axis(
    array,
    kernel, 
    axis=1 , 
    mode='same', 
    correct_edge_effects=True,
    multicore_pref=False, 
    verbose=False
    ):
    '''
    Convolves an array with a kernel along a defined axis
    Use a continguous array ( np.ascontinguousarray() ) 
    along axis=1 for top speed.
    if multicore_pref==True, array must be 2-D
    
    Also try 'convolve_numba' which is a manual implemenatation
    of a convolution, and of similar speed (sometimes faster)
    when using continguous arrays along axis=1, but it makes NaNs
    at edges instead of the various mode options in numpy.convolve

    if you want more speed see (try using CuPy.convolve):
        https://numba.discourse.group/t/numba-convolutions/33/4
        https://github.com/randompast/python-convolution-comparisons

    RH 2021
    
    Args:
        array (np.ndarray): 
            array you wish to convolve
        kernel (np.ndarray): 
            array to be used as the convolutional kernel (see numpy.convolve documentation)
        axis (int): 
            axis to convolve array along. NOT USED IF multicore_pref==True
        mode (str): 
            see numpy.convolve documentation. Can be 'valid', 'same', 'full'
        correct_edge_effects (bool):
            Whether or not to correct for edge effects.
            Here, correcting for edge effects means to
             normalize each time point by the number of
             samples actually used in the convolution 
             at each time point
            
    Returns:
        output (np.ndarray): input array convolved with kernel
    '''
    tic = time.time()

    axis = array.ndim + axis if axis<0 else axis  ## convert negative axis to positive axis

    if array.ndim == 1:
        multicore_pref = False
    if multicore_pref:
        def convFun_axis0(iter):
            return np.convolve(array[:,iter], kernel, mode=mode)
        def convFun_axis1(iter):
            return np.convolve(array[iter,:], kernel, mode=mode)

        kernel = np.ascontiguousarray(kernel)
        if axis==0:
            output_list = parallel_helpers.map_parallel(convFun_axis0, [range(array.shape[1])], method='multithreading', n_workers=-1, prog_bar=verbose)
        if axis==1:
            output_list = parallel_helpers.map_parallel(convFun_axis1, [range(array.shape[0])], method='multithreading', n_workers=-1, prog_bar=verbose)
        
        if verbose:
            print(f'ThreadPool elapsed time : {round(time.time() - tic , 2)} s. Now unpacking list into array.')
        
        if axis==0:
            output = np.array(output_list).T
        if axis==1:
            output = np.array(output_list)

    else:
        output = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis=axis, arr=array)

    if correct_edge_effects:
        trace_norm = np.convolve(
            a=np.ones(array.shape[axis]),
            v=kernel,
            mode=mode
            )

        trace_norm_padded = indexing.pad_with_singleton_dims(trace_norm, n_dims_pre=axis, n_dims_post=array.ndim-axis-1)

        output = output / trace_norm_padded

    if verbose:
        print(f'Calculated convolution. Total elapsed time: {round(time.time() - tic,2)} seconds')

    return output
    

def threshold(
    array, 
    thresh_max=None, 
    thresh_min=None, 
    val_max=None, 
    val_min=None, 
    inPlace_pref=False):
    """
    Thresholds values in an array and sets them to defined values
    RH 2021

    Args:
        array (np.ndarray):  
            the mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D gaussian
        thresh_max (float): 
            values in array above this are set to val_max
        thresh_min (float): 
            values in array below this are set to val_min
        val_max (float): 
            values in array above thresh_max are set to this
        val_min (float): 
            values in array above thresh_min are set to this
        inPlace_pref (bool): 
            whether to do the calculation 'in place', and change the local input variable directly

    Return:
        output_array (np.ndarray): 
            same as input array but with values thresheld
    """
    if val_max is None:
        val_max = thresh_max
    if val_min is None:
        val_min = thresh_min

    if inPlace_pref:
        output_array = array
    else:
        output_array = copy.deepcopy(array)

    if thresh_max is None:
        output_array[output_array < thresh_min] = val_min
    elif thresh_min is None:
        output_array[output_array > thresh_max] = val_max
    else:
        output_array[output_array < thresh_min] = val_min
        output_array[output_array > thresh_max] = val_max
    return output_array


def scale_between(
    x, 
    low=0, 
    high=1, 
    axis=0, 
    low_percentile=None, 
    high_percentile=None, 
    clip=True,
):
    '''
    Scales the first (or more) dimension of an array to be between 
    lower and upper bounds.
    RH 2024

    Args:
        x (ndarray):
            Any dimensional array. First dimension is scaled
        lower (scalar):
            lower bound for scaling
        upper (scalar):
            upper bound for scaling
        axis (int or tuple):
            axis to scale along
        low_percentile (scalar):
            Ranges between 0-100. Defines what percentile of
            the data to set as the lower bound
        high_percentile (scalar):
            Ranges between 0-100. Defines what percentile of
            the data to set as the upper bound
        clip (bool):
            If true then data is clipped to be between lower
            and upper. Only meaningful if lower_percentile or
            upper_percentile is not None.

    Returns:
        x_out (ndarray):
            Scaled array
    '''
    if isinstance(x, torch.Tensor):
        percentile = lambda x, p: torch.quantile(x, q=p/100, axis=axis, keepdims=True)
        nanmin = lambda x: torch_helpers.nanmin(x, axis=axis, keepdims=True)[0]
        nanmax = lambda x: torch_helpers.nanmax(x, axis=axis, keepdims=True)[0]
        clip = lambda x, low, high: torch.clamp(x, min=low, max=high)
    elif isinstance(x, np.ndarray):
        percentile = lambda x, p: np.percentile(x, q=p, axis=axis, keepdims=True)
        nanmin = lambda x, axis: np.nanmin(x, axis=axis, keepdims=True)
        nanmax = lambda x: np.nanmax(x, axis=axis, keepdims=True)
        clip = lambda x, low, high: np.clip(x, a_min=low, a_max=high)

    if low_percentile is not None:
        lowest_val = percentile(x, low_percentile)
    else:
        lowest_val = nanmin(x)
    if high_percentile is not None:
        highest_val = percentile(x, high_percentile)
    else:
        highest_val = nanmax(x)

    x_out = ((x - lowest_val) * (high - low) / (highest_val - lowest_val) ) + low

    if clip:
        x_out = clip(x_out, low, high)

    return x_out


def rolling_percentile_pd(
    X, 
    ptile=50, 
    window=21, 
    interpolation='linear', 
    multiprocessing_pref=True,
    prog_bar=False,
    **kwargs_rolling
):
    '''
    Computes a rolling percentile over one dimension 
     (defaults to dim 1 / rows).
    This function is currently just a wrapper for pandas'
     rolling library.
    Input can be pandas DataFrame or numpy array, and output
     can also be either.
    I tried to accelerate this with multithreading and numba and
    they don't seem to help or work. Also tried the new 
    rolling_quantiles stuff (https://github.com/marmarelis/rolling-quantiles)
    and saw only modest speed ups at the cost of all the
    parameters. I'm still not sure if anyone is using efficient
    insertion sorting.
    RH 2021

    Args:
        X (numpy.ndarray OR pd.core.frame.DataFrame):
            Input array of signals. Calculation done over
            dim 1 (rows).
        ptile (scalar):
            Percentile. 0-100.
        window (scalar):
            Size of window in samples. Ideally odd integer.
        interpolation (string):
            For details see: pandas.core.window.rolling.Rolling.quantile
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.rolling.Rolling.quantile.html
            Can be: linear, lower, higher, nearest, midpoint
        multiprocessing_pref (bool):
            Whether to use multiprocessing to speed up the calculation.
        prog_bar (bool):
            Whether to show a progress bar. For multiprocessing.
        **kwargs_rolling (dict):
            kwargs for pandas.DataFrame.rolling function call.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
            Includes: min_periods, center, axis, win_type, closed

    Returns:
        output (numpy.ndarray OR pd.core.frame.DataFrame):
            Output array of signals.
    '''
    
    if 'min_periods' not in kwargs_rolling:
        kwargs_rolling['min_periods'] = 1
    if 'center' not in kwargs_rolling:
        kwargs_rolling['center'] = True
    if 'axis' not in kwargs_rolling:
        kwargs_rolling['axis'] = 1
    if 'win_type' not in kwargs_rolling:
        kwargs_rolling['win_type'] = None
    if 'closed' not in kwargs_rolling:
        kwargs_rolling['closed'] = None

    from functools import partial
    # _rolling_ptile_pd_helper_partial = partial(_rolling_ptile_pd_helper, win=int(window), ptile=ptile, kwargs_rolling=kwargs_rolling, interpolation=interpolation)
    ## Avoid using partial because it doesn't work with multiprocessing

    if multiprocessing_pref:
        from .parallel_helpers import map_parallel
        import multiprocessing as mp
        batches = list(indexing.make_batches(X, num_batches=mp.cpu_count()))
        n_batches = len(batches)
        ## Make args as a list of iterables, each with length n_batches. This is the format for map and map_parallel
        args = [
            batches,
            [int(window)] * n_batches,
            [ptile] * n_batches,
            [kwargs_rolling] * n_batches,
            [interpolation] * n_batches
        ]
        output = map_parallel(
            _rolling_ptile_pd_helper,
            args,
            method='multiprocessing',
            n_workers=-1,
            prog_bar=prog_bar,
        )
        output = np.concatenate(output, axis=0)
    else:
        output = _rolling_ptile_pd_helper(X, win=int(window), ptile=ptile, kwargs_rolling=kwargs_rolling, interpolation=interpolation)

    return output
def _rolling_ptile_pd_helper(X, win, ptile, kwargs_rolling, interpolation='linear'):
    return pd.DataFrame(X).rolling(
        window=win, 
        **kwargs_rolling
    ).quantile(
        ptile/100, 
        numeric_only=True, 
        interpolation=interpolation, 
        # engine='numba', 
        # engine_kwargs={'nopython': True, 'nogil': True, 'parallel': True}
    ).to_numpy()

def rolling_percentile_rq(x_in, window, ptile=10, stride=1, center=True):
    import rolling_quantiles as rq
    pipe = rq.Pipeline( rq.LowPass(window=window, quantile=(ptile/100), subsample_rate=stride) )
    lag = int(np.floor(pipe.lag))
    if center:
        return pipe.feed(x_in)[lag:]
    else:
        return pipe.feed(x_in)
def rolling_percentile_rq_multicore(x_in, window, ptile, stride=1, center=True, n_workers=None):
    import rolling_quantiles as rq
    return multiprocessing_pool_along_axis(x_in, rolling_percentile_rq, n_workers=n_workers, axis=0, **{'window': window , 'ptile': ptile, 'stride': stride, 'center': center} )


def event_triggered_traces(
    arr, 
    idx_triggers, 
    win_bounds=[-100,0], 
    dim=0,
    verbose=1,
):
    """
    Makes event triggered traces along last dimension.
    New version using torch.
    RH 2022
    
    Args:
        arr (np.ndarray or torch.Tensor):
            Input array. Dimension 'dim' will be 
             aligned to the idx specified in 'idx_triggers'.
        idx_triggers (list of int, np.ndarray, or torch.Tensor):
            1-D boolean array or 1-D index array.
            True values or idx values are trigger events.
        win_bounds (size 2 integer list, tuple, or np.ndarray):
            2 value integer array. win_bounds[0] should
             be negative and is the number of samples prior
             to the event that the window starts. 
             win_bounds[1] is the number of samples 
             following the event.
            Events that would have a window extending
             before or after the bounds of the length
             of the trace are discarded.
        dim (int):
            Dimension of 'arr' to align to 'idx_triggers'.
        verbose (int):
            0: no print statements
            1: print warnings
            2: print warnings and info
        
     Returns:
        et_traces (np.ndarray or torch.Tensor):
             Event Triggered Traces. et_traces.ndim = arr.ndim+1.
             Last two dims are the event triggered traces.
             Shape: (len(idx_triggers), len(window), lengths of other dimensions besides 'dim')
        xAxis (np.ndarray or torch.Tensor):
            x-axis of the traces. Aligns with dimension
             et_traces.shape[1]
        windows (np.ndarray or torch.Tensor):
            Index array of the windows used.
    """
    from warnings import warn

    ## Error checking
    assert isinstance(dim, int), "dim must be int"
    assert isinstance(arr, (np.ndarray, torch.Tensor)), "arr must be np.ndarray or torch.Tensor"
    assert isinstance(idx_triggers, (list, np.ndarray, torch.Tensor)), "idx_triggers must be list, np.ndarray, or torch.Tensor"
    assert isinstance(win_bounds, (list, tuple, np.ndarray)), "win_bounds must be list, tuple, or np.ndarray"

    ## Convert stuff to torch Tensors
    dtype_in = 'np' if isinstance(arr, np.ndarray) else 'torch' if isinstance(arr, torch.Tensor) else 'unknown'
    if isinstance(arr, np.ndarray):
        arr = torch.as_tensor(arr)
    if isinstance(idx_triggers, np.ndarray):
        idx_triggers = torch.as_tensor(idx_triggers)
    if isinstance(win_bounds, np.ndarray):
        win_bounds = torch.as_tensor(win_bounds)

    ## Warn if idx_triggers are not integers
    if idx_triggers.dtype != torch.long:
        warn("idx_triggers should be integers. Converting to integers.")
        idx_triggers = idx_triggers.type(torch.long)

    ## Warn if idx_triggers are likely boolean (all values are isin 0 and 1)
    if torch.all(torch.isin(idx_triggers, torch.tensor([0,1]))):
        warn("idx_triggers are likely in boolean format. Please convert to index format via np.where or torch.where or similar")
    
    ## if idx_triggers is length 0, return empty arrays
    if len(idx_triggers)==0:
        print("idx_triggers is length 0. Returning empty arrays.") if verbose>0 else None
        shape_out = list(arr.shape)
        # replace dim with window length
        shape_out[dim] = win_bounds[1] - win_bounds[0]
        # insert 0 at dim
        shape_out.insert(dim, 0)
        
        arr_out = torch.empty(shape_out, dtype=arr.dtype)
        xAxis_out = torch.arange(win_bounds[0], win_bounds[1], dtype=torch.long)
        windows_out = torch.empty((0, win_bounds[1]-win_bounds[0]), dtype=torch.long)
        
        return tuple(v.numpy() if dtype_in=='np' else v for v in (arr_out, xAxis_out, windows_out))
    
    ## Find x-axis
    xAxis = torch.arange(win_bounds[0], win_bounds[1], dtype=torch.long)  ## x-axis for each window relative to trigger
    
    dim = arr.ndim + dim if dim<0 else dim  ## convert negative dim to positive dim
    
    idx_triggers_clean = torch.as_tensor(idx_triggers[~torch.isnan(idx_triggers)], dtype=torch.long)  ## remove nans from idx_triggers and convert to torch.long

    windows = torch.stack([xAxis + i for i in torch.as_tensor(idx_triggers_clean, dtype=torch.long)], dim=0)  ## make windows. shape = (n_triggers, len_window)
    win_toInclude = (torch.any(windows<0, dim=1)==0) * (torch.any(windows>=arr.shape[dim], dim=1)==0)  ## boolean array of windows that are within the bounds of the length of 'dim'
    n_win_excluded = torch.sum(win_toInclude==False)  ## number of windows excluded due to window bounds. Only used for printing currently
    windows = windows[win_toInclude]  ## windows after pruning out windows that are out of bounds
    n_windows = windows.shape[0]  ## number of windows. Only used for printing currently

    print(f'number of triggers excluded due to window bounds:     {n_win_excluded}') if (n_win_excluded>0) and (verbose>1) else None
    print(f'number of triggers included and within window bounds: {len(windows)}') if verbose>2 else None

    shape = list(arr.shape)  ## original shape
    dims_perm = [dim] + list(range(dim)) + list(range(dim+1, len(shape)))  ## new dims for indexing. put dim at dim 0
    arr_perm = arr.permute(*dims_perm)  ## permute to put 'dim' to dim 0
    arr_idx = arr_perm.index_select(dim=0, index=windows.reshape(-1))  ## index out windows along dim 0
    rs = list(arr_perm.shape[1:]) + [n_windows, win_bounds[1]-win_bounds[0]]  ## new shape for unflattening. 'dim' will be moved to dim -1, then reshaped to n_windows x len_window
    arr_idx_rs = arr_idx.permute(*(list(range(1, arr_idx.ndim)) + [0])).reshape(*rs)  ## permute to put current 'dim' (currently dim 0) to end, then reshape

    arr_out = arr_idx_rs.numpy() if dtype_in=='np' else arr_idx_rs
    xAxis_out = xAxis.numpy() if dtype_in=='np' else xAxis
    windows_out = windows.numpy() if dtype_in=='np' else windows

    return arr_out, xAxis_out, windows_out


def make_sorted_event_triggered_average(
    arr, 
    idx_triggers, 
    win_bounds, 
    dim=0,
    cv_group_size=2, 
    test_frac=0.5, 
    show_plot=False
):
    '''
    Makes a sorted event triggered average plot
    RH 2021
    
    Args:
        arr (np.ndarray):
            Input array. Last dimension will be aligned
             to boolean True values in 'trigger_signal'.
        trigger_signal (boolean np.ndarray):
            1-D boolean array. True values are trigger
             events.
             Same as in event_triggered_traces.
        win_bounds (size 2 integer np.ndarray):
            2 value integer array. win_bounds[0] is the
             number of samples prior to the event that
             the window starts. win_bounds[1] is the 
             number of samples following the event.
            Events that would have a window extending
             before or after the bounds of the length
             of the trace are discarded.
            Same as in event_triggered_traces.
        dim (int):
            Axis/dimension to align to trigger_signal.
        cv_group_size (int):
            Number of samples per group. Uses sklearn's
             model_selection.GroupShuffleSplit.
            Same as in cross_validation.group_split
         test_frac (scalar):
             Fraction of samples in test set.
             Same as in cross_validation.group_split
        trigger_signal_is_idx (bool):
            If True then trigger_signal is an index array.
            If False then trigger_signal is a boolean array.
            Use an index array if there are multiple events
             with the same index, else they will be
             collapsed when this is 'True'.
         show_plot (bool):
             Whether or not to show the plot
     
     Returns:
        mean_traces_sorted (np.ndarray):
            Output traces/image. Size (arr.shape[0],
             win_bounds[1]-win_bounds[0])
            Shows the event triggered average of the 
             test set, sorted by the peak times found 
             in the training set.
        peaks_train_sorted (np.ndarray):
            Indices of the peaks for each row. Training set.
            Size (arr.shape[0],)
        peaks_test_sorted (np.ndarray):
            Indices of the peaks for each row. Test set.
            Size (arr.shape[0],)
        cv_idx (list of 2 lists):
            List of 2 lists.
            Outer list entries: Splits
            Inner list entries: Train, Test indices
            Same as return 'cv_idx' in 
             cross_validation.group_split
        et_traces (np.ndarray):
             Event Triggered Traces. et_traces.ndim = 
              arr.ndim+1. Last dimension is new and is
              the event number axis. Note that events 
              near edges are discarded if window extends
              past edge bounds.
        xAxis (np.ndarray):
            x-axis of the traces. Aligns with dimension
             et_traces.shape[-2]
        windows (np.ndarray):
            Index array of the windows used.       
    '''

    et_traces, xAxis, windows = event_triggered_traces(
        arr, 
        idx_triggers=idx_triggers,
        win_bounds=win_bounds, 
        dim=dim,       
    )

    ## if there are no triggers, return empty arrays and Nones
    if len(idx_triggers)==0:
        return None, None, None, None, et_traces, xAxis, windows

    cv_idx = cross_validation.group_split(1, et_traces.shape[1], cv_group_size, test_size=test_frac)

    mean_traces_train = np.nanmean(et_traces[:,cv_idx[0][0]], axis=1)
    mean_traces_test =  np.nanmean(et_traces[:,cv_idx[0][1]], axis=1)

    peaks_train = np.argmax(mean_traces_train, axis=1)
    peaks_test = np.argmax(mean_traces_test, axis=1)
    mean_traces_sorted = mean_traces_test[np.argsort(peaks_train)]
    
    if show_plot:
        plt.figure()
        plt.imshow(mean_traces_sorted, aspect='auto', 
                   extent=(xAxis[0], xAxis[-1],
                          mean_traces_sorted.shape[0], 0),
#                    vmin=-1, vmax=3
                  )

    peaks_train_sorted = np.sort(peaks_train)
    peaks_test_sorted = np.sort(peaks_test)
    return mean_traces_sorted, peaks_train_sorted, peaks_test_sorted, cv_idx, et_traces, xAxis, windows
    

def simple_smooth(arr, x=None, mu=0, sig=1, axis=0, mode='same', correct_edge_effects=True):
    '''
    Simple smoothing function.
    Uses convolve_along_axis math_functions.gaussian to 
     convolve over the any dimension.
    RH 2022

    Args:
        arr (np.ndarray or torch.Tensor):
            Input array.
        x (np.ndarray or torch.Tensor):
            x-axis for kernel.
            see math_functions.gaussian
        mu (scalar):
            Mean of gaussian.
            see math_functions.gaussian
        sig (scalar):
            Standard deviation of gaussian.
            see math_functions.gaussian
        axis (int):
            Axis to convolve over.
        mode (str):
            Mode of convolution.
            'valid' or 'same' or 'full'
        correct_edge_effects (bool):
            Whether or not to correct for edge effects.
            Here, correcting for edge effects means to
             normalize each time point by the number of
             samples actually used in the convolution 
             at each time point
        
    Returns:
        arr_smooth (np.ndarray or torch.Tensor):
            Smoothed array.
    '''
    gaus = gaussian(sig=sig, x=x, mu=mu, plot_pref=False)

    arr_conv = convolve_along_axis(
        array=arr, 
        kernel=gaus,
        axis=axis, 
        mode=mode, 
        correct_edge_effects=correct_edge_effects,
        multicore_pref=True, 
        verbose=False
    )
    
    return arr_conv


class Convolver_1d():
    """
    Class for 1D convolution.
    Uses torch.nn.functional.conv1d.
    Stores the convolution and edge correction kernels
     for repeated use.
    RH 2023
    """
    def __init__(
        self,
        kernel,
        length_x: int=None,
        dtype=torch.float32,
        pad_mode: str='same',
        correct_edge_effects: bool=True,
        device='cpu',
    ):
        """
        Args:
            kernel (np.ndarray or torch.Tensor):
                1D array to convolve with.
            length_x (int):
                Length of the array to be convolved.
                Must not be None if pad_mode is not 'valid'.
            pad_mode (str):
                Mode for padding.
                See torch.nn.functional.conv1d for details.
            correct_edge_effects (bool):
                Whether or not to correct for edge effects.
            device (str):
                Device to use.
        """
        self.pad_mode = pad_mode
        self.dtype = dtype

        ## convert kernel to torch tensor
        self.kernel = torch.as_tensor(kernel, dtype=dtype, device=device)[None,None,:]

        ## compute edge correction kernel
        if pad_mode != 'valid':
            assert length_x is not None, "Must provide length_x if pad_mode is not 'valid'"
            assert length_x >= kernel.shape[0], "length_x must be >= kernel.shape[0]"
            
            self.trace_correction = torch.conv1d(
                input=torch.ones((1,1,length_x), dtype=dtype, device=device),
                weight=self.kernel,
                padding=pad_mode,
            )[0,0,:] if correct_edge_effects else None
        else:
            self.trace_correction = None
            
    def convolve(self, arr) -> torch.Tensor:
        """
        Convolve array with kernel.
        Args:
            arr (np.ndarray or torch.Tensor):
                Array to convolve.
                Convolution performed along the last axis.
                ndim must be 1 or 2 or 3.
        """
        ## make array 3D by adding singleton dimensions if necessary
        ndim = arr.ndim
        if ndim == 1:
            arr = arr[None,None,:]
        elif ndim == 2:
            arr = arr[None,:,:]
        assert arr.ndim == 3, "Array must be 1D or 2D or 3D"

        ## convolve along last axis
        out = torch.conv1d(
            input=torch.as_tensor(arr, dtype=self.dtype, device=self.kernel.device),
            weight=self.kernel,
            padding=self.pad_mode,
        )

        ## correct for edge effects
        if self.trace_correction is not None:
            out = out / self.trace_correction[None,None,:]
            
        ## remove singleton dimensions if necessary
        if ndim == 1:
            out = out[0,0,:]
        elif ndim == 2:
            out = out[0,:,:]
        return out
    
    def __call__(self, arr):
        return self.convolve(arr)
    def __repr__(self) -> str:
        return f"Convolver_1d(kernel shape={self.kernel.shape}, pad_mode={self.pad_mode})"
    

@njit(parallel=False, fastmath=True, nogil=True)
def _helper_dampening_filter_numba(X: np.ndarray, X_diff: np.ndarray, dampening_factor: float):
    """
    Numba helper function for dampening_filter_numba.
    """
    Y = np.zeros_like(X)
    for i in range(1, len(X)):
        Y[i] = Y[i-1]*dampening_factor + X_diff[i-1]
    return Y
def dampening_filter(X: np.ndarray, dampening_factor: float = 0.9):
    """
    Dampening filter. Applies a recursive dampening filter that pulls values
    back towards zero.
    function: y[t] = y[t-1]*d + (x[t] - x[t-1])
    numba implementation appears to be faster than torch

    RH 2024

    Args:
        X (np.ndarray):
            Input array. Dampening will be performed
            along the first dimension (columns).
        dampening_factor (float):
            Dampening factor. Should typically be between 0 and 1.

    Returns:
        Y (np.ndarray):
            Dampened array.
    """
    X_diff = np.diff(X, axis=0)
    Y = _helper_dampening_filter_numba(X, X_diff, dampening_factor)
    return Y


## Import jax if available
try:
    import jax
    import jax.numpy as jnp

    def dampening_step(carry, x_curr):
        """
        Perform a single step of the dampening process.
        
        Args:
            carry: A tuple containing the previous output y (y_prev), the previous input x (x_prev),
                and the dampening factor.
            x_curr: Current value of x.
        
        Returns:
            Updated carry and the current y value.
        """
        y_prev, x_prev, dampening_factor = carry
        
        # # Handle division by zero safely. Ensure x_prev is not zero before dividing.
        # # This avoids the ambiguous truth value error by not using arrays in conditions.
        # division_result = 0 if x_prev == 0 else y_prev / x_prev
        
        # Calculate the current y value using the given formula.
        y_curr = y_prev * dampening_factor + (x_curr - x_prev) * 1
        
        # Update the carry for the next iteration.
        carry = (y_curr, x_curr, dampening_factor)
        return carry, y_curr

    @jax.jit
    def dampening_filter_jax(X, dampening_factor=0.9):
        """
        Applies a recursive dampening filter on a series of data points using JAX.
        Utilizes `lax.scan` for efficiently applying the dampening step function across the input series.
        
        Args:
            X (jnp.ndarray): Input series on which the dampening filter is applied.
            dampening_factor (float): Factor used in the dampening calculation.
        
        Returns:
            jnp.ndarray: The series after applying the dampening filter.
        """
        # Initial conditions: y is zero and x is the first element of the series.
        # The carry contains the initial y, initial x, and the dampening factor.
        initial_carry = (jnp.zeros_like(X[0]), X[0], dampening_factor)
        
        # Apply the dampening step across the series. Skip the first element since it's part of the initial conditions.
        _, Y = jax.lax.scan(dampening_step, initial_carry, X[1:])
        
        # Include the initial y value at the start of the series.
        Y = jnp.concatenate([jnp.zeros_like(X[:1]), Y])
        
        return Y

except ImportError:
    pass
        
####################################
######## PYTORCH algorithms ########
####################################

def convolve_torch(X, kernels, **conv1d_kwargs):
    """
    Convolution of X with kernels
    RH 2021

    Args:
        X (torch.Tensor):
            N-D array. Convolution will be performed
             along first dimension (columns).
            Dims 1+ are convolved independently and 
             increase the dimensionality of the output.
        kernels (torch.Tensor):
            N-D array. Convolution will be performed
             along first dimension (columns).
            Dims 1+ are convolved independently and 
             increase the dimensionality of the output.
        conv1d_kwargs (dict or keyword args):
            Keyword arguments for 
             torch.nn.functional.conv1d.
            See torch.nn.functional.conv1d for details.
            You can use padding='same'
        
    Returns:
        output (torch.Tensor):
            N-D array. Convolution of X with kernels
    """

    if isinstance(X, torch.Tensor)==False:
        X = torch.from_numpy(X)
    if isinstance(kernels, torch.Tensor)==False:
        kernels = torch.from_numpy(kernels)

    if kernels.dtype != X.dtype:
        kernels = kernels.type(X.dtype)

    X_dims = list(X.shape)
    t_dim = X_dims[0]

    kernel_dims = list(kernels.shape)
    w_dim = [kernel_dims[0]]

    conv_out_shape = [-1] + X_dims[1:] + kernel_dims[1:] 
    
    X_rshp = X.reshape((t_dim, 1, -1)).permute(2,1,0) # flatten non-time dims and shape to (non-time dims, 1, time dims) (D X R, 1, t)
    kernel_rshp = kernels.reshape(w_dim + [1, -1]).permute(2,1,0) # flatten rank + complex dims and shape to (rank X complex dims, 1, W) (R X C, 1, W)

    convolved = torch.nn.functional.conv1d(X_rshp, kernel_rshp, **conv1d_kwargs)  
   
    convolved_rshp = convolved.permute(2, 0, 1).reshape((conv_out_shape)) # (T, D, R, C)

    return convolved_rshp


def phase_shift(signal, shift_angle=90, deg_or_rad='deg', axis=0):
    """
    Shifts the frequency angles of a signal by a given amount.
    This is the functional version. It can be faster if needing
     to initialize multiple times.
    Setting shift_angle=90 with deg_or_rad='deg' will output the
     imaginary part of an analytic signal (i.e. a Hilbert transform).
    Numpy / Torch are both compatible
    RH 2021

    Args:
        signal (np.ndarray or torch.Tensor):
            The signal to be shifted
        shift_angle (float):
            The amount to shift the angle by
        deg_or_rad (str):
            Whether the shift_angle is in degrees or radians
        axis (int):
            The axis to shift along
    
    Returns:
        output (np.ndarray or torch.Tensor):
            The shifted signal
    """
    if type(signal) is torch.Tensor:
        fft, ifft, ones, cat, ceil, floor = torch.fft.fft, torch.fft.ifft, torch.ones, torch.cat, np.ceil, np.floor
    else:
        fft, ifft, ones, cat, ceil, floor = np.fft.fft, np.fft.ifft, np.ones, np.concatenate, np.ceil, np.floor

    if deg_or_rad == 'deg':
        shift_angle = np.deg2rad(shift_angle)

    half_len_minus = int(ceil(signal.shape[axis]/2))
    half_len_plus = int(floor(signal.shape[axis]/2))
    angle_mask = cat(([-ones(half_len_minus), ones(half_len_plus)])) * shift_angle

    signal_fft = fft(signal, axis=axis)
    mag, ang = cartesian2polar(signal_fft)
    ang_shifted = ang + angle_mask

    signal_fft_shifted = polar2cartesian(mag, ang_shifted)
    signal_shifted = ifft(signal_fft_shifted, axis=axis)
    return signal_shifted

class Phase_Shifter():
    def __init__(self, signal_len, discard_imaginary_component=True, device='cpu', dtype=torch.float32, pin_memory=False):
        """
        Initializes the shift_signal_angle_obj class.
        This is the object version. It can be faster than
         the functional version if needing to call it 
         multiple times.
        Setting shift_angle=90 with deg_or_rad='deg' will output the
         imaginary part of an analytic signal (i.e. a Hilbert transform).
        See __call__ for more details.
        RH 2021

        Args:
            signal_len (int):
                The shape of the signal to be shifted.
                The first (0_th) dimension must be the shift dimension.
            discard_imaginary_component (bool):
                Whether to discard the imaginary component of the signal
            device (str):
                The device to put self.angle_mask on
            dtype (torch.dtype):
                The dtype to use for self.angle_mask
            pin_memory (bool):
                Whether to pin self.angle_mask to memory
        """

        self.signal_len = signal_len
        signal_len = torch.as_tensor(signal_len)
        half_len_minus = int(torch.ceil(signal_len/2))
        half_len_plus = int(torch.floor(signal_len/2))
        self.angle_mask = torch.cat([
            -torch.ones(half_len_minus, dtype=dtype, device=device, pin_memory=pin_memory),
             torch.ones(half_len_plus,  dtype=dtype, device=device, pin_memory=pin_memory)
             ])
        self.discard_imaginary_component = discard_imaginary_component

    def __call__(self, signal, shift_angle=90, deg_or_rad='deg'):
        """
        Shifts the frequency angles of a signal by a given amount.
        A signal containing multiple frequecies will see each 
         frequency shifted independently by the shift_angle.
        Functions on first dimension only.
        RH 2021

        Args:
            signal (torch.Tensor):
                The signal to be shifted
            shift_angle (float):
                The amount to shift the angle by
            deg_or_rad (str):
                Whether the shift_angle is in degrees or radians
            dim (int):
                The axis to shift along
            
        Returns:
            output (torch.Tensor):
                The shifted signal
        """
        
        if shift_angle == 0:
            return signal
        
        shift_angle = np.deg2rad(shift_angle) if deg_or_rad == 'deg' else shift_angle

        signal_fft = torch.fft.fft(signal, dim=0) # convert to spectral domain
        mag, ang = torch.abs(signal_fft), torch.angle(signal_fft) # extract magnitude and angle
        ang_shifted = ang + (self.angle_mask.reshape([len(self.angle_mask)] + [1]*(len(signal.shape)-1))) * shift_angle # shift the angle. The bit in the middle is for matching the shape of the signal
        signal_fft_shifted = mag * torch.exp(1j*ang_shifted) # remix magnitude and angle
        signal_shifted = torch.fft.ifft(signal_fft_shifted, dim=0) # convert back to signal domain
        if self.discard_imaginary_component:
            signal_shifted = torch.real(signal_shifted) # discard imaginary component
        return signal_shifted


##############################################################
######### NUMBA implementations of simple algorithms #########
##############################################################

@njit(parallel=True)
def percentile_numba(X, ptile):
    '''
    Parallel (multicore) Percentile. Uses numba
    RH 2021

    Args:
        X (ndarray):
            2-D array. Percentile will be calculated
            along second dimension (rows)
        ptile (scalar 0-100):
            Percentile
    
    Returns:
        X_ptile (ndarray):
            Percentiles of X
    '''

    X_ptile = np.zeros(X.shape[0])
    for ii in prange(X.shape[0]):
        X_ptile[ii] = np.percentile(X[ii,:] , ptile)
    return X_ptile


@jit(parallel=True, nopython=True)
def zscore_numba(array):
    '''
    Parallel (multicore) Z-Score. Uses numba.
    Computes along second dimension (axis=1) for speed
    Best to input a contiguous array.
    RH 2021

    Args:
        array (ndarray):
            2-D array. Zscore will be calculated
            along second dimension (rows)
    
    Returns:
        output_array (ndarray):
            2-D array. Z-Scored array
    '''

    output_array = np.zeros_like(array)
    for ii in prange(array.shape[0]):
        array_tmp = array[ii,:]
        output_array[ii,:] = (array_tmp - np.mean(array_tmp)) / np.std(array_tmp)
    return output_array


@njit(parallel=True)
def conv1d_alongAxis_helper(X, k_rev):
    y = np.empty_like(X)
    y.fill(np.nan)
    k_hs = k_rev.size//2
    for ii in prange(X.shape[0]):
        for i in prange( k_hs , X.shape[1]-(k_hs) ):
            y[ii, i] = np.dot(X[ii, 0+i-k_hs : 1+i+k_hs], k_rev)
    return y
def convolve_numba(X, k, axis=1):
    '''
    Convolves an array with a kernel along a defined axis
    if multicore_pref==True, array must be 2-D and 
    convolution is performed along dim-0. A 1-D array is 
    okay if you do X=array[:,None]
    RH 2021
    
    Args:
        array (np.ndarray): 
            array you wish to convolve
        kernel (np.ndarray): 
            array to be used as the convolutional kernel (see numpy.convolve documentation)
            LENGTH MUST BE AN ODD NUMBER
        axis (int): 
            axis to convolve array along. NOT USED IF multicore_pref==True

    Returns:
        output (np.ndarray): input array convolved with kernel
    '''
    if len(k)%2==0:
        raise TypeError('RH WARNING: k must have ODD LENGTH')

    if axis==0:
        X = X.T
    k_rev = np.ascontiguousarray(np.flip(k), dtype=X.dtype)
    y = conv1d_alongAxis_helper(X, k_rev)

    if axis==0:
        return y.T
    else:
        return y


@njit(parallel=True)
def conv1d_numba(X, k):
    '''
    remember to flip k
    '''
    y = np.empty_like(X)
    y.fill(np.nan)
    k_hs = k.size//2
    for ii in prange( k_hs , len(X)-(k_hs) ):
        y[ii] = np.dot(X[0+ii-k_hs : 1+ii+k_hs], k)
    return y

##############################################################
###################### FFT Convolution #######################
##############################################################

# @torch.jit.script
def next_fast_len(size: int):
    """
    Taken from PyTorch Forecasting:
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining = remaining // n
        if remaining == 1:
            return next_size
        next_size += 1

# @torch.jit.script
def apply_padding_mode(
    conv_result: torch.Tensor, 
    x_length: int, 
    y_length: int, 
    mode: str = "valid",
) -> torch.Tensor:
    """
    This is adapted from torchaudio.functional._apply_convolve_mode. \n
    NOTE: This function has a slight change relative to torchaudio's version.
    For mode='same', ceil rounding is used. This results in fftconv matching the
    result of conv1d. However, this then results in it not matching the result of
    scipy.signal.fftconvolve. This is a tradeoff. The difference is only a shift
    in 1 sample when y_length is even. This phenomenon is a result of how conv1d
    handles padding, and the fact that conv1d is actually cross-correlation, not
    convolution. \n

    RH 2024

    Args:
        conv_result (torch.Tensor):
            Result of the convolution.
            Padding applied to last dimension.
        x_length (int):
            Length of the first input.
        y_length (int):
            Length of the second input.
        mode (str):
            Padding mode to use.

    Returns:
        torch.Tensor:
            Result of the convolution with the specified padding mode.
    """
    n = x_length + y_length - 1
    valid_convolve_modes = ["full", "valid", "same"]
    if mode == "full":
        return conv_result
    elif mode == "valid":
        len_target = max(x_length, y_length) - min(x_length, y_length) + 1
        idx_start = (n - len_target) // 2
        return conv_result[..., idx_start : idx_start + len_target]
    elif mode == "same":
        # idx_start = (conv_result.size(-1) - x_length) // 2  ## This is the original line from torchaudio
        idx_start = math.ceil((n - x_length) / 2)  ## This line is different from torchaudio
        return conv_result[..., idx_start : idx_start + x_length]
    else:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")


# @torch.jit.script
def fftconvolve(
    y: torch.Tensor, 
    x: Optional[torch.Tensor]=None,
    mode: str='valid',
    n: Optional[int]=None,
    fast_length: bool=False,
    x_fft: Optional[torch.Tensor]=None,
    return_real: bool=True,
):
    """
    Convolution using the FFT method. \n
    This is adapted from of torchaudio.functional.fftconvolve that handles
    complex numbers. Code is added for handling complex inputs. \n
    NOTE: For mode='same' and y length even, torch's conv1d convention is used,
    which pads 1 more at the end and 1 fewer at the beginning (which is
    different from numpy/scipy's convolve). See apply_padding_mode for more
    details. \n

    RH 2024

    Args:
        y (torch.Tensor):
            Second input. (kernel) \n
            Convolution performed along the last dimension.
        x (torch.Tensor):
            First input. (signal) \n
            Convolution performed along the last dimension.\n
            If None, x_fft must be provided.
        mode (str):
            Padding mode to use. ['full', 'valid', 'same']
        fast_length (bool):
            Whether to use scipy.fftpack.next_fast_len to 
             find the next fast length for the FFT.
            Set to False if you want to use backpropagation.
        n (int):
            Length of the fft domain. If None, n is computed from x and y.\n
            If n is less than the length of x and y, then the output will be
            truncated.
        x_fft (torch.Tensor):
            FFT of x. If None, x is used to compute the FFT.\n
            If x is provided, x_fft must be None. If x_fft is provided, x must
            be None.
        return_real (bool):
            Whether to return the real part of the convolution.
            If False, the complex result is returned as well.

    Returns:
        torch.Tensor:
            Result of the convolution.
    """
    ## Compute the convolution
    n_original = x.shape[-1] + y.shape[-1] - 1
    if x_fft is None:
        if n is None:
            n = n_original
            # n = scipy.fftpack.next_fast_len(n_original) if fast_length else n_original
            if fast_length:
                n = next_fast_len(n_original)
            else:
                n = n_original
        else:
            n = n
        x_fft = torch.fft.fft(x, n=n, dim=-1)            
    else:
        n = x_fft.shape[-1] if x_fft is not None else n
    
    y_fft = torch.fft.fft(torch.flip(y, dims=(-1,)), n=n, dim=-1)
    f = x_fft * y_fft
    fftconv_xy = torch.fft.ifft(f, n=n, dim=-1)
    fftconv_xy = fftconv_xy.real if return_real else fftconv_xy
    return apply_padding_mode(
        conv_result=fftconv_xy,
        x_length=x.shape[-1],
        y_length=y.shape[-1],
        mode=mode,
    )

class FFTConvolve(torch.nn.Module):
    def __init__(
        self, 
        x: Optional[torch.Tensor]=None, 
        n: Optional[int]=None, 
        next_fast_length: bool=False,
        use_x_fft: bool=True,
        return_real: bool=True,
    ):
        super(FFTConvolve, self).__init__()
        if x is not None:
            self.set_x_fft(x=x, n=n, next_fast_length=next_fast_length)
        else:
            self.n = None
            self.x_fft = None

        self.use_x_fft = use_x_fft
        self.return_real = return_real

    def set_x_fft(self, x: torch.Tensor, n: Optional[int]=None, next_fast_length: bool=False):
        if next_fast_length:
            self.n = next_fast_len(size=n)
        self.x_fft = torch.fft.fft(x, n=self.n, dim=-1).contiguous()

        ## Check for any NaNs or inf or weird values in x_fft
        if torch.any(torch.isnan(self.x_fft)):
            raise ValueError(f"x_fft has NaNs")
        if torch.any(torch.isinf(self.x_fft)):
            raise ValueError(f"x_fft has infs")
        if torch.any(torch.abs(self.x_fft) > 1e6):
            raise ValueError(f"x_fft has values > 1e6")

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str='same',
        n: Optional[int]=None,
        fast_length: Union[int, bool]=False,
        x_fft: Optional[torch.Tensor]=None,
        return_real: bool=None,
    ) -> torch.Tensor:
        x_fft = self.x_fft if x_fft is None else x_fft
        return_real = self.return_real if return_real is None else return_real
        n = self.n if n is None else n
        return fftconvolve(
            x=x, 
            y=y, 
            mode=mode, 
            n=n, 
            fast_length=fast_length, 
            x_fft=x_fft if self.use_x_fft else None,
            return_real=return_real,
        )
