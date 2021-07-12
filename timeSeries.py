'''
Table of Contents

Functions and Interdependencies:
    convolve_along_axis
        - parallel_helpers.multithreading
    threshold
    scale_between

    percentile_numba
    zscore_numba
    convolve_numba
'''

from logging import warn
import numpy as np
import scipy.stats
import time
from matplotlib import pyplot as plt
import copy
from numba import njit, jit, prange
import pandas as pd
import rolling_quantiles as rq

from . import parallel_helpers
from .parallel_helpers import multiprocessing_pool_along_axis
from . import cross_validation



def convolve_along_axis(array , kernel , axis , mode , multicore_pref=False , verbose=False):
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
        array (np.ndarray): array you wish to convolve
        kernel (np.ndarray): array to be used as the convolutional kernel (see numpy.convolve documentation)
        axis (int): axis to convolve array along. NOT USED IF multicore_pref==True
        mode (str): see numpy.convolve documentation. Can be 'valid', 'same', 'full'
    Returns:
        output (np.ndarray): input array convolved with kernel
    '''
    tic = time.time()
    if multicore_pref:
        def convFun_axis0(iter):
            return np.convolve(array[:,iter], kernel, mode=mode)
        def convFun_axis1(iter):
            return np.convolve(array[iter,:], kernel, mode=mode)

        kernel = np.ascontiguousarray(kernel)
        if axis==0:
            output_list = parallel_helpers.multithreading(convFun_axis0, range(array.shape[1]), workers=None)
        if axis==1:
            output_list = parallel_helpers.multithreading(convFun_axis1, range(array.shape[0]), workers=None)
        
        if verbose:
            print(f'ThreadPool elapsed time : {round(time.time() - tic , 2)} s. Now unpacking list into array.')
        
        if axis==0:
            output = np.array(output_list).T
        if axis==1:
            output = np.array(output_list)

    else:
        output = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode=mode), axis=axis, arr=array)

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
        array (np.ndarray):  the mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D gaussian
        thresh_max (number, scalar): values in array above this are set to val_max
        thresh_min (number, scalar): values in array below this are set to val_min
        val_max (number, scalar): values in array above thresh_max are set to this
        val_min (number, scalar): values in array above thresh_min are set to this
        inPlace_pref (bool): whether to do the calculation 'in place', and change the local input variable directly

    Return:
        output_array (np.ndarray): same as input array but with values thresheld
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


def scale_between(x, lower=0, upper=1, axes=0, lower_percentile=None, upper_percentile=None, crop_pref=True):
    '''
    Scales the first (or more) dimension of an array to be between 
    lower and upper bounds.
    RH 2021

    Args:
        x (ndarray):
            Any dimensional array. First dimension is scaled
        lower (scalar):
            lower bound for scaling
        upper (scalar):
            upper bound for scaling
        axes (tuple):
            UNTESTED for values other than 0.
            It should work for tuples defining axes, so long
            as the axes are sequential starting from 0
            (eg. (0,1,2,3) ).
        lower_percentile (scalar):
            Ranges between 0-100. Defines what percentile of
            the data to set as the lower bound
        upper_percentile (scalar):
            Ranges between 0-100. Defines what percentile of
            the data to set as the upper bound
        crop_pref (bool):
            If true then data is cropped to be between lower
            and upper. Only meaningful if lower_percentile or
            upper_percentile is not None.

    Returns:
        x_out (ndarray):
            Scaled array
    '''

    if lower_percentile is not None:
        lowest_val = np.percentile(x, lower_percentile, axis=axes)
    else:
        lowest_val = np.min(x, axis=axes)
    if upper_percentile is not None:
        highest_val = np.percentile(x, upper_percentile, axis=axes)
    else:
        highest_val = np.max(x, axis=axes)

    x_out = ((x - lowest_val) * (upper - lower) / (highest_val - lowest_val) ) + lower

    if crop_pref:
        x_out[x_out < lower] = lower
        x_out[x_out > upper] = upper

    return x_out


def rolling_percentile_pd(X, ptile=50, window=21, interpolation='linear', output_type='numpy', **kwargs):
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
            dim 1 (rows) by default.
        ptile (scalar):
            Percentile. 0-100.
        window (scalar):
            Size of window in samples. Ideally odd integer.
        interpolation (string):
            For details see: pandas.core.window.rolling.Rolling.quantile
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.rolling.Rolling.quantile.html
            Can be: linear, lower, higher, nearest, midpoint
        output_type (string):
            Either 'numpy' or 'pandas'
            If 'numpy', then output is a continguous array
        **kwargs (dict):
            kwargs for pandas.DataFrame.rolling function call.
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
            Includes: min_periods, center, axis, win_type, closed

    Returns:
        output (numpy.ndarray OR pd.core.frame.DataFrame):
            Output array of signals.
    '''
    
    if 'min_periods' not in kwargs:
        kwargs['min_periods'] = 1
    if 'center' not in kwargs:
        kwargs['center'] = True
    if 'axis' not in kwargs:
        kwargs['axis'] = 1
    if 'win_type' not in kwargs:
        kwargs['win_type'] = None
    if 'closed' not in kwargs:
        kwargs['closed'] = None
        
    if not isinstance(X, pd.core.frame.DataFrame):
        X = pd.DataFrame(X)
    output = X.rolling(window=window, **kwargs).quantile(ptile/100, interpolation=interpolation)
    if output_type == 'numpy':
        output = np.ascontiguousarray(output)
    return output


def rolling_percentile_rq(x_in, window, ptile=10, stride=1, center=True):
    pipe = rq.Pipeline( rq.LowPass(window=window, quantile=(ptile/100), subsample_rate=stride) )
    lag = int(np.floor(pipe.lag))
    if center:
        return pipe.feed(x_in)[lag:]
    else:
        return pipe.feed(x_in)
def rolling_percentile_rq_multicore(x_in, window, ptile, stride=1, center=True, n_workers=None):
    return multiprocessing_pool_along_axis(x_in, rolling_percentile_rq, n_workers=None, axis=0, **{'window': window , 'ptile': ptile, 'stride': stride, 'center': False} )


def event_triggered_traces(arr, trigger_signal, win_bounds):
    '''
    Makes event triggered traces along last dimension
    RH 2021
    
    Args:
        arr (np.ndarray):
            Input array. Last dimension will be 
             aligned to boolean True values in 
             'trigger_signal'
        trigger_signal (boolean np.ndarray):
            1-D boolean array. True values are trigger
             events
        win_bounds (size 2 integer np.ndarray):
            2 value integer array. win_bounds[0] is the
             number of samples prior to the event that
             the window starts. win_bounds[1] is the 
             number of samples following the event.
            Events that would have a window extending
             before or after the bounds of the length
             of the trace are discarded.
     
     Returns:
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
            
            
    '''
    def bounds_to_win(x_pos, win_bounds):
        return x_pos + np.arange(win_bounds[0], win_bounds[1])
    def make_windows(x_pos, win_bounds):
        return np.apply_along_axis(bounds_to_win, 0, tuple([x_pos]), win_bounds).T

    axis=arr.ndim-1
    len_axis = arr.shape[axis]

    windows = make_windows(np.nonzero(trigger_signal)[0], win_bounds)
    win_toInclude = (np.sum(windows<0, axis=1)==0) * (np.sum(windows>len_axis, axis=1)==0)
    win_toExclude = win_toInclude==False
    n_win_included = np.sum(win_toInclude)
    n_win_excluded = np.sum(win_toExclude)
    windows = windows[win_toInclude]


    windows_flat = np.reshape(windows, (windows.size))

    axes_all = np.arange(arr.ndim)
    axes_non = axes_all[axes_all != axis]
    et_traces_flat = np.take_along_axis(arr, np.expand_dims(np.array(windows_flat, dtype=np.int64), axis=tuple(axes_non)), axis=axis)

    new_shape = np.array(et_traces_flat.shape)
    new_shape[axis] = new_shape[axis] / windows.shape[1]
    new_shape = np.concatenate((new_shape, np.array([windows.shape[1]])))
    et_traces = np.reshape(et_traces_flat, new_shape)
    
    xAxis = np.arange(win_bounds[0], win_bounds[1])
    
    return et_traces, xAxis, windows


def make_sorted_event_triggered_average(arr, trigger_signal, win_bounds, cv_group_size=2, test_frac=0.5, show_plot=False):
    '''
    Makes a sorted event triggered average plot
    RH 2021
    
    Args:
        arr (np.ndarray):
            Input array. Last dimension will be aligned
             to boolean True values in 'trigger_signal'.
             Same as in event_triggered_traces.
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
        cv_group_size (int):
            Number of samples per group. Uses sklearn's
             model_selection.GroupShuffleSplit.
            Same as in cross_validation.group_split
         test_frac (scalar):
             Fraction of samples in test set.
             Same as in cross_validation.group_split
         show_plot (bool):
             Whether or not to show the plot
     
     Returns:
        mean_traces_sorted (np.ndarray):
            Output traces/image. Size (arr.shape[0],
             win_bounds[1]-win_bounds[0])
            Shows the event triggered average of the 
             test set, sorted by the peak times found 
             in the training set.
        et_traces (np.ndarray):
            All event triggered traces.
            Same as return 'et_traces' in 
             timeSeries.event_triggered_traces
        cv_idx (list of 2 lists):
            List of 2 lists.
            Outer list entries: Splits
            Inner list entries: Train, Test indices
            Same as return 'cv_idx' in 
             cross_validation.group_split
            
    '''

    et_traces = event_triggered_traces(arr, trigger_signal, win_bounds)

    cv_idx = cross_validation.group_split(1, et_traces[0].shape[1], cv_group_size, test_size=test_frac)

    mean_traces_train = et_traces[0][:,cv_idx[0][0],:].mean(1)
    mean_traces_test = et_traces[0][:,cv_idx[0][1],:].mean(1)

    mean_traces_sorted = mean_traces_test[np.argsort(np.argmax(mean_traces_train,axis=1))]
    
    if show_plot:
        plt.figure()
        plt.imshow(mean_traces_sorted, aspect='auto', 
                   extent=(et_traces[1][0], et_traces[1][-1],
                          mean_traces_sorted.shape[0], 0),
#                    vmin=-1, vmax=3
                  )
    return mean_traces_sorted, et_traces, cv_idx


def widen_boolean(arr, n_before, n_after, axis=None):
    '''
    Widens boolean events by n_before and n_after.    
    RH 2021    

    Args:
        arr (np.ndarray):
            Input array. Widening will be applied
             to the last dimension.
        n_before (int):
            Number of samples before 'True' values
             that will also be set to 'True'.
        n_after (int):
            Number of samples after 'True' values
             that will also be set to 'True'.
        axis (int):
            Axis to apply the event widening.
             If None then arr should be a 1-D array.
    
    Returns:
        widened arr (np.ndarray):
            Output array. Same as input arr, but
             with additional 'True' values before
             and after initial 'True' values.
    '''
    
    kernel = np.zeros(np.max(np.array([n_before, n_after])) * 2 + 1)
    kernel_center = int(np.ceil(len(kernel) / 2))
    kernel[kernel_center - (n_before+1): kernel_center] = 1
    kernel[kernel_center: kernel_center + n_after] = 1
    
    if axis is None:
        return scipy.signal.convolve(arr, kernel/np.sum(kernel), mode='same')
    else:
        return np.apply_along_axis(lambda m: scipy.signal.convolve(m, kernel/np.sum(kernel), mode='same'),
                                                              axis=axis, arr=arr)


@njit
def idx2bool(idx, length):
    '''
    Converts a vector of indices to a boolean vector.
    RH 2021

    Args:
        idx (np.ndarray):
            1-D array of indices.
        length (int):
            Length of boolean vector.
    
    Returns:
        bool_vec (np.ndarray):
            1-D boolean array.
    '''
    out = np.zeros(length)
    out[idx] = True
    return out


def moduloCounter_to_linearCounter(trace, modulus, modulus_value, diff_thresh=None, plot_pref=False):
    '''
    Converts a trace of modulo counter values to a linear counter.
    Useful for converting a pixel clock with a modulus
     to total times. Use this for FLIR camera top pixel
     stuff.
    The function basically just finds where the modulus
     events occur in the trace and adds 'modulus_value'
     to the next element in the trace.
    RH 2021

    Args:
        trace (np.ndarray):
            1-D array of modulo counter values.
        modulus (scalar):
            Modulus of the counter. Values in trace
             should range from 0 to modulus-1.
        modulus_value (scalar):
            Multiplier for the modulus counter. The
             value of a modulus event.
        diff_thresh (scalar):
            Threshold for defining a modulus event.
            Should typically be a negative value
             smaller than 'modulus', but larger
             than the difference between consecutive
             trace values.
        plot_pref (bool):
            Whether or not to plot the trace.

    Returns:
        linearCounter (np.ndarray):
            1-D array of linearized counter values.
    '''

    if diff_thresh is None:
        diff_thresh = -modulus/2

    diff_trace = np.diff(np.double(trace))
    mod_times = np.where(diff_trace<diff_thresh)[0]


    mod_times_bool = np.zeros(len(trace))
    mod_times_bool[mod_times+1] = modulus_value
    mod_times_steps = np.cumsum(mod_times_bool)
    trace_times = (trace/modulus)*modulus_value + mod_times_steps

    if plot_pref:
        plt.figure()
        plt.plot(trace)
        plt.plot(mod_times , trace[mod_times] , 'o')

        plt.figure()
        plt.plot(mod_times_steps)
        plt.plot(trace_times)
    
    return trace_times


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
            along first dimension (columns)
        ptile (scalar 0-100):
            Percentile
    
    Returns:
        X_ptile (ndarray):
            1-D array. Percentiles of X
    '''

    X_ptile = np.zeros(X.shape[0])
    for ii in prange(X.shape[0]):
        X_ptile[ii] = np.percentile(X[ii,:] , ptile)
    return X_ptile


@jit(parallel=True)
def zscore_numba(array):
    '''
    Parallel (multicore) Z-Score. Uses numba.
    Computes along second dimension (axis=1) for speed
    Best to input a contiguous array.
    RH 2021

    Args:
        array (ndarray):
            2-D array. Percentile will be calculated
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


def convolve_numba(X, k, axis=1):
    '''
    Convolves an array with a kernel along a defined axis
    if multicore_pref==True, array must be 2-D and 
    convolution is performed along dim-0. A 1-D array is 
    okay if you do X=array[:,None]
    Faster and more memory efficient than that above
    function 'zscore_multicore' for massive arrays
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
    @njit(parallel=True)
    def conv(X, k_rev):
        y = np.empty_like(X)
        y.fill(np.nan)
        k_hs = k_rev.size//2
        for ii in prange(X.shape[0]):
            for i in prange( k_hs , X.shape[1]-(k_hs+1) ):
                y[ii, i] = np.dot(X[ii, 0+i-k_hs : 1+i+k_hs], k_rev)
        return y
    
    if axis==0:
        X = X.T
    k_rev = np.ascontiguousarray(np.flip(k), dtype=X.dtype)
    y = conv(X, k_rev)

    if axis==0:
        return y.T
    else:
        return y

    
@njit(parallel=True)
def var_numba(X):
    Y = np.zeros(X.shape[0], dtype=X.dtype)
    for ii in prange(X.shape[0]):
        Y[ii] = np.var(X[ii,:])
    return Y


@njit(parallel=True)
def min_numba(X):
    output = np.zeros(X.shape[0])
    for ii in prange(X.shape[0]):
        output[ii] = np.min(X[ii])
    return output


@njit(parallel=True)
def max_numba(X):
    output = np.zeros(X.shape[0])
    for ii in prange(X.shape[0]):
        output[ii] = np.max(X[ii])
    return output

@njit(parallel=True)
def round_numba(x):
    output = np.zeros_like(x)
    for ii in prange(x.shape[0]):
        for jj in prange(x.shape[1]):
            output[ii,jj] = np.round(x[ii,jj])
    return output