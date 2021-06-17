'''
Table of Contents

Functions and Interdependencies:
    convolve_along_axis
        - parallel_helpers.multithreading
    gaussian
    gaussian_kernel_2D
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

from . import parallel_helpers


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
    

def gaussian(x, mu, sig , plot_pref=False):
    '''
    A gaussian function (normalized similarly to scipy's function)
    RH 2021
    
    Args:
        x (np.ndarray): 1-D array of the x-axis of the kernel
        mu (float): center position on x-axis
        sig (float): standard deviation (sigma) of gaussian
        plot_pref (boolean): True/False or 1/0. Whether you'd like the kernel plotted
    Returns:
        gaus (np.ndarray): gaussian function (normalized) of x
        params_gaus (dict): dictionary containing the input params
    '''

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x-mu)/sig, 2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'$\mu$={mu}, $\sigma$={sig}')
    
    params_gaus = {
        "x": x,
        "mu": mu,
        "sig": sig,
    }

    return gaus , params_gaus


def gaussian_kernel_2D(center = (5, 5), image_size = (11, 11), sig = 1):
    """
    Generate a 2D or 1D gaussian kernel
    
    Args:
        center (tuple):  the mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D gaussian
        image_size (tuple): The total image size (width, height). Make second value 0 to make 1D gaussian
        sig (scalar): The sigma value of the gaussian
    
    Return:
        kernel (np.ndarray): 2D or 1D array of the gaussian kernel
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel


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


def rolling_percentile(X, ptile=50, window=21, interpolation='linear', output_type='numpy', **kwargs):
    '''
    Computes a rolling percentile over one dimension 
     (defaults to dim 1 / rows).
    Uses pandas' rolling library.
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
            along first dimension (columns)
    
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