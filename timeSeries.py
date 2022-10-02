from this import d
import numpy as np
import time
from matplotlib import pyplot as plt
import copy
from numba import njit, jit, prange
import pandas as pd
import torch

from . import parallel_helpers
from .parallel_helpers import multiprocessing_pool_along_axis
from . import cross_validation
from .math_functions import polar2real, real2polar, gaussian
from . import indexing



def convolve_along_axis(
    array,
    kernel, 
    axis=1 , 
    mode='same', 
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
    Returns:
        output (np.ndarray): input array convolved with kernel
    '''
    tic = time.time()

    if array.ndim == 1:
        multicore_pref = False
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


def scale_between(
    x, 
    lower=0, 
    upper=1, 
    axes=0, 
    lower_percentile=None, 
    upper_percentile=None, 
    crop_pref=True,
    verbose=False
    ):
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
        verbose (bool):
            Whether or not to print the highest and lowest values.

    Returns:
        x_out (ndarray):
            Scaled array
    '''

    if lower_percentile is not None:
        lowest_val = np.percentile(x, lower_percentile, axis=axes, keepdims=True)
    else:
        lowest_val = np.nanmin(x, axis=axes, keepdims=True)
    if upper_percentile is not None:
        highest_val = np.percentile(x, upper_percentile, axis=axes, keepdims=True)
    else:
        highest_val = np.nanmax(x, axis=axes, keepdims=True)

    if verbose:
        print(f'Highest value: {highest_val}')
        print(f'Lowest value: {lowest_val}')

    x_out = ((x - lowest_val) * (upper - lower) / (highest_val - lowest_val) ) + lower

    if crop_pref:
        np.putmask(x_out, x_out < np.squeeze(lower), lower)
        np.putmask(x_out, x_out > np.squeeze(upper), upper)

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
    import rolling_quantiles as rq
    pipe = rq.Pipeline( rq.LowPass(window=window, quantile=(ptile/100), subsample_rate=stride) )
    lag = int(np.floor(pipe.lag))
    if center:
        return pipe.feed(x_in)[lag:]
    else:
        return pipe.feed(x_in)
def rolling_percentile_rq_multicore(x_in, window, ptile, stride=1, center=True, n_workers=None):
    import rolling_quantiles as rq
    return multiprocessing_pool_along_axis(x_in, rolling_percentile_rq, n_workers=None, axis=0, **{'window': window , 'ptile': ptile, 'stride': stride, 'center': False} )


def event_triggered_traces(arr, trigger_signal, win_bounds, trigger_signal_is_idx=False):
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
        win_bounds (size 2 integer list or np.ndarray):
            2 value integer array. win_bounds[0] should
             be negative and is the number of samples prior
             to the event that the window starts. 
             win_bounds[1] is the number of samples 
             following the event.
            Events that would have a window extending
             before or after the bounds of the length
             of the trace are discarded.
        trigger_signal_is_idx (bool):
            If True then trigger_signal is an index array.
            If False then trigger_signal is a boolean array.
            Use an index array if there are multiple events
             with the same index, else they will be
             collapsed when this is 'True'.
     
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

    if trigger_signal_is_idx:
        windows = make_windows(trigger_signal[np.isnan(trigger_signal)==0], win_bounds)
    else:
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


def make_sorted_event_triggered_average(
    arr, 
    trigger_signal, 
    win_bounds, 
    cv_group_size=2, 
    test_frac=0.5, 
    trigger_signal_is_idx=False, 
    show_plot=False
):
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

    et_traces = event_triggered_traces(arr, trigger_signal, win_bounds, trigger_signal_is_idx=trigger_signal_is_idx)

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
    

def simple_smooth(arr, x=None, mu=0, sig=1, axis=0, mode='same', correct_edge_effects=True):
    '''
    Simple smoothing function.
    Uses convolve_torch and math_functions.gaussian to 
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
        multicore_pref=True, 
        verbose=False
        )
    
    if correct_edge_effects:
        trace_norm = np.convolve(
            a=np.ones(arr.shape[axis]),
            v=gaus,
            mode=mode
            )

        trace_norm_padded = indexing.pad_with_singleton_dims(trace_norm, n_dims_pre=axis, n_dims_post=arr.ndim-axis-1)

        arr_conv_corrected = arr_conv / trace_norm_padded

        return arr_conv_corrected
    else:
        return arr_conv

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
    mag, ang = real2polar(signal_fft)
    ang_shifted = ang + angle_mask

    signal_fft_shifted = polar2real(mag, ang_shifted)
    signal_shifted = ifft(signal_fft_shifted, axis=axis)
    return signal_shifted

class phase_shifter():
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
    

@njit(parallel=True)
def mean_numba(X):
    Y = np.zeros(X.shape[0], dtype=X.dtype)
    for ii in prange(X.shape[0]):
        Y[ii] = np.mean(X[ii,:])
    return Y


@njit(parallel=True)
def var_numba(X):
    Y = np.zeros(X.shape[0], dtype=X.dtype)
    for ii in prange(X.shape[0]):
        Y[ii] = np.var(X[ii,:])
    return Y


@njit(parallel=True)
def std_numba(X):
    Y = np.zeros(X.shape[0], dtype=X.dtype)
    for ii in prange(X.shape[0]):
        Y[ii] = np.std(X[ii,:])
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
