import numpy as np
import scipy.stats
import time
from matplotlib import pyplot as plt
import copy

from . import parallel_helpers


def zscore_multicore(array , verbose=True):
    """
    calculates the zscore using parallel processing along the first dimension of a 2-D array.
    Currently using ConcurrentFutures multithreading
    RH 2021
    
    Args:
        array (np.ndarray): 2-D array
        verbose (boolean): True/False or 1/0. Whether you'd like printed updates
    Returns:
        output_array (np.ndarray): array after computed zscore along first dimension
    """
    tic = time.time()

    def zscoreFun(iter):
        return scipy.stats.zscore(array[:,iter])
    output_list = parallel_helpers.multithreading(zscoreFun, range(array.shape[1]), workers=None)
    if verbose:
            print(f'ThreadPool elapsed time : {round(time.time() - tic , 2)} s. Now unpacking list into array.')
    output_array = np.array(output_list).T
    if verbose:
        print(f'Calculated zscores. Total elapsed time: {round(time.time() - tic,2)} seconds')
    return output_array


def convolve_along_axis(array , kernel , axis , mode , multicore_pref=False , verbose=False):
    '''
    Convolves an array with a kernel along a defined axis
    if multicore_pref==True, array must be 2-D and convolution is performed along dim-0
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
        def convFun(iter):
            return np.convolve(array[:,iter], kernel, mode=mode)

        output_list = parallel_helpers.multithreading(convFun, range(array.shape[1]), workers=None)
        if verbose:
            print(f'ThreadPool elapsed time : {round(time.time() - tic , 2)} s. Now unpacking list into array.')
        output = np.array(output_list).T

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

def make_dFoF(
    F, 
    Fneu=None, 
    neuropil_fraction=0.7, 
    percentile_baseline=30, 
    multicore_pref=False, 
    verbose=True):
    """
    calculates the dF/F and other signals. Designed for Suite2p data.
    If Fneu is left empty or =None, then no neuropil subtraction done.
    See S2p documentation for more details
    RH 2021
    
    Args:
        F (Path): raw fluorescence values of each ROI. dims(time, ...)
        Fneu (np.ndarray): Neuropil signals corresponding to each ROI. dims match F.
        neuropil_fraction (float): value, 0-1, of neuropil signal (Fneu) to be subtracted off of ROI signals (F)
        percentile_baseline (float/int): value, 0-100, of percentile to be subtracted off from signals
        verbose (boolean): True/False or 1/0. Whether you'd like printed updates
    Returns:
        dFoF (np.ndarray): array, dF/F
        dF (np.ndarray): array
        F_neuSub (np.ndarray): F with neuropil subtracted
        F_baseline (np.ndarray): 1-D array of size F.shape[1]. Baseline value for each ROI
    """
    tic = time.time()

    if Fneu is None:
        F_neuSub = F
    else:
        F_neuSub = F - neuropil_fraction*Fneu

    if multicore_pref:
        def ptileFun(iter):
            return np.percentile(F_neuSub[:,iter] , percentile_baseline)

        output_list = parallel_helpers.multithreading(ptileFun, range(F_neuSub.shape[1]) , workers=None)

        if verbose:
            print(f'ThreadPool elapsed time : {round(time.time() - tic , 2)} s. Now unpacking list into array.')
        F_baseline = np.array(output_list).T

    else:
        F_baseline = np.percentile(F_neuSub , percentile_baseline , axis=0)
    dF = F_neuSub - F_baseline
    dFoF = dF / F_baseline
    if verbose:
        print(f'Calculated dFoF. Total elapsed time: {round(time.time() - tic,2)} seconds')
    
    return dFoF , dF , F_neuSub , F_baseline