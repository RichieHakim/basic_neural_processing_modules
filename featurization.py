'''
Table of Contents

Functions and Interdependencies:
    make_cosine_kernels
    amplitude_basis_expansion
'''

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import time

def make_cosine_kernels(y=None,
                        y_resolution=500,
                        y_range=None, 
                        n_kernels=6, 
                        crop_first_and_last_kernels=True, 
                        warping_curve=None, 
                        plot_pref=1):
    '''
    Makes a set of cosines offset by pi/2.
    This function is useful for doing amplitude basis expansions.
    The outputs of this function can be used as a look up table
    for some one dimensional signal to effectively make its 
    representation nonlinear and high dimensional.

    This function works great up to ~100 kernels, then some
    rounding errors mess up the range slightly, but not badly.
    RH 2021

    Args:
        y (ndarray):
            1-D array. Used only to obtain the min and max for 
            setting 'y_range'. If 'y_range' is not None, then 
            this is unused.
        y_range (2 entry array):
            [min , max] of 'xAxis_of_curves'. Effectively defines
            the range of the look up table that will be applied.
        y_resolution (int):
            Sets the size (and therefore resolution) of the
            output array and xAxis_of_curves.
        n_kernels (int):
            Number of cosine kernels to be made. Increasing this
            also decreases the width of individual kernels.
        crop_first_and_last_kernels (bool):
            Preference of whether the first and last kernels
            should be cropped at their peak. Doing this
            maintains a constant sum over the entire range of
            kernels
        warping_curve (1-D array):
            A curve used for warping (via interpolation) the 
            shape of the cosine kernels. This allows for
            non-uniform widths of kernels. Make this array
            large (>=100000) values to keep interpolation smooth.
        plot_pref (int 0-2):
            set to 0: plot nothing
            set to 1: show output curves
            set to 2: show intermediate provessing curves

    Returns:
        kernels_output (ndarray):
            The output cosine kernels
        LUT (1-D array):
            Look Up Table.
            The look up table defined by 'y_range' or the
            min/max of 'y'. Use this axis to expand some signal
            'y' using the kernel bases functions
    '''

    if y is None:
        y = np.arange(0, 1, 0.1)

    if y_range is None:
        y_range = np.array([np.min(y) , np.max(y)])

    if warping_curve is None:
        warping_curve = np.arange(1000000)

    y_resolution_highRes = y_resolution * 10
    bases_highRes = np.zeros((y_resolution_highRes, n_kernels))

    cos_width_highRes = int((bases_highRes.shape[0] / (n_kernels+1))*2)
    cos_kernel_highRes = (np.cos(np.linspace(-np.pi, np.pi, cos_width_highRes)) + 1)/2

    for ii in range(n_kernels):
        bases_highRes[int(cos_width_highRes*(ii/2)) : int(cos_width_highRes*((ii/2)+1)) , ii] = cos_kernel_highRes

    if crop_first_and_last_kernels:
        bases_highRes_cropped = bases_highRes[int(cos_width_highRes/2):-int(cos_width_highRes/2)]
    else:
        bases_highRes_cropped = bases_highRes

    WC_norm = warping_curve - np.min(warping_curve)
    WC_norm = (WC_norm/np.max(WC_norm)) * (bases_highRes_cropped.shape[0]-1)

    f_interp = scipy.interpolate.interp1d(np.arange(bases_highRes_cropped.shape[0]),
                                          bases_highRes_cropped, axis=0)

    kernels_output = f_interp(WC_norm[np.uint64(np.round(np.linspace(0, len(WC_norm)-1, y_resolution)))])

    LUT = np.linspace(y_range[0] , y_range[1], y_resolution)

    if plot_pref==1:
        fig, axs = plt.subplots(1)
        axs.plot(LUT, kernels_output)
        axs.set_xlabel('y_range look up axis')
        axs.set_title('kernels_warped')
    if plot_pref>=2:
        fig, axs = plt.subplots(6, figsize=(5,15))
        axs[0].plot(bases_highRes)
        axs[0].set_title('kernels')
        axs[1].plot(bases_highRes_cropped)
        axs[1].set_title('kernels_cropped')
        axs[2].plot(warping_curve)
        axs[2].set_title('warping_curve')
        axs[3].plot(WC_norm)
        axs[3].set_title('warping_curve_normalized')
        axs[4].plot(LUT, kernels_output)
        axs[4].set_title('kernels_warped')
        axs[4].set_xlabel('y_range look up axis')
        axs[5].plot(np.sum(kernels_output, axis=1))
        axs[5].set_ylim([0,1.1])
        axs[5].set_title('sum of kernels')
        
    return kernels_output , LUT


def amplitude_basis_expansion(y, LUT, kernels):
    '''
    Performs amplitude basis expansion of one or more arrays using a set
    of kernels.
    Use a function like 'make_cosine_kernels' to make 'LUT' and 'kernels'
    RH 2021

    Args:
        y (ndarray): 
            1-D or 2-D array. First dimension is samples (eg time) and second 
            dimension is feature number (eg neuron number). 
            The values of the array should be scaled to be within the range of
            'LUT'. You can use 'timeSeries.scale_between' for scaling.
        LUT (1-D array):
            Look Up Table. This is the conversion between y-value and 
            index of kernel. Basically the value of y at each sample point 
            is compared to LUT and the index of the closest match determines
            the index kernels to use, therefore resulting in an amplitude 
            output for each kernel. This array is output from a function
            like 'make_cosine_kernels'
        kernels (ndarray):
            Basis functions/kernels to use for expanding 'y'.
            shape: (len(LUT) , n_kernels)
            Output of this function will be based on where the y-values
            land within these kernels. This array is output from a function
            like 'make_cosine_kernels'
    
    Returns:
        y_expanded (ndarray):
            Basis-expanded 'y'.
            shape: (y.shape[0], y.shape[1], kernels.shape[1])
    '''

    tic = time.time()
    LUT_array = np.tile(LUT, (len(y),1)).T
    
    print(f'computing basis expansion')
    LUT_idx = np.argmin(
        np.abs(LUT_array[:,:,None] - y),
        axis=0)

    y_expanded = kernels[LUT_idx,:]

    print(f'finished in {round(time.time() - tic, 3)} s')
    print(f'output array size: {y_expanded.shape}')

    return y_expanded


def make_distance_image(center_idx, vid_height, vid_width):
    """
    creates a matrix of cartesian coordinate distances from the center
    RH 2021
    
    Args:
        center_idx (list): chosen center index
        vid_height (int): height of the video in pixels
        vid_width (int): width of the video in pixels

    Returns:
        distance_image (np.ndarray): array of distances to the center index

    """

    x, y = np.meshgrid(range(vid_width), range(vid_height))  # note dim 1:X and dim 2:Y
    return np.sqrt((y - int(center_idx[1])) ** 2 + (x - int(center_idx[0])) ** 2)


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


def make_cosine_taurus(offset, width):
    l = (offset + width)*2 + 1
    c_idx = (l-1)/2
    cosine = np.cos(np.linspace((-np.pi) , (np.pi), width)) + 1
    cosine = np.concatenate((np.zeros(offset), cosine))
    dist_im = make_distance_image([c_idx , c_idx], l, l)
    taurus = cosine[np.searchsorted(np.arange(len(cosine)), dist_im, side='left')-1]
    return taurus