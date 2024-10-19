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


def amplitude_basis_expansion(y, LUT, kernels, device='cpu', verbose=False):
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

    import torch

    if isinstance(y, np.ndarray):
        flag_numpy = True
    else:
        flag_numpy = False

    y = torch.as_tensor(y, device=device)
    LUT = torch.as_tensor(LUT, device=device)
    kernels = torch.as_tensor(kernels, device=device)


    tic = time.time()
    LUT_array = torch.tile(LUT, (len(y),1)).T
    
    print(f'computing basis expansion') if verbose else None
    LUT_idx = torch.argmin(
        torch.abs(LUT_array[:,:,None] - y),
        dim=0)

    y_expanded = kernels[LUT_idx,:]

    print(f'finished in {round(time.time() - tic, 3)} s') if verbose else None
    print(f'output array size: {y_expanded.shape}') if verbose else None

    if flag_numpy:
        y_expanded = y_expanded.cpu().numpy()

    return y_expanded


def make_distance_grid(shape=(512,512), p=2, idx_center=None, return_axes=False, use_fftshift_center=False):
    """
    Creates a matrix of distances from the center.
    Can calculate the Minkowski distance for any p.
    RH 2023
    
    Args:
        shape (Tuple[int, int, ...]):
            Shape of the n-dimensional grid (i,j,k,...)
            If a shape value is odd, the center will be the center
             of that dimension. If a shape value is even, the center
             will be between the two center points.
        p (int):
            Order of the Minkowski distance.
            p=1 is the Manhattan distance
            p=2 is the Euclidean distance
            p=inf is the Chebyshev distance
        idx_center Optional[Tuple[int, int, ...]]:
            The index of the center of the grid. If None, the center is
            assumed to be the center of the grid. If provided, the center
            will be set to this index. This is useful for odd shaped grids
            where the center is not obvious.
        return_axes (bool):
            If True, return the axes of the grid as well. Return will be a
            tuple.
        use_fft_center (bool):
            If True, the center of the grid will be the center of the FFT
            grid. This is useful for FFT operations where the center is
            assumed to be the top left corner.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            distance_image (np.ndarray): 
                array of distances to the center index
            axes (Optional[np.ndarray]):
                axes of the grid as well. Only returned if return_axes=True

    """
    if use_fftshift_center:
        ## Find idx wheter freq=0. Use np.fft.fftfreq
        freqs_h, freqs_w = np.fft.fftshift(np.fft.fftfreq(shape[0])), np.fft.fftshift(np.fft.fftfreq(shape[1]))
        idx_center = (np.argmin(np.abs(freqs_h)), np.argmin(np.abs(freqs_w)))

    shape = np.array(shape)
    if idx_center is not None:
        axes = [np.linspace(-idx_center[i], shape[i] - idx_center[i] - 1, shape[i]) for i in range(len(shape))]
    else:
        axes = [np.arange(-(d - 1) / 2, (d - 1) / 2 + 0.5) for d in shape]
    grid = np.stack(
        np.meshgrid(*axes, indexing="ij"),
        axis=0,
    )
    if idx_center is not None:
        grid_dist = np.linalg.norm(
            grid ,
            ord=p,
            axis=0,
        )
    else:
        grid_dist = np.linalg.norm(
            grid,
            ord=p,
            axis=0,
        )

    return grid_dist if not return_axes else (grid_dist, axes)

def gaussian_kernel_2D(image_size=(11, 11), sig=1, center=None):
    """
    Generate a 2D or 1D gaussian kernel
    RH 2021
    
    Args:
        image_size (tuple): 
            The total image size (width, height). Make second value 0 to make 1D gaussian
        sig (scalar): 
            The sigma value of the gaussian
        center (tuple):  
            The mean position (X, Y) - where high value expected. 0-indexed.
            Make second value 0 to make 1D gaussian.
            If None, assume center of image.
    
    Return:
        kernel (np.ndarray): 
            2D or 1D array of the gaussian kernel
    """
    # If center is not provided, assume it is the middle of the image
    if center is None:
        center = (image_size[0] // 2, image_size[1] // 2)

    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel

def cosine_kernel_2D(center=(5,5), image_size=(11,11), width=5):
    """
    Generate a 2D cosine kernel
    RH 2021
    
    Args:
        center (tuple):  
            The mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D
        image_size (tuple): 
            The total image size (width, height). Make second value 0 to make 1D
        width (scalar): 
            The full width of one cycle of the cosine
    
    Return:
        k_cos (np.ndarray): 
            2D or 1D array of the cosine kernel
    """
    x, y = np.meshgrid(range(image_size[1]), range(image_size[0]))  # note dim 1:X and dim 2:Y
    dist = np.sqrt((y - int(center[1])) ** 2 + (x - int(center[0])) ** 2)
    dist_scaled = (dist/(width/2))*np.pi
    dist_scaled[np.abs(dist_scaled > np.pi)] = np.pi
    k_cos = (np.cos(dist_scaled) + 1)/2
    return k_cos


def make_cosine_taurus(offset, width):
    l = (offset + width)*2 + 1
    c_idx = (l-1)/2
    cosine = np.cos(np.linspace((-np.pi) , (np.pi), width)) + 1
    cosine = np.concatenate((np.zeros(offset), cosine))
    dist_im = make_distance_grid([c_idx , c_idx])
    taurus = cosine[np.searchsorted(np.arange(len(cosine)), dist_im, side='left')-1]
    return taurus

def helper_shift(X, shift, fill_val=0):
    X_shift = np.empty_like(X)
    if shift>0:
        X_shift[:shift] = fill_val
        X_shift[shift:] = X[:-shift]
    elif shift<0:
        X_shift[shift:] = fill_val
        X_shift[:shift] = X[-shift:]
    else:
        X_shift[:] = X
    return X_shift
def shift_along_axis(X, shift, fill_val=0, axis=0):
    return np.apply_along_axis(helper_shift, axis, X, shift, fill_val)


def mspline_grid(order, num_basis_funcs, nt):
    """
    Generate a set of M-spline basis functions with evenly
    spaced knots.

    Ramsay, J. O. (1988). Monotone regression splines in action.
    Statistical science, 3(4), 425-441.

    Stolen with love from Alex Williams: https://gist.github.com/ahwillia/511097a0968bf05a2579db0eab353393

    Parameters
    ----------
    order : int
        Order parameter of the splines.
    num_basis_funcs : int
        Number of desired basis functions. Note that we
        require num_basis_funcs >= order.
    nt : int
        Number of points to evaluate the basis functions.
    Returns
    -------
    spine_basis : array
        Matrix with shape (num_basis_funcs, nt), holding the
        desired spline basis functions.
    """

    # Determine number of interior knots.
    num_interior_knots = num_basis_funcs - order
    
    if num_interior_knots < 0:
        raise ValueError(
            "Spline `order` parameter cannot be larger "
            "than `num_basis_funcs` parameter."
        )

    # Fine grid of numerically evaluated points.
    x = np.linspace(0, 1 - 1e-6, nt)

    # Set of spline knots. We need to add extra knots to
    # the end to handle boundary conditions for higher-order
    # spline bases. See Ramsay (1988) cited above.
    #
    # Note - this is poorly explained on most corners of the
    # internet that I've found.
    knots = np.concatenate((
        np.zeros(order - 1),
        np.linspace(0, 1, num_interior_knots + 2),
        np.ones(order - 1),
    ))

    # Evaluate and stack each basis function.
    return np.row_stack(
        [mspline(x, order, i, knots) for i in range(num_basis_funcs)]
    )
def mspline(x, k, i, T):
    """
    Compute M-spline basis function `i` at points `x` for a spline
    basis of order-`k` with knots `T`.
    Parameters
    ----------
    x : array
        Vector holding points to evaluate the spline.
    """

    # Boundary conditions.
    if (T[i + k] - T[i]) < 1e-6:
        return np.zeros_like(x)

    # Special base case of first-order spline basis.
    elif k == 1:
        v = np.zeros_like(x)
        v[(x >= T[i]) & (x < T[i + 1])] = 1 / (T[i + 1] - T[i])
        return v

    # General case, defined recursively
    else:
        return k * (
            (x - T[i]) * mspline(x, k - 1, i, T)
            + (T[i + k] - x) * mspline(x, k - 1, i + 1, T)
        ) / ((k-1) * (T[i + k] - T[i]))
    

def make_scaled_wave_basis(
    mother, 
    lens_waves, 
    lens_windows=None, 
    interp_kind='cubic', 
    fill_value=0,
):
    """
    Generates a set of wavelet-like basis functions by scaling a mother wavelet
    to different sizes. \n
    
    Note that this does not necessarily result in a true
    orthogonal 'wavelet' basis set. This function uses interpolation to adjust
    the mother wavelet's size, making it suitable for creating filter banks with
    various frequency resolutions.

    RH 2024

    Parameters:
    - mother (np.ndarray): 
        A 1D numpy array representing the mother wavelet used as the basis for scaling.
    - lens_waves (int, list, tuple, np.ndarray): 
        The lengths of the output waves. Can be a single integer or a list/array of integers.
    - lens_windows (int, list, tuple, np.ndarray, None): 
        The window lengths for each of the output waves. If None, defaults to
        the values in lens_waves. Can be a single integer (applied to all waves)
        or a list/array of integers corresponding to each wave length.
    - interp_kind (str): 
        Specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic', 'cubic', where 'zero', 'slinear',
        'quadratic' and 'cubic' refer to a spline interpolation of zeroth,
        first, second or third order).
    - fill_value (float): 
        Value used to fill in for requested points outside of the domain of the
        x_mother. Can be anything from scipy.interpolate.interp1d. If not
        provided, defaults to 0.

    Returns:
        (tuple):
        - waves (list): 
            List of the scaled wavelets.
        - xs (list):
            List of the x-values for each of the scaled wavelets.

    Example:
    ```
    mother_wave = np.cos(np.linspace(-2*np.pi, 2*np.pi, 10000, endpoint=True))
    lens_waves = [50, 100, 200]
    lens_windows = [100, 200, 400]
    waves, xs = make_scaled_wave_basis(mother_wave, lens_waves, lens_windows)
    ```
    """
    assert isinstance(mother, np.ndarray), "mother must be a 1D array"
    assert mother.ndim == 1, "mother must be a 1D array"
    
    arraylikes = (list, tuple, np.ndarray)
    if isinstance(lens_waves, arraylikes):
        lens_waves = np.array(lens_waves, dtype=int)
        if lens_windows is None:
            lens_windows = lens_waves
        if isinstance(lens_windows, int):
            lens_windows = np.array([lens_windows] * len(lens_waves), dtype=int)
        if isinstance(lens_windows, arraylikes):
            assert len(lens_waves) == len(lens_windows), "lens_waves and lens_windows must have the same length"
            lens_windows = np.array(lens_windows, dtype=int)
        else:
            raise ValueError("lens_windows must be an int or an array-like")
    elif isinstance(lens_waves, int):
        if lens_windows is None:
            lens_windows = lens_waves
        if isinstance(lens_windows, int):
            lens_waves = np.array([lens_waves], dtype=int)
            lens_windows = np.array([lens_windows], dtype=int)
        if isinstance(lens_windows, arraylikes):
            lens_waves = np.array([lens_waves] * len(lens_windows), dtype=int)
            lens_windows = np.array(lens_windows, dtype=int)
        else:
            raise ValueError("lens_windows must be an int or an array-like")
    else:
        raise ValueError("lens_waves must be an int or an array-like")


    x_mother = np.linspace(start=0, stop=1, num=len(mother), endpoint=True) - 0.5

    interpolator = scipy.interpolate.interp1d(
        x=x_mother,
        y=mother, 
        kind=interp_kind, 
        fill_value=fill_value, 
        bounds_error=False, 
        assume_sorted=True,
    )

    waves = []
    xs = []
    for i_wave, (l_wave, l_window) in enumerate(zip(lens_waves, lens_windows)):
        x_wave = (np.linspace(start=0, stop=1, num=l_window, endpoint=True) - 0.5) * (l_window / l_wave)
        wave = interpolator(x_wave)
        waves.append(wave)
        xs.append(x_wave)

    return waves, xs
