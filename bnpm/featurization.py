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


def make_distance_grid(shape=(512,512), p=2):
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

    Returns:
        distance_image (np.ndarray): 
            array of distances to the center index

    """
    axes = [np.arange(-(d - 1) / 2, (d - 1) / 2 + 0.5) for d in shape]
    grid_dist = np.linalg.norm(
        np.stack(
            np.meshgrid(*axes),
            axis=0,
        ),
        ord=p,
        axis=0,
    )

    return grid_dist

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


class Toeplitz_convolution2d:
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix 
     multiplication method.
    Allows for SPARSE 'x' inputs. 'k' should remain dense.
    Ideal when 'x' is very sparse (density<0.01), 'x' is small
     (shape <(1000,1000)), 'k' is small (shape <(100,100)), and
     the batch size is large (e.g. 1000+).
    Generally faster than scipy.signal.convolve2d when convolving mutliple
     arrays with the same kernel. Maintains low memory footprint by
     storing the toeplitz matrix as a sparse matrix.

    See: https://stackoverflow.com/a/51865516 and https://github.com/alisaaalehi/convolution_as_multiplication
     for a nice illustration.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html 
     for 1D version.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html#scipy.linalg.matmul_toeplitz 
     for potential ways to make this implementation faster.

    Test with: tests.test_toeplitz_convolution2d()
    RH 2022
    """
    def __init__(
        self,
        x_shape,
        k,
        mode='same',
        dtype=None,
    ):
        """
        Initialize the convolution object.
        Makes the Toeplitz matrix and stores it.

        Args:
            x_shape (tuple):
                The shape of the 2D array to be convolved.
            k (np.ndarray):
                2D kernel to convolve with
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
            dtype (np.dtype):
                The data type to use for the Toeplitz matrix.
                Ideally, this matches the data type of the input array.
                If None, then the data type of the kernel is used.
        """
        self.k = k = np.flipud(k.copy())
        self.mode = mode
        self.x_shape = x_shape
        self.dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        self.so = so = size_output_array = ( (k.shape[0] + x_shape[0] -1), (k.shape[1] + x_shape[1] -1))  ## 'size out' is the size of the output array

        ## make the toeplitz matrices
        t = toeplitz_matrices = [scipy.sparse.diags(
            diagonals=np.ones((k.shape[1], x_shape[1]), dtype=self.dtype) * k_i[::-1][:,None], 
            offsets=np.arange(-k.shape[1]+1, 1), 
            shape=(so[1], x_shape[1]),
            dtype=self.dtype,
        ) for k_i in k[::-1]]  ## make the toeplitz matrices for the rows of the kernel
        tc = toeplitz_concatenated = scipy.sparse.vstack(t + [scipy.sparse.dia_matrix((t[0].shape), dtype=self.dtype)]*(x_shape[0]-1))  ## add empty matrices to the bottom of the block due to padding, then concatenate

        ## make the double block toeplitz matrix
        self.dt = double_toeplitz = scipy.sparse.hstack([self._roll_sparse(
            x=tc, 
            shift=(ii>0)*ii*(so[1])  ## shift the blocks by the size of the output array
        ) for ii in range(x_shape[0])]).tocsr()
    
    def __call__(
        self,
        x,
        batching=True,
        mode=None,
    ):
        """
        Convolve the input array with the kernel.

        Args:
            x (np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix):
                Input array(s) (i.e. image(s)) to convolve with the kernel
                If batching==False: Single 2D array to convolve with the kernel.
                    shape: (self.x_shape[0], self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
                If batching==True: Multiple 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, self.x_shape[0]*self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
            batching (bool):
                If False, x is a single 2D array.
                If True, x is a 2D array where each row is a flattened 2D array.
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
                Overrides the mode set in __init__.

        Returns:
            out (np.ndarray or scipy.sparse.csr_matrix):
                If batching==True: Multiple convolved 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, height*width)
                    type: np.ndarray or scipy.sparse.csc_matrix
                If batching==False: Single convolved 2D array of shape (height, width)
        """
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        issparse = scipy.sparse.issparse(x)
        
        if batching:
            x_v = x.T  ## transpose into column vectors
        else:
            x_v = x.reshape(-1, 1)  ## reshape 2D array into a column vector
        
        if issparse:
            x_v = x_v.tocsc()
        
        out_v = self.dt @ x_v  ## if sparse, then 'out_v' will be a csc matrix
            
        ## crop the output to the correct size
        if mode == 'full':
            p_t = 0
            p_b = self.so[0]+1
            p_l = 0
            p_r = self.so[1]+1
        if mode == 'same':
            p_t = (self.k.shape[0]-1)//2
            p_b = -(self.k.shape[0]-1)//2
            p_l = (self.k.shape[1]-1)//2
            p_r = -(self.k.shape[1]-1)//2

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        if mode == 'valid':
            p_t = (self.k.shape[0]-1)
            p_b = -(self.k.shape[0]-1)
            p_l = (self.k.shape[1]-1)
            p_r = -(self.k.shape[1]-1)

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        
        if batching:
            idx_crop = np.zeros((self.so), dtype=np.bool_)
            idx_crop[p_t:p_b, p_l:p_r] = True
            idx_crop = idx_crop.reshape(-1)
            out = out_v[idx_crop,:].T
        else:
            if issparse:
                out = out_v.reshape((self.so)).tocsc()[p_t:p_b, p_l:p_r]
            else:
                out = out_v.reshape((self.so))[p_t:p_b, p_l:p_r]  ## reshape back into 2D array and crop
        return out
    
    def _roll_sparse(
        self,
        x,
        shift,
    ):
        """
        Roll columns of a sparse matrix.
        """
        out = x.copy()
        out.row += shift
        return out