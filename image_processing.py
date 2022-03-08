import numpy as np
import cv2
import copy

from . import indexing


def find_registration_transformation(
    im_template, 
    im_moving,
    warp_mode='MOTION_HOMOGRAPHY',
    n_iter=5000,
    termination_eps=1e-10,
    mask=None,
    gaussFiltSize=1
):
    """
    Find the transformation between two images.
    Only homography is supported for now.
    RH 2022

    Args:
        im_template (np.ndarray):
            Template image
        im_moving (np.ndarray):
            Moving image
        warp_mode (str):
            warp mode.
            See cv2.findTransformECC for more info.
            MOTION_TRANSLATION sets a translational motion model; warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated.
            MOTION_EUCLIDEAN sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is 2x3.
            MOTION_AFFINE sets an affine motion model (DEFAULT); six parameters are estimated; warpMatrix is 2x3.
            MOTION_HOMOGRAPHY sets a homography as a motion model; eight parameters are estimated;`warpMatrix` is 3x3.
        n_iter (int):
            Number of iterations
        termination_eps (float):
            Termination epsilon.
            Threshold of the increment in the correlation
             coefficient between two iterations
        mask (np.ndarray):
            Binary mask. If None, no mask is used.
            Regions where mask is zero are ignored 
             during the registration.
        gaussFiltSize (int):
            gaussian filter size. If 0, no gaussian 
             filter is used.
    
    Returns:
        warp_matrix (np.ndarray):
            Warp matrix. See cv2.findTransformECC for more info.
            Can be applied using cv2.warpAffine or 
             cv2.warpPerspective.
    """
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iter,
        termination_eps,
    )
    print("running findTransformECC")
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(
        templateImage=im_template, 
        inputImage=im_moving, 
        warpMatrix=warp_matrix,
        motionType=warp_mode, 
        criteria=criteria, 
        inputMask=mask, 
        gaussFiltSize=gaussFiltSize
    )
    return warp_matrix


def apply_warp_transform(
    im_in,
    warp_matrix,
    interpolation_method=cv2.INTER_LINEAR, 
    borderMode=cv2.BORDER_CONSTANT, 
    borderValue=0
):
    """
    Apply a warp transform to an image.
    RH 2022

    Args:
        im_in (np.ndarray):
            Input image
        warp_matrix (np.ndarray):
            Warp matrix. See cv2.findTransformECC for more info.
        interpolation_method (int):
            Interpolation method.
            See cv2.warpAffine for more info.
        borderMode (int):
            Border mode.
            Whether to use a constant border value or not.
            See cv2.warpAffine for more info.
        borderValue (int):
            Border value.

    Returns:
        im_out (np.ndarray):
            Output image
    """
    im_out = cv2.warpPerspective(
        src=im_in,
        M=warp_matrix,
        dsize=(im_in.shape[1], im_in.shape[0]), 
        dst=copy.copy(im_in), 
        flags=interpolation_method, 
        borderMode=borderMode, 
        borderValue=borderValue
    )
    return im_out

def stack_to_RGB(images):
    """
    Convert a stack of images to RGB.
    RH 2022

    Args:
        images (list of np.ndarray):
            List of images.
            Can be between 1 and 3 images.
            Can also be a single image.

    Returns:
        im_out (np.ndarray):
            RGB image.
    """
    if isinstance(images, np.ndarray):
        images = [images]
    
    im_out = np.stack(images, axis=2)
    appended_images = np.stack([images[0]*0] * (3 - len(images)), axis=2)
    im_out = np.concatenate([im_out, appended_images], axis=2)

    return im_out

def bin_array(array, bin_widths=[2,3,4], method='append', function=np.nanmean, function_kwargs=None):
    """
    Bins an array of arbitrary shape along the
     first N dimensions. Works great for images.
    Works by iteratively reshaping and applying
     the defined function (ie averaging).
    Function requires pad_with_singleton_dims function.
    RH 2022

    Args:
        array (np.ndarray):
            Input array.
        bin_widths (list of int):
            List of bin widths for first N dimensions.
        method (str):
            Method for binning.
            'append' appends NaNs to the end of
             each dimension that is not divisible
             by the bin width.
            'prepend' prepends NaNs to the beginning
             of each dimension that is not divisible
             by the bin width.
            'post_crop' crops the array to be divisible
             by the bin width by cropping off the end.
            'post_crop' crops the array to be divisible
             by the bin width by cropping off the 
             beginning.
        function (function):
            Function to apply to each bin.
            Must at least take array and axis as arguments.
            Doesn't need to handle NaNs if 
             method==pre or post_crop.
            Typically: 
                np.nanmean
                np.nanmedian
                np.nanstd
                np.nanvar
                np.nanmax
                np.nanmin
                np.nanpercentile (requires percentile kwarg)
                np.nanquantile (requires percentile kwarg)
        function_kwargs (dict):
            Keyword arguments for function.

    Returns:
        array_out (np.ndarray):
            Output array.
    """
    
    # A cute function for flattening a list of lists.
    flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]

    s = list(array.shape)

    arr_out = copy.deepcopy(array)
    for n, w in enumerate(bin_widths):
        if arr_out.shape[n] % w != 0:
            if method=='append':
                s_pad = copy.copy(s)
                s_pad[n] = w - (arr_out.shape[n] % w)
                arr_out = np.concatenate(
                    [arr_out, np.zeros(s_pad)*np.nan],
                    axis=n
                )
            
            if method=='prepend':
                s_pad = copy.copy(s)
                s_pad[n] = w - (arr_out.shape[n] % w)
                arr_out = np.concatenate(
                    [np.zeros(s_pad)*np.nan, arr_out],
                    axis=n
                )
                
            if method=='post_crop':
                s_crop = indexing.pad_with_singleton_dims(np.arange((arr_out.shape[n]//w)*w), n_dims_pre=n, n_dims_post=array.ndim-(n+1))
                arr_out = np.take_along_axis(arr_out, s_crop, axis=n)

            if method=='pre_crop':
                s_crop = indexing.pad_with_singleton_dims(np.arange((arr_out.shape[n]//w)*w) + arr_out.shape[n]%w, n_dims_pre=n, n_dims_post=array.ndim-(n+1))
                arr_out = np.take_along_axis(arr_out, s_crop, axis=n)
        
        s_n = list(arr_out.shape)
        s_n[n] = [w, arr_out.shape[n] // w]
        s_n = flatten_list(s_n)
        arr_out = np.reshape(arr_out, s_n, order='F')
        arr_out = function(arr_out, axis=n, **function_kwargs)
        
    return arr_out