import multiprocessing as mp

import numpy as np
import cv2

import copy
import torch
import torchvision
from tqdm.notebook import tqdm

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
    ## assert that the inputs are numpy arrays of dtype np.uint8
    assert isinstance(im_template, np.ndarray) and im_template.dtype == np.uint8
    assert isinstance(im_moving, np.ndarray) and im_moving.dtype == np.uint8    

    if warp_mode == 'MOTION_HOMOGRAPHY':
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    elif warp_mode == 'MOTION_AFFINE':
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif warp_mode == 'MOTION_EUCLIDEAN':
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif warp_mode == 'MOTION_TRANSLATION':
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        raise ValueError(f"warp_mode {warp_mode} not recognized")
    

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
    if warp_matrix.shape == (2, 3):
        im_out = cv2.warpAffine(
            src=im_in,
            M=warp_matrix,
            dsize=(im_in.shape[1], im_in.shape[0]),
            dst=copy.copy(im_in),
            flags=interpolation_method + cv2.WARP_INVERSE_MAP,
            borderMode=borderMode,
            borderValue=borderValue
        )
        
    elif warp_matrix.shape == (3, 3):
        im_out = cv2.warpPerspective(
            src=im_in,
            M=warp_matrix,
            dsize=(im_in.shape[1], im_in.shape[0]), 
            dst=copy.copy(im_in), 
            flags=interpolation_method + cv2.WARP_INVERSE_MAP, 
            borderMode=borderMode, 
            borderValue=borderValue
        )

    else:
        raise ValueError(f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3) or (3, 3)")
    
    return im_out


def affine_matrix_to_flow_field(warp_matrix, x, y, return_remappingIdx=False):
    """
    Convert an affine warp matrix (2x3) into a flow field (2D).
    RH 2023
    
    Args:
        warp_matrix (np.ndarray or torch.Tensor): 
            Warp matrix of shape (2, 3).
        x (int): 
            Width of the desired flow field.
        y (int): 
            Height of the desired flow field.
        return_remappingIdx (bool):
            Whether to return the remapping indices.
            These are just the flow field + the x and y coordinates.
    
    Returns:
        field (np.ndarray or torch.Tensor): 
            Flow field of shape (x, y, 2) representing the x and y displacements.
    """
    assert warp_matrix.shape == (2, 3), f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3)"
    assert isinstance(x, int) and isinstance(y, int), f"x and y must be integers"
    assert x > 0 and y > 0, f"x and y must be positive"

    if isinstance(warp_matrix, torch.Tensor):
        stack, meshgrid, arange, hstack, ones, float32 = torch.stack, torch.meshgrid, torch.arange, torch.hstack, torch.ones, torch.float32
        stack_partial = lambda x: stack(x, dim=0)
    elif isinstance(warp_matrix, np.ndarray):
        stack, meshgrid, arange, hstack, ones, float32 = np.stack, np.meshgrid, np.arange, np.hstack, np.ones, np.float32
        stack_partial = lambda x: stack(x, axis=0)
    else:
        raise ValueError(f"warp_matrix must be a torch.Tensor or np.ndarray")

    # create the grid
    mesh = stack_partial(meshgrid(arange(x, dtype=np.float32), arange(y, dtype=np.float32)))
    mesh_coords = hstack((mesh.reshape(2,-1).T, ones((x*y, 1))))
    # mesh_coords = np.concatenate((mesh, np.ones((1, y, x))), axis=0).reshape(3, -1).T
    
    # warp the grid
    mesh_coords_warped = mesh_coords @ warp_matrix.T
    
    # reshape the warped grid
    remapIdx = mesh_coords_warped.T.reshape(2, y, x)

    # compute the flow field if desired
    if return_remappingIdx:
        return remapIdx
    else:
        field = remapIdx - mesh
        return field


@torch.jit.script
def phase_correlation_helper(
    im_template,
    im_moving,
    mask_fft=None, 
    compute_maskFFT: bool=False, 
    template_precomputed: bool=False,
    eps: float=1e-17,
):
    if im_template.ndim == 2:
        im_template = im_template[None, ...]
    if im_moving.ndim == 2:
        im_moving = im_moving[None, ...]
        return_2D = True
    else:
        return_2D = False
    if compute_maskFFT:
        mask_fft = mask_fft[None, ...]

    dims = (-2, -1)
        
    if compute_maskFFT:
        mask_fft = torch.fft.fftshift(mask_fft/mask_fft.sum(), dim=dims)
        fft_template = torch.conj(torch.fft.fft2(im_template, dim=dims) * mask_fft) if not template_precomputed else im_template
        fft_moving = torch.fft.fft2(im_moving, dim=dims) * mask_fft
    else:
        fft_template = torch.conj(torch.fft.fft2(im_template, dim=dims)) if not template_precomputed else im_template
        fft_moving = torch.fft.fft2(im_moving, dim=dims)

    R = fft_template[:,None,:,:] * fft_moving[None,:,:,:]
    R /= torch.abs(R) + eps
    
    cc = torch.fft.fftshift(torch.fft.ifft2(R, dim=dims), dim=dims).real.squeeze()
    
    return cc if not return_2D else cc[0]
def phase_correlation(
    im_template, 
    im_moving,
    mask_fft=None, 
    template_precomputed=False, 
    device='cpu'
):
    """
    Perform phase correlation on two images.
    Uses pytorch for speed
    RH 2022
    
    Args:
        im_template (np.ndarray or torch.Tensor):
            Template image(s).
            If ndim=2, a single image is assumed.
                shape: (height, width)
            if ndim=3, multiple images are assumed, dim=0 is the batch dim.
                shape: (batch, height, width)
                dim 0 should either be length 1 or the same as im_moving.
            If template_precomputed is True, this is assumed to be:
             np.conj(np.fft.fft2(im_template, axis=(1,2)) * mask_fft)
        im_moving (np.ndarray or torch.Tensor):
            Moving image(s).
            If ndim=2, a single image is assumed.
                shape: (height, width)
            if ndim=3, multiple images are assumed, dim=0 is the batch dim.
                shape: (batch, height, width)
                dim 0 should either be length 1 or the same as im_template.
        mask_fft (np.ndarray or torch.Tensor):
            Mask for the FFT.
            Shape: (height, width)
            If None, no mask is used.
        template_precomputed (bool):
            If True, im_template is assumed to be:
             np.conj(np.fft.fft2(im_template, axis=(1,2)) * mask_fft)
        device (str):
            Device to use.
    
    Returns:
        cc (np.ndarray):
            Phase correlation coefficient.
            Middle of image is zero-shift.
            Last two dims are frame height and width.
    """
    if isinstance(im_template, np.ndarray):
        im_template = torch.from_numpy(im_template).to(device)
        return_numpy = True
    else:
        return_numpy = False
    if isinstance(im_moving, np.ndarray):
        im_moving = torch.from_numpy(im_moving).to(device)
    if isinstance(mask_fft, np.ndarray):
        mask_fft = torch.from_numpy(mask_fft).to(device)
    if isinstance(mask_fft, torch.Tensor):
        if mask_fft.device != device:
            mask_fft = mask_fft.to(device)

    cc = phase_correlation_helper(
        im_template=im_template,
        im_moving=im_moving,
        mask_fft=mask_fft if mask_fft is not None else torch.as_tensor([1], device=device),
        compute_maskFFT=(mask_fft is not None),
        template_precomputed=template_precomputed,
    )

    if return_numpy:
        cc = cc.cpu().numpy()
    return cc

@torch.jit.script 
def phaseCorrelationImage_to_shift_helper(cc_im):
    cc_im_shape = cc_im.shape
    cc_im = cc_im[None,:] if cc_im.ndim==2 else cc_im
    height, width = cc_im.shape[-2:]
    vals_max, idx = torch.max(cc_im.reshape(cc_im_shape[0], cc_im_shape[1]*cc_im_shape[2]), dim=1)
    shift_x_raw = idx % cc_im_shape[2]
    shift_y_raw = torch.floor(idx / cc_im_shape[2]) % cc_im_shape[1]
    shifts_y_x = torch.stack(((torch.floor(torch.as_tensor(height)/2) - shift_y_raw) , (torch.ceil(torch.as_tensor(width)/2) - shift_x_raw)), dim=1)
    return shifts_y_x, vals_max
def phaseCorrelationImage_to_shift(cc_im):
    """
    Convert phase correlation image to pixel shift values.
    RH 2022

    Args:
        cc_im (np.ndarray):
            Phase correlation image.
            Middle of image is zero-shift.
            Shape: (height, width) or (batch, height, width)

    Returns:
        shifts (np.ndarray):
            Pixel shift values (y, x).
    """
    assert cc_im.ndim in [2,3], "cc_im must be 2D or 3D"
    cc_im = torch.as_tensor(cc_im)
    shifts_y_x, cc_max = phaseCorrelationImage_to_shift_helper(cc_im)
    return shifts_y_x, cc_max


def make_Fourier_mask(
    frame_shape_y_x=(512,512),
    bandpass_spatialFs_bounds=[1/128, 1/3],
    order_butter=5,
    mask=None,
    dtype_fft=torch.complex64,
    plot_pref=False,
    verbose=False,
):
    """
    Make a Fourier domain mask for the phase correlation.
    Used in BWAIN.

    Args:
        frame_shape_y_x (Tuple[int]):
            Shape of the images that will be passed through
                this class.
        bandpass_spatialFs_bounds (tuple): 
            (lowcut, highcut) in spatial frequency
            A butterworth filter is used to make the mask.
        order_butter (int):
            Order of the butterworth filter.
        mask (np.ndarray):
            If not None, use this mask instead of making one.
        plot_pref (bool):
            If True, plot the absolute value of the mask.

    Returns:
        mask_fft (torch.Tensor):
            Mask in the Fourier domain.
    """
    from .other_peoples_code import get_nd_butterworth_filter

    bandpass_spatialFs_bounds = list(bandpass_spatialFs_bounds)
    bandpass_spatialFs_bounds[0] = max(bandpass_spatialFs_bounds[0], 1e-9)
    
    if (isinstance(mask, (np.ndarray, torch.Tensor))) or ((mask != 'None') and (mask is not None)):
        mask = torch.as_tensor(mask, dtype=dtype_fft)
        mask = mask / mask.sum()
        mask_fftshift = torch.fft.fftshift(mask)
        print(f'User provided mask of shape: {mask.shape} was normalized to sum=1, fftshift-ed, and converted to a torch.Tensor')
    else:
        wfilt_h = get_nd_butterworth_filter(
            shape=frame_shape_y_x, 
            factor=bandpass_spatialFs_bounds[0], 
            order=order_butter, 
            high_pass=True, 
            real=False,
        )
        wfilt_l = get_nd_butterworth_filter(
            shape=frame_shape_y_x, 
            factor=bandpass_spatialFs_bounds[1], 
            order=order_butter, 
            high_pass=False, 
            real=False,
        )

        kernel = torch.as_tensor(
            wfilt_h * wfilt_l,
            dtype=dtype_fft,
        )

        mask = kernel / kernel.sum()
        # self.mask_fftshift = torch.fft.fftshift(self.mask)
        mask_fftshift = mask
        mask_fftshift = mask_fftshift.contiguous()

        if plot_pref and plot_pref!='False':
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(
                torch.abs(kernel.cpu()).numpy(), 
                # clim=[0,1],
            )
        if verbose:
            print(f'Created Fourier domain mask. self.mask_fftshift.shape: {mask_fftshift.shape}. Images input to find_translation_shifts will now be masked in the FFT domain.')

    return mask_fftshift


def find_translation_shifts(im1, im2, mask_fft=None, device='cpu', dtype=torch.float16):
    """
    Convenience function that combines `phase_correlation`
     and `phaseCorrelationImage_to_shift`
    
    Useful for matlab calls.
    """
    if mask_fft == 'None':
        mask_fft = None

    im1_t = torch.as_tensor(im1).type(dtype).to(device)
    im2_t = torch.as_tensor(im2).type(dtype).to(device)
    cc = phase_correlation(
        im1_t, 
        im2_t, 
        mask_fft=None,
        template_precomputed=False, 
        device=device
    )
    y_x, cc_max = phaseCorrelationImage_to_shift(cc)
    return y_x.cpu().numpy(), cc_max.cpu().numpy()

def mask_image_border(im, border_inner, border_outer, mask_value=0):
    """
    Mask an image with a border.
    RH 2022

    Args:
        im (np.ndarray):
            Input image.
        border_inner (int):
            Inner border width.
            Number of pixels in the center to mask.
            Value is the edge length of the center square.
        border_outer (int):
            Outer border width.
            Number of pixels in the border to mask.
        mask_value (float):
            Value to mask with.
    
    Returns:
        im_out (np.ndarray):
            Output image.
    """
    ## Find the center of the image
    height, width = im.shape
    center_y = cy = int(np.floor(height/2))
    center_x = cx = int(np.floor(width/2))

    ## make edge_lengths
    center_edge_length = cel = int(np.ceil(border_inner/2))
    outer_edge_length = oel = int(border_outer)

    ## Mask the center
    im[cy-cel:cy+cel, cx-cel:cx+cel] = mask_value
    ## Mask the border
    im[:oel, :] = mask_value
    im[-oel:, :] = mask_value
    im[:, :oel] = mask_value
    im[:, -oel:] = mask_value
    return im

def clahe(im, grid_size=50, clipLimit=0, normalize=True):
    """
    Perform Contrast Limited Adaptive Histogram Equalization (CLAHE)
     on an image.
    RH 2022

    Args:
        im (np.ndarray):
            Input image
        grid_size (int):
            Grid size.
            See cv2.createCLAHE for more info.
        clipLimit (int):
            Clip limit.
            See cv2.createCLAHE for more info.
        normalize (bool):
            Whether to normalize the output image.
        
    Returns:
        im_out (np.ndarray):
            Output image
    """
    im_tu = (im / im.max())*(2**16) if normalize else im
    im_tu = im_tu/10
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid_size, grid_size))
    im_c = clahe.apply(im_tu.astype(np.uint16))
    return im_c


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
    if im_out.shape[2] != 3:
        appended_images = np.stack([images[0]*0] * (3 - len(images)), axis=2)
        im_out = np.concatenate([im_out, appended_images], axis=2)

    return im_out


def change_hsv(image, hsv_gain=[1,1,1], hsv_offset=[0,0,0], in_place=False, return_float=False):
    """
    Change the hue, saturation, and value of an rgb image.
    Note: Gain is applied first, then offset.
    Note: Intermediate image variables are np.uint8.
    RH 2022
    
    Args:
        image (np.ndarray):
            Input image (RGB). Shape: (H, W, 3)
            If image is of type float, it is assumed
             to be in the range [0, 1].
        hsv_gain (list of float):
            Gain for hue, saturation, and value.
        hsv_offset (list of float):
            Offset for hue, saturation, and value.
        in_place (bool):
            Whether to change the image in place or not.
            If False, a copy of the image is returned.
        return_float (bool):
            Whether to return the image as a float or not.
            If True, the image is returned as a float in
             the range [0, 1].
            If False, the image is returned as a uint8 in
             range [0, 255].
            
    Returns:
        im_out (np.ndarray):
            Output image
    """
    
    out = image if in_place else copy.copy(image)
    out = out*255 if out.dtype.kind == 'f' else out
    
    out = (out[...,(2,1,0)]).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    out = out.astype(np.float64)
    out *= np.array(hsv_gain)[None,None,:]
    out += np.array(hsv_offset)[None,None,:]
    out[out>255] = 255
    out = out.astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)[...,(2,1,0)]

    return out if return_float==False else out.astype(np.float32)/255


def bin_array(array, bin_widths=[2,3,4], method='append', function=np.nanmean, function_kwargs={}):
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
            'pre_crop' crops the array to be divisible
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

    arr_out = copy.copy(array)
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


def flatten_channels_along_frames(images):
    """
    Concatenates channels along frame dimension.
    Opposite of reravel_channels.
    RH 2022

    Args:
        images: 
            Input images or video.
            shape(frames, height, width, channels)
    
    Returns:
        output images:
            Reshaped images.
            shape(frames*channels, height, width)
    """
    return images.transpose(1,2,0,3).reshape(images.shape[1], images.shape[2], images.shape[0]*images.shape[3]).transpose(2,0,1)

def reravel_channels(images, n_channels=3):
    """
    De-concatenates channels from the frame dimension to a new dimension.
    Opposite of flatten_channels_along_frames.
    RH 2022
    
    Args:
        images:
            Input images or video. Channels should be concatenated in an
             interleaved fashion along the frame dimension.
            shape(frames*channels, height, width)
        n_channels:
            Number of channels.

    Returns:
        output images:
            Reshaped images.
            shape(frames, height, width, channels)
    """
    return images.transpose(1,2,0).reshape(images.shape[1], images.shape[2], images.shape[0]//n_channels, n_channels).transpose(2,0,1,3)


def center_crop_images(images, height_width=None):
    """
    Crops images around the center.
    RH 2022

    Args:
        images:
            Input images or video.
            shape(frames, height, width, channels)
        height_width:
            2-tuple or list of height and width to crop to.
            If None, then set to the minimum of the height or width
             of the images (making the images square).

    Returns:
        output images:
            Cropped images.
    """
    if height_width is None:
        minimum_shape = min(images.shape[1], images.shape[2])
        height = minimum_shape
        width  = minimum_shape
    else:
        height = height_width[0]
        width  = height_width[1]
        
    center_idx = [images.shape[1]//2, images.shape[2]//2]
    height_half = height//2
    width_half  = width//2
    
    return images[:, center_idx[0] - height_half : center_idx[0] + height_half, center_idx[1] - width_half : center_idx[1] + width_half, :]

def center_pad_images(images, height_width=None):
    """
    Pads images around the center.
    RH 2022

    Args:
        images:
            Input images or video.
            shape(frames, height, width, channels)
        height_width:
            2-tuple or list of height and width to pad to.
            If None, then set to the maximum of the height or width
             of the images (making the images square).

    Returns:
        output images:
            Padded images.
    """
    if height_width is None:
        maximum_shape = max(images.shape[1], images.shape[2])
        height = maximum_shape
        width = maximum_shape
    else:
        height = height_width[0]
        width = height_width[1]
        
    pad_height = (height - images.shape[1]) // 2
    pad_width  = (width - images.shape[2]) // 2
    images_out = np.zeros((images.shape[0], height, width, images.shape[3]))

    images_out[:, pad_height : pad_height + images.shape[1], pad_width : pad_width + images.shape[2], :] = images
    
    return images_out


def add_text_to_images(images, text, position=(10,10), font_size=1, color=(255,255,255), line_width=1, font=None, show=False, frameRate=30):
    """
    Add text to images using cv2.putText()
    RH 2022

    Args:
        images (np.array):
            frames of video or images.
            shape: (n_frames, height, width, n_channels)
        text (list of lists):
            text to add to images.
            Outer list: one element per frame.
            Inner list: each element is a line of text.
        position (tuple):
            (x,y) position of text (top left corner)
        font_size (int):
            font size of text
        color (tuple):
            (r,g,b) color of text
        line_width (int):
            line width of text
        font (str):
            font to use.
            If None, then will use cv2.FONT_HERSHEY_SIMPLEX
            See cv2.FONT... for more options
        show (bool):
            if True, then will show the images with text added.

    Returns:
        images_with_text (np.array):
            frames of video or images with text added.
    """
    import cv2
    import copy
    
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    
    images_cp = copy.copy(images)
    for i_f, frame in enumerate(images_cp):
        for i_t, t in enumerate(text[i_f]):
            cv2.putText(frame, t, [position[0] , position[1] + i_t*font_size*30], font, font_size, color, line_width)
        if show:
            cv2.imshow('add_text_to_images', frame)
            cv2.waitKey(int(1000/frameRate))
    
    if show:
        cv2.destroyWindow('add_text_to_images')
    return images_cp


def add_image_overlay(
    images_overlay,
    images_underlay,
    position_topLeft=(0,0),
    height_width=(100,100),
    interpolation=None,
):
    """
    Add an image/video overlay to an image/video underlay.
    RH 2022

    Args:
        images_overlay (np.array):
            frames of video or images.
            shape: (n_frames, height, width, n_channels)
        images_underlay (np.array):
            frames of video or images.
            shape: (n_frames, height, width, n_channels)
        position_topLeft (tuple):
            (y,x) position of overlay (top left corner of overlay)
        height_width (tuple):
            (height, width) of overlay
            Images will be resized to this size.
        interpolation (str):
            interpolation method to use.
            
    Returns:
        images_with_overlay (np.array):
            frames of video or images with overlay added.
    """
    
    if interpolation is None:
        interpolation = torchvision.transforms.InterpolationMode.BICUBIC
        
    def resize_torch(images, new_shape=[100,100], interpolation=interpolation):
        resize = torchvision.transforms.Resize(new_shape, interpolation=interpolation, max_size=None, antialias=None)
        return resize(torch.as_tensor(images)).numpy()

    im_hw = images_underlay[0].shape

    pos_all = np.array([position_topLeft[0], position_topLeft[0]+height_width[0], position_topLeft[1], position_topLeft[1]+height_width[1]])
    pos_all[pos_all < 0] = 0
    pos_all[1] = min(im_hw[0], pos_all[1])
    pos_all[3] = min(im_hw[1], pos_all[3])

    rs_size = [pos_all[1]-pos_all[0], pos_all[3]-pos_all[2]]

    images_rs = resize_torch(images_overlay.transpose(0,3,1,2), new_shape=rs_size, interpolation=interpolation).transpose(0,2,3,1)

    images_out = copy.copy(images_underlay)
    images_out[:, pos_all[0]:pos_all[1], pos_all[2]:pos_all[3],:] = images_rs
    
    return images_out


def apply_shifts_along_axis(
    images, 
    xShifts=[0], 
    yShifts=[0], 
    workers=-1,
    prog_bar=True
):
    """
    Apply shifts along a given axis.
    Useful for applying motion correction shifts to a video.
    RH 2022

    Args:
        images (np.ndarray):
            Sequence of images or video.
            shape: (n_frames, height, width, n_channels)
             or (n_frames, height, width)
        xShifts (np.ndarray):
            Shifts to apply along x axis.
            shape: (n_frames,)
        yShifts (np.ndarray):
            Shifts to apply along y axis.
            shape: (n_frames,)
        prog_bar (bool):
            if True, then will show a progress bar.

    Returns:
        images_shifted (np.ndarray):
            Shifted images.
            shape: (n_frames, height, width, n_channels)
             or (n_frames, height, width)
    """
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp

    def apply_shifts_frame(frame, xShift, yShift):
        M = np.float32([[1,0,xShift],[0,1,yShift]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    workers = mp.cpu_count() if workers == -1 else workers
    n_args = len(images)

    if workers == 1:
        return np.stack([apply_shifts_frame(*args) for args in zip(images, xShifts, yShifts)], axis=0)

    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(tqdm(ex.map(apply_shifts_frame, *(images, xShifts, yShifts)), total=n_args, disable=prog_bar!=True)), axis=0)