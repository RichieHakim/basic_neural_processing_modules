from functools import partial
import typing
from typing import Union, List, Any, Tuple, Optional, Dict, Callable, Iterable, Sequence

import numpy as np
import cv2

import copy
import torch
import torchvision
from tqdm.notebook import tqdm
import scipy.interpolate
import scipy.sparse

from . import indexing, featurization, parallel_helpers, spectral


def find_geometric_transformation(
    im_template, 
    im_moving,
    warp_mode='euclidean',
    n_iter=5000,
    termination_eps=1e-10,
    mask=None,
    gaussFiltSize=1
):
    """
    Find the transformation between two images.
    Wrapper function for cv2.findTransformECC
    RH 2022

    Args:
        im_template (np.ndarray):
            Template image
            dtype must be: np.uint8 or np.float32
        im_moving (np.ndarray):
            Moving image
            dtype must be: np.uint8 or np.float32
        warp_mode (str):
            warp mode.
            See cv2.findTransformECC for more info.
            'translation': sets a translational motion model; warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated.
            'euclidean':   sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is 2x3.
            'affine':      sets an affine motion model (DEFAULT); six parameters are estimated; warpMatrix is 2x3.
            'homography':  sets a homography as a motion model; eight parameters are estimated;`warpMatrix` is 3x3.
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
    LUT_modes = {
        'translation': cv2.MOTION_TRANSLATION,
        'euclidean': cv2.MOTION_EUCLIDEAN,
        'affine': cv2.MOTION_AFFINE,
        'homography': cv2.MOTION_HOMOGRAPHY,
    }
    assert warp_mode in LUT_modes.keys(), f"warp_mode must be one of {LUT_modes.keys()}. Got {warp_mode}"
    warp_mode = LUT_modes[warp_mode]
    if warp_mode in [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE]:
        shape_eye = (2, 3)
    elif warp_mode == cv2.MOTION_HOMOGRAPHY:
        shape_eye = (3, 3)
    else:
        raise ValueError(f"warp_mode {warp_mode} not recognized (should not happen)")
    warp_matrix = np.eye(*shape_eye, dtype=np.float32)

    ## assert that the inputs are numpy arrays of dtype np.uint8
    assert isinstance(im_template, np.ndarray) and (im_template.dtype == np.uint8 or im_template.dtype == np.float32), f"im_template must be a numpy array of dtype np.uint8 or np.float32. Got {type(im_template)} of dtype {im_template.dtype}"
    assert isinstance(im_moving, np.ndarray) and (im_moving.dtype == np.uint8 or im_moving.dtype == np.float32), f"im_moving must be a numpy array of dtype np.uint8 or np.float32. Got {type(im_moving)} of dtype {im_moving.dtype}"
    ## cast mask to bool then to uint8
    if mask is not None:
        assert isinstance(mask, np.ndarray), f"mask must be a numpy array. Got {type(mask)}"
        if np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.uint8):
            pass
        else:
            mask = (mask != 0).astype(np.uint8)
    
    ## make gaussFiltSize odd
    gaussFiltSize = int(np.ceil(gaussFiltSize))
    gaussFiltSize = gaussFiltSize + (gaussFiltSize % 2 == 0)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iter,
        termination_eps,
    )
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
    Wrapper function for cv2.warpAffine and cv2.warpPerspective
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
            dst=copy.deepcopy(im_in),
            flags=interpolation_method + cv2.WARP_INVERSE_MAP,
            borderMode=borderMode,
            borderValue=borderValue
        )
        
    elif warp_matrix.shape == (3, 3):
        im_out = cv2.warpPerspective(
            src=im_in,
            M=warp_matrix,
            dsize=(im_in.shape[1], im_in.shape[0]), 
            dst=copy.deepcopy(im_in), 
            flags=interpolation_method + cv2.WARP_INVERSE_MAP, 
            borderMode=borderMode, 
            borderValue=borderValue
        )

    else:
        raise ValueError(f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3) or (3, 3)")
    
    return im_out


def warp_matrix_to_remappingIdx(
    warp_matrix, 
    x, 
    y, 
):
    """
    Convert an warp matrix (2x3 or 3x3) into remapping indices (2D).
    RH 2023
    
    Args:
        warp_matrix (np.ndarray or torch.Tensor): 
            Warp matrix of shape (2, 3) [affine] or (3, 3) [homography].
        x (int): 
            Width of the desired remapping indices.
        y (int): 
            Height of the desired remapping indices.
        
    Returns:
        remapIdx (np.ndarray or torch.Tensor): 
            Remapping indices of shape (x, y, 2) representing the x and y displacements
             in pixels.
    """
    assert warp_matrix.shape in [(2, 3), (3, 3)], f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3) or (3, 3)"
    assert isinstance(x, int) and isinstance(y, int), f"x and y must be integers"
    assert x > 0 and y > 0, f"x and y must be positive"

    if isinstance(warp_matrix, torch.Tensor):
        stack, meshgrid, arange, hstack, ones, float32, array = torch.stack, torch.meshgrid, torch.arange, torch.hstack, torch.ones, torch.float32, torch.as_tensor
        stack_partial = lambda x: stack(x, dim=0)
    elif isinstance(warp_matrix, np.ndarray):
        stack, meshgrid, arange, hstack, ones, float32, array = np.stack, np.meshgrid, np.arange, np.hstack, np.ones, np.float32, np.array
        stack_partial = lambda x: stack(x, axis=0)
    else:
        raise ValueError(f"warp_matrix must be a torch.Tensor or np.ndarray")

    # create the grid
    mesh = stack_partial(meshgrid(arange(x, dtype=float32), arange(y, dtype=float32)))
    mesh_coords = hstack((mesh.reshape(2,-1).T, ones((x*y, 1), dtype=float32)))
    
    # warp the grid
    mesh_coords_warped = (mesh_coords @ warp_matrix.T)
    mesh_coords_warped = mesh_coords_warped[:, :2] / mesh_coords_warped[:, 2:3] if warp_matrix.shape == (3, 3) else mesh_coords_warped  ## if homography, divide by z
    
    # reshape the warped grid
    remapIdx = mesh_coords_warped.T.reshape(2, y, x)

    # permute the axes to (x, y, 2)
    remapIdx = remapIdx.permute(1, 2, 0) if isinstance(warp_matrix, torch.Tensor) else remapIdx.transpose(1, 2, 0)

    return remapIdx


def remap_images(
    images,
    remappingIdx,
    backend="torch",
    interpolation_method='linear',
    border_mode='constant',
    border_value=0,
    device='cpu',
):
    """
    Apply remapping indices to a set of images.
    Remapping indices are like flow fields, but instead of describing
     the displacement of each pixel, they describe the index of the pixel
     to sample from.
    RH 2023

    Args:
        images (np.ndarray or torch.Tensor):
            Images to be warped.
            Shape (N, C, H, W) or (C, H, W) or (H, W).
        remappingIdx (np.ndarray or torch.Tensor):
            Remapping indices. Describes the index of the pixel to 
             sample from.
            Shape (H, W, 2).
        backend (str):
            Backend to use. Either "torch" or "cv2".
        interpolation_method (str):
            Interpolation method to use.
            Can be: 'linear', 'nearest', 'cubic', 'lanczos'
            See cv2.remap or torch.nn.functional.grid_sample for details.
        borderMode (str):
            Border mode to use.
            Can be: 'constant', 'reflect', 'replicate', 'wrap'
            See cv2.remap for details.
        borderValue (float):
            Border value to use.
            See cv2.remap for details.

    Returns:
        warped_images (np.ndarray or torch.Tensor):
            Warped images.
            Shape (N, C, H, W) or (C, H, W).
    """
    # Check inputs
    assert isinstance(images, (np.ndarray, torch.Tensor)), f"images must be a np.ndarray or torch.Tensor"
    assert isinstance(remappingIdx, (np.ndarray, torch.Tensor)), f"remappingIdx must be a np.ndarray or torch.Tensor"
    if images.ndim == 2:
        images = images[None, None, :, :]
    elif images.ndim == 3:
        images = images[None, :, :, :]
    elif images.ndim != 4:
        raise ValueError(f"images must be a 2D, 3D, or 4D array. Got shape {images.shape}")
    assert remappingIdx.ndim == 3, f"remappingIdx must be a 3D array of shape (H, W, 2). Got shape {remappingIdx.shape}"

    # Check backend
    if backend not in ["torch", "cv2"]:
        raise ValueError("Invalid backend. Supported backends are 'torch' and 'cv2'.")
    if backend == 'torch':
        if isinstance(images, np.ndarray):
            images = torch.as_tensor(images, device=device, dtype=torch.float32)
        elif isinstance(images, torch.Tensor):
            images = images.to(device=device).type(torch.float32)
        if isinstance(remappingIdx, np.ndarray):
            remappingIdx = torch.as_tensor(remappingIdx, device=device, dtype=torch.float32)
        elif isinstance(remappingIdx, torch.Tensor):
            remappingIdx = remappingIdx.to(device=device).type(torch.float32)
        interpolation = {
            'linear': 'bilinear',
            'nearest': 'nearest',
            'cubic': 'bicubic',
            'lanczos': 'lanczos',
        }[interpolation_method]
        border = {
            'constant': 'zeros',
            'reflect': 'reflection',
            'replicate': 'replication',
            'wrap': 'circular',
        }[border_mode]
        ## Convert remappingIdx to normalized grid
        normgrid = cv2RemappingIdx_to_pytorchFlowField(remappingIdx)

        # Apply remappingIdx
        warped_images = torch.nn.functional.grid_sample(
            images, 
            normgrid[None,...],
            mode=interpolation, 
            padding_mode=border, 
            align_corners=True,  ## align_corners=True is the default in cv2.remap. See documentation for details.
        )

    elif backend == 'cv2':
        assert isinstance(images, np.ndarray), f"images must be a np.ndarray when using backend='cv2'"
        assert isinstance(remappingIdx, np.ndarray), f"remappingIdx must be a np.ndarray when using backend='cv2'"
        ## convert to float32 if not uint8
        images = images.astype(np.float32) if images.dtype != np.uint8 else images
        remappingIdx = remappingIdx.astype(np.float32) if remappingIdx.dtype != np.uint8 else remappingIdx

        interpolation = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }[interpolation_method]
        borderMode = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE,
            'wrap': cv2.BORDER_WRAP,
        }[border_mode]

        # Apply remappingIdx
        def remap(ims):
            out = np.stack([cv2.remap(
                im,
                remappingIdx[..., 0], 
                remappingIdx[..., 1], 
                interpolation=interpolation, 
                borderMode=borderMode, 
                borderValue=border_value,
            ) for im in ims], axis=0)
            return out
        warped_images = np.stack([remap(im) for im in images], axis=0)

    return warped_images.squeeze()


def remap_sparse_images(
    ims_sparse: Union[scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]],
    remappingIdx: np.ndarray,
    method: str = 'linear',
    fill_value: float = 0,
    dtype: Union[str, np.dtype] = None,
    safe: bool = True,
    n_workers: int = -1,
    verbose: bool = True,
) -> List[scipy.sparse.csr_matrix]:
    """
    Remaps a list of sparse images using the given remap field.
    RH 2023

    Args:
        ims_sparse (Union[scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]]): 
            A single sparse image or a list of sparse images.
        remappingIdx (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field. It
            should be the same size as the images in ims_sparse.
        method (str): 
            Interpolation method to use. See ``scipy.interpolate.griddata``.
            Options are:
            \n
            * ``'linear'``
            * ``'nearest'``
            * ``'cubic'`` \n
            (Default is ``'linear'``)
        fill_value (float): 
            Value used to fill points outside the convex hull. (Default is
            ``0.0``)
        dtype (Union[str, np.dtype]): 
            The data type of the resulting sparse images. Default is ``None``,
            which will use the data type of the input sparse images.
        safe (bool): 
            If ``True``, checks if the image is 0D or 1D and applies a tiny
            Gaussian blur to increase the image width. (Default is ``True``)
        n_workers (int): 
            Number of parallel workers to use. Default is *-1*, which uses all
            available CPU cores.
        verbose (bool):
            Whether or not to use a tqdm progress bar. (Default is ``True``)

    Returns:
        (List[scipy.sparse.csr_matrix]): 
            ims_sparse_out (List[scipy.sparse.csr_matrix]): 
                A list of remapped sparse images.

    Raises:
        AssertionError: If the image and remappingIdx have different spatial
        dimensions.
    """
    # Ensure ims_sparse is a list of sparse matrices
    ims_sparse = [ims_sparse] if not isinstance(ims_sparse, list) else ims_sparse

    # Assert that all images are sparse matrices
    assert all(scipy.sparse.issparse(im) for im in ims_sparse), "All images must be sparse matrices."
    
    # Assert and retrieve dimensions
    dims_ims = ims_sparse[0].shape
    dims_remap = remappingIdx.shape
    assert dims_ims == dims_remap[:-1], "Image and remappingIdx should have same spatial dimensions."
    
    dtype = ims_sparse[0].dtype if dtype is None else dtype
    
    if safe:
        conv2d = featurization.Toeplitz_convolution2d(
            x_shape=(dims_ims[0], dims_ims[1]),
            k=np.array([[0   , 1e-8, 0   ],
                        [1e-8, 1,    1e-8],
                        [0   , 1e-8, 0   ]], dtype=dtype),
            dtype=dtype,
        )

    def warp_sparse_image(
        im_sparse: scipy.sparse.csr_matrix,
        remappingIdx: np.ndarray,
        method: str = method,
        fill_value: float = fill_value,
        safe: bool = safe
    ) -> scipy.sparse.csr_matrix:
        
        # Convert sparse image to COO format
        im_coo = scipy.sparse.coo_matrix(im_sparse)

        # Get coordinates and values from COO format
        rows, cols = im_coo.row, im_coo.col
        data = im_coo.data

        if safe:
            # can't use scipy.interpolate.griddata with 1d values
            is_horz = np.unique(rows).size == 1
            is_vert = np.unique(cols).size == 1

            # check for diagonal pixels 
            # slope = rise / run --- don't need to check if run==0 
            rdiff = np.diff(rows)
            cdiff = np.diff(cols)
            is_diag = np.unique(cdiff / rdiff).size == 1 if not np.any(rdiff==0) else False
            
            # best practice to just convolve instead of interpolating if too few pixels
            is_smol = rows.size < 3 

            if is_horz or is_vert or is_smol or is_diag:
                # warp convolved sparse image directly without interpolation
                return warp_sparse_image(im_sparse=conv2d(im_sparse, batching=False), remappingIdx=remappingIdx)

        # Get values at the grid points
        try:
            grid_values = scipy.interpolate.griddata(
                points=(rows, cols), 
                values=data, 
                xi=remappingIdx[:,:,::-1], 
                method=method, 
                fill_value=fill_value,
            )
        except Exception as e:
            raise Exception(f"Error interpolating sparse image. Something is either weird about one of the input images or the remappingIdx. Error: {e}")
        
        # Create a new sparse image from the nonzero pixels
        warped_sparse_image = scipy.sparse.csr_matrix(grid_values, dtype=dtype)
        warped_sparse_image.eliminate_zeros()
        return warped_sparse_image
    
    wsi_partial = partial(warp_sparse_image, remappingIdx=remappingIdx)
    ims_sparse_out = parallel_helpers.map_parallel(func=wsi_partial, args=[ims_sparse,], method='multithreading', n_workers=n_workers, prog_bar=verbose)
    return ims_sparse_out


def remap_points(
    points: np.ndarray, 
    remappingIdx: np.ndarray,
    interpolation: str = 'linear',
    fill_value: float = None,
) -> np.ndarray:
    """
    Remaps a set of points using an index map.

    Args:
        points (np.ndarray): 
            Array of points to be remapped. It should be a 2D array with the
            shape *(n_points, 2)*, where each point is represented by a pair of
            floating point coordinates within the image.
        remappingIdx (np.ndarray): 
            Index map for the remapping. It should be a 3D array with the shape
            *(height, width, 2)*. The data type should be a floating point
            subtype.
        interpolation (str):
            Interpolation method to use.
            See scipy.interpolate.RegularGridInterpolator. Can be:
                * ``'linear'``
                * ``'nearest'``
                * ``'slinear'``
                * ``'cubic'``
                * ``'quintic'``
                * ``'pchip'``
        fill_value (float, optional):
            Value used to fill points outside the convex hull. If ``None``, values
            outside the convex hull are extrapolated.

    Returns:
        (np.ndarray): 
            points_remap (np.ndarray): 
                Remapped points array. It has the same shape as the input.
    """
    ### Assert points is a 2D numpy.ndarray of shape (n_points, 2) and that all points are within the image and that points are float
    assert isinstance(points, np.ndarray), 'points must be a numpy.ndarray'
    assert points.ndim == 2, 'points must be a 2D numpy.ndarray'
    assert points.shape[1] == 2, 'points must be of shape (n_points, 2)'
    assert np.issubdtype(points.dtype, np.floating), 'points must be a float subtype'

    assert isinstance(remappingIdx, np.ndarray), 'remappingIdx must be a numpy.ndarray'
    assert remappingIdx.ndim == 3, 'remappingIdx must be a 3D numpy.ndarray'
    assert remappingIdx.shape[2] == 2, 'remappingIdx must be of shape (height, width, 2)'
    assert np.issubdtype(remappingIdx.dtype, np.floating), 'remappingIdx must be a float subtype'

    ## Make grid of indices for image remapping
    dims = remappingIdx.shape
    x_arange, y_arange = np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32)

    ## Use RegularGridInterpolator to remap points
    warper = scipy.interpolate.RegularGridInterpolator(
        points=(y_arange, x_arange),
        values=remappingIdx,
        method=interpolation,
        bounds_error=False,
        fill_value=fill_value,
    )
    points_remap = warper(xi=(points[:, 1], points[:, 0]))

    return points_remap


def invert_remappingIdx(
    remappingIdx: np.ndarray, 
    method: str = 'linear', 
    fill_value: typing.Optional[float] = np.nan
) -> np.ndarray:
    """
    Inverts a remapping index field.
    Requires assumption that the remapping index field is:
    - invertible or bijective / one-to-one.
    - non oc
    Example:
        Define 'remap_AB' as a remapping index field that warps
         image A onto image B. Then, 'remap_BA' is the remapping
         index field that warps image B onto image A. This function
         computes 'remap_BA' given 'remap_AB'.
        
    RH 2023

    Args:
        remappingIdx (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field.
        method (str):
            Interpolation method to use.
            See scipy.interpolate.griddata.
            Options are 'linear', 'nearest', 'cubic'.
        fill_value (float, optional):
            Value used to fill points outside the convex hull.

    Returns:
        remappingIdx_inv (np.ndarray): 
            An array of shape (H, W, 2) representing the inverse remap field.
    """
    H, W, _ = remappingIdx.shape
    
    # Create the meshgrid of the original image
    grid = np.mgrid[:H, :W][::-1].transpose(1,2,0).reshape(-1, 2)
    
    # Flatten the original meshgrid and remappingIdx
    remapIdx_flat = remappingIdx.reshape(-1, 2)
    
    # Interpolate the inverse mapping using griddata
    map_BA = scipy.interpolate.griddata(
        points=remapIdx_flat, 
        values=grid, 
        xi=grid, 
        method=method,
        fill_value=fill_value,
    ).reshape(H,W,2)
    
    return map_BA

def invert_warp_matrix(warp_matrix: np.ndarray) -> np.ndarray:
    """
    Invert a given warp matrix (2x3 or 3x3) for A->B to compute the warp matrix for B->A.
    RH 2023

    Args:
        warp_matrix (numpy.ndarray): 
            A 2x3 or 3x3 numpy array representing the warp matrix.

    Returns:
        numpy.ndarray: 
            The inverted warp matrix.
    """
    if warp_matrix.shape == (2, 3):
        # Convert 2x3 affine warp matrix to 3x3 by appending [0, 0, 1] as the last row
        warp_matrix_3x3 = np.vstack((warp_matrix, np.array([0, 0, 1])))
    elif warp_matrix.shape == (3, 3):
        warp_matrix_3x3 = warp_matrix
    else:
        raise ValueError("Input warp_matrix must be of shape (2, 3) or (3, 3)")

    # Compute the inverse of the 3x3 warp matrix
    inverted_warp_matrix_3x3 = np.linalg.inv(warp_matrix_3x3)

    if warp_matrix.shape == (2, 3):
        # Convert the inverted 3x3 warp matrix back to 2x3 by removing the last row
        inverted_warp_matrix = inverted_warp_matrix_3x3[:2, :]
    else:
        inverted_warp_matrix = inverted_warp_matrix_3x3

    return inverted_warp_matrix


def compose_remappingIdx(
    remap_AB: np.ndarray,
    remap_BC: np.ndarray,
    method: str = 'linear',
    fill_value: typing.Optional[float] = np.nan,
    bounds_error: bool = False,
) -> np.ndarray:
    """
    Composes two remapping index fields using scipy.interpolate.interpn.
    Example:
        Define 'remap_AB' as a remapping index field that warps
        image A onto image B. Define 'remap_BC' as a remapping
        index field that warps image B onto image C. This function
        computes 'remap_AC' given 'remap_AB' and 'remap_BC'.
    RH 2023

    Args:
        remap_AB (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field.
        remap_BC (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field.
        method (str, optional): 
            The interpolation method to use, default is 'linear'.
        fill_value (float, optional): 
            The value to use for points outside the interpolation domain,
             default is np.nan.
        bounds_error (bool, optional):
            If True, when interpolated values are requested outside of
             the domain of the input data, a ValueError is raised.
             
    """
    # Get the shape of the remap fields
    H, W, _ = remap_AB.shape
    
    # Combine the x and y components of remap_AB into a complex number
    # This is done to simplify the interpolation process
    AB_complex = remap_AB[:,:,0] + remap_AB[:,:,1]*1j

    # Perform the interpolation using interpn
    AC = scipy.interpolate.interpn(
        (np.arange(H), np.arange(W)), 
        AB_complex, 
        remap_BC.reshape(-1, 2)[:, ::-1], 
        method=method, 
        bounds_error=bounds_error, 
        fill_value=fill_value
    ).reshape(H, W)

    # Split the real and imaginary parts of the interpolated result to get the x and y components
    remap_AC = np.stack((AC.real, AC.imag), axis=-1)

    return remap_AC


def compose_transform_matrices(
    matrix_AB: np.ndarray, 
    matrix_BC: np.ndarray
) -> np.ndarray:
    """
    Composes two transformation matrices.
    Example:
        Define 'matrix_AB' as a transformation matrix that warps
        image A onto image B. Define 'matrix_BC' as a transformation
        matrix that warps image B onto image C. This function
        computes 'matrix_AC' given 'matrix_AB' and 'matrix_BC'.
    RH 2023

    Args:
        matrix_AB (np.ndarray): 
            An array of shape (2, 3) or (3, 3) representing the transformation matrix.
        matrix_BC (np.ndarray): 
            An array of shape (2, 3) or (3, 3) representing the transformation matrix.

    Returns:
        matrix_AC (np.ndarray): 
            An array of shape (2, 3) or (3, 3) representing the composed transformation matrix.

    Raises:
        AssertionError: If the input matrices are not of shape (2, 3) or (3, 3).
    """
    assert matrix_AB.shape in [(2, 3), (3, 3)], "Matrix AB must be of shape (2, 3) or (3, 3)."
    assert matrix_BC.shape in [(2, 3), (3, 3)], "Matrix BC must be of shape (2, 3) or (3, 3)."

    # If the input matrices are (2, 3), extend them to (3, 3) by adding a row [0, 0, 1]
    if matrix_AB.shape == (2, 3):
        matrix_AB = np.vstack((matrix_AB, [0, 0, 1]))
    if matrix_BC.shape == (2, 3):
        matrix_BC = np.vstack((matrix_BC, [0, 0, 1]))

    # Compute the product of the extended matrices
    matrix_AC = matrix_AB @ matrix_BC

    # If the resulting matrix is (3, 3) and has the last row [0, 0, 1], convert it back to a (2, 3) matrix
    if (matrix_AC.shape == (3, 3)) and np.allclose(matrix_AC[2], [0, 0, 1]):
        matrix_AC = matrix_AC[:2, :]

    return matrix_AC


def shifts_to_remappingIdx(
    shifts, 
    im_shape=(512, 512), 
    edge_method='clip', 
    edge_value=0
):
    """
    Convert a set of shifts to remapping indices.
    RH 2023

    Args:
        shifts (np.ndarray or torch.Tensor):
            Shifts (displacements) of all pixels in the image.
            Shape: (N, 2) or (2,). Last dimension is (x, y).
        im_shape (tuple):
            Shape of the image. Shape: (H, W).
            Used to create the grid of indices.
        edge_method (str):
            Method to use for handling edges.
            Can be:
                - 'none': Allow for indices outside the image.
                - 'clip': Clip the indices to the min and max values of the
                  image size.
                - 'constant': Use a constant value defined by `edge_value`.
                - 'reflect': Reflect the image at the edges.
                - 'wrap': Wrap the image around.
        edge_value (float):
            Value to use for the 'constant' edge method.

    Returns:
        ri (np.ndarray or torch.Tensor):
            Remapping indices.
            Describes the index of the pixel in the original
            image that should be mapped to the new pixel.
            Shape: (N, H, W, 2). Last dimension is (x, y).
    """
    ## get functions
    if isinstance(shifts, np.ndarray):
        stack, meshgrid, arange, zeros, clip, full, abs, array, int64, mod = np.stack, np.meshgrid, np.arange, np.zeros, np.clip, np.full, np.abs, np.array, np.int64, np.mod
    elif isinstance(shifts, torch.Tensor):
        stack, meshgrid, arange, zeros, clip, full, abs, array, int64, mod = torch.stack, torch.meshgrid, torch.arange, torch.zeros, torch.clamp, torch.full, torch.abs, torch.as_tensor, torch.int64, torch.remainder

    ## check inputs
    ### check shifts
    assert isinstance(shifts, (np.ndarray, torch.Tensor)), f"shifts must be a np.ndarray or torch.Tensor. Got {type(shifts)}"
    assert shifts.ndim in [1, 2], f"shifts must be a 1D or 2D array. Got {shifts.ndim}D"
    if shifts.ndim == 1:
        shifts = shifts[None, :]
    assert shifts.shape[-1] == 2, f"shifts must have shape (N, 2) or (2,). Got shape {shifts.shape}"
    ### check im_shape
    assert isinstance(im_shape, (tuple, list, np.ndarray, torch.Tensor)), f"im_shape must be a tuple, list, np.ndarray, or torch.Tensor. Got {type(im_shape)}"
    if isinstance(im_shape, (tuple, list)):
        im_shape = array(im_shape, dtype=int64)
    assert im_shape.ndim == 1, f"im_shape must be 1D. Got {im_shape.ndim}D"
    assert im_shape.shape[0] == 2, f"im_shape must have length 2. Got length {im_shape.shape[0]}"

    # Create a grid of indices
    grid = stack(meshgrid(arange(im_shape[1]), arange(im_shape[0]), indexing='xy'), axis=-1)  ## (H, W, 2). Last dimension is (x, y).
    
    # Apply shifts
    ri = grid - shifts[:, None, None, :]  ## (N, H, W, 2). Last dimension is (x, y).

    # Handle edges
    if edge_method == 'none':
        pass
    elif edge_method == 'clip':
        ri = clip(ri, array(0), im_shape - 1)
    elif edge_method == 'constant':
        mask = (ri < 0) | (ri >= im_shape)
        ri = clip(ri, array(0), im_shape - 1)
        ri[mask] = edge_value
    elif edge_method == 'reflect':
        # Reflect mode: Index wraps around to the other side of the image
        reflect = lambda x, max_val: abs((x + max_val) % (2 * max_val) - max_val)
        ri = reflect(ri, im_shape - 1)
    elif edge_method == 'wrap':
        # Wrap around
        for dim in [0, 1]:  # x and y dimensions
            ri[..., dim] = mod(ri[..., dim], im_shape[dim])
    else:
        raise ValueError(f"Invalid edge method: {edge_method}")

    return ri

def make_idx_grid(im):
    """
    Helper function to make a grid of indices for an image.
    Used in flowField_to_remappingIdx and remappingIdx_to_flowField.
    """
    if isinstance(im, torch.Tensor):
        stack, meshgrid, arange = partial(torch.stack, dim=-1), partial(torch.meshgrid, indexing='xy'), partial(torch.arange, device=im.device, dtype=im.dtype)
    elif isinstance(im, np.ndarray):
        stack, meshgrid, arange = partial(np.stack, axis=-1), partial(np.meshgrid, indexing='xy'), partial(np.arange, dtype=im.dtype)
    return stack(meshgrid(arange(im.shape[1]), arange(im.shape[0]))) # (H, W, 2). Last dimension is (x, y).
def flowField_to_remappingIdx(ff):
    """
    Convert a flow field to a remapping index.
    WARNING: Technically, it is not possible to convert a flow field
     to a remapping index, since the remapping index describes an
     interpolation mapping, while the flow field describes a displacement.
    RH 2023

    Args:
        ff (np.ndarray or torch.Tensor): 
            Flow field.
            Describes the displacement of each pixel.
            Shape (H, W, 2). Last dimension is (x, y).

    Returns:
        ri (np.ndarray or torch.Tensor):
            Remapping index.
            Describes the index of the pixel in the original
             image that should be mapped to the new pixel.
            Shape (H, W, 2)
    """
    ri = ff + make_idx_grid(ff)
    return ri
def remappingIdx_to_flowField(ri):
    """
    Convert a remapping index to a flow field.
    WARNING: Technically, it is not possible to convert a flow field
     to a remapping index, since the remapping index describes an
     interpolation mapping, while the flow field describes a displacement.
    RH 2023

    Args:
        ri (np.ndarray or torch.Tensor):
            Remapping index.
            Describes the index of the pixel in the original
             image that should be mapped to the new pixel.
            Shape (H, W, 2). Last dimension is (x, y).

    Returns:
        ff (np.ndarray or torch.Tensor):
            Flow field.
            Describes the displacement of each pixel.
            Shape (H, W, 2)
    """
    ff = ri - make_idx_grid(ri)
    return ff
def cv2RemappingIdx_to_pytorchFlowField(ri):
    """
    Convert remapping indices from the OpenCV format to the PyTorch format.
    cv2 format: Displacement is in pixels relative to the top left pixel
     of the image.
    PyTorch format: Displacement is in pixels relative to the center of
     the image.
    RH 2023

    Args:
        ri (np.ndarray or torch.Tensor): 
            Remapping indices.
            Each pixel describes the index of the pixel in the original
             image that should be mapped to the new pixel.
            Shape (H, W, 2). Last dimension is (x, y).

    Returns:
        normgrid (np.ndarray or torch.Tensor):
            "Flow field", in the PyTorch format.
            Technically not a flow field, since it doesn't describe
             displacement. Rather, it is a remapping index relative to
             the center of the image.
            Shape (H, W, 2). Last dimension is (x, y).
    """
    assert isinstance(ri, torch.Tensor), f"ri must be a torch.Tensor. Got {type(ri)}"
    im_shape = torch.flipud(torch.as_tensor(ri.shape[:2], dtype=torch.float32, device=ri.device))  ## (W, H)
    normgrid = ((ri / (im_shape[None, None, :] - 1)) - 0.5) * 2  ## PyTorch's grid_sample expects grid values in [-1, 1] because it's a relative offset from the center pixel. CV2's remap expects grid values in [0, 1] because it's an absolute offset from the top-left pixel.
    ## note also that pytorch's grid_sample expects align_corners=True to correspond to cv2's default behavior.
    return normgrid

def pytorchFlowField_to_cv2RemappingIdx(normgrid):
    """
    Convert remapping indices from the PyTorch format to the OpenCV format.
    cv2 format: Displacement is in pixels relative to the top left pixel
     of the image.
    PyTorch format: Displacement is in pixels relative to the center of
     the image.
    RH 2024

    Args:
        normgrid (np.ndarray or torch.Tensor):
            "Flow field", in the PyTorch format.
            Technically not a flow field, since it doesn't describe
             displacement. Rather, it is a remapping index relative to
             the center of the image.
            Shape (H, W, 2). Last dimension is (x, y).

    Returns:
        ri (np.ndarray or torch.Tensor): 
            Remapping indices.
            Each pixel describes the index of the pixel in the original
             image that should be mapped to the new pixel.
            Shape (H, W, 2). Last dimension is (x, y).
    """
    assert isinstance(normgrid, torch.Tensor), f"normgrid must be a torch.Tensor. Got {type(normgrid)}"
    im_shape = torch.flipud(torch.as_tensor(normgrid.shape[:2], dtype=torch.float32, device=normgrid.device))  ## (W, H)
    ri = ((normgrid / 2) + 0.5) * (im_shape[None, None, :] - 1)
    return ri

def resize_remappingIdx(
    ri: Union[np.ndarray, torch.Tensor], 
    new_shape: Tuple[int, int],
    interpolation: str = 'BILINEAR',
) -> Union[np.ndarray, torch.Tensor]:
    """
    Resize a remapping index field. This function both resizes the shape of the
    actual remappingIdx arrays and scales the values to match the new shape.
    RH 2024

    Args:
        ri (np.ndarray or torch.Tensor): 
            Remapping index field(s). Describes the index of the pixel in the
            original image that should be mapped to the new pixel. Shape (H, W,
            2) or (B, H, W, 2). Last dimension is (x, y).
        new_shape (Tuple[int, int]):
            New shape of the remapping index field.
            Shape (H', W').
        interpolation (str): 
            The interpolation method to use. See ``torchvision.transforms.Resize`` 
            for options. \n
                * ``'NEAREST'``: Nearest neighbor interpolation
                * ``'NEAREST_EXACT'``: Nearest neighbor interpolation
                * ``'BILINEAR'``: Bilinear interpolation
                * ``'BICUBIC'``: Bicubic interpolation
        antialias (bool): 
            If ``True``, antialiasing will be used. (Default is ``False``)                

    Returns:
        ri_resized (np.ndarray or torch.Tensor):
            Resized remapping index field.
            Shape (H', W', 2). Last dimension is (x, y).
    """
    assert isinstance(ri, (np.ndarray, torch.Tensor)), f"ri must be a np.ndarray or torch.Tensor. Got {type(ri)}"
    assert ri.ndim in [3, 4], f"ri must have shape (H, W, 2) or (B, H, W, 2). Got shape {ri.shape}"
    assert ri.shape[-1] == 2, f"ri must have shape (H, W, 2). Got shape {ri.shape}"
    assert isinstance(new_shape, (tuple, list, np.ndarray, torch.Tensor)), f"new_shape must be a tuple, list, np.ndarray, or torch.Tensor. Got {type(new_shape)}"
    assert len(new_shape) == 2, f"new_shape must have length 2. Got length {len(new_shape)}"
    
    new_shape = (int(new_shape[0]), int(new_shape[1]))
    
    if ri.ndim == 3:
        ri = ri[None, ...]
        return_3D = True
    else:
        return_3D = False
    hw_ri = ri.shape[1:3]
    
    if isinstance(ri, np.ndarray):
        ri = torch.as_tensor(ri)
        return_numpy = True
    else:
        return_numpy = False
    device = ri.device

    offsets = torch.as_tensor([(new_shape[0] - 1) / (hw_ri[0] - 1), (new_shape[1] - 1) / (hw_ri[1] - 1)], dtype=torch.float32, device=device)[None, None, None, ...]

    ri_resized = resize_images(
        images=ri.permute(3, 0, 1, 2),
        new_shape=new_shape,
        interpolation=interpolation,
    ).permute(1, 2, 3, 0) * offsets

    if return_numpy:
        ri_resized = ri_resized.cpu().numpy()
    if return_3D:
        ri_resized = ri_resized[0]
    return ri_resized


# @torch.jit.script
# def phase_correlation_helper(
#     im_template,
#     im_moving,
#     mask_fft=None, 
#     compute_maskFFT: bool=False, 
#     template_precomputed: bool=False,
#     eps: float=1e-17,
# ):
#     if im_template.ndim == 2:
#         im_template = im_template[None, ...]
#     if im_moving.ndim == 2:
#         im_moving = im_moving[None, ...]
#         return_2D = True
#     else:
#         return_2D = False
#     if compute_maskFFT:
#         mask_fft = mask_fft[None, ...]

#     dims = (-2, -1)
        
#     if compute_maskFFT:
#         mask_fft = torch.fft.fftshift(mask_fft/mask_fft.sum(), dim=dims)
#         fft_template = torch.conj(torch.fft.fft2(im_template, dim=dims) * mask_fft) if not template_precomputed else im_template
#         fft_moving = torch.fft.fft2(im_moving, dim=dims) * mask_fft
#     else:
#         fft_template = torch.conj(torch.fft.fft2(im_template, dim=dims)) if not template_precomputed else im_template
#         fft_moving = torch.fft.fft2(im_moving, dim=dims)

#     R = fft_template[:,None,:,:] * fft_moving[None,:,:,:]
#     R /= torch.abs(R) + eps
    
#     cc = torch.fft.fftshift(torch.fft.ifft2(R, dim=dims), dim=dims).real.squeeze()
    
#     return cc if not return_2D else cc[0]
# def phase_correlation(
#     im_template, 
#     im_moving,
#     mask_fft=None, 
#     template_precomputed=False, 
#     device='cpu'
# ):
#     """
#     Perform phase correlation on two images.
#     Uses pytorch for speed
#     RH 2022
    
#     Args:
#         im_template (np.ndarray or torch.Tensor):
#             Template image(s).
#             If ndim=2, a single image is assumed.
#                 shape: (height, width)
#             if ndim=3, multiple images are assumed, dim=0 is the batch dim.
#                 shape: (batch, height, width)
#                 dim 0 should either be length 1 or the same as im_moving.
#             If template_precomputed is True, this is assumed to be:
#              np.conj(np.fft.fft2(im_template, axis=(1,2)) * mask_fft)
#         im_moving (np.ndarray or torch.Tensor):
#             Moving image(s).
#             If ndim=2, a single image is assumed.
#                 shape: (height, width)
#             if ndim=3, multiple images are assumed, dim=0 is the batch dim.
#                 shape: (batch, height, width)
#                 dim 0 should either be length 1 or the same as im_template.
#         mask_fft (np.ndarray or torch.Tensor):
#             Mask for the FFT.
#             Shape: (height, width)
#             If None, no mask is used.
#         template_precomputed (bool):
#             If True, im_template is assumed to be:
#              np.conj(np.fft.fft2(im_template, axis=(1,2)) * mask_fft)
#         device (str):
#             Device to use.
    
#     Returns:
#         cc (np.ndarray):
#             Phase correlation coefficient.
#             Middle of image is zero-shift.
#             Last two dims are frame height and width.
#     """
#     if isinstance(im_template, np.ndarray):
#         im_template = torch.from_numpy(im_template).to(device)
#         return_numpy = True
#     else:
#         return_numpy = False
#     if isinstance(im_moving, np.ndarray):
#         im_moving = torch.from_numpy(im_moving).to(device)
#     if isinstance(mask_fft, np.ndarray):
#         mask_fft = torch.from_numpy(mask_fft).to(device)
#     if isinstance(mask_fft, torch.Tensor):
#         if mask_fft.device != device:
#             mask_fft = mask_fft.to(device)

#     cc = phase_correlation_helper(
#         im_template=im_template,
#         im_moving=im_moving,
#         mask_fft=mask_fft if mask_fft is not None else torch.as_tensor([1], device=device),
#         compute_maskFFT=(mask_fft is not None),
#         template_precomputed=template_precomputed,
#     )

#     if return_numpy:
#         cc = cc.cpu().numpy()
#     return cc


def phase_correlation(
    im_template: Union[np.ndarray, torch.Tensor],
    im_moving: Union[np.ndarray, torch.Tensor],
    mask_fft: Optional[Union[np.ndarray, torch.Tensor]] = None,
    return_filtered_images: bool = False,
    eps: float = 1e-8,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform phase correlation on two images. Calculation performed along the
    last two axes of the input arrays (-2, -1) corresponding to the (height,
    width) of the images.
    RH 2024

    Args:
        im_template (np.ndarray): 
            The template image(s). Shape: (..., height, width). Can be any
            number of dimensions; last two dimensions must be height and width.
        im_moving (np.ndarray): 
            The moving image. Shape: (..., height, width). Leading dimensions
            must broadcast with the template image.
        mask_fft (Optional[np.ndarray]): 
            2D array mask for the FFT. If ``None``, no mask is used. Assumes mask_fft is
            fftshifted. (Default is ``None``)
        return_filtered_images (bool): 
            If set to ``True``, the function will return filtered images in
            addition to the phase correlation coefficient. (Default is
            ``False``)
        eps (float):
            Epsilon value to prevent division by zero. (Default is ``1e-8``)
    
    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): tuple containing:
            cc (np.ndarray): 
                The phase correlation coefficient.
            fft_template (np.ndarray): 
                The filtered template image. Only returned if
                return_filtered_images is ``True``.
            fft_moving (np.ndarray): 
                The filtered moving image. Only returned if
                return_filtered_images is ``True``.
    """
    fft2, fftshift, ifft2 = torch.fft.fft2, torch.fft.fftshift, torch.fft.ifft2
    abs, conj = torch.abs, torch.conj
    axes = (-2, -1)

    return_numpy = isinstance(im_template, np.ndarray)
    im_template = torch.as_tensor(im_template)
    im_moving = torch.as_tensor(im_moving)

    fft_template = fft2(im_template, dim=axes)
    fft_moving   = fft2(im_moving, dim=axes)

    if mask_fft is not None:
        mask_fft = torch.as_tensor(mask_fft)
        # Normalize and shift the mask
        mask_fft = fftshift(mask_fft / mask_fft.sum(), dim=axes)
        mask = mask_fft[tuple([None] * (im_template.ndim - 2) + [slice(None)] * 2)]
        fft_template *= mask
        fft_moving *= mask

    # Compute the cross-power spectrum
    R = fft_template * conj(fft_moving)

    # Normalize to obtain the phase correlation function
    R /= abs(R) + eps  # Add epsilon to prevent division by zero

    # Compute the magnitude of the inverse FFT to ensure symmetry
    # cc = abs(fftshift(ifft2(R, dim=axes), dim=axes))
    # Compute the real component of the inverse FFT (not symmetric)
    cc = fftshift(ifft2(R, dim=axes), dim=axes).real

    if return_filtered_images == False:
        return cc.cpu().numpy() if return_numpy else cc
    else:
        if return_numpy:
            return (
                cc.cpu().numpy(), 
                abs(ifft2(fft_template, dim=axes)).cpu().numpy(), 
                abs(ifft2(fft_moving, dim=axes)).cpu().numpy()
            )
        else:
            return cc, abs(ifft2(fft_template, dim=axes)), abs(ifft2(fft_moving, dim=axes))


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

def mask_image_border(
    im, 
    border_outer=None, 
    border_inner=None, 
    mask_value=0
):
    """
    Mask an image with a border.
    RH 2022

    Args:
        im (np.ndarray):
            Input image.
        border_outer (int or tuple(int)):
            Outer border width.
            Number of pixels along the border to mask.
            If None, don't mask the border.
            If tuple of ints, then (top, bottom, left, right).
        border_inner (int):
            Inner border width.
            Number of pixels in the center to mask. Will be a square.
            Value is the edge length of the center square.
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

    ## Mask the center
    if border_inner is not None:
        ## make edge_lengths
        center_edge_length = cel = int(np.ceil(border_inner/2)) if border_inner is not None else 0
        im[cy-cel:cy+cel, cx-cel:cx+cel] = mask_value
    ## Mask the border
    if border_outer is not None:
        ## make edge_lengths
        if isinstance(border_outer, int):
            border_outer = (border_outer, border_outer, border_outer, border_outer)
        
        im[:border_outer[0], :] = mask_value
        im[-border_outer[1]:, :] = mask_value
        im[:, :border_outer[2]] = mask_value
        im[:, -border_outer[3]:] = mask_value

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
    
    out = image if in_place else copy.deepcopy(image)
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

    arr_out = copy.deepcopy(array)
    for n, w in enumerate(bin_widths):
        if arr_out.shape[n] % w != 0:
            if method=='append':
                s_pad = copy.deepcopy(s)
                s_pad[n] = w - (arr_out.shape[n] % w)
                arr_out = np.concatenate(
                    [arr_out, np.zeros(s_pad)*np.nan],
                    axis=n
                )
            
            if method=='prepend':
                s_pad = copy.deepcopy(s)
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
    
    images_cp = copy.deepcopy(images)
    for i_f, frame in enumerate(images_cp):
        for i_t, t in enumerate(text[i_f]):
            fn_putText = lambda frame_gray: cv2.putText(frame_gray, t, [position[0] , position[1] + i_t*font_size*30], font, font_size, color, line_width)
            if frame.ndim == 3:
                [fn_putText(frame[:,:,ii]) for ii in range(frame.shape[2])]
            else:
                fn_putText(frame)
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

    images_out = copy.deepcopy(images_underlay)
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
    

def resize_images(
    images: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]], 
    new_shape: Tuple[int, int] = (100,100),
    interpolation: str = 'BILINEAR',
    antialias: bool = False,
    device: Optional[str] = None,
    return_numpy: Optional[bool] = None,
) -> np.ndarray:
    """
    Resizes images using the ``torchvision.transforms.Resize`` method.
    RH 2023

    Args:
        images (Union[np.ndarray, List[np.ndarray]], torch.Tensor, List[torch.Tensor]): 
            Images or frames of a video. Can be 2D, 3D, or 4D. 
            * For a 2D array: shape is *(height, width)*
            * For a 3D array: shape is *(n_frames, height, width)*
            * For a 4D array: shape is *(n_frames, n_channels, height, width)*
        new_shape (Tuple[int, int]): 
            The desired height and width of resized images as a tuple. 
            (Default is *(100, 100)*)
        interpolation (str): 
            The interpolation method to use. See ``torchvision.transforms.Resize`` 
            for options.
            * ``'NEAREST'``: Nearest neighbor interpolation
            * ``'NEAREST_EXACT'``: Nearest neighbor interpolation
            * ``'BILINEAR'``: Bilinear interpolation
            * ``'BICUBIC'``: Bicubic interpolation
        antialias (bool): 
            If ``True``, antialiasing will be used. (Default is ``False``)
        device Optional[str]:
            The device to use for ``torchvision.transforms.Resize``. If None,
            will use the device of the input images. (Default is ``None``)
        return_numpy Optional[bool]:
            If ``True``, then will return a numpy array. Otherwise, will return
            a torch tensor on the defined device. If None, will return a numpy
            array only if the input is a numpy array. (Default is ``None``)
            
    Returns:
        (np.ndarray): 
            images_resized (np.ndarray): 
                Frames of video or images with overlay added.
    """
    ## Convert images to torch tensor
    if isinstance(images, list):
        if isinstance(images[0], np.ndarray):
            device = device if device is not None else 'cpu'
            images = torch.stack([torch.as_tensor(im, device=device) for im in images], dim=0)
            return_numpy = True if return_numpy is None else return_numpy
    elif isinstance(images, np.ndarray):
        device = device if device is not None else 'cpu'
        images = torch.as_tensor(images, device=device)
        return_numpy = True if return_numpy is None else return_numpy
    elif isinstance(images, torch.Tensor):
        images = images.to(device=device)
    else:
        raise ValueError(f"images must be a np.ndarray or torch.Tensor or a list of np.ndarray or torch.Tensor. Got {type(images)}")        
    
    ## Convert images to 4D
    def pad_to_4D(ims):
        if ims.ndim == 2:
            ims = ims[None, None, :, :]
        elif ims.ndim == 3:
            ims = ims[None, :, :, :]
        elif ims.ndim != 4:
            raise ValueError(f"images must be a 2D, 3D, or 4D array. Got shape {ims.shape}")
        return ims
    ndim_orig = images.ndim
    images = pad_to_4D(images)
    
    ## Get interpolation method
    try:
        interpolation = getattr(torchvision.transforms.InterpolationMode, interpolation.upper())
    except Exception as e:
        raise Exception(f"Invalid interpolation method. See torchvision.transforms.InterpolationMode for options. Error: {e}")

    resizer = torchvision.transforms.Resize(
        size=new_shape,
        interpolation=interpolation,
        antialias=antialias,
    ).to(device=device)
    images_resized = resizer(images)
       
    ## Convert images back to original shape
    def unpad_to_orig(ims, ndim_orig):
        if ndim_orig == 2:
            ims = ims[0,0,:,:]
        elif ndim_orig == 3:
            ims = ims[0,:,:,:]
        elif ndim_orig != 4:
            raise ValueError(f"images must be a 2D, 3D, or 4D array. Got shape {ims.shape}")
        return ims
    images_resized = unpad_to_orig(images_resized, ndim_orig)
        
    ## Convert images to numpy
    if return_numpy == True:
        images_resized = images_resized.detach().cpu().numpy()
    
    return images_resized


def image_to_uint8(
    images, 
    dtype_intermediate=np.float32,
    clip=True,
):
    """
    Convert any dtype images to uint8.
    RH 2023

    Args:
        images (np.ndarray):
            Input images.
            If dtype is integer, then the range is assumed to be [0, max_val].
            If dtype is float, then the range is assumed to be [0, 1].
        dtype_intermediate (np.dtype):
            Intermediate dtype to use for conversion.
        clip (bool):
            If True, then will clip values to [0, 255]. Otherwise, over and
            underflow will be allowed.

    Returns:
        images_uint8 (np.ndarray):
            Output images.
            Shape matches input images.
    """
    ## Get dtype range
    dtype = images.dtype
    if np.issubdtype(dtype, np.integer):
        min_val = 0
        max_val = np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        min_val = 0
        max_val = 1
    else:
        raise ValueError(f"images must be an integer or float dtype. Got {dtype}")
    
    ## Convert to uint8
    images = images.astype(dtype_intermediate)
    images = (images - min_val) / (max_val - min_val) * 255
    images = images.clip(0, 255).astype(np.uint8) if clip else images.astype(np.uint8)

    return images


class ImageAlignmentChecker:
    """
    Class to check the alignment of images using phase correlation.
    RH 2024

    Args:
        hw (Tuple[int, int]): 
            Height and width of the images.
        radius_in (Union[float, Tuple[float, float]]): 
            Radius of the pixel shift / offset that can be considered as
            'aligned'. Used to create the 'in' filter which is an image of a
            small centered circle that is used as a filter and multiplied by
            the phase correlation images. If a single value is provided, the
            filter will be a circle with radius 0 to that value; it will be
            converted to a tuple representing a bandpass filter (0, radius_in).
        radius_out (Union[float, Tuple[float, float]]):
            Similar to radius_in, but for the 'out' filter, which defines the
            'null distribution' for defining what is 'aligned'. Should be a
            value larger than the expected maximum pixel shift / offset. If a
            single value is provided, the filter will be a donut / taurus
            starting at that value and ending at the edge of the smallest
            dimension of the image; it will be converted to a tuple representing
            a bandpass filter (radius_out, min(hw)).
        order (int):
            Order of the butterworth bandpass filters used to define the 'in'
            and 'out' filters. Larger values will result in a sharper edges, but
            values higher than 5 can lead to collapse of the filter.
        device (str):
            Torch device to use for computations. (Default is 'cpu')

    Attributes:
        hw (Tuple[int, int]): 
            Height and width of the images.
        order (int):
            Order of the butterworth bandpass filters used to define the 'in'
            and 'out' filters.
        device (str):
            Torch device to use for computations.
        filt_in (torch.Tensor):
            The 'in' filter used for scoring the alignment.
        filt_out (torch.Tensor):
            The 'out' filter used for scoring the alignment.
    """
    def __init__(
        self,
        hw: Tuple[int, int],
        radius_in: Union[float, Tuple[float, float]],
        radius_out: Union[float, Tuple[float, float]],
        order: int = 5,
        device: str = 'cpu',
    ):
        ## Set attributes
        ### Convert to torch.Tensor
        self.hw = tuple(hw)

        ### Set other attributes
        self.order = int(order)
        self.device = str(device)
        ### Set filter attributes
        if isinstance(radius_in, (int, float, complex)):
            radius_in = (float(0.0), float(radius_in))
        elif isinstance(radius_in, (tuple, list, np.ndarray, torch.Tensor)):
            radius_in = tuple(float(r) for r in radius_in)
        else:
            raise ValueError(f'radius_in must be a float or tuple of floats. Found type: {type(radius_in)}')
        if isinstance(radius_out, (int, float, complex)):
            radius_out = (float(radius_out), float(min(self.hw)) / 2)
        elif isinstance(radius_out, (tuple, list, np.ndarray, torch.Tensor)):
            radius_out = tuple(float(r) for r in radius_out)
        else:
            raise ValueError(f'radius_out must be a float or tuple of floats. Found type: {type(radius_out)}')

        ## Make filters
        self.filt_in, self.filt_out = (torch.as_tensor(make_2D_frequency_filter(
            hw=self.hw,
            low=bp[0],
            high=bp[1],
            order=order,
        ), dtype=torch.float32, device=device) for bp in [radius_in, radius_out])
    
    def score_alignment(
        self,
        images: Union[np.ndarray, torch.Tensor],
        images_ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Score the alignment of a set of images using phase correlation. Computes
        the stats of the center ('in') of the phase correlation image over the
        stats of the outer region ('out') of the phase correlation image.
        RH 2024

        Args:
            images (Union[np.ndarray, torch.Tensor]): 
                A 3D array of images. Shape: *(n_images, height, width)*
            images_ref (Optional[Union[np.ndarray, torch.Tensor]]):
                Reference images to compare against. If provided, the images
                will be compared against these images. If not provided, the
                images will be compared against themselves. (Default is
                ``None``)

        Returns:
            (Dict): 
                Dictionary containing the following keys:
                * 'mean_out': 
                    Mean of the phase correlation image weighted by the
                    'out' filter
                * 'mean_in': 
                    Mean of the phase correlation image weighted by the
                    'in' filter
                * 'ptile95_out': 
                    95th percentile of the phase correlation image multiplied by
                    the 'out' filter
                * 'max_in': 
                    Maximum value of the phase correlation image multiplied by
                    the 'in' filter
                * 'std_out': 
                    Standard deviation of the phase correlation image weighted by
                    the 'out' filter
                * 'std_in': 
                    Standard deviation of the phase correlation image weighted by
                    the 'in' filter
                * 'max_diff': 
                    Difference between the 'max_in' and 'ptile95_out' values
                * 'z_in': 
                    max_diff divided by the 'std_out' value
                * 'r_in': 
                    max_diff divided by the 'ptile95_out' value
        """
        def _fix_images(ims):
            assert isinstance(ims, (np.ndarray, torch.Tensor, list, tuple)), f'images must be np.ndarray, torch.Tensor, or a list/tuple of np.ndarray or torch.Tensor. Found type: {type(ims)}'
            if isinstance(ims, (list, tuple)):
                assert all(isinstance(im, (np.ndarray, torch.Tensor)) for im in ims), f'images must be np.ndarray or torch.Tensor. Found types: {set(type(im) for im in ims)}'
                assert all(im.ndim == 2 for im in ims), f'images must be 2D arrays (height, width). Found shapes: {set(im.shape for im in ims)}'
                if isinstance(ims[0], np.ndarray):
                    ims = np.stack([np.array(im) for im in ims], axis=0)
                else:
                    ims = torch.stack([torch.as_tensor(im) for im in ims], dim=0)
            else:
                if ims.ndim == 2:
                    ims = ims[None, :, :]
                assert ims.ndim == 3, f'images must be a 3D array (n_images, height, width). Found shape: {ims.shape}'
                assert ims.shape[1:] == self.hw, f'images must have shape (n_images, {self.hw[0]}, {self.hw[1]}). Found shape: {ims.shape}'

            ims = torch.as_tensor(ims, dtype=torch.float32, device=self.device)
            return ims

        images = _fix_images(images)
        images_ref = _fix_images(images_ref) if images_ref is not None else images
        
        pc = phase_correlation(images_ref[None, :, :, :], images[:, None, :, :])  ## All to all phase correlation. Shape: (n_images, n_images, height, width)

        ## metrics
        filt_in, filt_out = self.filt_in[None, None, :, :], self.filt_out[None, None, :, :]
        mean_out = (pc * filt_out).sum(dim=(-2, -1)) / filt_out.sum(dim=(-2, -1))
        mean_in =  (pc * filt_in).sum(dim=(-2, -1))  / filt_in.sum(dim=(-2, -1))
        ptile95_out = torch.quantile((pc * filt_out).reshape(pc.shape[0], pc.shape[1], -1)[:, :, filt_out.reshape(-1) > 1e-3], 0.95, dim=-1)
        max_in = (pc * filt_in).amax(dim=(-2, -1))
        std_out = torch.sqrt(torch.mean((pc - mean_out[:, :, None, None])**2 * filt_out, dim=(-2, -1)))
        std_in = torch.sqrt(torch.mean((pc - mean_in[:, :, None, None])**2 * filt_in, dim=(-2, -1)))

        max_diff = max_in - ptile95_out
        z_in = max_diff / std_out
        r_in = max_diff / ptile95_out

        outs = {
            'pc': pc.cpu().numpy(),
            'mean_out': mean_out,
            'mean_in': mean_in,
            'ptile95_out': ptile95_out,
            'max_in': max_in,
            'std_out': std_out,
            'std_in': std_in,
            'max_diff': max_diff,
            'z_in': z_in,  ## z-score of in value over out distribution
            'r_in': r_in,
        }

        outs = {k: val.cpu().numpy() if isinstance(val, torch.Tensor) else val for k, val in outs.items()}
        
        return outs
    
    def __call__(
        self,
        images: Union[np.ndarray, torch.Tensor],
    ):
        """
        Calls the `score_alignment` method. See `self.score_alignment` docstring
        for more info.
        """
        return self.score_alignment(images)


def make_2D_frequency_filter(
    hw: tuple,
    low: float = 5,
    high: float = 6,
    order: int = 3,
):
    """
    Make a filter for scoring the alignment of images using phase correlation.
    RH 2024

    Args:
        hw (tuple): 
            Height and width of the images.
        low (float): 
            Low cutoff frequency for the bandpass filter. Units are in
            pixels.
        high (float): 
            High cutoff frequency for the bandpass filter. Units are in
            pixels.
        order (int): 
            Order of the butterworth bandpass filter. (Default is *3*)

    Returns:
        (np.ndarray): 
            Filter for scoring the alignment. Shape: *(height, width)*
    """
    ## Make a distance grid starting from the fftshifted center
    grid = featurization.make_distance_grid(shape=hw, p=2, use_fftshift_center=True)

    ## Make the number of datapoints for the kernel large
    n_x = max(hw) * 10

    fs = max(hw) * 1
    low = max(0, low)
    high = min((max(hw) / 2) - 1, high)
    b, a = spectral.design_butter_bandpass(lowcut=low, highcut=high, fs=fs, order=order, plot_pref=False)
    w, h = scipy.signal.freqz(b, a, worN=n_x)
    x_kernel = (fs * 0.5 / np.pi) * w
    kernel = np.abs(h)

    ## Interpolate the kernel to the distance grid
    filt = np.interp(
        x=grid,
        xp=x_kernel,
        fp=kernel,
    )

    return filt
