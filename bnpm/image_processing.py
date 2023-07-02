from functools import partial
import typing

import numpy as np
import cv2

import copy
import torch
import torchvision
from tqdm.notebook import tqdm
import scipy.interpolate
import scipy.sparse

from . import indexing, featurization, parallel_helpers


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
    assert images.shape[-2] == remappingIdx.shape[0], f"images H ({images.shape[-2]}) must match remappingIdx H ({remappingIdx.shape[0]})"
    assert images.shape[-1] == remappingIdx.shape[1], f"images W ({images.shape[-1]}) must match remappingIdx W ({remappingIdx.shape[1]})"

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
    ims_sparse: typing.Union[scipy.sparse.spmatrix, typing.List[scipy.sparse.spmatrix]],
    remappingIdx: np.ndarray,
    method: str = 'linear',
    fill_value: float = 0,
    dtype: typing.Union[str, np.dtype] = None,
    safe: bool = True,
    n_workers: int = -1,
    verbose=True,
) -> typing.List[scipy.sparse.csr_matrix]:
    """
    Remaps a list of sparse images using the given remap field.

    Args:
        ims_sparse (scipy.sparse.spmatrix or List[scipy.sparse.spmatrix]): 
            A single sparse image or a list of sparse images.
        remappingIdx (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field. It
             should be the same size as the images in ims_sparse.
        method (str): 
            Interpolation method to use. 
            See scipy.interpolate.griddata.
            Options are 'linear', 'nearest', 'cubic'.
        fill_value (float, optional): 
            Value used to fill points outside the convex hull. 
        dtype (np.dtype): 
            The data type of the resulting sparse images. 
            Default is None, which will use the data type of the input
             sparse images.
        safe (bool): 
            If True, checks if the image is 0D or 1D and applies a tiny
             Gaussian blur to increase the image width.
        n_workers (int): 
            Number of parallel workers to use. 
            Default is -1, which uses all available CPU cores.
        verbose (bool):
            Whether or not to use a tqdm progress bar.

    Returns:
        ims_sparse_out (List[scipy.sparse.csr_matrix]): 
            A list of remapped sparse images.

    Raises:
        AssertionError: If the image and remappingIdx have different spatial dimensions.
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

        # Account for 1d images by convolving image with tiny gaussian kernel to increase image width
        if safe:
            ## append if there are < 3 nonzero pixels
            if (np.unique(rows).size == 1) or (np.unique(cols).size == 1) or (rows.size < 3):
                return warp_sparse_image(im_sparse=conv2d(im_sparse, batching=False), remappingIdx=remappingIdx)

        # Get values at the grid points
        grid_values = scipy.interpolate.griddata(
            points=(rows, cols), 
            values=data, 
            xi=remappingIdx[:,:,::-1], 
            method=method, 
            fill_value=fill_value,
        )

        # Create a new sparse image from the nonzero pixels
        warped_sparse_image = scipy.sparse.csr_matrix(grid_values, dtype=dtype)
        warped_sparse_image.eliminate_zeros()

        return warped_sparse_image
    
    wsi_partial = partial(warp_sparse_image, remappingIdx=remappingIdx)
    ims_sparse_out = parallel_helpers.map_parallel(func=wsi_partial, args=(ims_sparse,), method='multithreading', workers=n_workers, prog_bar=verbose)
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
    images, 
    new_shape=(100,100),
    interpolation='linear',
    antialias=False,
    align_corners=False,
    device='cpu',
):
    """
    Resize images.
    Uses torch.nn.functional.interpolate.
    RH 2023

    Args:
        images (np.array or list):
            frames of video or images.
            Can be 2D, 3D, or 4D
            2D shape: (height, width)
            3D shape: (n_frames, height, width)
            4D shape: (n_frames, height, width, n_channels)
        new_shape (tuple):
            (height, width) of resized images.
        interpolation (str):
            interpolation method to use.
            See torch.nn.functional.interpolate for options.
        antialias (bool):
            if True, then will use antialiasing.
        align_corners (bool):
            if True, then will align corners.
            See torch.nn.functional.interpolate for details.
        device (str):
            device to use for torch.nn.functional.interpolate.
            
    Returns:
        images_resized (np.array):
            frames of video or images with overlay added.
    """
    if isinstance(images, list):
        images = np.stack(images, axis=0)
    
    if images.ndim == 2:
        images = images[None,:,:]
    elif images.ndim == 3:
        images = images
    elif images.ndim == 4:
        images = images.transpose(0,3,1,2)
    else:
        raise ValueError('images must be 2D, 3D, or 4D.')
    
    images_torch = torch.as_tensor(images, device=device)
    images_torch = torch.nn.functional.interpolate(
        images_torch, 
        size=tuple(np.array(new_shape, dtype=int)), 
        mode=interpolation,
        align_corners=align_corners,
        recompute_scale_factor=None,
        antialias=antialias,
    )
    images_resized = images_torch.cpu().numpy()
    
    if images.ndim == 2:
        images_resized = images_resized[0,:,:]
    elif images.ndim == 3:
        images_resized = images_resized
    elif images.ndim == 4:
        images_resized = images_resized.transpose(0,2,3,1)
    
    return images_resized