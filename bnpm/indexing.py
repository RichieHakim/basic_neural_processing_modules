from typing import Optional, Union, Iterator
import math

import numpy as np
from numba import jit, njit, prange
import copy
import scipy.signal
import matplotlib.pyplot as plt
import torch


def widen_boolean(arr, n_before, n_after, axis=None):
    '''
    Widens boolean events by n_before and n_after.    
    RH 2021    

    Args:
        arr (np.ndarray):
            Input array. Widening will be applied
             to the last dimension.
        n_before (int):
            Number of samples before 'True' values
             that will also be set to 'True'.
        n_after (int):
            Number of samples after 'True' values
             that will also be set to 'True'.
        axis (int):
            Axis to apply the event widening.
             If None then arr should be a 1-D array.
    
    Returns:
        widened arr (np.ndarray, dtype=bool):
            Output array. Same as input arr, but
             with additional 'True' values before
             and after initial 'True' values.
    '''
    
    kernel = np.zeros(np.max(np.array([n_before, n_after])) * 2 + 1)
    kernel_center = int(np.ceil(len(kernel) / 2))
    kernel[kernel_center - (n_before+1): kernel_center] = 1
    kernel[kernel_center: kernel_center + n_after] = 1
    kernel = kernel / np.mean(kernel)
    
    if axis is None:
        return np.bool_(scipy.signal.convolve(arr, kernel, mode='same'))
    else:
        return np.bool_(np.apply_along_axis(lambda m: scipy.signal.convolve(m, kernel, mode='same'),
                                                              axis=axis, arr=arr))


def widen_bool_sparse(
    b: np.ndarray,
    n_before: int,
    n_after: int,
):
    """
    Widen True values in a boolean array by n_before and n_after samples.
    Best to use when there are few transitions between low and high in the boolean array.
    RH 2025

    Args:
        b: np.ndarray
            Boolean array
        n_before: int
            Number of samples to widen before each True value in b
        n_after: int
            Number of samples to widen after each True value in b
            
    Returns:
        np.ndarray
            Boolean array with True values widened by n_before and n_after samples
    """
    ## Find rising and falling edges
    d = np.diff(b.astype(np.int64), prepend=0)
    idx_rise = np.where(d>0)[0]
    idx_fall = np.where(d<0)[0]
    
    out = b.copy()
    for i, r in enumerate(idx_rise):
        out[r-n_before:r+n_after+1] = True
    for i, f in enumerate(idx_fall):
        out[f-n_before:f+n_after+1] = True
    return out


def set_short_bool_to_low(
    b: np.ndarray,
    n: int,
):
    """
    Set all True values in b (a boolean array) to False if they are
    part of a sequence of True values shorter than n
    RH 2025
    
    Args:
        b: np.ndarray
            Boolean array
        n: int
            Minimum length of True values to keep
            
    Returns:
        np.ndarray
            Boolean array with short True sequences set to False
    """
    ## Find rising and falling edges
    d = np.diff(b.astype(np.int64), prepend=0)
    idx_rise = np.where(d>0)[0]
    idx_fall = np.where(d<0)[0]
    
    out = b.copy()
    if (idx_rise.size > 0) and (idx_fall.size > 0):
        ## Add edge boundaries
        if (out[0] == True) and (idx_rise[0] != 0):
            idx_rise = np.concatenate(([0], idx_rise))
        if (out[-1] == True) and (idx_fall[-1] != out.size-1):
            idx_fall = np.concatenate((idx_fall, [out.size-1]))

        ## match rising and falling edges
        if idx_fall[0] < idx_rise[0]:
            idx_fall = idx_fall[1:]
        if idx_fall.size < idx_rise.size:
            idx_rise = idx_rise[:-1]
            
        for r, f in zip(idx_rise, idx_fall):
            if f-r < n:
                out[r:f] = False
    return out


# @njit
def idx2bool(idx, length=None):
    '''
    Converts a vector of indices to a boolean vector.
    RH 2021

    Args:
        idx (np.ndarray):
            1-D array of indices.
        length (int):
            Length of boolean vector.
            If None then length will be set to
             the maximum index in idx + 1.
    
    Returns:
        bool_vec (np.ndarray):
            1-D boolean array.
    '''
    ## remove NaNs
    idx = idx[~np.isnan(idx)].astype(np.int64)
    if length is None:
        length = np.uint64(np.max(idx) + 1)
    out = np.zeros(length, dtype=np.bool8)
    out[idx] = True
    return out


def bool2idx(bool_vec):
    '''
    Converts a boolean vector to indices.
    RH 2021

    Args:
        bool_vec (np.ndarray):
            1-D boolean array.
    
    Returns:
        idx (np.ndarray):
            1-D array of indices.
    '''
    return np.where(bool_vec)[0]


@njit
def binary_search(arr, lb, ub, val):
    '''
    Recursive binary search
    adapted from https://www.geeksforgeeks.org/python-program-for-binary-search/
    RH 2021
    
    Args:
        arr (sorted list):
            1-D array of numbers that are already sorted.
            To use numba's jit features, arr MUST be a
            typed list. These include:
                - numpy.ndarray (ideally np.ascontiguousarray)
                - numba.typed.List
        lb (int):
            lower bound index.
        ub (int):
            upper bound index.
        val (scalar):
            value being searched for
    
    Returns:
        output (int):
            index of val in arr.
            returns -1 if value is not present
            
    Demo:
        # Test array
        arr = np.array([ 2, 3, 4, 10, 40 ])
        x = 100

        # Function call
        result = binary_search(arr, 0, len(arr)-1, x)

        if result != -1:
            print("Element is present at index", str(result))
        else:
            print("Element is not present in array")
    '''
    # Check base case
    if ub >= lb:
 
        mid = (ub + lb) // 2
 
        # If element is present at the middle itself
        if arr[mid] == val:
            return mid
 
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > val:
            return binary_search(arr, lb, mid - 1, val)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, ub, val)
 
    else:
        # Element is not present in the array
        return -1


def get_last_True_idx(input_array):
    '''
    for 1-d arrays only. gets idx of last entry
    that == True
    RH 2021
    '''
    nz = np.nonzero(input_array)[0]
    # print(nz.size)
    if nz.size==0:
        raise ValueError('No True values in array')
    else:
        output = np.max(nz)
#     print(output)
    return output

def get_nth_True_idx(input_array, n):
    '''
    for 1-d arrays only. gets idx of nth True entry.
    nth is zero-indexed.
    RH 2022
    '''
    nz = np.nonzero(input_array)[0]
    # print(nz.size)
    if nz.size==0:
        raise ValueError('No True values in array')
    else:
        output = nz[n]
    return output


def make_batches(
    iterable, 
    batch_size=None, 
    num_batches=None, 
    min_batch_size=0, 
    return_idx=False, 
    length=None,
    idx_start=0,
    randomize_batch_indices=False,
    shuffle_batch_order=False,
    shuffle_iterable_order=False,
):
    """
    Make batches of data or any other iterable.
    RH 2021-2024

    Args:
        iterable (iterable):
            Iterable to be batched.
        batch_size (int):
            Size of each batch.\n
            if None, then ``batch_size`` based on ``num_batches``.
        num_batches (int):
            Number of batches to make.\n
            if None, then ``num_batches`` based on ``batch_size``.
        min_batch_size (int):
            Minimum size of each batch. Set to ``-1`` to equal ``batch_size``.
        return_idx (bool):
            Whether to also return the slice indices of the batches. Returns will
            be the tuple: ``(iterable[idx], [idx_start, idx_end])`` where
            idx_start is inclusive and idx_end is exclusive. Use like:
            slice(idx_start, idx_end). This output doesn't make sense when
            randomizing or shuffling.
        length (int):
            Length of the iterable. Determines ``idx_end``. If ``None``, then
            length is ``len(iterable)`` This is useful if you want to make
            batches of something that doesn't have a ``__len__`` method.
        idx_start (int):
            Starting index of the iterable.
        randomize_batch_indices
            Whether to randomize the transition indices between batches.\n
            So ``[(1,2,3), (4,5,6),]`` can become ``[(1,), (2,3,4), (5,6),]``.
        shuffle_batch_order (bool):
            Whether to shuffle the order in which batches are yielded.
        shuffle_iterable_order (bool):
            Whether to shuffle the order of the contents of the iterable before
            batching.
    
    Returns:
        output (iterable):
            batches of iterable
    """

    ## Prepare parameters
    l = len(iterable) if length is None else length

    assert sum([(batch_size is None), (num_batches is None)]) == 1, 'Exactly one of batch_size or num_batches must be specified. The other must be None.'
    if batch_size is None:
        assert isinstance(num_batches, int), 'num_batches must be an integer'
        batch_size = int(math.ceil(l / num_batches))
    if num_batches is None:
        assert isinstance(batch_size, int), 'batch_size must be an integer'
        num_batches = int(math.ceil(l / batch_size))

    if min_batch_size == -1:
        min_batch_size = batch_size

    assert batch_size > 0, 'batch_size must be greater than 0'
    assert min_batch_size <= batch_size, 'min_batch_size must be less than or equal to batch_size'
    assert num_batches > 0, 'num_batches must be greater than 0'    

    ## Make slices
    offset = np.random.randint(0, batch_size) if randomize_batch_indices else 0
    idx_slices = [
        slice(s, min(e, l), None) for s, e in (zip(
            np.arange(idx_start + offset, l, batch_size), 
            np.arange(idx_start + batch_size + offset, l + batch_size + offset, batch_size)
        ))]
    if offset > 0:
        idx_slices = [slice(0, offset, None)] + idx_slices

    ## Remove small batches
    if min_batch_size > 0:
        idx_slices = [idx for idx in idx_slices if (idx.stop - idx.start) >= min_batch_size]

    ## Shuffle batch order
    if shuffle_batch_order:
        idx_slices = [idx_slices[ii] for ii in np.random.permutation(len(idx_slices))]

    ## Crop to num_batches
    idx_slices = idx_slices[:num_batches]

    ## Shuffle iterable contents order
    if shuffle_iterable_order:
        order = np.random.permutation(l)
    else:
        order = range(l)

    ## Yield batches
    if return_idx:
        for idx in idx_slices:
            yield iterable[order[idx]], [idx.start, idx.stop]
    else:
        for idx in idx_slices:
            yield iterable[order[idx]]


@njit
def find_nearest_idx(array, value):
    '''
    Finds the value and index of the nearest
     value in an array.
    RH 2021, 2024
    
    Args:
        array (np.ndarray):
            Array of values to search through.
        value (scalar):
            Value to search for.

    Returns:
        array_idx (int):
            Index of the nearest value in array.
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx-1
    else:
        return idx
@njit(parallel=True)
def find_nearest_array(array, values, max_diff=None):
    '''
    Finds the values and indices of the nearest
     values in an array.
    RH 2021, 2024

    Args:
        array (np.ndarray):
            Array of values to search through.
        values (np.ndarray):
            Values to search for.

    Returns:
        array_idx (np.ndarray):
            Indices of the nearest values in array.
        array_val (np.ndarray):
            Values of the nearest values in array.
        diff (np.ndarray):
            Differences between the values and the
             nearest values in array.
    '''
    assert array.ndim == 1, 'array must be 1-D'
    assert values.ndim == 1, 'values must be 1-D'
    
    vals_nearest = np.zeros(values.shape if array.size > 0 else (0,), dtype=array.dtype)
    idx_nearest  = np.zeros(values.shape if array.size > 0 else (0,), dtype=np.int64)
    diff_nearest = np.zeros(values.shape if array.size > 0 else (0,), dtype=array.dtype)
    
    if array.size > 0:
        for ii in prange(len(values)):
            idx_nearest[ii] = find_nearest_idx(array , values[ii])

        vals_nearest = array[idx_nearest]
        diff_nearest = np.abs(vals_nearest - values)

        if max_diff is not None:
            bool_keep = diff_nearest <= max_diff
            vals_nearest = vals_nearest[bool_keep]
            idx_nearest = idx_nearest[bool_keep]
            diff_nearest = diff_nearest[bool_keep]
            
    return vals_nearest, idx_nearest, diff_nearest


def pad_with_singleton_dims(array, n_dims_pre=0, n_dims_post=0):
    arr_out = copy.copy(np.array(array))
    
    for n in range(n_dims_pre):
        arr_out = np.expand_dims(arr_out, 0)
    for n in range(n_dims_post):
        arr_out = np.expand_dims(arr_out, -1)
    return arr_out


def index_with_nans(values, indices):
    """
    Indexes an array with a list of indices, allowing for NaNs in the indices.
    RH 2022
    
    Args:
        values (np.ndarray):
            Array to be indexed.
        indices (Union[List[int], np.ndarray]):
            1D list or array of indices to use for indexing. Can contain NaNs.
            Datatype should be floating point. NaNs will be removed and values
            will be cast to int.

    Returns:
        np.ndarray:
            Indexed array. Positions where `indices` was NaN will be filled with
            NaNs.
    """
    indices = np.array(indices, dtype=float) if not isinstance(indices, np.ndarray) else indices
    values = np.concatenate((np.full(shape=values.shape[1:], fill_value=np.nan, dtype=values.dtype)[None,...], values), axis=0)
    idx = indices.copy() + 1
    idx[np.isnan(idx)] = 0
    
    return values[idx.astype(np.int64)]


def shift_pad(array, shift=1, axis=-1, pad_val=0, in_place=False):
    """
    Pads an array with a constant value.
    Allows for shifting along any axis.
    RH 2022

    Args:
        array (np.ndarray):
            array to shift and pad
        shift (int):
            number of elements to shift
        axis (int):
            axis to shift along
        pad_val (any):
            value to pad with
        in_place (bool):
            whether to shift and pad in place

    Returns:
        output (np.ndarray):
            shifted and padded array
    """
    if shift==0:
        return array
        
    if shift > 0:
        idx = np.arange(array.shape[axis])[:-shift]
    if shift < 0:
        idx = np.arange(array.shape[axis])[-shift:]
    
    if axis==-1:
        axis_to_nix = array.shape[-1]
    else:
        axis_to_nix = axis
                
    dims_to_append = list(array.shape)
    dims_to_append[axis_to_nix] = np.abs(shift)
    
    padding = np.ones(dims_to_append) * pad_val
    
    if in_place:
        out = array
    else:
        out = None
    
    if shift > 0:
        arr_shifted = np.concatenate(
            ( padding, np.take(array, idx, axis=axis)),
            axis=axis,
            out=out
        )
    if shift < 0:
        arr_shifted = np.concatenate(
            ( np.take(array, idx, axis=axis),   padding ),
            axis=axis,
            out=out
        )
        
    return arr_shifted
    

def off_diagonal(x):
    """
    Returns the off-diagonal elements of a matrix as a vector.
    RH 2022

    Args:
        x (np.ndarray or torch tensor):
            square matrix to extract off-diagonal elements from.

    Returns:
        output (np.ndarray or torch tensor):
            off-diagonal elements of x.
    """
    n, m = x.shape
    assert n == m
    return x.reshape(-1)[:-1].reshape(n - 1, n + 1)[:, 1:].reshape(-1)


def batched_unfold(
    tensor: torch.Tensor, 
    dimension: int,
    size: int,
    step: int, 
    batch_size: int = 10, 
    pad_value: Optional[Union[int, float]] = None,
) -> Iterator[torch.Tensor]:
    """
    Generates batches of overlapping windows from a tensor in a batched manner.
    This function mimics the behavior of `torch.Tensor.unfold` but allows for
    processing in smaller, more manageable batches. Using
    `torch.cat(list(batched_unfold(...)), dim=dimension)` should reproduce the
    result of `tensor.unfold(dimension, size, step)`. Note: The last batch may
    be smaller than the specified batch size due to the remainder of the
    division.

    RH 2024

    Args:
        tensor (torch.Tensor): 
            The input tensor from which to generate unfolded windows.
        dimension (int): 
            The dimension along which to unfold the tensor.
        size (int): 
            The size of each window to unfold.
        step (int): 
            The step size between the starts of each window.
        batch_size (int): 
            The number of windows to include in each batch. (Default is 10)
        pad_value (Optional[Union[int, float]]):
            The value to use for padding the last batch if it is smaller than
            the specified batch size. If None, the last batch will not be
            padded. (Default is None)


    Yields:
        torch.Tensor: 
            A batch of unfolded windows from the tensor.
    """
    n_samples = tensor.shape[dimension]
    if step >= n_samples:
        raise ValueError("Step size must be less than the tensor size along the specified dimension.")
    if size > n_samples:
        raise ValueError("Window size must be less than or equal to the tensor size along the specified dimension.")

    # Calculate end points
    idx_start_last = ((n_samples - size) // step) * step
    # idx_last_start = max(idx_last_start, 0)
    # idx_last_end = idx_last_start + size - 1  ## Inclusive

    idx_starts_all = range(0, idx_start_last + 1, step)
    # idx_ends_all = list((idx + size for idx in idx_starts_all))  ## Exclusive

    ## Generate batches
    for i_batch, idx_start in enumerate(idx_starts_all[::batch_size]):
        idx_end = idx_start + (step * (batch_size - 1)) + size
        n_samples_pad = 0
        if idx_end > n_samples:
            if pad_value is None:
                idx_end = n_samples
            else:
                n_samples_batch = n_samples - idx_start
                idx_start_last_batch = int((math.ceil((n_samples_batch - size) / step)) * step)
                idx_end = idx_start + idx_start_last_batch + size
                idx_end = min(idx_end, n_samples + size)
                n_samples_pad = idx_end - n_samples
            
        len_batch = idx_end - idx_start

        if n_samples_pad > 0:
            shape_pad = list(tensor.shape)
            shape_pad[dimension] = n_samples_pad
            x_batch = torch.cat((
                tensor.narrow(
                    dim=dimension, 
                    start=idx_start, 
                    length=n_samples - idx_start,
                ), 
                torch.full(
                    size=shape_pad,
                    fill_value=pad_value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            ))
        else:
            x_batch = tensor.narrow(dimension, idx_start, len_batch)

        yield x_batch.unfold(dimension, size, step)


def find_longest_true_sequence(arr):
    """
    Finds the longest sequence of True values in a boolean array.
    RH 2024

    Args:
        arr (np.ndarray):
            1-D boolean array

    Returns:
        (int, int):
            Index of the first True value in the longest sequence
            Length of the longest sequence
    """
    assert arr.dtype == np.bool_, 'Input array must be boolean'
    idx = np.where(arr)[0]
    if len(idx) == 0:
        return None
    idx_diff = np.diff(idx, prepend=-1, append=arr.size)
    idx_diff = np.split(idx_diff, np.where(idx_diff > 1)[0])
    idx_diff = np.array([len(x) for x in idx_diff])
    idx_max = np.argmax(idx_diff)
    return idx[idx_max], idx_diff[idx_max]


def find_longest_true_sequence_circular(arr):
    """
    Finds the longest sequence of True values in a boolean array, assuming
    that the array is circular. This function is useful for finding the
    longest sequence of True values in a circular array, such as a circular
    buffer.
    RH 2024

    Args:
        arr (np.ndarray):
            1-D boolean array

    Returns:
        (int, int):
            Index of the first True value in the longest sequence
            Length of the longest sequence
    """
    assert arr.dtype == np.bool_, 'Input array must be boolean'
    idx = np.where(arr)[0]
    if len(idx) == 0:
        return None
    idx_diff = np.diff(idx, prepend=-1, append=arr.size)
    idx_diff = np.split(idx_diff, np.where(idx_diff > 1)[0])
    idx_diff = np.array([len(x) for x in idx_diff])
    ## if the following conditions are true then join the first sequence onto the end of the last sequence
    if len(idx_diff) > 1:
        if idx_diff[0][0] == 0 and idx_diff[-1][-1] == len(arr) - 1:
            ## join the first sequence onto the end of the last sequence
            idx_diff[-1] = np.concatenate((idx_diff[-1], idx_diff[0]))
            idx_diff = idx_diff[1:]
    idx_max = np.argmax(idx_diff)
    return idx[idx_max], idx_diff[idx_max]


def find_longest_true_sequence_circular(arr):
    """
    Finds the longest sequence of True values in a boolean array, assuming
    that the array is circular. This function is useful for finding the
    longest sequence of True values in a circular array, such as a circular
    buffer.
    RH 2024

    Args:
        arr (np.ndarray):
            1-D boolean array

    Returns:
        (int, int):
            Index of the first True value in the longest sequence
            Length of the longest sequence
    """
    assert arr.dtype == np.bool_, 'Input array must be boolean'
    idx = np.where(arr)[0]
    if len(idx) == 0:
        return None
    idx_diff = np.diff(idx, prepend=-1, append=arr.size + 1)
    idx_diff_split = np.split(idx, np.where(idx_diff > 1)[0])
    ## if the following conditions are true then join the first sequence onto the end of the last sequence
    idx_diff_split = [t for idx in idx_diff_split if len(t:=np.atleast_1d(idx)) > 0]
    if len(idx_diff_split) > 1:
        if idx_diff_split[0][0] == 0 and idx_diff_split[-1][-1] == len(arr) - 1:
            ## join the first sequence onto the end of the last sequence
            idx_diff_split[-1] = np.concatenate((idx_diff_split[-1], idx_diff_split[0]))
            idx_diff_split = idx_diff_split[1:]

    idx_diff = np.array([len(x) for x in idx_diff_split])
    idx_max = np.argmax(idx_diff)
    return idx_diff_split[idx_max][0], idx_diff[idx_max]


def find_all_true_sequences_circular(arr):
    """
    Finds the longest sequence of True values in a boolean array, assuming
    that the array is circular. This function is useful for finding the
    longest sequence of True values in a circular array, such as a circular
    buffer.
    RH 2024

    Args:
        arr (np.ndarray):
            1-D boolean array

    Returns:
        (int, int):
            Index of the first True value in the longest sequence
            Length of the longest sequence
    """
    assert arr.dtype == np.bool_, 'Input array must be boolean'
    idx = np.where(arr)[0]
    if len(idx) == 0:
        return None
    idx_diff = np.diff(idx, prepend=-1, append=arr.size + 1)
    idx_diff_split = np.split(idx, np.where(idx_diff > 1)[0])
    ## if the following conditions are true then join the first sequence onto the end of the last sequence
    idx_diff_split = [t for idx in idx_diff_split if len(t:=np.atleast_1d(idx)) > 0]
    if len(idx_diff_split) > 1:
        if idx_diff_split[0][0] == 0 and idx_diff_split[-1][-1] == len(arr) - 1:
            ## join the first sequence onto the end of the last sequence
            idx_diff_split[-1] = np.concatenate((idx_diff_split[-1], idx_diff_split[0]))
            idx_diff_split = idx_diff_split[1:]
            
    return idx_diff_split


#######################################
############ SPARSE STUFF #############
#######################################

def sparse_mask(x, mask_sparse, do_safety_steps=True):
    """
    Masks a sparse matrix with the non-zero elements of another
     sparse matrix.
    RH 2022

    Args:
        x (scipy.sparse.csr_matrix):
            sparse matrix to mask
        mask_sparse (scipy.sparse.csr_matrix):
            sparse matrix to mask with
        do_safety_steps (bool):
            whether to do safety steps to ensure that things
             are working as expected.

    Returns:
        output (scipy.sparse.csr_matrix):
            masked sparse matrix
    """
    if do_safety_steps:
        m = mask_sparse.copy()
        m.eliminate_zeros()
    else:
        m = mask_sparse
    return (m!=0).multiply(x)

def scipy_sparse_to_torch_coo(sp_array):
    import torch

    coo = scipy.sparse.coo_matrix(sp_array)
    
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def pydata_sparse_to_torch_coo(sp_array):
    import sparse
    import torch

    coo = sparse.COO(sp_array)
    
    values = coo.data
#     indices = np.vstack((coo.row, coo.col))
    indices = coo.coords

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))
    
def pydata_sparse_to_spconv(sp_array):
    import sparse
    import torch
    import spconv

    coo = sparse.COO(sp_array)
    idx_raw = torch.as_tensor(coo.coords.T, dtype=torch.int32)
    idx = torch.hstack((torch.zeros((idx_raw.shape[0],1)) , idx_raw)).type(torch.int32)
    spconv_array = rois_sp_spconv = spconv.SparseConvTensor(
        features=coo.reshape((coo.shape[0], -1)).T,
        indices=idx,
        spatial_shape=coo.shape, 
        batch_size=1
    )
    return spconv_array

def sparse_convert_spconv_to_scipy(sp_arr):
    import sparse

    coo = sparse.COO(
        coords=sp_arr.indices.T.to('cpu'),
        data=sp_arr.features.squeeze().to('cpu'),
        shape=[sp_arr.batch_size] + sp_arr.spatial_shape
    )
    return coo.reshape((coo.shape[0], -1)).to_scipy_sparse().tocsr()

def torch_to_torchSparse(s):
    import torch_sparse

    # return torch_sparse.from_torch_sparse(s)

    return torch_sparse.tensor.SparseTensor(
        row=s.indices()[0],
        col=s.indices()[1],
        value=s.values(),
        sparse_sizes=s.shape,
    )

# def pydata_sparse_to_torchSparse(s, shape=None):
#     import torch
#     import sparse
#     import torch_sparse

#     return torch_sparse.from_torch_sparse(pydata_sparse_to_torch_coo(s).coalesce())

    # coords = [torch.LongTensor(c) for c in s.coords]
    # vals = torch.as_tensor(s)

    # return torch_sparse.tensor.SparseTensor(
    #     row=coords[0],
    #     col=coords[1],
    #     value=vals,
    #     sparse_sizes=shape,
    # )

def denseDistances_to_knnDistances(denseDistanceMatrix, k=1023, epsilon=1e-9):
    """
    Converts a dense distance matrix to a sparse kNN distance matrix.
    Largest values are sparsened away. Zeros are set to epsilon.
    Useful for converting custom distance matrices into a format that
     can be used by things like sklearn's nearest neighbors algorithms.
    RH 2022

    Args:
        denseDistanceMatrix (np.ndarray):
            Dense distance matrix
        k (int):
            Number of nearest neighbors to find
        epsilon (float):
            Small number to add to distances because entries with
             distance 0 are ignored by scipy's sparse csr_matrix.
            It can be subtracted out later.

    Returns:
        output (scipy.sparse.csr_matrix):
            Sparse kNN distance matrix
    """
    X = denseDistanceMatrix + epsilon
    k_lowest = np.argsort(X, axis=1)[:,:k]
    kl_bool = np.stack([idx2bool(k_l, length=X.shape[1]) for k_l in k_lowest])
    X[~kl_bool] = 0
    return scipy.sparse.csr_matrix(X), kl_bool


class scipy_sparse_csr_with_length(scipy.sparse.csr_matrix):
    """
    A scipy sparse matrix with a length attribute.
    Useful when needing to iterate over the rows of a sparse matrix.
    This class gives a length attribute to the sparse matrix.
    RH 2022
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = self.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return self.__class__(super().__getitem__(key))


def find_nonredundant_idx(s):
    """
    Finds the indices of the nonredundant entries in a sparse matrix.
    Useful when you are manually populating a spare matrix and want to
     know which entries you have already populated.
    RH 2022

    Args:
        s (scipy.sparse.coo_matrix):
            Sparse matrix. Should be in coo format.

    Returns:
        idx_unique (np.ndarray):
            Indices of the nonredundant entries
    """
    if s.getformat() != 'coo':
        s = s.coo()
    idx_rowCol = np.vstack((s.row, s.col)).T
    u, idx_u = np.unique(idx_rowCol, axis=0, return_index=True)
    return idx_u
def remove_redundant_elements(s, inPlace=False):
    """
    Removes redundant entries from a sparse matrix.
    Useful when you are manually populating a spare matrix and want to
     remove redundant entries.
    RH 2022

    Args:
        s (scipy.sparse.coo_matrix):
            Sparse matrix. Should be in coo format.
        inPlace (bool):
            If True, the input matrix is modified in place.
            If False, a new matrix is returned.

    Returns:
        s (scipy.sparse.coo_matrix):
            Sparse matrix with redundant entries removed.
    """
    idx_nonRed = find_nonredundant_idx(s)
    if inPlace:
        s_out = s
    else:
        s_out = scipy.sparse.coo_matrix(s.shape)
    s_out.row = s.row[idx_nonRed]
    s_out.col = s.col[idx_nonRed]
    s_out.data = s.data[idx_nonRed]
    return s_out

def merge_sparse_arrays(s_list, idx_list, shape_full, remove_redundant=True, elim_zeros=True):
    """
    Merges a list of square sparse arrays into a single square sparse array.
    Note that no selection is performed for removing redundant entries;
     just whatever is selected by np.unique is kept.

    Args:
        s_list (list of scipy.sparse.csr_matrix):
            List of sparse arrays to merge.
            Each array can be of any shape.
        idx_list (list of np.ndarray int):
            List of arrays of integers. Each array should be of the same
             length as the corresponding array in s_list and should contain
             integers in the range [0, shape_full[0]). These integers
             represent the row/col indices in the full array.
        shape_full (tuple of int):
            Shape of the full array.
        remove_redundant (bool):
            If True, redundant entries are removed from the output array.
            If False, redundant entries are kept.

    Returns:
        s_full (scipy.sparse.csr_matrix):
    """
    row, col, data = np.array([]), np.array([]), np.array([])
    for s, idx in zip(s_list, idx_list):
        s_i = s.tocsr() if s.getformat() != 'csr' else s
        s_i.eliminate_zeros() if elim_zeros else s_i
        idx_grid = np.meshgrid(idx, idx)
        row = np.concatenate([row, (s_i != 0).multiply(idx_grid[0]).data])
        col = np.concatenate([col, (s_i != 0).multiply(idx_grid[1]).data])
        data = np.concatenate([data, s_i.data])
    s_full = scipy.sparse.coo_matrix((data, (row, col)), shape=shape_full)
    if remove_redundant:
        remove_redundant_elements(s_full, inPlace=True)
    return s_full


def sparse_to_dense_fill(arr_s, fill_val=0.):
    """
    Converts a sparse array to a dense array and fills
     in sparse entries with a fill value.
    """
    import sparse
    s = sparse.COO(arr_s)
    s.fill_value = fill_val
    return s.todense()


####################################################################################################
##################################### TENSOR OPERATIONS ############################################
####################################################################################################

def cp_to_kruskal(cp):
    """
    Converts a list (of length n_modes) of 2D arrays (of shape (len_dim, rank))
     [CP format] to a list (of length rank) of lists (of length n_modes) of
     vectors (of length len_dim) [Kruskal format].
    RH 2022

    Args:
        cp (list of np.ndarray):
            List of 2D arrays in CP format.
            Tensorly uses this format for their 'cp' format.

    Returns:
        k (list of list of np.ndarray):
            List of lists of vectors in Kruskal format.
    """
    rank = cp[0].shape[1]
    k = [[cp[m][:,r] for m in range(len(cp))] for r in range(rank)]
    return k

def kruskal_to_dense(k, weights=None):
    """
    Converts a list (of length rank) of lists (of length n_modes) of
     vectors (of length len_dim) [Kruskal format] to a dense tensor
     (of shape (len_dim, len_dim, ...))
    RH 2022

    Args:
        k (list of list of np.ndarray):
            List of lists of vectors in Kruskal format.

    Returns:
        dense (np.ndarray):
            Dense tensor
    """
    assert isinstance(k, list), 'k must be a list'
    assert isinstance(k[0], list), 'k must be a list of lists'
    assert all([len(k[0]) == len(k[r]) for r in range(len(k))]), 'each mode must have the same size'


    ## check if numpy or torch
    if isinstance(k[0][0], np.ndarray):
        zeros = np.zeros
        einsum = np.einsum
        weights = np.array(weights).astype(k[0][0].dtype) if weights is not None else np.ones(len(k)).astype(k[0][0].dtype)
    elif isinstance(k[0][0], torch.Tensor):
        zeros = torch.zeros
        einsum = torch.einsum
        weights = torch.as_tensor(weights).type(k[0][0].dtype).to(k[0][0].device) if weights is not None else torch.ones(len(k)).type(k[0][0].dtype).to(k[0][0].device)

    rank = len(k)
    n_modes = len(k[0])
    dense = zeros([k[0][m].shape[0] for m in range(n_modes)]).type(k[0][0].dtype).to(k[0][0].device)
    for r in range(rank):
        ## take the outer product of the n vectors in each mode and add to the dense tensor
        str_einsum = ','.join([chr(97+m) for m in range(n_modes)]) + '->' + ''.join([chr(97+m) for m in range(n_modes)])
        dense += einsum(str_einsum, *k[r]) * weights[r]

    return dense

def cp_to_dense(cp, weights=None):
    """
    Converts a list (of length n_modes) of 2D arrays (of shape (len_dim, rank))
     [CP format] to a dense tensor (of shape (len_dim, len_dim, ...))
    RH 2022

    Args:
        cp (list of np.ndarray):
            List of 2D arrays in CP format.
            Tensorly uses this format for their 'cp' format.

    Returns:
        dense (np.ndarray):
            Dense tensor
    """
    rank = cp[0].shape[1]
    n_modes = len(cp)
    str_einsum = ','.join([chr(97+m)+'r' for m in range(n_modes)]) + '->' + ''.join([chr(97+m) for m in range(n_modes)])
    if weights is None:
        weights = np.ones(rank)

    ## check if numpy or torch
    if isinstance(cp[0], np.ndarray):
        einsum = np.einsum
        weights = np.array(weights).astype(cp[0].dtype)
    elif isinstance(cp[0], torch.Tensor):
        einsum = torch.einsum
        weights = torch.as_tensor(weights).type(cp[0].dtype).to(cp[0].device)

    dense = einsum(str_einsum, *[cp[m] * weights for m in range(n_modes)])
    
    return dense


####################################################################################################
################################## NATIVE PYTHON OPERATIONS ########################################
####################################################################################################

def native_argsort(l):
    """
    Native Python argsort. Returns the indices that would sort a list.
    RH 2024

    Args:
        l (list):
            List to sort

    Returns:
        (list):
            List of indices that would sort the input list.
    """
    return sorted(range(len(l)), key=lambda k: l[k])

def native_argmax(l):
    """
    Native Python argmax. Returns the index of the maximum value in a list.
    RH 2024

    Args:
        l (list):
            List to find the argmax of

    Returns:
        (int):
            Index of the maximum value in the list.
    """
    return max(range(len(l)), key=lambda k: l[k])

def native_argmin(l):
    """
    Native Python argmin. Returns the index of the minimum value in a list.
    RH 2024

    Args:
        l (list):
            List to find the argmin of

    Returns:
        (int):
            Index of the minimum value in the list.
    """
    return min(range(len(l)), key=lambda k: l[k])