import numpy as np
from numba import jit, njit, prange
import copy
import scipy.signal
import matplotlib.pyplot as plt

from collections.abc import MutableMapping
import itertools



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
        return np.bool8(scipy.signal.convolve(arr, kernel, mode='same'))
    else:
        return np.bool8(np.apply_along_axis(lambda m: scipy.signal.convolve(m, kernel, mode='same'),
                                                              axis=axis, arr=arr))


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


def moduloCounter_to_linearCounter(trace, modulus, modulus_value, diff_thresh=None, plot_pref=False):
    '''
    Converts a (sawtooth) trace of modulo counter
     values to a linear counter.
    Useful for converting a pixel clock with a modulus
     to total times. Use this for FLIR camera top pixel
     stuff.
    The function basically just finds where the modulus
     events occur in the trace and adds 'modulus_value'
     to the next element in the trace.
    RH 2021

    Args:
        trace (np.ndarray):
            1-D array of modulo counter values.
        modulus (scalar):
            Modulus of the counter. Values in trace
             should range from 0 to modulus-1.
        modulus_value (scalar):
            Multiplier for the modulus counter. The
             value of a modulus event.
        diff_thresh (scalar):
            Threshold for defining a modulus event.
            Should typically be a negative value
             smaller than 'modulus', but larger
             than the difference between consecutive
             trace values.
        plot_pref (bool):
            Whether or not to plot the trace.

    Returns:
        linearCounter (np.ndarray):
            1-D array of linearized counter values.
    '''

    if diff_thresh is None:
        diff_thresh = -modulus/2

    diff_trace = np.diff(np.double(trace))
    mod_times = np.where(diff_trace<diff_thresh)[0]


    mod_times_bool = np.zeros(len(trace))
    mod_times_bool[mod_times+1] = modulus_value
    mod_times_steps = np.cumsum(mod_times_bool)
    trace_times = (trace/modulus)*modulus_value + mod_times_steps

    if plot_pref:
        plt.figure()
        plt.plot(trace)
        plt.plot(mod_times , trace[mod_times] , 'o')

        plt.figure()
        plt.plot(mod_times_steps)
        plt.plot(trace_times)
    
    return trace_times


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


def make_batches(iterable, batch_size=None, num_batches=5, min_batch_size=0, return_idx=False):
    """
    Make batches of data or any other iterable.
    RH 2021

    Args:
        iterable (iterable):
            iterable to be batched
        batch_size (int):
            size of each batch
            if None, then batch_size based on num_batches
        num_batches (int):
            number of batches to make
        min_batch_size (int):
            minimum size of each batch
        return_idx (bool):
            whether to return the indices of the batches.
            output will be [start, end] idx
    
    Returns:
        output (iterable):
            batches of iterable
    """
    l = len(iterable)
    
    if batch_size is None:
        batch_size = np.int64(np.ceil(l / num_batches))
    
    for start in range(0, l, batch_size):
        end = min(start + batch_size, l)
        if (end-start) < min_batch_size:
            break
        else:
            if return_idx:
                yield iterable[start:end], [start, end]
            else:
                yield iterable[start:end]


@njit
def find_nearest(array, value):
    '''
    Finds the value and index of the nearest
     value in an array.
    RH 2021
    
    Args:
        array (np.ndarray):
            Array of values to search through.
        value (scalar):
            Value to search for.

    Returns:
        array_idx (int):
            Index of the nearest value in array.
        array_val (scalar):
            Value of the nearest value in array.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx] , idx
@njit(parallel=True)
def find_nearest_array(array, values):
    '''
    Finds the values and indices of the nearest
     values in an array.
    RH 2021

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
    '''
    vals_nearest = np.zeros_like(values)
    idx_nearest = np.zeros_like(values)
    for ii in prange(len(values)):
        vals_nearest[ii] , idx_nearest[ii] = find_nearest(array , values[ii])
    return vals_nearest, idx_nearest


def pad_with_singleton_dims(array, n_dims_pre=0, n_dims_post=0):
    arr_out = copy.copy(np.array(array))
    
    for n in range(n_dims_pre):
        arr_out = np.expand_dims(arr_out, 0)
    for n in range(n_dims_post):
        arr_out = np.expand_dims(arr_out, -1)
    return arr_out


class lazy_repeat_item():
    """
    Makes a lazy iterator that repeats an item.
     RH 2021
    """
    def __init__(self, item, pseudo_length=None):
        """
        Args:
            item (any object):
                item to repeat
            pseudo_length (int):
                length of the iterator.
        """
        self.item = item
        self.pseudo_length = pseudo_length

    def __getitem__(self, i):
        """
        Args:
            i (int):
                index of item to return.
                Ignored if pseudo_length is None.
        """
        if self.pseudo_length is None:
            return self.item
        elif i < self.pseudo_length:
            return self.item
        else:
            raise IndexError('Index out of bounds')


    def __len__(self):
        return self.pseudo_length

    def __repr__(self):
        return repr(self.item)


def index_with_nans(values, indices):
    """
    Creates an array of values with the same shape
     as indices, but with nans where indices are NaN.
    RH 2022

    Args:
        values (np.ndarray):
            values to index from
        indices (np.ndarray, dtype=float):
            indices to index into values

    Returns:
        output (np.ndarray):
            array of values indexed by indices
    """
    values = np.concatenate((np.array([np.nan]), values))
    indices += 1
    indices[np.isnan(indices)] = 0
    
    return values[indices.astype(np.int64)]


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
    

######################################
############ PURE PYTHON #############
######################################

def flatten_list(irregular_list):
    """
    Flattens a list of lists into a single list.
    Stolen from https://stackabuse.com/python-how-to-flatten-list-of-lists/
    RH 2022

    Args:
        irregular_list (list):
            list of lists to flatten

    Returns:
        output (list):
            flattened list
    """
    helper = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]
    return helper(irregular_list)


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    """
    Flattens a dictionary of dictionaries into a 
     single dictionary.
    Stolen from https://stackoverflow.com/a/6027615
    RH 2022

    Args:
        d (Dict):
            dictionary to flatten
        parent_key (str):
            key to prepend to flattened keys
            IGNORE: USED INTERNALLY FOR RECURSION
        sep (str):
            separator to use between keys
            IGNORE: USED INTERNALLY FOR RECURSION

    Returns:
        output (Dict):
            flattened dictionary
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def deep_update_dict(dictionary, key, val, in_place=False):
    """
    Updates a dictionary with a new value.
    RH 2022

    Args:
        dictionary (Dict):
            dictionary to update
        key (list of str):
            Key to update
            List elements should be strings.
            Each element should be a hierarchical
             level of the dictionary.
            DEMO:
                deep_update_dict(params, ['dataloader_kwargs', 'prefetch_factor'], val)
        val (any):
            Value to update with
        in_place (bool):
            whether to update in place

    Returns:
        output (Dict):
            updated dictionary
    """
    def helper_deep_update_dict(d, key, val):
        if type(key) is str:
            key = [key]

        assert key[0] in d, f"RH ERROR, key: '{key[0]}' is not found"

        if type(key) is list:
            if len(key) > 1:
                helper_deep_update_dict(d[key[0]], key[1:], val)
            elif len(key) == 1:
                key = key[0]
                d.update({key:val})

    if in_place:
        helper_deep_update_dict(dictionary, key, val)
    else:
        d = copy.deepcopy(dictionary)
        helper_deep_update_dict(d, key, val)
        return d


def dict_shared_items(d1, d2):
    """
    Returns the matching items between two dictionaries.
    RH 2022
    """
    return {k: d1[k] for k in d1 if k in d2 and d1[k] == d2[k]}
def dict_diff_items(d1, d2):
    """
    Returns the differing items between two dictionaries.
    RH 2022
    """    
    return {k: d1[k] for k in d1 if k in d2 and d1[k] != d2[k]}
def dict_missing_keys(d1, d2):
    """
    Returns the keys in d1 that are missing in d2
    RH 2022
    """    
    return {k for k,v in d1.items() if k not in d2}


def find_differences_across_dictionaries(dicts):
    """
    Finds differences across many dictionaries.
    RH 2022

    Args:
        dicts (List):
            List of dictionaries to compare.

    Returns:
        params_unchanging (list of dicts):
            List of dictionary items that are the 
             same across all dictionaries.
        params_changing (list of dicts):
            List of dictionary items that are 
             different in at least one dictionary.
    """
    def get_binary_search_combos(n):
        combos = list(np.arange(n))
        if len(combos)%2 == 1:
            combos.append(combos[-1])
        combos = np.array(combos).reshape(len(combos)//2, 2)
        return combos

    ## flatten params to ease matching functions
    params_flat = [flatten_dict(param) for param in dicts]

    ## find unchanging params
    params_unchanging = copy.deepcopy(params_flat)
    while len(params_unchanging) > 1:
        combos = get_binary_search_combos(len(params_unchanging))
        params_unchanging = [dict_shared_items(params_unchanging[combo[0]], params_unchanging[combo[1]]) for combo in combos]
    params_unchanging = params_unchanging[0]

    ## find keys that are not unchanging
    mk = dict_missing_keys(params_flat[0], params_unchanging)

    ## make list dicts of changing params
    params_changing = [{k: params[k] for k in mk} for params in params_flat]
    
    return params_unchanging, params_changing


#######################################
############ SPARSE STUFF #############
#######################################

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
    import torch
    import spconv
    import scipy.sparse

    coo = sparse.COO(
        coords=sp_arr.indices.T.to('cpu'),
        data=sp_arr.features.squeeze().to('cpu'),
        shape=[sp_arr.batch_size] + sp_arr.spatial_shape
    )
    return coo.reshape((coo.shape[0], -1)).to_scipy_sparse().tocsr()


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