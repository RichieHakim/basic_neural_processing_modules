import numpy as np
from numba import jit, njit, prange
import copy
import scipy.signal
import matplotlib.pyplot as plt


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
def idx2bool(idx, length):
    '''
    Converts a vector of indices to a boolean vector.
    RH 2021

    Args:
        idx (np.ndarray):
            1-D array of indices.
        length (int):
            Length of boolean vector.
    
    Returns:
        bool_vec (np.ndarray):
            1-D boolean array.
    '''
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
    print(nz.size)
    if nz.size==0:
        output = len(input_array)-1
    else:
        output = np.max(nz)
#     print(output)
    return output


def make_batches(iterable, batch_size=None, num_batches=5, min_batch_size=0):
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


def pad_with_singleton_dims(array, n_dims):
    arr_out = copy.copy(array)
    while arr_out.ndim < n_dims:
        arr_out = np.expand_dims(arr_out, -1)
    return arr_out