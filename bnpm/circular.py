from typing import Union, Optional
import functools

import numpy as np
import torch
import matplotlib.pyplot as plt


from . import misc

"""
This module implements some modular arithmetic functions for circular data. For
circular stats, use pycircstat or pycircular.
"""


def _circ_operator(a, period: float = 2*np.pi):
    """
    Helper function for circular arithmetic. \n 
    This function is used to ensure that the output of a circular operation is
    always within the range [0, period). \n 
    RH 2024

    Args:
        a (float or np.ndarray or torch.Tensor):
            Input value
        period (float or np.ndarray or torch.Tensor):
            Period of the circle

    Returns:
        output (float or np.ndarray or torch.Tensor):
            Output value
    """
    return ((a + period/2) % period) - period/2
    

def circ_subtract(a, b, period: float = 2*np.pi):
    """
    Modular subtraction of ``b`` from ``a``. \n 
    This is equivalent to the distance from ``a`` to ``b`` on a circle. \n 
    RH 2024

    Args:
        a (float or np.ndarray or torch.Tensor):
            Minuend
        b (float or np.ndarray or torch.Tensor):
            Subtrahend
        period (float or np.ndarray or torch.Tensor):
            Period of the circle

    Returns:
        output (float or np.ndarray or torch.Tensor):
            Difference
    """
    return _circ_operator(a - b, period=period)


def circ_add(a, b, period: float = 2*np.pi):
    """
    Modular addition of ``a`` and ``b``. \n 
    This is equivalent to the sum of ``a`` and ``b`` on a circle. \n 
    RH 2024

    Args:
        a (float or np.ndarray or torch.Tensor):
            First addend
        b (float or np.ndarray or torch.Tensor):
            Second addend
        period (float or np.ndarray or torch.Tensor):
            Period of the circle

    Returns:
        output (float or np.ndarray or torch.Tensor):
            Sum
    """
    return _circ_operator(a + b, period=period)

@misc.wrapper_flexible_args(['dim', 'axis'])
def circ_diff(
    arr: Union[np.ndarray, torch.Tensor],
    period: Union[float, np.ndarray, torch.Tensor] = 2*np.pi,
    axis: int = -1,
    prepend: int = None,
    append: int = None,
    n: int = 1
):
    """
    Modular derivative (like np.diff) of an array. \n 
    Calculates the circular difference between adjacent elements of ``arr``. \n 
    RH 2024

    Args:
        arr (np.ndarray or torch.Tensor):
            Input array
        period (float or np.ndarray or torch.Tensor):
            Period of the circle
        axis (int):
            Axis along which to take the difference
        prepend (int):
            Number of elements to prepend to the result
        append (int):
            Number of elements to append to the result
        n (int):
            Number of times to take the difference

    Returns:

    """
    if isinstance(arr, np.ndarray):
        prepend = np._NoValue if prepend is None else prepend
        append = np._NoValue if append is None else append
        diff = functools.partial(np.diff, axis=axis, prepend=prepend, append=append)
    elif isinstance(arr, torch.Tensor):
        diff = functools.partial(torch.diff, dim=axis, prepend=prepend, append=append)
    else:
        raise TypeError("arr must be either np.ndarray or torch.Tensor")
    
    for _ in range(n):
        arr = diff(arr)
        arr = _circ_operator(arr, period=period)
    return arr


def moduloCounter_to_linearCounter(trace, modulus, modulus_value=None, diff_thresh=None, plot_pref=False):
    '''
    Converts a (sawtooth) trace of modulo counter values to a linear counter. \n
    Useful for converting a clock with a modulus to total times. \n
    The function basically just finds where the modulus events occur in the
    trace and adds 'modulus_value' to the next element in the trace. \n
    RH 2021

    Args:
        trace (np.ndarray):
            1-D array of modulo counter values.
        modulus (scalar):
            Modulus of the counter. Values in trace should range from 0 to
            modulus-1.
        modulus_value (scalar):
            Multiplier for the modulus counter. The value of a modulus event. If
            None, then modulus_value is set to ``modulus``.
        diff_thresh (scalar):
            Threshold in change between consecutive elements for defining a
            modulus event. \n
            Should typically be a negative value smaller than 'modulus', but
            larger than the variances between consecutive trace values. If None,
            then modulus_value is set to ``-modulus/2``.
        plot_pref (bool):
            Whether or not to plot the trace.

    Returns:
        linearCounter (np.ndarray):
            1-D array of linearized counter values.
    '''

    if diff_thresh is None:
        diff_thresh = -modulus/2

    if modulus_value is None:
        modulus_value = modulus

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
