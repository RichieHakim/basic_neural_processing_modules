import numpy as np
import torch
import matplotlib.pyplot as plt

def gaussian(x=None, mu=0, sig=1, plot_pref=False):
    '''
    A gaussian function (normalized similarly to scipy's function)
    RH 2021
    
    Args:
        x (np.ndarray): 1-D array of the x-axis of the kernel
        mu (float): center position on x-axis
        sig (float): standard deviation (sigma) of gaussian
        plot_pref (boolean): True/False or 1/0. Whether you'd like the kernel plotted
        
    Returns:
        gaus (np.ndarray): gaussian function (normalized) of x
        params_gaus (dict): dictionary containing the input params
    '''
    if x is None:
        x = np.linspace(-sig*5, sig*5, int(sig*7), endpoint=True)

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp((-((x-mu)/sig) **2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'$\mu$={mu}, $\sigma$={sig}')

    return gaus


def generalized_logistic_function(
    x, 
    a=0, 
    k=1, 
    b=1, 
    v=1, 
    q=1, 
    c=1,
    mu=0,
    ):
    '''
    Generalized logistic function
    See: https://en.wikipedia.org/wiki/Generalised_logistic_function
     for parameters and details
    RH 2021

    Args:
        a: the lower asymptote
        k: the upper asymptote when C=1
        b: the growth rate
        v: > 0, affects near which asymptote maximum growth occurs
        q: is related to the value Y (0). Center positions
        c: typically takes a value of 1
        mu: the center position of the function

    Returns:
        output:
            Logistic function
     '''
    return a + (k-a) / (c + q*np.exp(-b*(x-mu)))**(1/v)


def bounded_logspace(start, stop, num,):
    """
    Like np.logspace, but with a defined start and
     stop.
    NOTE: numpy.geomspace now has this functionality.
    RH 2022
    
    Args:
        start (float):
            First value in output array
        stop (float):
            Last value in output array
        num (int):
            Number of values in output array
            
    Returns:
        output (np.ndarray):
            Array of values
    """

    exp = 2  ## doesn't matter what this is, just needs to be > 1

    return exp ** np.linspace(np.log(start)/np.log(exp), np.log(stop)/np.log(exp), num, endpoint=True)


def polar2cartesian(mag, angle):
    """
    Converts a polar coordinates to cartesian coordinates
    RH 2021

    Args:
        mag (float or np.ndarray or torch.Tensor):
            Magnitude of the polar coordinates
        angle (float or np.ndarray or torch.Tensor):
            Angle of the polar coordinates
    
    Returns:
        output (float or np.ndarray or torch.Tensor):
    """
    if type(mag) is torch.Tensor:
        exp = torch.exp
    else:
        exp = np.exp
    return mag * exp(1j*angle)

def cartesian2polar(x):
    """
    Converts a cartesian coordinates to polar coordinates
    RH 2021

    Args:
        x (float or np.ndarray or torch.Tensor):
            Cartesian coordinates
        
    Returns:
        Magnitude (float or np.ndarray or torch.Tensor):
            Magnitude of the polar coordinates
        Angle (float or np.ndarray or torch.Tensor):
            Angle of the polar coordinates
    """
    if type(x) is torch.Tensor:
        abs, angle = torch.abs, torch.angle
    else:
        abs, angle = np.abs, np.angle
    return abs(x), angle(x)


def angle2rotmat(angle):
    """
    Converts an angle to a 2D rotation matrix
    RH 2024

    Args:
        angle (float or np.ndarray or torch.Tensor):
            Angle of rotation
        
    Returns:
        output (np.ndarray or torch.Tensor):
            2D rotation matrix
    """
    if type(angle) is torch.Tensor:
        cos, sin = torch.cos, torch.sin
        wrapper = torch.as_tensor
    else:
        cos, sin = np.cos, np.sin
        wrapper = np.array
    return wrapper([[(c:=cos(angle)), -(s:=sin(angle))], [s, c]])


def make_correlated_distributions_2D(means, stds, corrs, n_points_per_mode):
    """
    Makes correlated noisey distributions in 2D.
    RH 2022
    
    Args:
        means:
            List of lists.
            outer list: each mode
            inner list: means of each mode (2 entries)
        stds:
            List of lists.
            outer list: each mode
            inner list: stds of each mode (2 entries)
        corrs:
            List: correlations of each mode
        n_points_per_mode:
            List: number of points in each mode
            
    Returns:
        dist:
            The output data with all the distributions concatenated
    """
    for ii, (mean, std, corr) in enumerate(zip(means, stds, corrs)):
        cov = [
            [std[0]**2, std[0]*std[1]*corr],
            [std[0]*std[1]*corr, std[1]**2]
              ]
        if ii == 0:
            dist = np.random.multivariate_normal(mean, cov, n_points_per_mode[ii])
        else:
            dist = np.vstack((dist, np.random.multivariate_normal(mean, cov, n_points_per_mode[ii])))
            
    return dist


def linex(x, mu=1, a=1, b=1, c=1, d=1, e=1, f=1, g=1):
    """
    Linex (loss) function. 'Linear - exponential'.
    The curve below mu is linear, and the curve
     above mu is exponential.
    simple version: exp(x) - a*x - 1
    RH 2022

    Args:
        x (float or np.ndarray or torch.Tensor):
            Input data
        mu (float):
            Center of the linex function
        a (float):
            'Slope' or 'Stregth' of the linex function
        b (float):
            Non-linearity of the linex function
        c (float):
            Multiplier for the linex function

    Returns:
        Linex function output (float or np.ndarray or torch.Tensor)
    """
    # return c*(torch.exp(a*(x**d-mu**(d))) - a*(x**d-mu**(d)) - 1)**b
    return a*(b*(torch.exp(c*(x-mu))**d) - e*(x-mu)**f - 1)**g
    # return ((a*(x-mu)**c) - a*(x-mu)**(d) - 1)**b


def szudzik_encode(a, b):
    """
    Szudzik's function for encoding two integers into one.
    Output is a unique hash for each pair of integers.
    RH 2023

    Args:
        a (int or np.ndarray or torch.Tensor):
            First integer array.
            Must be non-negative.
        b (int or np.ndarray or torch.Tensor):
            Second integer array
            Must be non-negative.

    Returns:
        output (int or np.ndarray or torch.Tensor):
            Encoded integer array
    """

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        zeros_like = np.zeros_like
        dtype = np.int64
        assert np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer), "a and b must be integer types"
        assert np.all(a >= 0) and np.all(b >= 0), "a and b must be non-negative"
        a = a.astype(np.int64)
        b = b.astype(np.int64)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        zeros_like = torch.zeros_like
        dtype = torch.int64
        assert a.dtype.is_floating_point==False and b.dtype.is_floating_point==False, "a and b must be integer types"
        assert torch.all(a >= 0) and torch.all(b >= 0), "a and b must be non-negative"
        a = a.type(torch.int64)
        b = b.type(torch.int64)
    else:
        raise TypeError("a and b must be both np.ndarray or both torch.Tensor")
    
    assert a.shape == b.shape, "a and b must have the same shape"

    m = a >= b  ## mask for when a >= b
    target = zeros_like(a, dtype=dtype)
    target[m] = a[m] * a[m] + a[m] + b[m]
    target[~m] = a[~m] + b[~m] * b[~m]
    return target

def szudzik_decode(z):
    """
    Szudzik's function for decoding one integer into two.
    Output is a unique hash pair for each input integer.
    RH 2023

    Args:
        z (int or np.ndarray or torch.Tensor):
            Encoded integer array

    Returns:
        a (int or np.ndarray or torch.Tensor):
            First integer array.
            Must be non-negative.
        b (int or np.ndarray or torch.Tensor):
            Second integer array
            Must be non-negative.
    """

    if isinstance(z, np.ndarray):
        zeros_like = np.zeros_like
        dtype = np.int64
        floor = np.floor
        sqrt = np.sqrt
        assert np.issubdtype(z.dtype, np.integer), "z must be integer type"
    elif isinstance(z, torch.Tensor):
        zeros_like = torch.zeros_like
        dtype = torch.int64
        floor = torch.floor
        sqrt = torch.sqrt
        assert z.dtype.is_floating_point==False, "z must be integer type"
    else:
        raise TypeError("z must be np.ndarray or torch.Tensor")
    
    w = floor(sqrt(z))
    w = w.astype(dtype) if isinstance(w, np.ndarray) else w.type(dtype)

    a = zeros_like(z, dtype=dtype)
    b = zeros_like(z, dtype=dtype)
    m = z >= w * w + w
    a[m] = w[m]
    b[m] = z[m] - w[m] * w[m] - w[m]
    a[~m] = z[~m] - w[~m] * w[~m]
    b[~m] = w[~m]
    return a, b


def rand_log(low=1, high=100, size=(1,)):
    """
    Generate random numbers from a log-uniform distribution.
    RH 2023

    Args:
        low (float):
            Lower bound of the distribution
        high (float):
            Upper bound of the distribution
        size (tuple):
            Shape of the output array

    Returns:
        output (np.ndarray):
            Random numbers from the log-uniform distribution
    """
    if isinstance(size, int):
        size = (size,)
    return low*10**(np.random.rand(*size)*(np.log(high/low) / np.log(10)))


def make_odd(n, mode='up'):
    """
    Make a number odd.
    RH 2023

    Args:
        n (int):
            Number to make odd
        mode (str):
            'up' or 'down'
            Whether to round up or down to the nearest odd number

    Returns:
        output (int):
            Odd number
    """
    if n % 2 == 0:
        if mode == 'up':
            return n + 1
        elif mode == 'down':
            return n - 1
        else:
            raise ValueError("mode must be 'up' or 'down'")
    else:
        return n
def make_even(n, mode='up'):
    """
    Make a number even.
    RH 2023

    Args:
        n (int):
            Number to make even
        mode (str):
            'up' or 'down'
            Whether to round up or down to the nearest even number

    Returns:
        output (int):
            Even number
    """
    if n % 2 != 0:
        if mode == 'up':
            return n + 1
        elif mode == 'down':
            return n - 1
        else:
            raise ValueError("mode must be 'up' or 'down'")
    else:
        return n