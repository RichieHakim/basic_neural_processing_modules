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
        x = np.linspace(-sig*5, sig*5, sig*7, endpoint=True)

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp((-((x-mu)/sig) **2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'$\mu$={mu}, $\sigma$={sig}')

    return gaus


def generalised_logistic_function(x, a=0, k=1, b=1, v=1, q=1, c=1):
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

    Returns:
        output:
            Logistic function
     '''
    return a + (k-a) / (c + q*np.exp(-b*x))**(1/v)


# def bounded_exponential(x, bounds=[1/10,10], base=2):
#     """
#     Bounded exponential function
#     Computes an exponential function where when
#      x is 0, the output is bounds[0], and when
#      x is 1, the output is bounds[1]. The relative
#      probability of outputting bounds[0[ over bounds[1]
#      is base.
#     Useful for randomly sampling over large ranges of
#      values with an exponential resolution.
#     RH 2021

#     Args:
#         x (float or np.ndarray): 
#             Float or 1-D array of the x-axis
#         bounds (list):
#             List of two floats, the lower and upper
#              bounds
#         base (float):  
#             The relative probability of outputting
#              bounds[0] over bounds[1]
    
#     Returns:
#         output (float or np.ndarray):
#             The bounded exponential output
#     """
    
#     range_additive = bounds[1] - bounds[0]

#     return (((base**x - 1)/(base-1)) * range_additive) + bounds[0]

def bounded_logspace(start, stop, num,):
    """
    Like np.logspace, but with a defined start and
     stop.
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


def polar2real(mag, angle):
    """
    Converts a polar coordinates to real coordinates
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

def real2polar(x):
    """
    Converts a real coordinates to polar coordinates
    RH 2021

    Args:
        x (float or np.ndarray or torch.Tensor):
            Real coordinates
        
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


def Linex(x, mu=1, a=1, b=1, c=1):
    """
    Linex (loss) function.
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
    return c*(torch.exp(a*(x-mu)) - a*(x-mu) - 1)**b