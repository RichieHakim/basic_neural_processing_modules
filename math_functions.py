import numpy as np
import torch
import matplotlib.pyplot as plt

def gaussian(x, mu, sig , plot_pref=False):
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

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp((-((x-mu)/sig) **2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'$\mu$={mu}, $\sigma$={sig}')
    
    params_gaus = {
        "x": x,
        "mu": mu,
        "sig": sig,
    }

    return gaus , params_gaus


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


def bounded_exponential(x, bounds=[1/10,10], base=2):
    """
    Bounded exponential function
    Computes an exponential function where when
     x is 0, the output is bounds[0], and when
     x is 1, the output is bounds[1]. The relative
     probability of outputting bounds[0[ over bounds[1]
     is base.
    Useful for randomly sampling over large ranges of
     values with an exponential resolution.
    RH 2021

    Args:
        x (float or np.ndarray): 
            Float or 1-D array of the x-axis
        bounds (list):
            List of two floats, the lower and upper
             bounds
        base (float):  
            The relative probability of outputting
             bounds[0] over bounds[1]
    
    Returns:
        output (float or np.ndarray):
            The bounded exponential output
    """
    
    range_additive = bounds[1] - bounds[0]

    return (((base**x - 1)/(base-1)) * range_additive) + bounds[0]