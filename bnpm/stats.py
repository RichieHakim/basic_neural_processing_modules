import numpy as np
import scipy.stats

def ttest_paired_ratio(a, b):
    """
    Computes the ratio paired t-test between two arrays of data.
    This is useful for comparing ratios of data, and can be
     thought of as a geometric mean paired t-test.
    The ratio t-test takes the logarithm of the ratio of
     a vs. b and then tests the null hypothesis that the
     mean is zero.
    
    RH 2023

    Args:
        a (np.ndarray or torch.Tensor):
            Array of data. Paired 1-to-1 with b.
        b (np.ndarray or torch.Tensor):
            Array of data. Paired 1-to-1 with a.
    """
    # Compute ratio
    ratio = a/b
    # Compute log of ratio
    log_ratio = np.log(ratio)
    # Compute mean of log ratio
    mean_log_ratio = np.mean(log_ratio)
    # Compute standard error of log ratio
    stderr_log_ratio = scipy.stats.sem(log_ratio)
    # Compute t-statistic
    t_stat = mean_log_ratio/stderr_log_ratio
    # Compute p-value
    p_val = scipy.stats.t.sf(np.abs(t_stat), len(a)-1)*2

    ## above is equivalent to:
    # t_stat, p_val = scipy.stats.ttest_rel(np.log(a/b), np.zeros_like(a))
    return p_val


def geometric_mean(a):
    """
    Computes the geometric mean of an array of data.
    This is useful for computing the geometric mean of ratios.
    
    RH 2023

    Args:
        a (np.ndarray or torch.Tensor):
            Array of data.
    """
    return np.exp(np.mean(np.log(a)))


def sparsity(a, axis=0):
    """
    A normalized dispersion index. Varies from 0 (dense) to 1 (sparse).
    Computes the lifetime sparseness of an array of data.
    From Vinje & Gallant 2000; and Willmore, Mazer, & Gallant 2011.
    
    RH 2023

    Args:
        a (np.ndarray or torch.Tensor):
            Array of data.
        axis (int):
            Axis along which to compute sparsity.

    Returns:
        float: Sparsity value. 0 (dense) to 1 (sparse).
    """
    return 1 - ((a.mean(axis) ** 2) / (a ** 2).mean(axis))


def error_interval(
    data, 
    axis=0,
    error="sem",
    confidence=0.95, 
):
    """
    Calculate the mean and the confidence interval of the mean.

    Args:
    data (np.ndarray): 
        The input data. Can be a list or a numpy array.
    axis (int): 
        Axis along which to compute the mean and confidence interval. Default is 0.
    error (str):
        Type of error to compute. \n
            * "std" - Standard deviation.
            * "sem" - Standard error of the mean.
            * "ci" - Confidence interval. \n
    confidence (float): 
        The confidence level for the interval. Only used if error is "ci".

    Returns:
    tuple:
        mean (float): 
            The mean of the data along the specified axis.
        lower (float):
            The lower bound of the confidence interval.
        upper (float):
            The upper bound of the confidence interval.
    """

    assert confidence > 0 and confidence < 1, "Confidence level must be between 0 and 1."
    assert isinstance(data, np.ndarray), "Data must be a numpy array."

    # Calculate the size along the specified axis
    n = data.shape[axis]

    # Compute the mean along the specified axis

    if error in ["std", "sem", "ci"]:
        m = np.mean(data, axis=axis)
    
        # Compute the standard deviation along the specified axis
        sd = np.std(data, axis=axis)
        if error in ["sem", "ci"]:
            # Compute the standard error of the mean along the specified axis
            se = sd / np.sqrt(n)
        if error == "ci":
            # Compute the margin of error
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            mean, lower, upper = m, m-h, m+h
        if error == "sem":
            mean, lower, upper = m, m-se, m+se
        if error == "std":
            mean, lower, upper = m, m-sd, m+sd
    else:
        raise ValueError("Error must be one of 'std', 'sem', or 'ci'.")
    
    return mean, lower, upper