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
