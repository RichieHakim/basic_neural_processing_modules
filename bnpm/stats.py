import numpy as np
import scipy.stats
import torch

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


def geometric_mean(a, axis=0, nan_policy="omit", zero_policy="propagate"):
    """
    Computes the geometric mean of an array of data.
    This is useful for computing the geometric mean of ratios.
    
    RH 2023

    Args:
        a (np.ndarray or torch.Tensor):
            Array of data.
        axis (int):
            Axis along which to compute the geometric mean.
        nan_policy (str):
            Defines how to handle nan values. \n
                * "omit" - Ignore nan values.
                * "propagate" - Propagate nan values.
                * "raise" - Raise an error if nan values are present.
        zero_policy (str):
            Defines how to handle zero values. \n
                * "raise" - Raise an error if zero values are present.
                * "propagate" - Propagate zero values.
                * "nan" - Replace zero values with nan.
    """
    if isinstance(a, (np.ndarray, list)):
        mean, nanmean, isnan, exp, log, nan = np.mean, np.nanmean, np.isnan, np.exp, np.log, np.nan
    elif isinstance(a, torch.Tensor):
        mean, nanmean, isnan, exp, log, nan = torch.mean, torch.nanmean, torch.isnan, torch.exp, torch.log, torch.nan
    else:
        raise ValueError("Data must be a numpy array or a torch tensor.")
    
    if zero_policy == "raise":
        if (a == 0).any():
            raise ValueError("Data contains zero values.")
    elif zero_policy == "nan":
        a[a == 0] = nan
    elif zero_policy == "propagate":
        pass

    if nan_policy == "omit":
        mean = nanmean
    elif nan_policy == "propagate":
        mean = mean
    elif nan_policy == "raise":
        mean = mean
        if isnan(a).any():
            raise ValueError("Data contains nan values.")
        
    return exp(mean(log(a), axis=axis))


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
    nan_policy="omit"
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
    nan_policy (str):
        Defines how to handle nan values. \n
            * "omit" - Ignore nan values.
            * "propagate" - Propagate nan values.
            * "raise" - Raise an error if nan values are present.

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

    ## Prepare functions for handling nan values
    if nan_policy == "omit":
        mean, std = np.nanmean, np.nanstd
    elif nan_policy == "propagate":
        mean, std = np.mean, np.std
    elif nan_policy == "raise":
        mean, std = np.mean, np.std
        if np.isnan(data).any():
            raise ValueError("Data contains nan values.")

    # Calculate the size along the specified axis
    n = data.shape[axis]

    # Compute the mean along the specified axis
    if error in ["std", "sem", "ci"]:
        m = mean(data, axis=axis)
    
        # Compute the standard deviation along the specified axis
        sd = std(data, axis=axis)
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


def zscore_to_pvalue(z, two_tailed=True):
    """
    Convert a z-score to a p-value.

    Args:
    z (float): 
        The z-score.
    two_tailed (bool): 
        If True, return a two-tailed p-value. If False, return a one-tailed
        p-value.

    Returns:
        float:
            The p-value.
    """
    if two_tailed:
        return 2 * scipy.stats.norm.sf(np.abs(z))
    else:
        return scipy.stats.norm.sf(np.abs(z))
    

def pvalue_to_zscore(p, two_tailed=True):
    """
    Convert a p-value to a z-score.

    Args:
    p (float): 
        The p-value.
    two_tailed (bool): 
        If True, the p-value is two-tailed. If False, the p-value is one-tailed.

    Returns:
        float:
            The z-score.
    """
    if two_tailed:
        return scipy.stats.norm.ppf(1 - p/2)
    else:
        return scipy.stats.norm.ppf(1 - p)
    

def multiTest_two_sample_independence(
    a,
    b,
) -> dict:
    """
    Perform multiple tests for two-sample independence.
    This function performs the following tests: \n
        * Two-sample t-test
            * equal variance, unequal variance
            * two-tailed, greater, less
        * Mann-Whitney U test
            * two-tailed, greater, less

    Args:
        a (np.ndarray or torch.Tensor): 
            The first samples.
        b (np.ndarray or torch.Tensor):
            The second samples.

    Returns:
        dict:
            A dictionary containing the results of the tests.
    """
    # Perform two-sample t-test
    ttest_ind = scipy.stats.ttest_ind(a, b, equal_var=True)
    ttest_ind_unequal = scipy.stats.ttest_ind(a, b, equal_var=False)
    ttest_ind_greater = scipy.stats.ttest_ind(a, b, alternative="greater")
    ttest_ind_unequal_greater = scipy.stats.ttest_ind(a, b, equal_var=False, alternative="greater")
    ttest_ind_less = scipy.stats.ttest_ind(a, b, alternative="less")
    ttest_ind_unequal_less = scipy.stats.ttest_ind(a, b, equal_var=False, alternative="less")

    # Perform Mann-Whitney U test
    mannwhitneyu = scipy.stats.mannwhitneyu(a, b, alternative="two-sided")
    mannwhitneyu_greater = scipy.stats.mannwhitneyu(a, b, alternative="greater")
    mannwhitneyu_less = scipy.stats.mannwhitneyu(a, b, alternative="less")

    return {
        "ttest_ind": ttest_ind,
        "ttest_ind_unequal": ttest_ind_unequal,
        "ttest_ind_greater": ttest_ind_greater,
        "ttest_ind_unequal_greater": ttest_ind_unequal_greater,
        "ttest_ind_less": ttest_ind_less,
        "ttest_ind_unequal_less": ttest_ind_unequal_less,
        "mannwhitneyu": mannwhitneyu,
        "mannwhitneyu_greater": mannwhitneyu_greater,
        "mannwhitneyu_less": mannwhitneyu_less,
    }