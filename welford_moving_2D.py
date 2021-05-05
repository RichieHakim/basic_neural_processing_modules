import numpy as np
from tqdm import trange

#################################################################################
######################### START OF HELPER FUNCTIONS #############################
#################################################################################


def update_varSum(idx_new, series, win_size, prevVal=None):
    """
    Returns the power sum average based on the blog post from
    Subliminal Messages.  Use the power sum average to help derive the running
    variance.
    sources: http://subluminal.wordpress.com/2008/07/31/running-standard-deviations/

    Keyword arguments:
    idx_new     --  current index or location of the value in the series
    series  --  list or tuple of data to average
    win_size  -- number of values to include in average
    prevVal    --  previous powersumavg (n - 1) of the series.
    """

    if win_size < 1:
        raise ValueError("period must be 1 or greater")

    if idx_new < 0:
        idx_new = 0

    if np.any(prevVal == None):
        if idx_new > 0:
            raise ValueError("pval of None invalid when bar > 0")

        prevVal = np.zeros(series.shape[1])

    newVal = np.double(series[idx_new,:])

    if idx_new < win_size:
        result = prevVal + (newVal * newVal - prevVal) / (idx_new + 1.0)

    else:
        oldVal = np.double(series[idx_new - win_size,:])
        result = prevVal + (((newVal * newVal) - (oldVal * oldVal)) / win_size)

    return result

def varSum_to_var(idx_new, series, win_size, mean_current, varSum):
    """
    Returns the running variance based on a given time period.
    sources: http://subluminal.wordpress.com/2008/07/31/running-standard-deviations/

    Keyword arguments:
    idx_new     --  current index or location of the value in the series
    series  --  list or tuple of data to average
    mean_current    --  current average of the given period
    varSum -- current powersumavg of the given period
    """
    if win_size < 1:
        raise ValueError("period must be 1 or greater")

    if idx_new <= 0:
        return 0.0

    if np.any(mean_current == None):
        raise ValueError("asma of None invalid when bar > 0")

    if np.any(varSum == None):
        raise ValueError("powsumavg of None invalid when bar > 0")

    windowsize = idx_new + 1.0
    if windowsize >= win_size:
        windowsize = win_size

    return (varSum * windowsize - windowsize * mean_current * mean_current) / windowsize

def running_mean(idx_new, series, mean_old):
    """
    Returns the cumulative or unweighted simple moving average.
    Avoids sum of series per call.

    Keyword arguments:
    idx_new     --  current index or location of the value in the series
    series  --  list or tuple of data to average
    mean_old  --  previous average (n - 1) of the series.
    """

    if idx_new <= 0:
        return series[0,:]

    return mean_old + ((series[idx_new,:] - mean_old) / (idx_new + 1.0))

def update_mean(idx_new, series, win_size, mean_old):
    """
    Returns the running simple moving average - avoids sum of series per call.

    Keyword arguments:
    idx_new     --  current index or location of the value in the series
    series  --  list or tuple of data to average
    win_size  --  number of values to include in average
    mean_old  --  previous simple moving average (n - 1) of the series
    """

    if win_size < 1:
        raise ValueError("period must be 1 or greater")

    if idx_new <= 0:
        return series[0,:]

    elif idx_new < win_size:
        return running_mean(idx_new, series, mean_old)

    return mean_old + ((series[idx_new,:] - series[idx_new - win_size,:]) / float(win_size))


#################################################################################
######################### END OF HELPER FUNCTIONS ###############################
###################### START OF PROCESSING FUNCTIONS ############################
#################################################################################


def make_rollingZScore(X , win_roll):

    X_mean_rolling = np.ones_like(X) * np.nan
    X_var_rolling = np.ones_like(X) * np.nan
    win_size_rollingBaseline = win_roll
    list_of_values = X
    varSum_old = None
    mean_old = None
    for idx in trange(len(list_of_values)):

        mean_new = update_mean(idx, list_of_values, win_size_rollingBaseline, mean_old)
        varSum_new = update_varSum(idx, list_of_values, win_size_rollingBaseline, varSum_old)
        var_new = varSum_to_var(idx, list_of_values, win_size_rollingBaseline, mean_new, varSum_new)

        X_mean_rolling[idx,:] = mean_new
        X_var_rolling[idx,:] = var_new

        mean_old = mean_new
        varSum_old = varSum_new

    eps = 1e-7
    X_mean_rolling[X_mean_rolling<eps] = eps
    X_zscore_roll = (list_of_values - X_mean_rolling)/np.sqrt(X_var_rolling)

    return X_zscore_roll