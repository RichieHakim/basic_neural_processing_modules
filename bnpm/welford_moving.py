import numpy as np
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
    pval    --  previous powersumavg (n - 1) of the series.
    """

    if win_size < 1:
        raise ValueError("period must be 1 or greater")

    if idx_new < 0:
        idx_new = 0

    if prevVal == None:
        if idx_new > 0:
            raise ValueError("pval of None invalid when bar > 0")

        prevVal = 0.0

    newVal = float(series[idx_new])

    if idx_new < win_size:
        result = prevVal + (newVal * newVal - prevVal) / (idx_new + 1.0)

    else:
        oldVal = float(series[idx_new - win_size])
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

    if mean_current == None:
        raise ValueError("asma of None invalid when bar > 0")

    if varSum == None:
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
        return series[0]

    return mean_old + ((series[idx_new] - mean_old) / (idx_new + 1.0))

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
        return series[0]

    elif idx_new < win_size:
        return running_mean(idx_new, series, mean_old)

    return mean_old + ((series[idx_new] - series[idx_new - win_size]) / float(win_size))
