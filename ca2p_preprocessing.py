'''
Table of Contents

Functions and Interdependencies:
    make_dFoF
        - timeSeries.percentile_numba
    calculate_noise_levels
    trace_quality_metrics
        - calculate_noise_levels
        - make_dFoF (outputs used as inputs to trace_quality_metrics)
'''

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from ScanImageTiffReader import ScanImageTiffReader

from varname import nameof

import time

from numba import jit, njit, prange

from .timeSeries import percentile_numba, var_numba, rolling_percentile_pd, rolling_percentile_rq_multicore, min_numba, max_numba


def make_dFoF(
    F, 
    Fneu=None, 
    neuropil_fraction=0.7, 
    percentile_baseline=30, 
    multicore_pref=False, 
    verbose=True):
    """
    calculates the dF/F and other signals. Designed for Suite2p data.
    If Fneu is left empty or =None, then no neuropil subtraction done.
    See S2p documentation for more details
    RH 2021
    
    Args:
        F (np.ndarray): 
            raw fluorescence values of each ROI. shape=(ROI#, time)
        Fneu (np.ndarray): 
            Neuropil signals corresponding to each ROI. dims match F.
        neuropil_fraction (float): 
            value, 0-1, of neuropil signal (Fneu) to be subtracted off of ROI signals (F)
        percentile_baseline (float/int): 
            value, 0-100, of percentile to be subtracted off from signals
        verbose (bool): 
            Whether you'd like printed updates
    Returns:
        dFoF (np.ndarray): 
            array, dF/F
        dF (np.ndarray): 
            array
        F_neuSub (np.ndarray): 
            F with neuropil subtracted
        F_baseline (np.ndarray): 
            1-D array of size F.shape[0]. Baseline value for each ROI
    """
    tic = time.time()

    if Fneu is None:
        F_neuSub = F
    else:
        F_neuSub = F - neuropil_fraction*Fneu

    if multicore_pref:
        F_baseline = percentile_numba(F_neuSub, percentile_baseline)
    else:
        F_baseline = np.percentile(F_neuSub , percentile_baseline , axis=1)
    dF = F_neuSub - F_baseline[:,None]
    dFoF = dF / F_baseline[:,None]
    if verbose:
        print(f'Calculated dFoF. Total elapsed time: {round(time.time() - tic,2)} seconds')
    
    return dFoF , dF , F_neuSub , F_baseline


@njit(parallel=True)
def calculate_noise_levels(dFoF, frame_rate):

    """"
    adapted from:  Peter Rupprecht (github: CASCADE by PTRRupprecht)
    
    Computes the noise levels for each neuron of the input matrix 'dFoF'.

    The noise level is computed as the median absolute dF/F difference
    between two subsequent time points. This is a outlier-robust measurement
    that converges to the simple standard deviation of the dF/F trace for
    uncorrelated and outlier-free dF/F traces.

    Afterwards, the value is divided by the square root of the frame rate
    in order to make it comparable across recordings with different frame rates.


    input: dFoF (matrix with n_neurons x time_points)
    output: vector of noise levels for all neurons

    """

    def abs_numba(X):
        Y = np.zeros_like(X)
        for ii in prange(len(X)):
            Y[ii] = abs(X[ii])
        return Y
    noise_levels = np.zeros(dFoF.shape[0])
    for ii in prange(dFoF.shape[0]):
        noise_levels[ii] = np.median(abs_numba(np.diff(dFoF[ii,:])))

    # noise_levels = np.median(np.abs(np.diff(dFoF, axis=1)), axis=1)
    noise_levels = noise_levels / np.sqrt(frame_rate) * 100    # scale noise levels to percent
    return noise_levels

def trace_quality_metrics(F, Fneu, dFoF, dF, F_neuSub, F_baseline,
                        percentile_baseline=30, Fs=30,
                        plot_pref=True, thresh=None):
    '''
    Some simple quality metrics for calcium imaging traces. Designed to
    work with Suite2p's outputs (F, Fneu) and the make_dFoF function
    above.
    RH 2021

    Args:
        F: 
            Fluorescence traces. 
            From S2p. shape=[Neurons, Time]
        Fneu: 
            Fluorescence Neuropil traces. 
            From S2p. shape=[Neurons, Time]
        dFoF:
            Normalized changes in fluorescence ('dF/F').
            From 'make_dFoF' above.
            ((F-Fneu) - F_base) / F_base . Where F_base is
            something like percentile((F-Fneu), 30)
        dF:
            Changes in fluorescence.
            From 'make_dFoF' above. ((F-Fneu) - F_base)
        F_neuSub:
            Neuropil subtracted fluorescence.
            From 'make_dFoF' above
        F_baseline:
            currently unused.
            From 'make_dFoF' above
        percentile_baseline:
            percentile to use as 'baseline'/'quiescence'.
            Use same value as in 'make_dFoF' above
        Fs:
            Framerate of imaging
    
    Returns:
        tqm: dict with the following fields:
            metrics:
                quality_metrics. Dict of all the 
                relevant output variables
            thresh:
                Some hardcoded thresholds for aboslute 
                cutoffs
            sign:
                Whether the thresholds for exclusion in tqm_thresh
                should be positive (>) or negative (<). Multiply
                by tqm_thresh to just do everything as (>).
        good_ROIs:
            ROIs that did not meet the exclusion creteria
    '''
    var_F = var_numba(F)
    var_Fneu = var_numba(Fneu)
    var_FneuSub = var_numba(F_neuSub)

    var_F[var_F==0] = np.nan
    var_ratio = var_Fneu / var_F

    EV_F_by_Fneu = 1 - (var_FneuSub / var_F)

    base_FneuSub = percentile_numba(F_neuSub, percentile_baseline)
    base_F = percentile_numba(F, percentile_baseline)
    
    noise_levels = calculate_noise_levels(dFoF, Fs)
    # noise_levels = np.median(np.abs(np.diff(dFoF, axis=1)), axis=1) # Use this line of code if numba is acting up
    noise_levels[np.abs(noise_levels) > 1e3] = np.nan

    max_dFoF = np.max(dFoF, axis=1)

    # currently hardcoding the rolling baseline window to be 10 minutes
    # rolling_baseline = rolling_percentile_pd(dFoF, ptile=percentile_baseline, window=int(Fs*60*2 + 1))
    rolling_baseline = rolling_percentile_rq_multicore(dFoF, ptile=percentile_baseline, window=int(Fs*60*10 + 1))
    baseline_var = var_numba(rolling_baseline)
    # baseline_range = max_numba(rolling_baseline) - min_numba(rolling_baseline)

    metrics = {
        'var_ratio': var_ratio,
        'EV_F_by_Fneu': EV_F_by_Fneu,
        'base_FneuSub': base_FneuSub,
        'base_F': base_F,
        'noise_levels': noise_levels,
        'max_dFoF': max_dFoF,
        'baseline_var': baseline_var,
    }

    # ############# HARD-CODED exclusion criteria ###############
    if thresh is None:
        thresh = {
                    'var_ratio': 1,
                    'EV_F_by_Fneu': 0.6,
                    'base_FneuSub': 0,
                    'base_F': 50,
                    'noise_levels': 12,
                    'max_dFoF': 50,
                    'baseline_var': 1,
                }
    # thresh = {
    # 'var_ratio': 3,
    # 'EV_F_by_Fneu': 1,
    # 'base_FneuSub': -1000,
    # 'base_F': -1000,
    # 'noise_levels': 500,
    # 'max_dFoF': 3000,
    # 'baseline_var': 1,
    # }

    sign = {
    'var_ratio': 1,
    'EV_F_by_Fneu': 1,
    'base_FneuSub': -1,
    'base_F': -1,
    'noise_levels': 1,
    'max_dFoF': 1,
    'baseline_var': 1,
    }

    # Exclude ROIs
    n_ROIs = len(list(metrics.values())[0])
    good_ROIs = np.ones(n_ROIs)
    classifications = dict()
    for ii, val in enumerate(metrics):
        if sign[val]==1:
            to_exclude = (metrics[val] > thresh[val]) + np.isnan(metrics[val]) # note that if NaN then excluded
        if sign[val]==-1:
            to_exclude = (metrics[val] < thresh[val]) + np.isnan(metrics[val])

        classifications[val] = 1-to_exclude
        good_ROIs[to_exclude] = 0

    # drop everything into a dict
    tqm = {
        "metrics": metrics,
        "thresh": thresh,
        "sign": sign,
        "classifications": classifications,
    }

    # plot
    if plot_pref:
        fig, axs = plt.subplots(len(tqm['metrics']), figsize=(7,10))
        for ii, val in enumerate(tqm['metrics']):
            if val=='noise_levels':
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==1)[0]], 300, histtype='step')
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==0)[0]], 300, histtype='step')
                axs[ii].set_xlim([0,20])
            elif val=='baseline_var':
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==1)[0]], 300, histtype='step')
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==0)[0]], 300, histtype='step')
                axs[ii].set_xlim(right=50)
                # axs[ii].set_xscale('log')
            else:
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==1)[0]], 300, histtype='step')
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==0)[0]], 300, histtype='step')

            axs[ii].title.set_text(f"{val}: {np.sum(tqm['classifications'][val]==0)} excl")
            axs[ii].set_yscale('log')

            axs[ii].plot(np.array([tqm['thresh'][val],tqm['thresh'][val]])  ,  np.array([0,100]), 'k')
        fig.legend(('thresh', 'included','excluded'))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.figure()
        plt.plot(good_ROIs)
        plt.plot(scipy.signal.savgol_filter(good_ROIs, 101,1))

    print(f'ROIs excluded: {int(np.sum(1-good_ROIs))} / {len(good_ROIs)}')
    print(f'ROIs included: {int(np.sum(good_ROIs))} / {len(good_ROIs)}')

    good_ROIs = np.array(good_ROIs, dtype=bool)
    return tqm, good_ROIs


def get_chan1_offset(path_to_tif):
    vol=ScanImageTiffReader(path_to_tif)
    md = vol.metadata().split("\n")
    for ii, val in enumerate(md):
        if val[:25] == 'SI.hScan2D.channelOffsets':
            chan1_offset = int(val[29:33])

    print(f'{chan1_offset=}')
    return chan1_offset
def get_metadata(path_to_tif):
    vol=ScanImageTiffReader(path_to_tif)
    md = vol.metadata().split("\n")
    print(md)
    return md