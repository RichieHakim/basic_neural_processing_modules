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
    from ScanImageTiffReader import ScanImageTiffReader

    vol=ScanImageTiffReader(path_to_tif)
    md = vol.metadata().split("\n")
    for ii, val in enumerate(md):
        if val[:25] == 'SI.hScan2D.channelOffsets':
            chan1_offset = int(val[29:33])

    print(f'{chan1_offset=}')
    return chan1_offset
def get_metadata(path_to_tif):
    from ScanImageTiffReader import ScanImageTiffReader
    
    vol=ScanImageTiffReader(path_to_tif)
    md = vol.metadata().split("\n")
    print(md)
    return md


####################################################
################ SUITE2p SPECIFIC ##################
####################################################

import pathlib
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

###############################################################################
############################## IMPORT STAT FILES ##############################
###############################################################################

def statFile_to_spatialFootprints(
    path_statFile=None, 
    statFile=None, 
    out_height_width=[36,36], 
    max_footprint_width=241, 
    plot_pref=True
    ):
    """
    Converts a stat file to a list of spatial footprint images.
    RH 2021

    Args:
        path_statFile (pathlib.Path or str):
            Path to the stat file.
            Optional: if statFile is provided, this
             argument is ignored.
        statFile (dict):
            Suite2p stat file dictionary
            Optional: if path_statFile is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.
    
    Returns:
        sf_all (list):
            List of spatial footprints images
    """
    assert out_height_width[0]%2 == 0 and out_height_width[1]%2 == 0 , "RH: 'out_height_width' must be list of 2 EVEN integers"
    assert max_footprint_width%2 != 0 , "RH: 'max_footprint_width' must be odd"
    if statFile is None:
        stat = np.load(path_statFile, allow_pickle=True)
    else:
        stat = statFile
    n_roi = stat.shape[0]
    
    # sf_big: 'spatial footprints' prior to cropping. sf is after cropping
    sf_big_width = max_footprint_width # make odd number
    sf_big_mid = sf_big_width // 2

    sf_big = np.zeros((n_roi, sf_big_width, sf_big_width))
    for ii in range(n_roi):
        sf_big[ii , stat[ii]['ypix'] - np.int16(stat[ii]['med'][0]) + sf_big_mid, stat[ii]['xpix'] - np.int16(stat[ii]['med'][1]) + sf_big_mid] = stat[ii]['lam'] # (dim0: ROI#) (dim1: y pix) (dim2: x pix)

    sf = sf_big[:,  
                sf_big_mid - out_height_width[0]//2:sf_big_mid + out_height_width[0]//2,
                sf_big_mid - out_height_width[1]//2:sf_big_mid + out_height_width[1]//2]
    if plot_pref:
        plt.figure()
        plt.imshow(np.max(sf, axis=0)**0.2)
        plt.title('spatial footprints cropped MIP^0.2')
    
    return sf

def import_multiple_stat_files(
    paths_statFiles=None, 
    dir_statFiles=None, 
    fileNames_statFiles=None, 
    out_height_width=[36,36], 
    max_footprint_width=241, 
    plot_pref=True
    ):
    """
    Imports multiple stat files.
    RH 2021 
    
    Args:
        paths_statFiles (list):
            List of paths to stat files.
            Elements can be either str or pathlib.Path.
        dir_statFiles (str):
            Directory of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        fileNames_statFiles (list):
            List of file names of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.

    Returns:
        stat_all (list):
            List of stat files.
    """
    if paths_statFiles is None:
        paths_statFiles = [pathlib.Path(dir_statFiles) / fileName for fileName in fileNames_statFiles]

    sf_all_list = [statFile_to_spatialFootprints(path_statFile=path_statFile,
                                                 out_height_width=out_height_width,
                                                 max_footprint_width=max_footprint_width,
                                                 plot_pref=plot_pref)
                  for path_statFile in paths_statFiles]
    return sf_all_list

def convert_multiple_stat_files(
    statFiles_list=None, 
    statFiles_dict=None, 
    out_height_width=[36,36], 
    max_footprint_width=241, 
    print_pref=False, 
    plot_pref=False
    ):
    """
    Converts multiple stat files to spatial footprints.
    RH 2021

    Args:
        statFiles_list (list):
            List of stat files.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.
    """
    if statFiles_dict is None:
        sf_all_list = [statFile_to_spatialFootprints(statFile=statFile,
                                                    out_height_width=out_height_width,
                                                    max_footprint_width=max_footprint_width,
                                                    plot_pref=plot_pref)
                    for statFile in statFiles_list]
    else:
        sf_all_list = []
        for key, stat in statFiles_dict.items():
            if print_pref:
                print(key)
            sf_all_list.append(statFile_to_spatialFootprints(statFile=stat,
                                                    out_height_width=out_height_width,
                                                    max_footprint_width=max_footprint_width,
                                                    plot_pref=plot_pref))
    return sf_all_list


def import_and_convert_to_CellReg_spatialFootprints(
    paths_statFiles, 
    frame_height=512, 
    frame_width=1024,
    dtype=np.uint8,
    ):
    """
    Imports and converts multiple stat files to spatial footprints
     suitable for CellReg.
    Output will be a list of arrays of shape (n_roi, height, width).
    RH 2022
    """

    isInt = np.issubdtype(dtype, np.integer)

    stats = [np.load(path, allow_pickle=True) for path in paths_statFiles]
    num_rois = [stat.size for stat in stats]
    sf_all_list = [np.zeros((n_roi, frame_height, frame_width), dtype) for n_roi in num_rois]
    for ii, stat in enumerate(stats):
        for jj, roi in enumerate(stat):
            if isInt:
                lam = dtype(roi['lam'] * np.iinfo(dtype).max)
            else:
                lam = roi['lam']
            sf_all_list[ii][jj, roi['ypix'], roi['xpix']] = lam
    return sf_all_list