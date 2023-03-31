import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import time
import multiprocessing as mp
import gc

from numba import jit, njit, prange

from pathlib import Path

# from .timeSeries import percentile_numba, var_numba, rolling_percentile_pd, rolling_percentile_rq_multicore, min_numba, max_numba


def make_dFoF(
    F, 
    Fneu=None, 
    neuropil_fraction=0.7, 
    percentile_baseline=30, 
    rolling_percentile_window=None,
    roll_centered=True,
    roll_stride=1,
    roll_interpolation='linear',
    channelOffset_correction=0,
    multicore_pref=False, 
    verbose=True,
    ):
    """
    Calculates the dF/F and other signals. Designed for Suite2p data.
    If Fneu is left empty or =None, then no neuropil subtraction done.
    See S2p documentation for more details
    RH 2021-2023
    
    Args:
        F (np.ndarray): 
            raw fluorescence values of each ROI. shape=(ROI#, time)
        Fneu (np.ndarray): 
            Neuropil signals corresponding to each ROI. dims match F.
        neuropil_fraction (float): 
            value, 0-1, of neuropil signal (Fneu) to be subtracted off of ROI signals (F)
        percentile_baseline (float/int): 
            value, 0-100, of percentile to be subtracted off from signals
        rolling_percentile_window (int):
            window size for rolling percentile. 
            NOTE: this value will be divided by stride_roll.
            If None, then a single percentile is calculated for the entire trace.
        roll_centered (bool):
            If True, then the rolling percentile is calculated with a centered window.
            If False, then the rolling percentile is calculated with a trailing window
             where the right edge of the window is the current timepoint.
        roll_stride (int):
            Stride for rolling percentile.
            NOTE: rolling_percentile_window will be divided by this value.
            If 1, then the rolling percentile is calculated
             at every timepoint. If 2, then the rolling percentile is calculated at every
             other timepoint, etc.
        roll_interpolation (str):
            Interpolation method for rolling percentile.
            Options: 'linear', 'nearest'
            See pandas.DataFrame.rolling for more details
        channelOffset_correction (float):
            value to be added to F and Fneu to correct for channel offset
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
    from .timeSeries import percentile_numba

    tic = time.time()

    roll_stride = int(roll_stride)

    F = torch.as_tensor(F, dtype=torch.float32) + channelOffset_correction
    Fneu = torch.as_tensor(Fneu, dtype=torch.float32) + channelOffset_correction if Fneu is not None else 0

    if Fneu is None:
        F_neuSub = F
    else:
        F_neuSub = F - neuropil_fraction*Fneu

    if rolling_percentile_window is None:
        F_baseline = percentile_numba(F_neuSub.numpy(), ptile=percentile_baseline) if multicore_pref else np.percentile(F_neuSub.numpy() , percentile_baseline , axis=1)
    else:
        from .timeSeries import rolling_percentile_pd
        F_baseline = rolling_percentile_pd(
            F_neuSub.numpy()[:,::roll_stride],
            ptile=percentile_baseline, 
            window=int(rolling_percentile_window / roll_stride), 
            multiprocessing_pref=multicore_pref, 
            prog_bar=verbose,
            center=roll_centered,
            interpolation=roll_interpolation,
        )
    F_baseline = torch.as_tensor(F_baseline, dtype=torch.float32)
    F_baseline = torch.tile(F_baseline[:,:,None], (1,1,roll_stride)).reshape((F_baseline.shape[0],-1))[:,:F_neuSub.shape[1]] if roll_stride>1 else F_baseline

    dF = F_neuSub - F_baseline[:,None] if F_baseline.ndim==1 else F_neuSub - F_baseline
    dFoF = dF / F_baseline[:,None] if F_baseline.ndim==1 else dF / F_baseline

    if verbose:
        print(f'Calculated dFoF. Total elapsed time: {round(time.time() - tic,2)} seconds')
    
    return dFoF.numpy() , dF.numpy() , F_neuSub.numpy() , F_baseline.numpy()


def import_s2p(dir_s2p):
    """
    Imports suite2p data
    """
    dir_s2p = Path(dir_s2p).resolve()

    try:
        F = np.load(dir_s2p / 'F.npy')
    except FileNotFoundError:
        print(f'F.npy not found in {dir_s2p}')
        F = None

    try:
        Fneu = np.load(dir_s2p / 'Fneu.npy')
    except FileNotFoundError:
        print(f'Fneu.npy not found in {dir_s2p}')
        Fneu = None

    try:
        iscell = np.load(dir_s2p / 'iscell.npy')
    except FileNotFoundError:
        print(f'iscell.npy not found in {dir_s2p}')
        iscell = None

    try:
        ops = np.load(dir_s2p / 'ops.npy', allow_pickle=True)[()]
    except FileNotFoundError:
        print(f'ops.npy not found in {dir_s2p}')
        ops = None

    try:
        spks = np.load(dir_s2p / 'spks.npy')
    except FileNotFoundError:
        print(f'spks.npy not found in {dir_s2p}')
        spks = None

    try:
        stat = np.load(dir_s2p / 'stat.npy', allow_pickle=True)
    except FileNotFoundError:
        print(f'stat.npy not found in {dir_s2p}')
        stat = None
        
    return F, Fneu, iscell, ops, spks, stat


@njit(parallel=True)
def peter_noise_levels(dFoF, frame_rate):

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
        noise_levels[ii] = np.median(abs_numba(np.diff(dFoF[ii,:],1)))

    # noise_levels = np.median(np.abs(np.diff(dFoF, axis=1)), axis=1)
    noise_levels = noise_levels / np.sqrt(frame_rate) * 100    # scale noise levels to percent
    return noise_levels

def snr_autoregressive(x, axis=1, center=True, standardize=True):
    """
    Calculate the SNR of an autoregressive signal.
    Relies on the assumption that the magnitude of the signal
     can be estimated as the correlation of a signal and
     its autoregressive component (corr(sig, roll(sig, 1))).
    
    """
    if center:
        x_norm = x - np.mean(x, axis=axis, keepdims=True)
    else:
        x_norm = x
    if standardize:
        x_norm = x_norm / np.std(x, axis=axis, keepdims=True)
    else:
        x_norm = x_norm
    var = ((x_norm**2).sum(axis) / x_norm.shape[axis])  ## total variance of each trace
    sig = ((x_norm * np.roll(x_norm, 1, axis=axis)).sum(axis) / x_norm.shape[axis])  ## signal variance of each trace based on assumption that trace = signal + noise, signal is autoregressive, noise is not autoregressive
    
    noise = var - sig
    
    return sig / noise, sig, noise

def trace_quality_metrics(
    F, Fneu, dFoF, dF, F_neuSub, F_baseline,
    percentile_baseline=30,
    Fs=30,
    plot_pref=True, 
    thresh=None,
    clip_range=(-1,1)
    ):
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
        plot_pref:
            Whether to plot the traces and metrics
        thresh:
            Dictionary of thresholds to use.
            If None, then use default values:
                'var_ratio': 1,
                'EV_F_by_Fneu': 0.6,
                'base_FneuSub': 0,
                'base_F': 50,
                'peter_noise_levels': 12,
                'max_dFoF': 50,
                'baseline_var': 1,
        clip_range (tuple of floats):
            Range of values to clip dFoF to for calculating
             peter_noise_levels and baseline_var.
    
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
    from .timeSeries import percentile_numba, var_numba, rolling_percentile_rq_multicore

    var_F = var_numba(F)
    var_Fneu = var_numba(Fneu)
    var_FneuSub = var_numba(F_neuSub)

    var_F[var_F==0] = np.nan
    var_ratio = var_Fneu / var_F

    EV_F_by_Fneu = 1 - (var_FneuSub / var_F)

    base_FneuSub = percentile_numba(F_neuSub, percentile_baseline)
    base_F = percentile_numba(F, percentile_baseline)

    
    dFoF_clip = np.clip(dFoF, clip_range[0], clip_range[1])

    peter_noise = peter_noise_levels(np.ascontiguousarray(dFoF_clip), Fs)
    # peter_noise = np.median(np.abs(np.diff(dFoF, axis=1)), axis=1) # Use this line of code if numba is acting up
    peter_noise[np.abs(peter_noise) > 1e3] = np.nan

    rich_nsr = 1 / snr_autoregressive(dFoF_clip, center=True, standardize=False)[0]


    # currently hardcoding the rolling baseline window to be 10 minutes
    # rolling_baseline = rolling_percentile_pd(dFoF, ptile=percentile_baseline, window=int(Fs*60*2 + 1))
    # dFoF_clipNaN = copy.copy(dFoF)
    # dFoF_clipNaN[dFoF_clipNaN <= clip_range[0]] = np.nan
    # dFoF_clipNaN[dFoF_clipNaN >= clip_range[1]] = np.nan
    dFoF_clipped = np.clip(dFoF, a_min=clip_range[0], a_max=clip_range[1])
    rolling_baseline = rolling_percentile_rq_multicore(dFoF_clipped, ptile=percentile_baseline, window=int(Fs*60*20 + 1))
    baseline_var = var_numba(rolling_baseline[:,int(Fs*60*5):-int(Fs*60*5)])
    # baseline_range = max_numba(rolling_baseline) - min_numba(rolling_baseline)

    max_dFoF = np.max(dFoF, axis=1)

    metrics = {
        'var_ratio': var_ratio,
        'EV_F_by_Fneu': EV_F_by_Fneu,
        'base_FneuSub': base_FneuSub,
        'base_F': base_F,
        'peter_noise_levels': peter_noise,
        'rich_nsr': rich_nsr,
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
                    'peter_noise_levels': 12,
                    'rich_nsr': 2,
                    'max_dFoF': 50,
                    'baseline_var': 1,
                }
    # thresh = {
    # 'var_ratio': 3,
    # 'EV_F_by_Fneu': 1,
    # 'base_FneuSub': -1000,
    # 'base_F': -1000,
    # 'peter_noise_levels': 500,
    # 'rich_nsr': 1,
    # 'max_dFoF': 3000,
    # 'baseline_var': 1,
    # }

    sign = {
    'var_ratio': 1,
    'EV_F_by_Fneu': 1,
    'base_FneuSub': -1,
    'base_F': -1,
    'peter_noise_levels': 1,
    'rich_nsr': 1,
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
            if val=='peter_noise_levels':
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==1)[0]], 300, histtype='step')
                axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==0)[0]], 300, histtype='step')
                axs[ii].set_xlim([0,20])
            # elif val=='baseline_var':
            #     axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==1)[0]], 300, histtype='step')
            #     axs[ii].hist(tqm['metrics'][val][np.where(good_ROIs==0)[0]], 300, histtype='step')
            #     axs[ii].set_xlim(right=50)
            #     # axs[ii].set_xscale('log')
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

    print(f'chan1_offset= {chan1_offset}')
    return chan1_offset
def get_ScanImage_tiff_metadata(path_to_tif, verbose=False):
    from ScanImageTiffReader import ScanImageTiffReader
    
    vol=ScanImageTiffReader(path_to_tif)
    md = vol.metadata().split("\n")
    print(md) if verbose else None
    return md

def import_tiffs_SI(
    paths, 
    downsample_factors=[1,1,1],
    clip_zero=False,
    dtype=None,
    verbose=True,
):
    """
    Imports a tif file using ScanImageTiffReader
    RH 2023

    Args:
        paths (list):
            List of paths to tif files.
        downsample_factors (list):
            List of downsample factors for each dimension.
            [frames, height, width]
            Calls .image_processing.bin_array(frames, downsample_factors, function=partial(np.mean, axis=0))
        clip_zero (bool):
            Whether to clip negative values to zero.
        dtype (np.dtype):
            Data type to cast to.
        verbose (bool):
            Whether to show tqdm progress bar.

    Returns:
        vol (list):
            List of ScanImageTiffReader objects.
    """
    from ScanImageTiffReader import ScanImageTiffReader
    from .image_processing import bin_array
    from functools import partial

    ## warn if dtype is unsigned integer type and clip_zero is False
    if (dtype is not None) and (np.issubdtype(dtype, np.unsignedinteger)) and (not clip_zero):
        print("Warning: dtype is unsigned integer type and clip_zero is False. This will result in overflow.")

    def get_frames(path):
        vol = ScanImageTiffReader(path)
        n_frames = 1 if len(vol.shape()) < 3 else vol.shape()[0]  # single page tiffs
        frames = vol.data(beg=0, end=n_frames)
        vol.close()
        del vol
        frames = np.clip(frames, 0, None) if clip_zero else frames
        frames = bin_array(frames, downsample_factors, function=partial(np.mean, axis=0)) if np.any(np.array(downsample_factors) > 1) else frames
        frames = frames.astype(dtype) if (dtype is not None) and (frames.dtype != dtype) else frames
        gc.collect()
        gc.collect()
        return frames
    
    out = [get_frames(path) for path in tqdm(paths, disable=not verbose)] if isinstance(paths, list) else get_frames(paths)

    return out

def dense_stack_to_sparse_stack_SI(
    stack_in, 
    scanimage_metadata=None,
    frames_to_discard_per_slice=30, 
    sparse_step_size_um=4,
    num_frames_per_slice=60, 
    num_slices=25, 
    num_volumes=10, 
    step_size_um=0.8, 
    verbose=True,
):
    """
    Converts a dense stack of images into a sparse stack of images.
    Depends on the indexing used by ScanImage for z-stacks.
    RH 2023

    Args:
        stack_in (np.ndarray):
            Input stack of images.
            Shape: [frames, height, width]
        scanimage_metadata (list of str):
            Metadata from ScanImage tiff file. Expected to be a list of strings.
            This list can be obtained using get_ScanImage_tiff_metadata().
             fill in the following args:
                - num_frames_per_slice
                - num_slices
                - num_volumes
                - step_size_um
        frames_to_discard_per_slice (int):
            Number of frames to discard per slice.
        sparse_step_size_um (float):
            Desired step size in microns for the sparse stack.
        num_frames_per_slice (int):
            Number of frames per slice.
            From SI z-stack params.
        num_slices (int):
            Number of slices.
            From SI z-stack params.
        num_volumes (int):
            Number of volumes.
            From SI z-stack params.
        step_size_um (float):
            Step size in microns.
            From SI z-stack params.

    Returns:
        stack_out (np.ndarray):
            Output stack of images.
            Shape: [frames, height, width]
        positions_z (np.ndarray):
            Z positions of each frame in the sparse stack.
            Shape: [frames]
        idx_slices (np.ndarray):
            Indices that were subsampled from the dense stack along the z-axis.
    """
    if scanimage_metadata is not None:
        meta_strings = {
            'num_frames_per_slice': 'SI.hStackManager.framesPerSlice',
            'num_slices': 'SI.hStackManager.actualNumSlices',
            'num_volumes': 'SI.hStackManager.actualNumVolumes',
            'step_size_um': 'SI.hStackManager.actualStackZStepSize',
        }
        meta_values = {key: [float(t[t.find('= ')+2:]) for t in scanimage_metadata if t[:len(m)]==m][0] for key, m in meta_strings.items()}
        num_frames_per_slice = int(meta_values['num_frames_per_slice'])
        num_slices = int(meta_values['num_slices'])
        num_volumes = int(meta_values['num_volumes'])
        step_size_um = float(meta_values['step_size_um'])

        if verbose:
            print(f"Args found from scanimage_metadata:")
            print(f"{'  num_frames_per_slice = ':<25}" + f"{num_frames_per_slice}") 
            print(f"{'  num_slices = ':<25}" + f"{num_slices}") 
            print(f"{'  num_volumes = ':<25}" + f"{num_volumes}") 
            print(f"{'  step_size_um = ':<25}" + f"{step_size_um}") 
            print('')

    range_slices = num_slices * step_size_um
    range_idx_half = int((range_slices / 2) // sparse_step_size_um)
    step_numIdx = int(sparse_step_size_um // step_size_um)
    idx_center = int(num_slices // 2)
    idx_slices = [idx_center + n for n in np.arange(-range_idx_half*step_numIdx, range_idx_half*step_numIdx + 1, step_numIdx, dtype=np.int64)]
    assert (min(idx_slices) >= 0) and (max(idx_slices) <= num_slices), f"RH ERROR: The range of slice indices expected is greater than the number of slices available: {idx_slices}"
    positions_z = [(idx - idx_center)*step_size_um for idx in idx_slices]
    positions_z = [np.round(z, decimals=10) for z in positions_z]
    
    slices_rs = np.reshape(stack_in, (num_frames_per_slice, num_slices, num_volumes, stack_in.shape[1], stack_in.shape[2]), order='F')
    slices_rs = slices_rs[frames_to_discard_per_slice:,:,:,:,:]
    slices_rs = np.mean(slices_rs, axis=(0, 2))

    stack_out = slices_rs[idx_slices]

    if verbose:
        print(f"{'stack_in.shape = ':<25}" + f"{stack_in.shape}") 
        print(f"{'stack_out.shape = ':<25}" + f"{stack_out.shape}") 
        print(f"{'positions_z = ':<25}" + ''.join([f'{z}, ' for z in positions_z])) 
        print(f"{'idx_slices = ':<25}" + f"{idx_slices}")

    return stack_out, positions_z, idx_slices


def find_zShifts(
    zstack,
    positions_z=None,
    path_to_tiff=None,
    frames=None,
    clip_zero=False,
    downsample_factors=[1,1,1],
    dtype=np.uint16,
    bandpass_spatialFs_bounds=(0.02, 0.3),
    order_butter=5,
    use_GPU=True,
    batch_size=70,
    resample_factor=100,
    sig=4.0,
    verbose=True,
):
    """
    Finds the z-shift of each frame in a stack of frames relative to a z-stack.
    RH 2023

    Args:
        zstack (np.ndarray):
            Z-stack of images.
            Shape: [num_frames, height, width]
        positions_z (list):
            List of z-positions for each frame in zstack.
            Shape: zstack.shape[0]
            Can be output of dense_stack_to_sparse_stack_SI().
        path_to_tiff (str):
            Path to tiff file.
        frames (np.ndarray):
            Stack of images.
            Shape: [num_frames, height, width]
        clip_zero (bool):
            Whether to clip values below zero.
            Only used if path_to_tiff is not None.
            Used in import_tiffs_SI().
        downsample_factors (list):
            List of factors to downsample each dimension of frames by.
            Only used if path_to_tiff is not None.
            Used in import_tiffs_SI().
        dtype (np.dtype):
            Data type to convert frames to.
            Only used if path_to_tiff is not None.
            Used in import_tiffs_SI().
        bandpass_spatialFs_bounds (tuple):
            Bounds of bandpass filter in spatial frequencies.
            Used in make_Fourier_mask().
        order_butter (int):
            Order of butterworth filter.
            Used in make_Fourier_mask().
        use_GPU (bool):
            Whether to use GPU.
        batch_size (int):
            Number of frames to process at once.
            Use smaller values if GPU or CPU memory is an issue.
        resample_factor (int):
            Factor to resample frames by.
            Used to up/resample the z-axis values.
        sig (float):
            Sigma of gaussian kernel used to smooth the z-axis values.
            Used in math_functions.gaussian().
        verbose (bool):
            Whether to print progress and plot stuff.
    """
    from .image_processing import make_Fourier_mask, phaseCorrelationImage_to_shift, phase_correlation
    from .torch_helpers import clear_cuda_cache, set_device
    from .indexing import make_batches
    from .timeSeries import convolve_along_axis
    from .math_functions import gaussian

    assert positions_z is not None, "RH ERROR: Must provide positions_z"

    if frames is None:
        frames = import_tiffs_SI(
            path_to_tiff, 
            downsample_factors=downsample_factors,
            clip_zero=clip_zero,
            dtype=dtype,
            verbose=verbose,
        )
    
    mask_fft = make_Fourier_mask(
        frame_shape_y_x=frames[0].shape,
        bandpass_spatialFs_bounds=bandpass_spatialFs_bounds,
        order_butter=order_butter,
        plot_pref=verbose,
        verbose=verbose,
    )

    clear_cuda_cache() if use_GPU else None
    DEVICE = set_device(use_GPU=use_GPU, verbose=verbose)

    def frames_to_zShift(frames_toUse, zstack_maskFFT):
        def shift_helper(cc):
            return torch.stack([phaseCorrelationImage_to_shift(c)[1] for c in cc], dim=0).T
        shifts = torch.cat([shift_helper(phase_correlation(
            im_template=zstack_maskFFT,
            im_moving=batch,
            template_precomputed=True,
            device=DEVICE,
        )) for batch in make_batches(frames_toUse.astype(np.float32), batch_size=batch_size)], dim=0)
        return shifts

    zstack_maskFFT = torch.as_tensor(np.conj(np.fft.fft2(zstack, axes=(-2,-1)) * mask_fft.numpy())).to(DEVICE).type(torch.complex64)

    z_cc = frames_to_zShift(frames, zstack_maskFFT[:]).cpu().numpy()

    z_cc_conv = convolve_along_axis(z_cc[:], gaussian(x=np.linspace(-int(sig), int(sig), num=int(sig*2)+1, endpoint=True), sig=sig, plot_pref=False), axis=1, mode='same')

    xAxis = np.linspace(0, z_cc_conv.shape[1]-1, num=z_cc_conv.shape[1]*int(resample_factor), endpoint=True)
    z_cc_interp = scipy.interpolate.interp1d(np.linspace(0, z_cc_conv.shape[1]-1, num=z_cc_conv.shape[1], endpoint=True), z_cc_conv, kind='quadratic', axis=1)(xAxis)

    pos_idx_sub = np.array(positions_z[:])
    zShift_interp = scipy.interpolate.interp1d(np.linspace(0, len(pos_idx_sub)-1, num=len(pos_idx_sub), endpoint=True), np.array(positions_z[:]))(np.linspace(0, len(pos_idx_sub)-1, num=len(pos_idx_sub)*int(resample_factor), endpoint=True))

    positions_interp = zShift_interp[np.argmax(z_cc_interp, axis=1)]

    clear_cuda_cache() if use_GPU else None

    if verbose:
        plt.figure()
        plt.plot(positions_z, z_cc[0])
        plt.plot(positions_z, z_cc_conv[0])
        plt.plot(zShift_interp, z_cc_interp[0])

    return positions_interp, zShift_interp, z_cc_interp


####################################################
################ SUITE2p SPECIFIC ##################
####################################################

import pathlib
import copy

import numpy as np
import matplotlib.pyplot as plt


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