'''
Table of Contents

Functions and Interdependencies:
    butter_bandpass
    butter_bandpass_filter
        - butter_bandpass
    mtaper_specgram
    simple_cwt
'''

import math
from re import S
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import torch
from . import math_functions, timeSeries
from tqdm.notebook import tqdm


def butter_bandpass(lowcut, highcut, fs, order=5, plot_pref=True):
    '''
    designs a butterworth bandpass filter.
    Found on a stackoverflow, but can't find it
     anymore.
    RH 2021

        Args:
            lowcut (scalar): 
                frequency (in Hz) of low pass band
            highcut (scalar):  
                frequency (in Hz) of high pass band
            fs (scalar): 
                sample rate (frequency in Hz)
            order (int): 
                order of the butterworth filter
        
        Returns:
            b (ndarray): 
                Numerator polynomial coeffs of the IIR filter
            a (ndarray): 
                Denominator polynomials coeffs of the IIR filter
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    
    if plot_pref:
        w, h = scipy.signal.freqz(b, a, worN=2000)
        plt.figure()
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.xlabel('frequency (Hz)')
        plt.ylabel('frequency response (a.u)')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, axis=-1, order=5, plot_pref=False):
    '''
    applies a butterworth bandpass filter
    RH 2021
    
        Args:
            data (ndarray): 
                data array. filtering done on 
                 defined axis 
            lowcut (scalar): 
                frequency (in Hz) of low pass band
            highcut (scalar): 
                frequency (in Hz) of high pass band
            fs (scalar): 
                sample rate (frequency in Hz)
            order (int): 
                order of the butterworth filter
        
        Returns:
            y (ndarray): 
                filtered data array
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, plot_pref=plot_pref)
    y = scipy.signal.lfilter(b, a, data, axis=axis)
    return y


def mtaper_specgram(
    signal,
    nw=2.5,
    ntapers=None,
    win_len=0.1,
    win_overlap=0.09,
    fs=int(192e3),
    clip=None,
    freq_res_frac=1,
    mode='psd',
    **kwargs
):
    """
    Multi-taper spectrogram
    RH 2021

        Args:
            signal (array type): 
                Signal.
            nw (float): 
                Time-bandwidth product
            ntapers (int): 
                Number of tapers (None to set to 2 * nw -1)
            win_len (float): 
                Window length in seconds
            win_overlap (float): 
                Window overlap in seconds
            fs (float): 
                Sampling rate in Hz
            clip (2-tuple of floats): 
                Normalize amplitudes to 0-1 using clips (in dB)
            freq_res_frac (float): 
                frequency resolution fraction. 
                generates nfft. If none then nfft=None,
                which makes nfft=win nfft=nperseg=len_samples. 
                else nfft = freq_resolution_frac * round(win_len * fs)
            mode (string): 
                mode of the scipy.signal.spectrogram to use. Can be
                'psd', 'complex', ‘magnitude’, ‘angle’, ‘phase’
            **kwargs: 
                Additional arguments for scipy.signal.spectrogram
        Returns:
            f (ndarray): 
                Frequency bin centers
            t (ndarray): 
                Time indices
            sxx (ndarray): 
                Spectrogram
    """
    len_samples = np.round(win_len * fs).astype("int")
    if freq_res_frac is None:
        nfft = None
    else:
        nfft = freq_res_frac*len_samples
    if ntapers is None:
        ntapers = int(nw * 2)
    overlap_samples = np.round(win_overlap * fs)
    sequences, r = scipy.signal.windows.dpss(
        len_samples, NW=nw, Kmax=ntapers, sym=False, norm=2, return_ratios=True
    )
    sxx_ls = None
    for sequence, weight in zip(sequences, r):
        f, t, sxx = scipy.signal.spectrogram(
            signal,
            fs=fs,
            window=sequence,
            nperseg=len_samples,
            noverlap=overlap_samples,
            nfft=nfft,
            detrend='constant',
            return_onesided=True, 
            scaling='density', 
            axis=-1, 
            mode=mode,
            **kwargs
        )
        if sxx_ls is None:
            sxx_ls = sxx * weight
        else:
            sxx_ls += np.abs(sxx * weight)
    sxx = sxx_ls / len(sequences)
    if clip is not None:
        sxx = 20 * np.log10(sxx)
        sxx = sxx - clip[0]
        sxx[sxx < 0] = 0
        sxx[sxx > (clip[1] - clip[0])] = clip[1] - clip[0]
        sxx /= clip[1] - clip[0]
    return f, t, sxx



def simple_cwt(
    X,
    freqs_toUse=None, 
    fs=30, 
    wavelet_type='cmor', 
    bwf=None, 
    cf=None, 
    psd_scaling=True, 
    plot_pref=True, 
    axis=-1):
    '''
    performs a simple continuous wavelet transform (cwt) using pywt.cwt
    RH 2021

        Args:
            X (ndarray): 
                data array
            freqs_toUse (1-D ndarray): 
                values of frequencies to perform cwt on
            fs (scalar): sample rate in Hz
            wavelet_type (string): 
                name of wavelet type to use. See pywt.wavelist() for all 
                possible inputs
            bwf (scalar): 
                bandwidth (in units of frequency). Used only if using complex 
                morlet ('cmor')
            cf (scalar): 
                center frequency. Used only if using complex morlet ('cmor')
            axis (int): 
                axis along which to perform cwt
            psd_scaling (bool): 
                preference of whether to scale the output to compute the power 
                spectral density or leave as raw output of pywt.cwt 
        Returns:
            coeff (ndarray): 
                output cwt array (with temporal dimension=='axis').
                A natural way to normalize output is to put it in units of
                'spectral density' = np.abs(coeff**2 / (1/freqs_toUse)[:,None])
                Another nice normalization is
                np.abs(coeff / (1/freqs_toUse)[:,None]**1)**1.5
    '''
    import pywt

    if wavelet_type=='cmor' and bwf is None:
        bwf = 2
    if wavelet_type=='cmor' and cf is None:
        cf = 1
    waveletname = f'{wavelet_type}{bwf}-{cf}'

    if freqs_toUse is None:
        freqs_toUse = np.logspace(np.log(fs/30), np.log(fs/2), 30, base=np.exp(1))
    scales = fs/freqs_toUse


    coeff, freq = pywt.cwt(data=X, 
                        scales=scales, 
                        wavelet=waveletname, 
                        sampling_period=1/fs,
                        axis=axis)

    if psd_scaling:
        coeff = np.abs(coeff**2 / (1/freqs_toUse)[:,None])

    if plot_pref:
        n_ticks = min(len(freq) , 10)
        tick_spacing = len(freq)//n_ticks
        ticks_toUse = np.arange(0,len(freq), tick_spacing)    
        fig, ax = plt.subplots()
        if psd_scaling:
            ax.imshow(coeff, aspect='auto', origin='lower')
        else:
            ax.imshow(np.abs(coeff), aspect='auto', origin='lower')
        ax.set_yticks(ticks_toUse)
        ax.set_yticklabels(np.round(freq[ticks_toUse],2))

    return coeff, freq


def torch_hilbert(x, dim=0):
    """
    Computes the analytic signal using the Hilbert transform.
    Based on scipy.signal.hilbert
    RH 2022
    
    Args:
        x (nd tensor):
            Signal data. Should be real.
        dim (int):
            Dimension along which to do the transformation.
    
    Returns:
        xa (nd tensor):
            Analytic signal of input x along dim
    """
    
    xf = torch.fft.fft(x, dim=dim)
    m = torch.zeros(x.shape[dim])
    n = x.shape[dim]
    if n % 2: ## then even
        m[0] = m[n//2] = 1
        m[1:n//2] = 2
    else:
        m[0] = 1
        m[1:(n+1)//2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[dim] = slice(None)
        m = m[tuple(ind)]

    return torch.fft.ifft(xf * m, dim=dim)


def make_VQT_filters(    
    Fs_sample=1000,
    Q_lowF=3,
    Q_highF=20,
    F_min=10,
    F_max=400,
    n_freq_bins=55,
    win_size=501,
    plot_pref=False
):
    """
    Creates a set of filters for use in the VQT algorithm.

    Set Q_lowF and Q_highF to be the same value for a 
     Constant Q Transform (CQT).
    Varying these values will varying the Q factor 
     logarithmically across the frequency range.

    RH 2022

    Args:
        Fs_sample (float):
            Sampling frequency of the signal.
        Q_lowF (float):
            Q factor to use for the lowest frequency.
        Q_highF (float):
            Q factor to use for the highest frequency.
        F_min (float):
            Lowest frequency to use.
        F_max (float):
            Highest frequency to use (inclusive).
        n_freq_bins (int):
            Number of frequency bins to use.
        win_size (int):
            Size of the window to use, in samples.
        plot_pref (bool):
            Whether to plot the filters.

    Returns:
        filters (Torch ndarray):
            Array of complex sinusoid filters.
            shape: (n_freq_bins, win_size)
        freqs (Torch array):
            Array of frequencies corresponding to the filters.
        wins (Torch ndarray):
            Array of window functions (gaussians)
             corresponding to each filter.
            shape: (n_freq_bins, win_size)
    """

    assert win_size%2==1, "RH Error: win_size should be an odd integer"
    
    freqs = math_functions.bounded_logspace(
        start=F_min,
        stop=F_max,
        num=n_freq_bins,
    )

    periods = 1 / freqs
    periods_inSamples = Fs_sample * periods

    sigma_all = math_functions.bounded_logspace(
        start=Q_lowF,
        stop=Q_highF,
        num=n_freq_bins,
    )
    sigma_all = sigma_all * periods_inSamples / 4

    wins = torch.stack([math_functions.gaussian(torch.arange(-win_size//2, win_size//2), 0, sig=sigma) for sigma in sigma_all])

    filts = torch.stack([torch.cos(torch.linspace(-np.pi, np.pi, win_size) * freq * (win_size/Fs_sample)) * win for freq, win in zip(freqs, wins)], dim=0)    
    filts_complex = torch_hilbert(filts.T, dim=0).T
    
    if plot_pref:
        plt.figure()
        plt.plot(freqs)
        plt.xlabel('filter num')
        plt.ylabel('frequency (Hz)')

        plt.figure()
        plt.imshow(wins / torch.max(wins, 1, keepdims=True)[0], aspect='auto')
        plt.title('windows (gaussian)')

        plt.figure()
        plt.plot(sigma_all)
        plt.xlabel('filter num')
        plt.ylabel('window width (sigma of gaussian)')    

        plt.figure()
        plt.imshow(filts / torch.max(filts, 1, keepdims=True)[0], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
        plt.title('filters (real component)')


        worN=win_size*4
        filts_freq = np.array([scipy.signal.freqz(
            b=filt,
            fs=Fs_sample,
            worN=worN,
        )[1] for filt in filts_complex])

        filts_freq_xAxis = scipy.signal.freqz(
            b=filts_complex[0],
            worN=worN,
            fs=Fs_sample
        )[0]

        plt.figure()
        plt.plot(filts_freq_xAxis, np.abs(filts_freq.T));
        plt.xscale('log')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('magnitude')

    return filts_complex, freqs, wins


class VQT():
    def __init__(
        self,
        Fs_sample=1000,
        Q_lowF=3,
        Q_highF=20,
        F_min=10,
        F_max=400,
        n_freq_bins=55,
        win_size=501,
        downsample_factor=4,
        DEVICE_compute='cpu',
        DEVICE_return='cpu',
        return_complex=False,
        filters=None,
        plot_pref=False,
        ):
        """
        Variable Q Transform.
        Class for applying the variable Q transform to signals.

        This function works differently than the VQT from 
         librosa or nnAudio. This one does not use iterative
         lowpass filtering. Instead, it uses a fixed set of 
         filters, and a Hilbert transform to compute the analytic
         signal. It can then take the envelope and downsample.
        
        Uses Pytorch for GPU acceleration, and allows gradients
         to pass through.

        Q: quality factor; roughly corresponds to the number 
         of cycles in a filter. Here, Q is the number of cycles
         within 4 sigma (95%) of a gaussian window.

        RH 2022

        Args:
            Fs_sample (float):
                Sampling frequency of the signal.
            Q_lowF (float):
                Q factor to use for the lowest frequency.
            Q_highF (float):
                Q factor to use for the highest frequency.
            F_min (float):
                Lowest frequency to use.
            F_max (float):
                Highest frequency to use.
            n_freq_bins (int):
                Number of frequency bins to use.
            win_size (int):
                Size of the window to use, in samples.
            downsample_factor (int):
                Factor to downsample the signal by.
                If the length of the input signal is not
                 divisible by downsample_factor, the signal
                 will be zero-padded at the end so that it is.
            DEVICE_compute (str):
                Device to use for computation.
            DEVICE_return (str):
                Device to use for returning the results.
            return_complex (bool):
                Whether to return the complex version of 
                 the transform. If False, then returns the
                 absolute value (envelope) of the transform.
                downsample_factor must be 1 if this is True.
            filters (Torch tensor):
                Filters to use. If None, will make new filters.
                shape: (n_freq_bins, win_size)
            plot_pref (bool):
                Whether to plot the filters.
        """

        assert all((return_complex==True, downsample_factor!=1))==False, "RH Error: if return_complex==True, then downsample_factor must be 1"

        if filters is not None:
            self.filters = filters
        else:
            self.filters, self.freqs, self.wins = make_VQT_filters(
                Fs_sample=Fs_sample,
                Q_lowF=Q_lowF,
                Q_highF=Q_highF,
                F_min=F_min,
                F_max=F_max,
                n_freq_bins=n_freq_bins,
                win_size=win_size,
                plot_pref=plot_pref
            )
        
        self.args = {}
        self.args['Fs_sample'] = Fs_sample
        self.args['Q_lowF'] = Q_lowF
        self.args['Q_highF'] = Q_highF
        self.args['F_min'] = F_min
        self.args['F_max'] = F_max
        self.args['n_freq_bins'] = n_freq_bins
        self.args['win_size'] = win_size
        self.args['downsample_factor'] = downsample_factor
        self.args['DEVICE_compute'] = DEVICE_compute
        self.args['DEVICE_return'] = DEVICE_return
        self.args['return_complex'] = return_complex
        self.args['plot_pref'] = plot_pref

    def _helper_ds(self, X, ds_factor):
        if ds_factor == 1:
            return X
        else:
            return torch.nn.functional.avg_pool1d(X.T, kernel_size=ds_factor, stride=ds_factor, ceil_mode=True).T

    def _helper_conv(self, arr, filters, take_abs, DEVICE):
        out =  timeSeries.convolve_torch(arr.to(DEVICE),  torch.real(filters.T).to(DEVICE), padding='same') + \
            1j*timeSeries.convolve_torch(arr.to(DEVICE), -torch.imag(filters.T).to(DEVICE), padding='same')
        if take_abs:
            return torch.abs(out)
        else:
            return out

    def __call__(self, X):
        """
        Forward pass of VQT.

        Args:
            X (Torch tensor):
                Input signal.
                shape: (n_channels, n_samples)

        Returns:
            Spectrogram (Torch tensor):
                Spectrogram of the input signal.
                shape: (n_channels, n_samples_ds, n_freq_bins)
        """
        return torch.stack([self._helper_ds(
            self._helper_conv(
                arr=arr, 
                filters=self.filters, 
                take_abs=(self.args['return_complex']==False),
                DEVICE=self.args['DEVICE_compute']
                ), 
            self.args['downsample_factor']).to(self.args['DEVICE_return']) for arr in tqdm(X)], dim=0)
