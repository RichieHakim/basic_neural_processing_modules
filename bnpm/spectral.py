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
from . import math_functions, timeSeries, indexing
from tqdm import tqdm


def design_butter_bandpass(lowcut, highcut, fs, order=5, plot_pref=True):
    '''
    designs a butterworth bandpass filter.
    Makes a lowpass filter if lowcut is 0.
    Makes a highpass filter if highcut is fs/2.
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

    if low <= 0:
        ## Make a lowpass filter
        b, a = scipy.signal.butter(N=order, Wn=high, btype='low')
    elif high >= 1:
        ## Make a highpass filter
        b, a = scipy.signal.butter(N=order, Wn=low, btype='high')
    else:
        b, a = scipy.signal.butter(N=order, Wn=[low, high], btype='band')
    
    if plot_pref:
        plot_digital_filter_response(b=b, a=a, fs=fs, worN=100000)
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
    b, a = design_butter_bandpass(lowcut, highcut, fs, order=order, plot_pref=plot_pref)
    y = scipy.signal.lfilter(b, a, data, axis=axis)
    return y


def design_fir_bandpass(lowcut, highcut, num_taps=30001, fs=30, window='hamming', plot_pref=True):
    '''
    designs a FIR bandpass filter.
    Makes a lowpass filter if lowcut is 0.
    Makes a highpass filter if highcut is fs/2.
    RH 2021

        Args:
            lowcut (scalar): 
                frequency (in Hz) of low pass band
            highcut (scalar):  
                frequency (in Hz) of high pass band
            num_taps (int): 
                number of taps in the filter
            fs (scalar): 
                sample rate (frequency in Hz)
        
        Returns:
            b (ndarray): 
                Numerator polynomial coeffs of the IIR filter
            a (ndarray): 
                Denominator polynomials coeffs of the IIR filter
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0:
        ## Make a lowpass filter
        b = scipy.signal.firwin(numtaps=num_taps, cutoff=high, window=window, pass_zero=True)
    elif high >= 1:
        ## Make a highpass filter
        b = scipy.signal.firwin(numtaps=num_taps, cutoff=low, window=window, pass_zero=False)
    else:
        b = scipy.signal.firwin(numtaps=num_taps, cutoff=[low, high], window=window, pass_zero=False)

    if plot_pref:
        plot_digital_filter_response(b=b, fs=fs, worN=100000)
    return b


def plot_digital_filter_response(b, a=None, fs=30, worN=100000, plot_pref=True):
    '''
    plots the frequency response of a digital filter
    RH 2021

        Args:
            b (ndarray): 
                Numerator polynomial coeffs of the IIR filter
            a (ndarray): 
                Denominator polynomials coeffs of the IIR filter
            fs (scalar): 
                sample rate (frequency in Hz)
            worN (int): 
                number of frequencies at which to evaluate the filter
    '''
    w, h = scipy.signal.freqz(b, a, worN=worN) if a is not None else scipy.signal.freqz(b, worN=worN)
    xAxis = (fs * 0.5 / np.pi) * w

    if plot_pref:
        plt.figure()
        plt.plot(xAxis, abs(h))
        plt.xlabel('frequency (Hz)')
        plt.ylabel('frequency response (a.u)')
        plt.xscale('log')

    return xAxis, abs(h)


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
                'psd', 'complex', 'magnitude', 'angle', 'phase'
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


def torch_hilbert(x, N=None, dim=0):
    """
    Computes the analytic signal using the Hilbert transform.
    Based on scipy.signal.hilbert
    RH 2022
    
    Args:
        x (nd tensor):
            Signal data. Must be real.
        N (int):
            Number of Fourier components to use.
            If None, then N = x.shape[dim]
        dim (int):
            Dimension along which to do the transformation.
    
    Returns:
        xa (nd tensor):
            Analytic signal of input x along dim
    """
    assert x.is_complex() == False, "x should be real"
    n = x.shape[dim] if N is None else N
    assert n >= 0, "N must be non-negative"

    xf = torch.fft.fft(input=x, n=n, dim=dim)
    m = torch.zeros(n, dtype=xf.dtype, device=xf.device)
    if n % 2: ## then even
        m[0] = m[n//2] = 1
        m[1:n//2] = 2
    else:
        m[0] = 1 ## then odd
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
     Constant Q Transform (CQT) filter set.
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
        padding='valid',
        DEVICE_compute='cpu',
        DEVICE_return='cpu',
        batch_size=1000,
        return_complex=False,
        filters=None,
        plot_pref=False,
        progressBar=True,
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
            padding (str):
                Padding to use for the signal.
                'same' will pad the signal so that the output
                 signal is the same length as the input signal.
                'valid' will not pad the signal. So the output
                 signal will be shorter than the input signal.
            DEVICE_compute (str):
                Device to use for computation.
            DEVICE_return (str):
                Device to use for returning the results.
            batch_size (int):
                Number of signals to process at once.
                Use a smaller batch size if you run out of memory.
            return_complex (bool):
                Whether to return the complex version of 
                 the transform. If False, then returns the
                 absolute value (envelope) of the transform.
                downsample_factor must be 1 if this is True.
            filters (Torch tensor):
                Filters to use. If None, will make new filters.
                Should be complex sinusoids.
                shape: (n_freq_bins, win_size)
            plot_pref (bool):
                Whether to plot the filters.
            progressBar (bool):
                Whether to show a progress bar.
        """
        ## Prepare filters
        if filters is not None:
            ## Use provided filters
            self.using_custom_filters = True
            self.filters = filters
        else:
            ## Make new filters
            self.using_custom_filters = False
            self.filters, self.freqs, self.wins = make_VQT_filters(
                Fs_sample=Fs_sample,
                Q_lowF=Q_lowF,
                Q_highF=Q_highF,
                F_min=F_min,
                F_max=F_max,
                n_freq_bins=n_freq_bins,
                win_size=win_size,
                plot_pref=plot_pref,
            )
        ## Gather parameters from arguments
        self.Fs_sample, self.Q_lowF, self.Q_highF, self.F_min, self.F_max, self.n_freq_bins, self.win_size, self.downsample_factor, self.padding, self.DEVICE_compute, \
            self.DEVICE_return, self.batch_size, self.return_complex, self.plot_pref, self.progressBar = \
                Fs_sample, Q_lowF, Q_highF, F_min, F_max, n_freq_bins, win_size, downsample_factor, padding, DEVICE_compute, DEVICE_return, batch_size, return_complex, plot_pref, progressBar

    def _helper_ds(self, X: torch.Tensor, ds_factor: int=4, return_complex: bool=False):
        if ds_factor == 1:
            return X
        elif return_complex == False:
            return torch.nn.functional.avg_pool1d(X, kernel_size=[int(ds_factor)], stride=ds_factor, ceil_mode=True)
        elif return_complex == True:
            ## Unfortunately, torch.nn.functional.avg_pool1d does not support complex numbers. So we have to split it up.
            ### Split X, shape: (batch_size, n_freq_bins, n_samples) into real and imaginary parts, shape: (batch_size, n_freq_bins, n_samples, 2)
            Y = torch.view_as_real(X)
            ### Downsample each part separately, then stack them and make them complex again.
            Z = torch.view_as_complex(torch.stack([torch.nn.functional.avg_pool1d(y, kernel_size=[int(ds_factor)], stride=ds_factor, ceil_mode=True) for y in [Y[...,0], Y[...,1]]], dim=-1))
            return Z

    def _helper_conv(self, arr, filters, take_abs, DEVICE):
        out = torch.complex(
            torch.nn.functional.conv1d(input=arr.to(DEVICE)[:,None,:], weight=torch.real(filters.T).to(DEVICE).T[:,None,:], padding=self.padding),
            torch.nn.functional.conv1d(input=arr.to(DEVICE)[:,None,:], weight=-torch.imag(filters.T).to(DEVICE).T[:,None,:], padding=self.padding)
        )
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
            x_axis (Torch tensor):
                New x-axis for the spectrogram in units of samples.
                Get units of time by dividing by self.Fs_sample.
            self.freqs (Torch tensor):
                Frequencies of the spectrogram.
        """
        if type(X) is not torch.Tensor:
            X = torch.as_tensor(X, dtype=torch.float32, device=self.DEVICE_compute)

        if X.ndim==1:
            X = X[None,:]

        ## Make iterator for batches
        batches = indexing.make_batches(X, batch_size=self.batch_size, length=X.shape[0])

        ## Make spectrograms
        specs = [self._helper_ds(
            X=self._helper_conv(
                arr=arr, 
                filters=self.filters, 
                take_abs=(self.return_complex==False),
                DEVICE=self.DEVICE_compute
                ), 
            ds_factor=self.downsample_factor,
            return_complex=self.return_complex,
            ).to(self.DEVICE_return) for arr in tqdm(batches, disable=(self.progressBar==False), leave=True, total=int(np.ceil(X.shape[0]/self.batch_size)))]
        specs = torch.cat(specs, dim=0)

        ## Make x_axis
        x_axis = torch.nn.functional.avg_pool1d(
            torch.nn.functional.conv1d(
                input=torch.arange(0, X.shape[-1], dtype=torch.float32)[None,None,:], 
                weight=torch.ones(1,1,self.filters.shape[-1], dtype=torch.float32) / self.filters.shape[-1], 
                padding=self.padding
            ),
            kernel_size=[int(self.downsample_factor)], 
            stride=self.downsample_factor, ceil_mode=True
            ).squeeze()
        
        return specs, x_axis, self.freqs

    def __repr__(self):
        if self.using_custom_filters:
            return f"VQT with custom filters"
        else:
            return f"VQT object with parameters: Fs_sample={self.Fs_sample}, Q_lowF={self.Q_lowF}, Q_highF={self.Q_highF}, F_min={self.F_min}, F_max={self.F_max}, n_freq_bins={self.n_freq_bins}, win_size={self.win_size}, downsample_factor={self.downsample_factor}, DEVICE_compute={self.DEVICE_compute}, DEVICE_return={self.DEVICE_return}, batch_size={self.batch_size}, return_complex={self.return_complex}, plot_pref={self.plot_pref}"
