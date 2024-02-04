import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import torch
from . import math_functions, indexing
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
    if n % 2 == 0: ## then even
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

