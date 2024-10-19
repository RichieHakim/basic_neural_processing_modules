from typing import Union, Tuple, List, Dict, Any, Optional
import functools
import math

import scipy.signal
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import torch
import opt_einsum

from . import circular
from . import misc
from . import torch_helpers
from . import timeSeries
from . import indexing


def design_butter_bandpass(lowcut, highcut, fs, order=5, plot_pref=False):
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


def design_fir_bandpass(lowcut, highcut, num_taps=30001, q=None, fs=30, window='hamming', plot_pref=True):
    '''
    designs a FIR bandpass filter.
    Makes a lowpass filter if lowcut is 0.
    Makes a highpass filter if highcut is fs/2.
    Apply filter with: `scipy.signal.lfilter(b, 1, data, axis=axis)`
    RH 2021

        Args:
            lowcut (scalar): 
                frequency (in Hz) of low pass band
            highcut (scalar):  
                frequency (in Hz) of high pass band
            num_taps (int): 
                number of taps in the filter. If None, then q is used.
            q (scalar):
                quality factor. \n
                ``num_taps = int(fs / f) * q``. \n
                where f is lowcut if lowcut > 0 else highcut. \n
                If None, then num_taps is used.
            fs (scalar): 
                sample rate (frequency in Hz)
            window (string):
                window to use for the FIR filter
                (see scipy.signal.firwin for all possible inputs)
            plot_pref (bool):
        
        Returns:
            b (ndarray): 
                FIR filter coefficients
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if q is not None:
        f = lowcut if lowcut > 0 else highcut
        num_taps = int((fs / lowcut) * q)
        num_taps = num_taps + 1 if num_taps % 2 == 0 else num_taps

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
    Multi-taper spectrogram.
    From Jeff Markowitz.

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


@torch.jit.script
def torch_hilbert(x: torch.Tensor, N: Optional[int] = None, dim: int = -1):
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
        ## pad m with singleton dimensions
        new_shape = [1] * x.ndim
        new_shape[dim] = -1
        m = m.reshape(new_shape)

    return torch.fft.ifft(xf * m, dim=dim)


def hilbert_fourier_domain(xf, axis=-1):
    """
    Computes the Fourier-transform of the analytic signal using the
    Fourier-transformed signal. Input is ``xf`` and output is ``xhf``. Uses the
    approach used in scipy.signal.hilbert.\n
    RH 2024
    
    Args:
        xf (torch.Tensor or np.ndarray):
            Fourier-transformed signal data. Should be complex and have negative
            frequencies. Generally, the output of torch.fft.fft or np.fft.fft.
        axis (int):
            Dimension along which to do the transformation.

    Returns:
        (nd tensor):
            Fourier-transformed analytic signal along axis.
    """
    if isinstance(xf, torch.Tensor):
        is_complex = torch.is_complex
        zeros = functools.partial(torch.zeros, device=xf.device)
    if isinstance(xf, np.ndarray):
        zeros, is_complex = np.zeros, np.iscomplexobj
    
    assert is_complex(xf), "xf should be complex"

    n = xf.shape[axis]

    m = zeros(n, dtype=xf.dtype)
    if n % 2 == 0: ## then even
        m[0] = m[n//2] = 1
        m[1:n//2] = 2
    else:
        m[0] = 1 ## then odd
        m[1:(n+1)//2] = 2

    if xf.ndim > 1:
        ind = [np.newaxis] * xf.ndim
        ind[axis] = slice(None)
        m = m[tuple(ind)]

    return xf * m        
        

def signal_angle_difference(x, y, center=True, window=None, axis=-1):
    """
    Computes the average angle difference between two signals.\n
    Calculated as the mean of the angle difference between the Hilbert
    transforms of the signals.\n
    Bound to the range [-pi, pi] (i.e. -180 to 180 degrees). Negative values
    indicate that x leads y.\n
    RH 2024
    
    Args:
        x (torch.Tensor or np.ndarray):
            Signal data. Must be complex.
        y (torch.Tensor or np.ndarray):
            Signal data. Must be complex.
        center (bool):
            Whether or not to center the signals.
        window (torch.Tensor or np.ndarray):
            If not None, then the signals are multiplied by the window after the
            Hilbert transform.
        dim (int):
            Dimension along which to do the transformation.
    
    Returns:
        angle_diff (nd tensor):
            Angle difference between x and y along dim
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        mean, circmean, hilbert = (functools.partial(fn, axis=axis) for fn in (torch.mean, torch_helpers.circmean, torch_hilbert))
        angle, pi = torch.angle, torch.pi
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        mean, circmean, hilbert = (functools.partial(fn, axis=axis) for fn in (np.mean, scipy.stats.circmean, scipy.signal.hilbert))
        angle, pi = np.angle, np.pi
    else:
        raise ValueError("x and y must be torch tensors or numpy arrays")
    
    fn_win = lambda x: x * window if window is not None else x
    
    if center:
        x, y = x - mean(x, keepdims=True), y - mean(y, keepdims=True)
    
    out = circmean((angle(fn_win(hilbert(x))) - angle(fn_win(hilbert(y)))))

    if out.ndim == 0:
        out = out - 2*pi if out > pi else out
    else:
        out[out > pi] -= 2*pi
    out = out[()]
    return out


@misc.wrapper_flexible_args(['dim', 'axis'])
def time_domain_reversal_in_fourier_domain(xf, axis=-1):
    """
    Reverses the Fourier-transformed input signal by effectively applying a
    time-domain reversal while remaining in the Fourier domain. This is done by
    applying the inverse shift operator. \n
    RH 2024

    Args:
        xf (torch.Tensor or np.ndarray):
            Fourier-transformed signal data. Should be complex and have negative
            frequencies. Generally, the output of torch.fft.fft or np.fft.fft.
        axis (int):
            Dimension along which to do the transformation.

    Returns:
        (nd tensor):
            Time-domain signal reversed along axis
    """
    if isinstance(xf, torch.Tensor):
        assert torch.is_complex(xf), "xf should be complex"
        arange = functools.partial(torch.arange, device=xf.device, dtype=torch_helpers.dtype_to_real(xf.dtype))
        exp, pi, conj = torch.exp, torch.pi, torch.conj
        resolution = lambda x: x.resolve_conj().resolve_neg()
    elif isinstance(xf, np.ndarray):
        arange = np.arange
        exp, pi, conj = np.exp, np.pi, np.conj
        resolution = lambda x: x
    
    ## Make the inverse shift operator
    ### The 'iso' is one period of a complex sinuoid. 
    ### It basically changes the ratio of real to imaginary parts in xf.
    n = xf.shape[axis]
    k = arange(n)
    w = exp(-2j * pi * k / n)
    ## Apply the inverse shift operator
    return resolution(conj(xf * w))


def filtfilt_simple_fft(
    x: Union[torch.Tensor, np.ndarray],
    kernel: Union[torch.Tensor, np.ndarray],
    fast_len: bool = True,
):
    """
    Applies a zero-phase filter to the input signal using the FFT method along
    the last dimension.\n
    Calculated as ``ifft(fft(x) * fft(kernel) * fft(kernel_flipped))``.\n
    This implementation is very fast and is suitable when using long kernels.\n
    NOTE: This is a simple implementation and does not handle edge effects.
    scipy.signal.filtfilt is recommended if speed is not a concern and/or if
    kernel length is similar in length to x.\n
    RH 2024

    Args:
        x (torch.Tensor or np.ndarray):
            Signal data. Convolution is done along the last dimension.
        kernel (torch.Tensor or np.ndarray):
            Filter kernel. Convolution is done along the last dimension. \n
            If not 1D, then shape should be broadcastable with x.
        fast_len (bool):
            Whether to use the fast length method.

    Returns:
        (nd tensor):
            Filtered signal
    """
    assert isinstance(x, torch.Tensor) or isinstance(x, np.ndarray), "x must be a torch tensor or numpy array"

    if isinstance(x, torch.Tensor) and isinstance(kernel, torch.Tensor):
        use_real = (torch.is_complex(x) == False) and (torch.is_complex(kernel) == False)
        fft, ifft = (functools.partial(fn, dim=-1) for fn in ((torch.fft.rfft, torch.fft.irfft) if use_real else (torch.fft.fft, torch.fft.ifft)))
        flip = functools.partial(torch.flip, dims=(-1,))
    elif isinstance(x, np.ndarray) and isinstance(kernel, np.ndarray):
        use_real = (np.iscomplexobj(x) == False) and (np.iscomplexobj(kernel) == False)
        fft, ifft = (functools.partial(fn, axis=-1) for fn in ((np.fft.rfft, np.fft.irfft) if use_real else (np.fft.fft, np.fft.ifft)))
        flip = functools.partial(np.flip, axis=-1)
    else:
        raise ValueError("x and kernel must be torch tensors or numpy arrays")
    
    f_flip = functools.partial(time_domain_reversal_in_fourier_domain, axis=-1)

    n = x.shape[-1] + kernel.shape[-1] - 1
    n = timeSeries.next_fast_len(n) if fast_len else n

    out = fft(x, n=n)  ## x_fft
    kernel_fft = fft(flip(kernel), n=n)
    out = out * kernel_fft  ## xk_fft_1
    out = out * f_flip(kernel_fft)  ## xk_fft_2
    out = ifft(out, n=n)  ## xk
    out = torch_helpers.slice_along_dim(
        X=out,
        dim=-1,
        idx=slice(0, x.shape[-1]),
    )
    out = out.real if use_real else out
    return out


@torch.jit.script
def _helper_time_domain_reversal_in_fourier_domain(xf: torch.Tensor, axis: int = -1):
    """
    Reverses the Fourier-transformed input signal by effectively applying a
    time-domain reversal while remaining in the Fourier domain. This is done by
    applying the inverse shift operator. \n
    RH 2024

    Args:
        xf (torch.Tensor or np.ndarray):
            Fourier-transformed signal data. Should be complex and have negative
            frequencies. Generally, the output of torch.fft.fft or np.fft.fft.
        axis (int):
            Dimension along which to do the transformation.

    Returns:
        (nd tensor):
            Time-domain signal reversed along axis
    """
    assert torch.is_complex(xf), "xf should be complex"
    
    ## Make the inverse shift operator
    ### The 'iso' is one period of a complex sinuoid. 
    ### It basically changes the ratio of real to imaginary parts in xf.
    n = xf.shape[axis]
    k = torch.arange(n, device=xf.device, dtype=torch_helpers.dtype_to_real(xf.dtype))
    w = torch.exp(-2j * torch.pi * k / n)
    ## Apply the inverse shift operator
    return torch.conj(xf * w).resolve_conj().resolve_neg()
@torch.jit.script
def torch_filtfilt_simple_fft(
    x: torch.Tensor,
    kernel: torch.Tensor,
    fast_len: bool = True,
):
    """
    Exactly the same as ``filtfilt_simple_fft`` but works with torch.jit.script.
    Applies a zero-phase filter to the input signal using the FFT method along
    the last dimension.\n
    Calculated as ``ifft(fft(x) * fft(kernel) * fft(kernel_flipped))``.\n
    This implementation is very fast and is suitable when using long kernels.\n
    NOTE: This is a simple implementation and does not handle edge effects.
    scipy.signal.filtfilt is recommended if speed is not a concern and/or if
    kernel length is similar in length to x.\n
    RH 2024

    Args:
        x (torch.Tensor or np.ndarray):
            Signal data. Convolution is done along the last dimension.
        kernel (torch.Tensor or np.ndarray):
            Filter kernel. Convolution is done along the last dimension. \n
            If not 1D, then shape should be broadcastable with x.
        fast_len (bool):
            Whether to use the fast length method.

    Returns:
        (nd tensor):
            Filtered signal
    """
    use_real = (torch.is_complex(x) == False) and (torch.is_complex(kernel) == False)

    n = x.shape[-1] + kernel.shape[-1] - 1
    n = timeSeries.next_fast_len(n) if fast_len else n

    out = torch.fft.fft(x, n=n, dim=-1)  ## x_fft
    kernel_fft = torch.fft.fft(torch.flip(kernel, dims=(-1,)), n=n, dim=-1)
    out = out * kernel_fft  ## xk_fft_1
    out = out * _helper_time_domain_reversal_in_fourier_domain(kernel_fft, axis=-1)  ## xk_fft_2
    out = torch.fft.ifft(out, n=n, dim=-1)  ## xk
    # out = torch_helpers.slice_along_dim(
    #     X=out,
    #     dim=-1,
    #     idx=slice(0, x.shape[-1]),
    # )
    out = out[..., :x.shape[-1]]
    out = out.real if use_real else out
    return out

def torch_coherence(
    x: torch.Tensor,
    y: torch.Tensor,
    fs: float = 1.0,
    window: str = 'hann',
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: str = 'constant',
    axis: int = -1,
    batch_size: Optional[int] = None,
    pad_last_segment: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the magnitude-squared coherence between two signals using a PyTorch
    implementation. This function gives identical results to the
    scipy.signal.coherence, but its much faster. A ``batch_size`` option allows
    for large arrays to be processed on GPU.\n    
    RH 2024
    
    Args:
        x (torch.Tensor): 
            First input signal.
        y (torch.Tensor): 
            Second input signal.
        fs (float): 
            Sampling frequency of the input signal. (Default is 1.0)
        window (str): 
            Type of window to apply. Supported window types are the same as
            `scipy.signal.get_window`. (Default is 'hann')
        nperseg (Optional[int]): 
            Length of each segment. (Default is ``None``, which uses ``len(x) //
            8``)
        noverlap (Optional[int]): 
            Number of points to overlap between segments. (Default is ``None``,
            which uses ``nperseg // 2``)
        nfft (Optional[int]): 
            Number of points in the FFT used for each segment. (Default is
            ``None``, which sets it equal to `nperseg`)
        detrend (str): 
            Specifies how to detrend each segment. Supported values are
            'constant' or 'linear'. (Default is 'constant')
        axis (int): 
            Axis along which the coherence is calculated. (Default is -1)
        batch_size (Optional[int]):
            Number of segments to process at once. Used to reduce memory usage.
            If None, then all segments are processed at once. Note that
            ``num_segments = (x.shape[axis] - nperseg) // (nperseg - noverlap) +
            1``. (Default is None)
        pad_last_segment (bool):
            Whether to pad the last segment with a value defined by this
            argument. If None, then no padding is done. (Default is None)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            freqs (torch.Tensor): 
                Frequencies for which the coherence is computed. Shape is
                ``nfft // 2 + 1``.
            coherence (torch.Tensor): 
                Magnitude-squared coherence values. Shape is x broadcasted with
                y and the specified axis set to ``nfft // 2 + 1``.

    Example:
        .. highlight:: python
        .. code-block:: python

            x = torch.randn(1024)
            y = torch.randn(1024)
            freqs, coherence = torch_coherence(x, y, fs=256)
    """
    ## Convert axis to positive
    axis = axis % len(x.shape)

    ## Checks
    ### Tensor checks
    assert isinstance(x, torch.Tensor), "x should be a torch tensor"
    assert isinstance(y, torch.Tensor), "y should be a torch tensor"
    ### Dims should either be the same or one of them should be 1
    if not (x.shape == y.shape):
        assert all([(x.shape[ii] == y.shape[ii]) or (1 in [x.shape[ii], y.shape[ii]]) for ii in range(len(x.shape))]), f"x and y should have the same shape or one of them should have shape 1 at each dimension. Found x.shape={x.shape} and y.shape={y.shape}"

    if nperseg is None:
        nperseg = len(x) // 8

    if noverlap is None:
        noverlap = nperseg // 2

    if nfft is None:
        nfft = nperseg

    if window is not None:
        window = scipy.signal.get_window(window, nperseg)
        window = torch.tensor(window, dtype=x.dtype, device=x.device)

    ## args should be greater than nfft
    assert all([arg.shape[axis] >= nfft for arg in (x, y)]), f"Signal length along axis should be greater than nfft. Found x.shape={x.shape} and y.shape={y.shape} and nfft={nfft}"

    ## Detrend the signals
    def detrend_constant(y, axis):
        y = y - torch.mean(y, axis=axis, keepdim=True)
        return y

    if detrend == 'linear':
        X_linearDetrendPrep = torch.ones(nperseg, 2, dtype=y.dtype, device=y.device)
        X_linearDetrendPrep[:, 1] = torch.arange(nperseg, dtype=y.dtype, device=y.device)
    def detrend_linear(y, axis):
        """
        Uses least squares approach to remove linear trend.
        """
        ## Move axis to end
        y = y.moveaxis(axis, -1)[..., None]
        ## Prepare the design matrix
        X = X_linearDetrendPrep[([None] * (len(y.shape) - 2))]
        ## Compute the coefficients
        # beta = torch.linalg.lstsq(X, y)[0] 
        ### Use closed form solution for least squares
        beta = torch.linalg.inv(X.transpose(-1, -2) @ X) @ X.transpose(-1, -2) @ y
        ## Remove the trend
        y = y - opt_einsum.contract('...ij, ...jk -> ...ik', X, beta)
        y = y[..., 0]
        ## Move axis back to original position (argsort y_dims_to)
        y = y.moveaxis(-1, axis)
        return y

    if detrend == 'constant':
        fn_detrend = detrend_constant
    elif detrend == 'linear':
        fn_detrend = detrend_linear
    else:
        raise ValueError(f"detrend must be 'constant' or 'linear'. Found {detrend}")
    
    ## Initialize the coherence arrays
    ### Get broadcasted dimensions: max(x, y) at each dimension, and nfft at axis
    x_shape = list(x.shape)
    y_shape = list(y.shape)
    out_shape = [max(x_shape[i], y_shape[i]) for i in range(len(x_shape))]
    out_shape[axis] = nfft // 2 + 1  ## rfft returns only non-negative frequencies (0 to fs/2 inclusive)
    
    ## Initialize sums for Welch's method
    ### Prepare complex dtype
    dtype_complex = torch_helpers.dtype_to_complex(x.dtype)
    f_cross = torch.zeros(out_shape, dtype=dtype_complex, device=x.device)
    psd1 = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    psd2 = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

    ## Prepare batch generator
    num_segments = (x.shape[axis] - nperseg) // (nperseg - noverlap) + 1
    batch_size = max(num_segments, 1) if batch_size is None else batch_size
    x_batches, y_batches = (indexing.batched_unfold(
        var, 
        dimension=axis, 
        size=nperseg, 
        step=nperseg - noverlap, 
        batch_size=batch_size,
        pad_value=pad_last_segment,
    ) for var in (x, y))
    
    process_segment = lambda x: torch.fft.rfft(fn_detrend(x, axis=-1) * window, n=nfft, dim=-1)  ## Note the broadcasting of 1-D window with last dimension of x
    ## Perform Welch's averaging of FFT segments
    for segs_x, segs_y in zip(x_batches, y_batches):
        segs_x = process_segment(segs_x)
        segs_y = process_segment(segs_y)
        f_cross += (torch.sum(torch.conj(segs_x) * segs_y, dim=axis).moveaxis(-1, axis)) / num_segments
        psd1 += (torch.sum((torch.conj(segs_x) * segs_x).real, dim=axis).moveaxis(-1, axis)) / num_segments
        psd2 += (torch.sum((torch.conj(segs_y) * segs_y).real, dim=axis).moveaxis(-1, axis)) / num_segments

    ## Compute coherence
    coherence = torch.abs(f_cross) ** 2 / (psd1 * psd2)

    ## Generate frequency axis
    freqs = torch.fft.rfftfreq(nfft, d=1 / fs)

    ## Take the positive part of the frequency spectrum
    ### NOTE: This is not necessary as the coherence is symmetric (always odd and real)
    # pos_mask = freqs >= 0
    # ### slice along axis
    # freqs = freqs[pos_mask]
    # coherence = torch_helpers.slice_along_dim(coherence, axis=axis, idx=pos_mask)
    
    return freqs, coherence


def spectrogram_magnitude_normalization(S: torch.Tensor, k: float = 0.05):
    """
    Normalize spectrogram by dividing by the total power across all frequencies
    at each time point. ``mag = mag / ((k * mean_mag_at_each_time) + (1-k))``.
    This formula differs slightly from the forumla often used in audio
    engineering in that it is a linear scaling, and does not apply only to data
    above a certain threshold.

    Args:
        spectrogram (torch.Tensor):
            A single spectrogram
            Spectrogram should be of shape: (n_points, n_freq_bins, n_samples)  
        k (float):
            Weighting factor for the normalization. 0 means no normalization. 1
            means every time point has the same power when summed across all
            frequencies.

    Returns:
        spectrogram (tuple or torch.Tensor):
            A single normalized spectrogram of same shape as input.
    """
    if k == 0:
        return S
    
    ## Check inputs
    if isinstance(S, torch.Tensor):
        mean, sum, polar, is_complex, abs, angle  = torch.mean, torch.sum, torch.polar, torch.is_complex, torch.abs, torch.angle
    elif isinstance(S, np.ndarray):
        mean, sum, is_complex, abs, angle = np.mean, np.sum, np.iscomplexobj, np.abs, np.angle
        polar = lambda mag, phase: mag * np.exp(1j * phase)

    assert S.ndim == 3, "Spectrogram should be of shape: (n_points, n_freq_bins, n_samples)"

    if is_complex(S) == False:
        s_mean = mean(sum(S, dim=1, keepdims=True), dim=0, keepdims=True)  ## Mean of the summed power across all frequencies and points. Shape (n_samples,)
        s_norm =  S / ((k * s_mean) + (1-k))  ## Normalize the spectrogram by the mean power across all frequencies and points. Shape (n_points, n_freq_bins, n_samples)
    
    elif is_complex(S) == True:
        s_mag = abs(S)
        s_phase = angle(S)
        s_mean = mean(sum(s_mag, dim=1, keepdims=True), dim=0, keepdims=True)  ## Mean of the summed power across all frequencies and points. Shape (n_samples,)
        s_mag = s_mag / ((k * s_mean) + (1-k))  ## Normalize the spectrogram by the mean power across all frequencies and points. Shape (n_points, n_freq_bins, n_samples)
        s_norm = polar(s_mag, s_phase)
    
    return s_norm



def ppc(phases, axis=-1):
    """
    Computes the pairwise phase consistency (PPC0) for a (set of) vector of
    phases. Based on Vinck et al. 2010, and the implementation in the FieldTrip
    toolbox:
    https://github.com/fieldtrip/fieldtrip/blob/d7403c6a6e8765b679ba8accc69f69a282fce6cf/contrib/spike/ft_spiketriggeredspectrum_stat.m#L442
    RH 2024

    Args:
        phases (np.ndarray): 
            Vector of phases in radians. Bound to the range [-pi, pi].
        axis (int):
            Axis along which to compute the pairwise phase consistency.

    Returns:
        float: 
            Pairwise phase consistency of the phases.
    """
    if isinstance(phases, torch.Tensor):
        sin, cos, abs, sum = torch.sin, torch.cos, torch.abs, torch.sum
    elif isinstance(phases, np.ndarray):
        sin, cos, abs, sum = np.sin, np.cos, np.abs, np.sum

    N = phases.shape[axis]
    if N < 2:
        raise ValueError("The input vector must contain at least two phase values.")

    # Compute pairwise phase consistency
    sinSum = abs(sum(sin(phases), axis=axis))
    cosSum = sum(cos(phases), axis=axis)
    return ((cosSum**2 + sinSum**2) - N) / (N * (N - 1))


@torch.jit.script
def torch_ppc(phases: torch.Tensor, axis: int = -1):
    """
    Exactly the same as ``ppc`` but works with torch.jit.script.
    Computes the pairwise phase consistency (PPC0) for a (set of) vector of
    phases. Based on Vinck et al. 2010, and the implementation in the FieldTrip
    toolbox:
    https://github.com/fieldtrip/fieldtrip/blob/d7403c6a6e8765b679ba8accc69f69a282fce6cf/contrib/spike/ft_spiketriggeredspectrum_stat.m#L442
    RH 2024

    Args:
        phases (np.ndarray): 
            Vector of phases in radians. Bound to the range [-pi, pi].
        axis (int):
            Axis along which to compute the pairwise phase consistency.

    Returns:
        float: 
            Pairwise phase consistency of the phases.
    """
    N = phases.shape[axis]
    if N < 2:
        raise ValueError("The input vector must contain at least two phase values.")

    # Compute pairwise phase consistency
    sinSum = torch.abs(torch.sum(torch.sin(phases), dim=axis))
    cosSum = torch.sum(torch.cos(phases), dim=axis)
    return ((cosSum**2 + sinSum**2) - N) / (N * (N - 1))


def ppc_windowed(phases, window, axis=-1):
    """
    Computes the pairwise phase consistency (PPC0) for a (set of) vector of
    phases using a windowed approach.
    RH 2024

    Args:
        phases (np.ndarray): 
            Matrix of phases in radians. Bound to the range [-pi, pi].
        window (int): 
            Vector or matrix of same length as phases along axis. Window to
            multiply by the sin and cos components.
            If 1-D: Shape should be (n_points,)
            If N-D: Shape must be broadcastable with phases. window.shape[axis]
            must be equal to phases.shape[axis]
        axis (int):
            Axis along which to compute the pairwise phase consistency.

    Returns:
        float: 
            Pairwise phase consistency of the phases.
    """
    if isinstance(phases, torch.Tensor):
        sin, cos, abs, sum = torch.sin, torch.cos, torch.abs, torch.sum
    elif isinstance(phases, np.ndarray):
        sin, cos, abs, sum = np.sin, np.cos, np.abs, np.sum

    N = phases.shape[axis]
    if N < 2:
        raise ValueError("The input vector must contain at least two phase values.")

    if window.ndim == 1:
        assert window.shape[0] == N, "Window should have the same length as the input phases."
    else:
        assert window.shape[axis] == N, "Window should have the same length as the input phases."

    # Compute pairwise phase consistency
    sinSum = abs(sum(sin(phases) * window, axis=axis))
    cosSum = sum(cos(phases) * window, axis=axis)
    return ((cosSum**2 + sinSum**2) - N) / (N * (N - 1))


def generate_multiphasic_sinewave(
    n_samples: int = 10000,
    n_periods: float = 1.0,
    n_waves: int = 3,
    return_x: bool = False,
    return_phases: bool = False,
):
    """
    Generates a multiphasic sine wave.
    RH 2024

    Args:
        n_samples (int): 
            Number of samples to generate.
        n_periods (float): 
            Number of periods in the sine wave.
        n_waves (int): 
            Number of sine waves to generate.
        return_x (bool): 
            If ``True``, returns the x-values along with the waves.
        return_phases (bool): 
            If ``True``, returns the phases along with the waves.

    Returns:
        (tuple): 
            Depending on the `return_x` and `return_phases` parameters, the
            function returns some combination of the following: \n
                * waves (np.ndarray): The generated sine waves.
                * x (np.ndarray): The x-values.
                * phases (np.ndarray): The phases of the sine waves.
    """
    x = np.linspace(0, n_periods * np.pi*2, n_samples)

    phases = np.stack([
        x - ii * np.pi * (2 / n_waves) for ii in range(n_waves)
    ])
    waves = np.cos(phases)
    
    if return_x and return_phases:
        return waves, x, phases
    elif return_x:
        return waves, x
    elif return_phases:
        return waves, phases
    else:
        return waves