import pytest
import torch
import numpy as np
import scipy.signal
from hypothesis import given, strategies as st

from ..spectral import torch_coherence

# Test with basic sinusoidal inputs
def test_basic_functionality():
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    y = np.sin(2 * np.pi * 5 * t + np.pi/4)  # 5 Hz sinusoid, phase shifted

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    # Use default parameters
    fs = 1.0
    nperseg = 256

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, nperseg=nperseg)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, nperseg=nperseg)

    # Check if the results are close enough
    assert np.allclose(coherence_pytorch.numpy(), coherence_scipy, atol=1e-2), "Coherence values do not match closely enough."

# Test varying sampling frequencies
@pytest.mark.parametrize("fs", [0.5, 1.0, 2.0, 10.0])
def test_varying_sampling_frequency(fs):
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    y = np.sin(2 * np.pi * 5 * t + np.pi/4)  # 5 Hz sinusoid, phase shifted

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    nperseg = 256

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, nperseg=nperseg)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, nperseg=nperseg)

    # Check if the results are close enough
    assert np.allclose(coherence_pytorch.numpy(), coherence_scipy, atol=1e-2), f"Coherence values do not match closely enough for fs={fs}."

# Test different window types
@pytest.mark.parametrize("window", ['hann', 'hamming', 'blackman'])
def test_different_window_types(window):
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    y = np.sin(2 * np.pi * 5 * t + np.pi/4)  # 5 Hz sinusoid, phase shifted

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    fs = 1.0
    nperseg = 256

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, window=window, nperseg=nperseg)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, window=window, nperseg=nperseg)

    # Check if the results are close enough
    assert np.allclose(coherence_pytorch.numpy(), coherence_scipy, atol=1e-2), f"Coherence values do not match closely enough with window type={window}."

# Test varying segment lengths
@pytest.mark.parametrize("nperseg", [128, 256, 512])
def test_varying_segment_lengths(nperseg):
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    y = np.sin(2 * np.pi * 5 * t + np.pi/4)  # 5 Hz sinusoid, phase shifted

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    fs = 1.0
    window = 'hann'

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, window=window, nperseg=nperseg)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, window=window, nperseg=nperseg)

    # Check if the results are close enough
    assert np.allclose(coherence_pytorch.numpy(), coherence_scipy, atol=1e-2), f"Coherence values do not match closely enough for segment length={nperseg}."

# Test varying overlap sizes
@pytest.mark.parametrize("noverlap", [0, 1, 2, 64, 128, 192])
def test_overlap_sizes(noverlap):
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    y = np.sin(2 * np.pi * 5 * t + np.pi/4)  # 5 Hz sinusoid, phase shifted

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    fs = 1.0
    nperseg = 256  # Fixed segment length for consistency in comparison

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, nperseg=nperseg, noverlap=noverlap)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Check if the results are close enough
    assert np.allclose(coherence_pytorch.numpy(), coherence_scipy, atol=1e-2), f"Coherence values do not match closely enough for overlap size={noverlap}."

# Test varying FFT lengths
@pytest.mark.parametrize("nfft", [256, 512, 1024])
def test_fft_lengths(nfft):
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    y = np.sin(2 * np.pi * 5 * t + np.pi/4)  # 5 Hz sinusoid, phase shifted

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    fs = 1.0
    nperseg = 256  # Maintain constant segment size to isolate the effect of nfft

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, nperseg=nperseg, nfft=nfft)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, nperseg=nperseg, nfft=nfft)

    # Check if the results are close enough
    assert np.allclose(coherence_pytorch.numpy(), coherence_scipy, atol=1e-2), f"Coherence values do not match closely enough for FFT length={nfft}."

# Test detrending methods
@pytest.mark.parametrize("detrend", ['constant', 'linear'])
def test_detrending_methods(detrend):
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t) + np.linspace(0, 1, 1000)  # Sinusoid with linear trend
    y = np.sin(2 * np.pi * 5 * t + np.pi/4) + np.linspace(1, 0, 1000)  # Sinusoid with inverse linear trend

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    fs = 1.0
    nperseg = 256

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, nperseg=nperseg, detrend=detrend)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, nperseg=nperseg, detrend=detrend)

    # Check if the results are close enough
    assert np.allclose(coherence_pytorch.numpy(), coherence_scipy, atol=1e-2), f"Coherence values do not match closely enough for detrend method={detrend}."

# Test multi-dimensional input
def test_multi_dimensional_input():
    np.random.seed(0)  # For reproducibility
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)  # 5 Hz sinusoid
    y = np.sin(2 * np.pi * 5 * t + np.pi/4)  # 5 Hz sinusoid, phase shifted

    # Extend to 2D by repeating the array
    x = np.tile(x, (10, 1))
    y = np.tile(y, (10, 1))

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)

    fs = 1.0
    nperseg = 256

    freqs_pytorch, coherence_pytorch = torch_coherence(x_torch, y_torch, fs=fs, nperseg=nperseg)
    freqs_scipy, coherence_scipy = scipy.signal.coherence(x, y, fs=fs, nperseg=nperseg, axis=1)

    # Check if the results are close enough, comparing each ensemble member's coherence
    for i in range(10):
        assert np.allclose(coherence_pytorch[i].numpy(), coherence_scipy[i], atol=1e-2), "Coherence values do not match for multi-dimensional input."
