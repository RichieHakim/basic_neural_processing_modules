#  basic_neural_processing_modules 
Personal library of functions used in analyzing neural data.
If you find a bug or just want to reach out: RichHakim@gmail.com


My favorites:
- **`featurization.Toeplitz_convolution2d`**
    - 1D and 2D convolution. Uses sparse Toeplitz matrix multiplication to speed up computation.
    - **Allows for sparse convolution.**
    - Same options as scipy.signal.convolve2d, but orders of magnitude faster in most cases.
- **`spectral.VQT`**
    - Variable-Q transform. Generates spectrograms with variable frequency resolution.
    - Comparable to librosa's VQT, but faster, more flexible, without approximations, with GPU support, and pytorch autograd compatible.


Other useful functions:
- Signal Processing:
    - `timeSeries.rolling_percentile_rq_multicore`
        - Fast rolling percentile calculation
    -  `timeSeries.event_triggered_traces`
        - Fast creation of a matrix of aligned traces relative to specified event times

- Machine Learning:
    - `decomposition.torch_PCA`
        - Fast standard PCA using PyTorch
    - `linear_regression.LinearRegression_sweep`
        - Performs linear regression with a variety of hyperparameters and methods (L1, L2, Logistic, optional GPU methods using cuml)
    - `misc.make_batches`
        - Creates batches of data or any other iterable
    - `similarity.orthogonalize` and `similarity.pairwise_orthogonalization`
        - Orthogonalize a matrix relative to a set of vectors using a Gram-Schmidt related process

- Miscellaneous
    - `misc.estimate_size_of_float_array`
        - Estimates the size of a float array in bytes
    - `image_processing.play_video_cv2`
        - Plays and/or saves a 3D array as a video using OpenCV
    - `h5_handling.simple_save` and `h5_handling.simple_load`
        - Simple lazy loading and saving of dictionaries as nested h5 files
    - `parallel_helpers.multiprocessing_pool_along_axis`
        - Easy parallelization of a function along an axis
    - `plotting_helpers.get_subplot_indices`
        - Returns the subscript indices of the subplots in a figure
    - `classification.squeeze_integers`
        - Removes the gaps between integers in an integer array (e.g. [-1, 2, 4, 5] -> [-1, 0, 1, 2])

Dependencies: \
```pip install matplotlib numpy scipy scikit-learn scikit-image tqdm scanimage-tiff-reader numba pandas scikit-learn scikit-image h5py hdfdict opencv-contrib-python ipywidgets opt_einsum rolling_quantiles pywavesurfer```

```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```







benchmarks
----------
### Toeplitz_convolution2d:

init_time: 0.001s,  call_time: 0.001s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((1, 1000), (1, 5))
init_time: 0.019s,  call_time: 0.054s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((1, 100000), (1, 5))
init_time: 0.234s,  call_time: 0.512s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((1, 1000000), (1, 5))
init_time: 0.001s,  call_time: 0.003s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((1, 10000), (1, 2))
init_time: 0.009s,  call_time: 0.013s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((1, 10000), (1, 20))
init_time: 0.101s,  call_time: 0.157s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((1, 10000), (1, 200))
init_time: 0.003s,  call_time: 0.001s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((2, 2), (10, 10))
init_time: 0.005s,  call_time: 0.001s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((16, 16), (10, 10))
init_time: 0.015s,  call_time: 0.002s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((64, 64), (10, 10))
init_time: 0.151s,  call_time: 0.497s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (10, 10))
init_time: 2.775s,  call_time: 8.743s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((1024, 1024), (10, 10))
init_time: 0.061s,  call_time: 0.126s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (5, 5))
init_time: 0.144s,  call_time: 0.499s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (5, 20))
init_time: 0.596s,  call_time: 1.921s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (5, 80))
init_time: 0.035s,  call_time: 0.021s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (2, 2))
init_time: 0.046s,  call_time: 0.073s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (4, 4))
init_time: 0.424s,  call_time: 1.256s,  density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (16, 16))
init_time: 10.359s, call_time: 36.875s, density: 0.0001, batch_size: 100000  conv_mode: 'full', shapes x, k: ((256, 256), (64, 64))
