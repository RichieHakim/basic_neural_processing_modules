#  basic_neural_processing_modules 
Personal library of functions used in analyzing neural data.
If you find a bug or just want to reach out: RichHakim@gmail.com

My favorite and most used functions:
- Signal Processing:
    - `featurization.Toeplitz_convolution2d`
        - Fast 1D and 2D convolution using sparse Toeplitz matrices.
        - Allows for sparse convolution.
    - `spectral.VQT`
        - Variable-Q transform.
        - Fast, accurate, and flexible. GPU method available.
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
        - Creates batches of data for training or loading
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

