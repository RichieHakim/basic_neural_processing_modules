#  basic_neural_processing_modules 
Personal library of functions used in analyzing neural data.
If you find a bug or just want to reach out: RichHakim@gmail.com

## Installation 
```
cd path/to/your/preferred/directory
git clone https://github.com/RichieHakim/basic_neural_processing_modules
cd path/to/basic_neural_processing_modules
pip install -e .
```
you can now import with: `import bnpm`

## Usage 
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
    - `path_helpers.find_paths`
        - Find paths to files and/or folders in a directory. Searches recursively using regex.
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