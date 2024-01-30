[![PyPI version](https://badge.fury.io/py/bnpm.svg)](https://badge.fury.io/py/bnpm)
[![Downloads](https://pepy.tech/badge/bnpm)](https://pepy.tech/project/bnpm)
[![repo size](https://img.shields.io/github/repo-size/RichieHakim/basic_neural_processing_modules)](https://github.com/RichieHakim/basic_neural_processing_modules/)

#  basic_neural_processing_modules 
Personal library of functions used in analyzing neural data.
If you find a bug or just want to reach out: RichHakim@gmail.com

## Installation 
Normal installation of `bnpm` does not install all possible dependencies; there are some specific functions that wrap libraries that may need to be installed separately on a case-by-case basis.

Install stable version:
```
pip install bnpm[core]
```

If installing on a server or any computer without graphics/display, after installing `bnpm`, please uninstall `opencv-contrib-python` and install `opencv-contrib-python-headless` instead. 
```
pip uninstall opencv-contrib-python
pip install opencv-contrib-python-headless
```

Install development version:
```
pip install git+https://github.com/RichieHakim/basic_neural_processing_modules.git
```

import with:
```
import bnpm
```


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
