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

If installing on a server or any computer without graphics/display, install using `core_cv2Headless`. If you accidentally installed the normal version, simply please uninstall `pip uninstall opencv-contrib-python` and install `pip install opencv-contrib-python-headless` instead. 


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
- **`automatic_regression`** module
    - Allows for easy and fast hyperparameter optimization of regression models
    - Any model with a `fit` and `predict` method can be used (e.g. `sklearn` and similar)
    - Uses `optuna` for hyperparameter optimization

Other useful functions:
- Signal Processing:
    - `timeSeries.rolling_percentile_rq_multicore`
        - Fast rolling percentile calculation
    -  `timeSeries.event_triggered_traces`
        - Fast creation of a matrix of aligned traces relative to specified event times

- Machine Learning:
    - `neural_networks` module
        - Has nice RNN regression and classification classes
    - `decomposition.torch_PCA`
        - Fast standard PCA using PyTorch
    - `similarity.orthogonalize`
        - Orthogonalize a matrix relative to a set of vectors using OLS or Gram-Schmidt process

- Miscellaneous
    - `path_helpers.find_paths`
        - Find paths to files and/or folders in a directory. Searches recursively using regex.
    - `image_processing.play_video_cv2`
        - Plays and/or saves a 3D array as a video using OpenCV
    - `h5_handling.simple_save` and `h5_handling.simple_load`
        - Simple lazy loading and saving of dictionaries as nested h5 files