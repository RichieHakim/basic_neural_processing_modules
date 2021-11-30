#  basic_neural_processing_modules 
Personal library of functions used in analyzing neural data.
If you find a bug or just want to reach out: RichHakim@gmail.com

My favorite and most used functions:
- Machine Learning:
    - `decomposition.torch_PCA`
        - standard PCA using PyTorch
    - `linear_regression.LinearRegression_sweep`
        - Performs linear regression with a variety of hyperparameters and methods (L1, L2, Logistic, GPU using cuml)
    - `misc.make_batches`
        - Creates batches of data for training or loading
    - `similarity.orthogonalize` and `similarity.pairwise_orthogonalization`
        - Orthogonalize a matrix relative to a set of vectors
    - `torch_helpers.tensor_sizeOnDisk`
        - Returns the size of a tensor on disk
    - `torch_helpers.set_device`
        - Sets the device for PyTorch

- Signal Processing:
    - `timeSeries.convolve_along_axis`
        - Convolve a matrix along a specified axis    
    - `timeSeries.rolling_percentile_rq_multicore`
        - Very fast rolling percentile calculation
    -  `timeSeries.event_triggered_traces`
        - Creates a matrix of aligned traces relative to specified event times

- Neuroscience:
    - `ca2p_preprocessing.make_dFoF`
        - High speed calculation of dF/F traces

- Miscellaneous
    - `misc.estimate_size_of_float_array`
        - Estimates the size of a float array in bytes
    - `h5_handling.simple_save` and `h5_handling.simple_load`
        - simple loading and saving of dictionaries as nested h5 files
    -`parallel_helpers.multiprocessing_pool_along_axis`
        - easy parallelization of a function along an axis
    - `plotting_helpers.get_subplot_indices`
        - returns the subscript indices of the subplots in a figure
