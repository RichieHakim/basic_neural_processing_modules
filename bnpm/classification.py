import numpy as np
import torch

def squeeze_integers(intVec, min_val: int=None):
    """
    Make integers in an array consecutive numbers
     starting from the smallest value. 
    ie. [7,2,7,4,-1,0] -> [3,2,3,1,-1,0].
    Useful for removing unused class IDs.
    This is v3.
    RH 2023
    
    Args:
        intVec (np.ndarray):
            1-D array of integers.
        min_val Optional[int]:
            If provided, then this overrides the minimum value.
    
    Returns:
        intVec_squeezed (np.ndarray):
            1-D array of integers with consecutive numbers
    """
    if isinstance(intVec, list):
        intVec = np.array(intVec, dtype=np.int64)
    if isinstance(intVec, np.ndarray):
        unique, arange = np.unique, np.arange
    elif isinstance(intVec, torch.Tensor):
        unique, arange = torch.unique, torch.arange
        
    u, inv = unique(intVec, return_inverse=True)  ## get unique values and their indices
    u_min = u.min() if min_val is None else min_val  ## get the smallest value
    u_s = arange(u_min, u_min + u.shape[0], dtype=u.dtype)  ## make consecutive numbers starting from the smallest value
    return u_s[inv]  ## return the indexed consecutive unique values

def confusion_matrix(y_hat, y_true):
    """
    Compute the confusion matrix from y_hat and y_true.
    y_hat should be either predictions or probabilities.
    RH 2021

    Args:
        y_hat (np.ndarray): 
            numpy array of predictions or probabilities. 
            Either PREDICTIONS: 2-D array of booleans
             ('one hots') or 1-D array of predicted 
             class indices.
            Or PROBABILITIES: 2-D array floats ('one hot
             like')
        y_true (np.ndarray):
            Either 1-D array of true class indices OR a
             precomputed onehot matrix.

    Returns:
        conf_mat (np.ndarray):
            2-D array of confusion matrix.
            Columns are true classes, rows are predicted.
            Note that this is the transpose of the
             sklearn convention.
    """
    n_classes = max(np.max(y_true)+1, np.max(y_hat)+1)
    if y_hat.ndim == 1:
        y_hat = idx_to_oneHot(y_hat, n_classes)
    if y_true.ndim == 1:
        y_true = idx_to_oneHot(y_true, n_classes)
    cmat = np.dot(y_hat.T, y_true)
    return cmat / np.sum(cmat, axis=0)[None,:]
    

def idx_to_oneHot(arr, n_classes: int=None):
    """
    Convert an array of class indices to matrix of
     one-hot vectors.
    RH 2021

    Args:
        arr (np.ndarray):
            1-D array of class indices.
            Values should be integers >= 0.
            Values will be used as indices in the
             output array.
        n_classes (int):
            Number of classes.
    
    Returns:
        oneHot (np.ndarray):
            2-D array of one-hot vectors.
            Shape is (len(arr), n_classes).
    """
    if type(arr) is np.ndarray:
        max = np.max
        zeros = np.zeros
        arange = np.arange
        dtype = np.bool_ if dtype is None else dtype
    if type(arr) is torch.Tensor:
        max = torch.max
        zeros = torch.zeros
        arange = torch.arange
    assert arr.ndim == 1

    if n_classes is None:
        n_classes = max(arr)+1
    
    ## Make an array of repeated arange(len(arr)) vector
    indices = arange(n_classes)[None,:]
    ## Compare each element of arr to each element of indices
    oneHot = arr[:,None] == indices
    return oneHot


def convert_stringArray_to_oneHot(seq, strs=None, safe=False):
    """
    Convert a sequence of string elements to a one-hot matrix.
    RH 2022

    Args:
        seq (str):
            Sequence of string elements.
            example ['A','C','G','T','A','C','G','T']
        strs (list):
            List of string elements to use.
            Number of output columns will be len(strs).
            example: ['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V']
            If None, will use np.unique(seq) to get all unique elements.
        safe (bool):
            If True, will check that all string elements in seq
             are in strs.

    Returns:
        oneHot (np.ndarray):
            2-D array of one-hot vectors.
            Columns correspond to string elements, rows to
             positions in seq.
    """
    seq_arr = np.char.upper(np.array(seq, dtype='U1'))

    if strs is None:
        strs = np.unique(seq_arr)
    strs = np.char.upper(np.array(strs, dtype='U1'))
    
    if not safe:
        assert np.all(np.isin(seq_arr, strs)), "Some characters in sequence are not in strs"
    
    nuc_ints = np.array(strs).view(np.uint32)
    seq_ints = seq_arr.view(np.uint32)
    oneHot = np.vstack([seq_ints==n for n in nuc_ints]).T
    return oneHot
