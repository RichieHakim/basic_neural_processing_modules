import numpy as np
import torch

def squeeze_integers(intVec):
    """
    Make integers in an array consecutive numbers
     starting from 0. ie. [7,2,7,4,1] -> [3,2,3,1,0].
    Useful for removing unused class IDs from y_true
     and outputting something appropriate for softmax.
    This is v2. The old version is busted.
    RH 2021
    
    Args:
        intVec (np.ndarray):
            1-D array of integers.
    
    Returns:
        intVec_squeezed (np.ndarray):
            1-D array of integers with consecutive numbers
    """
    uniques = np.unique(intVec)
    # unique_positions = np.arange(len(uniques))
    unique_positions = np.arange(uniques.min(), uniques.max()+1)
    return unique_positions[np.array([np.where(intVec[ii]==uniques)[0] for ii in range(len(intVec))]).squeeze()]

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
    
def idx_to_oneHot(arr, n_classes=None, dtype=None):
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
        dtype = np.bool8 if dtype is None else dtype
    elif type(arr) is torch.Tensor:
        max = torch.max
        zeros = torch.zeros
        arange = torch.arange
        dtype = torch.bool if dtype is None else dtype
    assert arr.ndim == 1

    if n_classes is None:
        n_classes = max(arr)+1
    oneHot = zeros((len(arr), n_classes), dtype=dtype)
    oneHot[arange(len(arr)), arr] = True
    return oneHot


def convert_stringArray_to_oneHot(seq, strs=['A','C','G','T'], safe=False):
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
    
    assert np.all(np.isin(seq_arr, strs)), "Some characters in sequence are not in allowable_chars" if safe else None
    
    nuc_ints = np.array(strs).view(np.uint32)
    seq_ints = seq_arr.view(np.uint32)
    oneHot = np.vstack([seq_ints==n for n in nuc_ints]).T
    return oneHot

