import numpy as np

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
    unique_positions = np.arange(len(uniques))
    return unique_positions[np.array([np.where(intVec[ii]==uniques)[0] for ii in range(len(intVec))]).squeeze()]

def confusion_matrix(y_hat, y_true):
    """
    Compute the confusion matrix from y_hat and y_true.
    y_hat should be either predictions ().
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
    """
    n_classes = np.max(y_true)+1
    if y_hat.ndim == 1:
        y_hat = idx_to_oneHot(y_hat, n_classes)
    cmat = y_hat.T @ idx_to_oneHot(y_true, n_classes)
    return cmat / np.sum(cmat, axis=0)[None,:]
    
def idx_to_oneHot(arr, n_classes=None):
    """
    Convert an array of class indices to matrix of
     one-hot vectors.
    RH 2021

    Args:
        arr (np.ndarray):
            1-D array of class indices.
        n_classes (int):
            Number of classes.
    
    Returns:
        oneHot (np.ndarray):
            2-D array of one-hot vectors.
    """
    if n_classes is None:
        n_classes = np.max(arr)+1
    oneHot = np.zeros((arr.size, n_classes))
    oneHot[np.arange(arr.size), arr] = 1
    return oneHot