'''
Table of Contents

Functions and Interdependencies:
    proj
    orthogonalize
        - proj
    OLS
    EV
    pairwise_similarity
    best_permutation
        - pairwise_similarity
    self_similarity_pairwise
        - best_permutation
'''
import numpy as np
import scipy.optimize
# import sklearn.decomposition
from numba import njit, prange, jit
import torch
from tqdm import tqdm

from . import indexing

import copy
import time
from functools import partial

def proj(v1, v2):
    '''
    Projects one or more vectors (columns of v1) onto one or more vectors (v2)
    RH 2021

    Args:
        v1 (ndarray): 
            vector set 1. Either a single vector or a 2-D array where the columns
             are the vectors
        v2 (ndarray): 
            vector set 2. Either a single vector or a 2-D array where the columns
             are the vectors
            If None, v2 is set to v1
    
    Returns:
        proj_vec (ndarray): 
            vector set 1 projected onto vector set 2. 
            shape: (v1.shape[0], v1.shape[1], v2.shape[1])
        proj_score (ndarray or scalar): 
            projection scores. shape: (v1.shape[1], v2.shape[1])
    '''
    if isinstance(v1, np.ndarray):
        from opt_einsum import contract
        einsum = contract
        norm = partial(np.linalg.norm, axis=0, keepdims=True)
    elif isinstance(v1, torch.Tensor):
        from torch import einsum
        norm = partial(torch.linalg.norm, dim=0, keepdim=True)

    if v2 is None:
        v2 = v1
    if v1.ndim < 2:
        v1 = v1[:,None]
    if v2.ndim < 2:
        v2 = v2[:,None]

    assert v1.shape[0] == v2.shape[0], f"v1.shape[0] must equal v2.shape[0]. Got v1.shape[0]={v1.shape[0]} and v2.shape[0]={v2.shape[0]}"

    u = v2 / norm(v2)
    proj_score = v1.T @ u 
    proj_vec = einsum('ik,jk->ijk', u, proj_score)

    return proj_vec, proj_score

def vector_angle(v1, v2=None, mode='cosine_similarity'):
    '''
    Calculates the angle between two vector sets.
    RH 2023

    Args:
        v1 (ndarray):
            vector set 1. Either a single vector or a 2-D array where the columns
             are the vectors
        v2 (ndarray):
            vector set 2. Either a single vector or a 2-D array where the columns
             are the vectors
            If None, v2 is set to v1
        mode (str):
            'deg': returns angle in degrees
            'rad': returns angle in radians
            'cosine_similarity': returns cosine similarity

    Returns:
        angle (ndarray or scalar):
            angle between v1 and v2. shape: (v1.shape[1], v2.shape[1])
    '''
    if isinstance(v1, np.ndarray):
        norm = partial(np.linalg.norm, axis=0, keepdims=True)
        arccos = np.arccos
        rad2deg = np.rad2deg
    elif isinstance(v1, torch.Tensor):
        norm = partial(torch.linalg.norm, dim=0, keepdim=True)
        arccos = torch.acos
        rad2deg = torch.rad2deg

    if v2 is None:
        v2 = v1
    if v1.ndim < 2:
        v1 = v1[:,None]
    if v2.ndim < 2:
        v2 = v2[:,None]

    assert v1.shape[0] == v2.shape[0], f"v1.shape[0] must equal v2.shape[0]. Got v1.shape[0]={v1.shape[0]} and v2.shape[0]={v2.shape[0]}"
    
    u1 = v1 / norm(v1)
    u2 = v2 / norm(v2)
    
    if mode == 'cosine_similarity':
        return u1.T @ u2
    elif mode == 'deg':
        return rad2deg(arccos(u1.T @ u2))
    elif mode == 'rad':
        return arccos(u1.T @ u2)
    else:
        raise ValueError(f"mode must be 'cosine_similarity', 'deg', or 'rad'. Got {mode}")

def orthogonalize(v1, v2, method='OLS', device='cpu', thresh_EVR_PCA=1e-15):
    '''
    Orthogonalizes one or more vectors (columns of v1) relative to another set 
     of vectors (v2). Subtracts the projection of v1 onto v2 off of v1.
    RH 2021-2023

    Args:
        v1 (ndarray): 
            vector set 1. (y_true)
            Either a single vector or a 2-D array where the columns
             are the vectors. shape: (n_samples, n_vectors)
        v2 (ndarray): 
            vector set 2. (y_pred)
            Either a single vector or a 2-D array where the columns
             are the vectors. shape: (n_samples, n_vectors)
        method (str):
            'serial': uses a gram-schmidt like iterative process to orthogonalize
             v1 off of v2. This method may have some minor numerical instability 
             issues when v2 contains many (>100) vectors.
            'OLS': uses OLS regression to orthogonalize v1 off of v2. This method 
             is numerically stable when v2 has many columns, and usually faster.
             However, OLS can have issues when v2 is not full rank or singular.
        device (str):
            Device to use for torch tensors. 
            Default is 'cpu'.
        thresh_EVR_PCA (scalar):
            Threshold for the Explained Variance Ratio (EVR) of the PCA components.
            Set so that it might act as a cutoff for redundant components.
            If the EVR of a component is below this threshold, it is not used to
             orthogonalize v1. This is useful when v2 is singular or nearly singular.
    
    Returns:
        v1_orth (ndarray): 
            vector set 1 with the projections onto vector set 2 subtracted off.
             Same size as v1.
        EVR (ndarray): 
            Explained Variance Ratio for each column of v1. 
            Amount of variance that all the vectors in v2 can explain for each 
            vector in v1.
            When v1 is z-scored, EVR is equivalent to pearsons R^2; 
            as in pairwise_similarity(OLS(v2, v1)[1] , v1)**2
        EVR_total (scalar):
            total amount of variance explained in v1 by v2
        pca_dict (dict):
            dictionary containing the PCA outputs:
                'comps': PCA components
                'scores': PCA scores
                'singVals': PCA singular values
                'EVR': PCA Explained Variance Ratio
                'PCs_above_thresh': boolean array indicating which PCs are above
                 the threshold for EVR
    '''
    if isinstance(v1, np.ndarray):
        v1 = torch.as_tensor(v1)
        return_numpy = True
    else:
        return_numpy = False
    if isinstance(v2, np.ndarray):
        v2 = torch.as_tensor(v2)

    v1 = v1.to(device)
    v2 = v2.to(device)

    if v1.ndim < 2:
        v1 = v1[:,None]
    if v2.ndim < 2:
        v2 = v2[:,None]
    
    # I'm pretty sure using PCA is fine for this.
    if v2.shape[1] > 1:
        from .decomposition import torch_pca
        comps, pc_scores, singVals, pc_EVR = torch_pca(
            X_in=v2, 
            rank=v2.shape[1], 
            mean_sub=True,  
            device=device, 
            return_cpu=False, 
            return_numpy=isinstance(v1, np.ndarray), 
            cuda_empty_cache=False
        )
        pca_dict = {
            'comps': comps,
            'scores': pc_scores,
            'singVals': singVals,
            'EVR': pc_EVR,
            'PCs_above_thresh': pc_EVR > thresh_EVR_PCA,
        }
        pc_scores_aboveThreshold = pcsat = pc_scores[:, pca_dict['PCs_above_thresh']]
    else:
        pcsat = v2
        pca_dict = None

    # decomp = sklearn.decomposition.PCA(n_components=v2.shape[1])
    # pc_scores = decomp.fit_transform(v2)

    v1_means = torch.mean(v1, dim=0)
    if method == 'serial':
        # Serial orthogonalization.
        v1_orth = copy.deepcopy(v1 - v1_means)
        for ii in range(pcsat.shape[1]):
            proj_vec = proj(v1_orth , pcsat[:,ii])[0]
            v1_orth = v1_orth.squeeze() - proj_vec.squeeze()
    elif method == 'OLS':
        # Ordinary Least Squares.
        X = torch.cat((pcsat, torch.ones((pcsat.shape[0], 1), dtype=pcsat.dtype, device=device)), dim=1)
        theta = torch.linalg.inv(X.T @ X) @ X.T @ (v1 - v1_means)
        y_rec = X @ theta
        v1_orth = v1 - y_rec + X[:,-1][:,None] * theta[-1]  ## this adds back the bias term

    v1_orth = v1_orth + v1_means

    EVR = 1 - (torch.var(v1_orth, dim=0) / torch.var(v1, dim=0))
    EVR_total = 1 - ( torch.sum(torch.var(v1_orth, dim=0), dim=0) / torch.sum(torch.var(v1, dim=0), dim=0) )

    if return_numpy:
        v1_orth = v1_orth.cpu().numpy()
        EVR = EVR.cpu().numpy()
        EVR_total = EVR_total.cpu().numpy()

        if pca_dict is not None:
            for key in pca_dict.keys():
                if isinstance(pca_dict[key], torch.Tensor):
                    pca_dict[key] = pca_dict[key].cpu().numpy()

    return v1_orth, EVR, EVR_total, pca_dict


@njit
def pair_orth_helper(v1, v2):
    """
    Helper function for main pairwise_orthogonalization
     function. Performs the pairwise orthogonalization
     by subtracting off v2 from v1. Uses numba to speed
     up the computation.
     v1 = v1 - proj(v1 onto v2)
    RH 2021

    Args:
        v1 (ndarray):
            Vector set 1. Columns are vectors.
        v2 (ndarray):
            Vector set 2. Columns are vectors.
    
    Returns:
        v1_orth (ndarray):
            v1 - proj(v1 onto v2)
    """
    return  v1 - (np.diag(np.dot(v1.T, v2)) / np.diag(np.dot(v2.T, v2))) * v2
def pairwise_orthogonalization(v1, v2, center:bool=True):
    """
    Orthogonalizes columns of v2 off of the columns of v1
     and returns the orthogonalized v1 and the explained
     variance ratio of v2 off of v1.
    v1: y_true, v2: y_pred
    Since it's just pairwise, there should not be any 
     numerical instability issues.
    RH 2021

    Args:
        v1 (ndarray):
            y_true
            Vector set 1. Either a single vector or a 2-D 
             array where the columns are the vectors.
        v2 (ndarray):  
            y_pred
            Vector set 2. Either a single vector or a 2-D
             array where the columns are the vectors.
        center (bool):
            Whether to center the vectors.
            Centering prevents negative EVR values.
    
    Returns:
        v1_orth (ndarray):
            Vector set 1 with the projections onto vector 
             set 2 subtracted off.
            Same size as v1.
        EVR (ndarray):
            Explained Variance Ratio for each column of v1.
            Amount of variance that all the vectors in v2 
             can explain for each vector in v1.
        EVR_total_weighted (scalar):
            Average amount of variance explained in v1 by v2
             weighted by the variance of each column of v1.
        EVR_total_unweighted (scalar):
            Average amount of variance explained in v1 by v2
    """
    assert v1.ndim == v2.ndim
    if v1.ndim==1:
        v1 = v1[:,None]
        v2 = v2[:,None]
    assert v1.shape[1] == v2.shape[1]
    assert v1.shape[0] == v2.shape[0]
    
    if center:
        v1 = v1 - np.mean(v1, axis=0)
        v2 = v2 - np.mean(v2, axis=0)
    
    v1_orth = pair_orth_helper(v1, v2)

    v1_var = np.var(v1, axis=0)
    EVR = 1 - (np.var(v1_orth, axis=0) / v1_var)

    EVR_total_weighted = np.sum(v1_var * EVR) / np.sum(v1_var)
    EVR_total_unweighted = np.mean(EVR)
    return v1_orth, EVR, EVR_total_weighted, EVR_total_unweighted


@torch.jit.script
def pairwise_orthogonalization_torch_helper(v1, v2, center:bool=True):
    # assert v1.ndim == v2.ndim
    if v1.ndim==1:
        v1 = v1[:,None]
    if v2.ndim==1:
        v2 = v2[:,None]
    # assert v1.shape[1] == v2.shape[1]
    assert v1.shape[0] == v2.shape[0]
    
    if center:
        v1 = v1 - torch.nanmean(v1, dim=0)
        v2 = v2 - torch.nanmean(v2, dim=0)

    # v1_orth = v1 - (torch.diag(torch.matmul(v1.T, v2)) / torch.diag(torch.matmul(v2.T, v2)))*v2
    v1_orth = v1 - (torch.nansum(v1 * v2, dim=0) / torch.nansum(v2 * v2, dim=0) )*v2

    v1_var = torch.var(v1, dim=0)
    EVR = 1 - (torch.var(v1_orth, dim=0) / v1_var)

    EVR_total_weighted = torch.nansum(v1_var * EVR) / torch.sum(v1_var)
    EVR_total_unweighted = torch.nanmean(EVR)
    return v1_orth, EVR, EVR_total_weighted, EVR_total_unweighted
def pairwise_orthogonalization_torch(v1, v2, center:bool=True, device='cpu'):
    """
    Orthogonalizes columns of v2 off of the columns of v1
     and returns the orthogonalized v1 and the explained
     variance ratio of v2 off of v1.
    Use center=True to get guarantee zero correlation between
     columns of v1_orth and v2.
    v1: y_true, v2: y_pred
    Since it's just pairwise, there should not be any 
     numerical instability issues.
    Same as pairwise_orthogonalization, but with 
     torch.jit.script instead of njit.
    RH 2021

    Args:
        v1 (ndarray):
            y_true
            Vector set 1. Either a single vector or a 2-D 
             array where the columns are the vectors.
        v2 (ndarray):  
            y_pred
            Vector set 2. Either a single vector or a 2-D
             array where the columns are the vectors.
        center (bool):
            Whether to center the vectors.
            Centering prevents negative EVR values.
    
    Returns:
        v1_orth (ndarray):
            Vector set 1 with the projections onto vector 
             set 2 subtracted off.
            Same size as v1.
        EVR (ndarray):
            Explained Variance Ratio for each column of v1.
            Amount of variance that all the vectors in v2 
             can explain for each vector in v1.
        EVR_total_weighted (scalar):
            Average amount of variance explained in v1 by v2
             weighted by the variance of each column of v1.
        EVR_total_unweighted (scalar):
            Average amount of variance explained in v1 by v2
    """
    if isinstance(v1, np.ndarray):
        v1 = torch.from_numpy(v1)
        return_numpy = True
    else:
        return_numpy = False
    if isinstance(v2, np.ndarray):
        v2 = torch.from_numpy(v2)
    v1 = v1.to(device)
    v2 = v2.to(device)
    v1_orth, EVR, EVR_total_weighted, EVR_total_unweighted = pairwise_orthogonalization_torch_helper(v1, v2, center=center)
    if return_numpy:
        v1_orth = v1_orth.cpu().numpy()
        EVR = EVR.cpu().numpy()
        EVR_total_weighted = EVR_total_weighted.cpu().numpy()
        EVR_total_unweighted = EVR_total_unweighted.cpu().numpy()
    return v1_orth, EVR, EVR_total_weighted, EVR_total_unweighted

def EV(y_true, y_pred):
    '''
    Explained Variance
    Calculating as 1 - (SS_residual_variance / SS_original_variance)
    Should be exactly equivalent to sklearn's 
    sklearn.metrics.explained_variance_score but slightly faster and provides
    most relevant outputs all together
    RH 2021

    Args:
        y_true (ndarray): 
            array with same size as y_pred. The columns are
            individual y vectors
        y_pred (ndarray):
            array with same size as y_true. The columns are the predicted
            y vectors
    
    Returns:
        EV (1-D array):
            explained variance of each y_true column by each corresponding
            y_pred column. Same as 
            sklearn.metrics.explained_variance_score(y_true, y_pred, multioutput='raw_values')
        EV_total_weighted (scalar):
            total weighted explained variance. Same as
            sklearn.metrics.explained_variance_score(y_true, y_pred, multioutput='variance_weighted')
        EV_total_unweighted (scalar):
            average of all EV values. Same as
            sklearn.metrics.explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    '''

    EV = 1 - np.sum((y_true - y_pred)**2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    y_true_var = np.var(y_true, axis=0)
    EV_total_weighted = np.sum( y_true_var* EV ) / np.sum(y_true_var)
    EV_total_unweighted = np.mean(EV)

    return EV , EV_total_weighted , EV_total_unweighted


def pairwise_similarity(v1 , v2=None , method='pearson' , ddof=1):
    '''
    Computes similarity matrices between two sets of vectors (columns within
    2-D arrays) using either covariance, Pearson correlation or cosine_similarity.
    
    Think of this function as a more general version of np.cov or np.corrcoef
    RH 2021

    Args:
        v1 (ndarray): 
            2-D array of column vectors.
        v2 (ndarray): 
            2-D array of column vectors to compare to vector_set1. If None, then
             the function is a type of autosimilarity matrix
        method (str):
            'cov' - covariance
            'pearson' or 'R' - Pearson correlation
            'cosine_similarity' - cosine similarity
        ddof (scalar/int):
            Used if method=='cov'. Define the degrees of freedom. Set to 1 for
            unbiased calculation, 0 for biased calculation.
    Returns:
        ouput (ndarray):
            similarity matrix dependent on method
    '''

    methods = ['cov', 'pearson', 'R', 'cosine_similarity']
    assert np.isin(method, methods), f'RH Error: method must be one of: {methods}'

    if v2 is None:
        v2 = v1
    
    if v1.ndim == 1:
        v1 = v1[:,None]
    if v2.ndim == 1:
        v2 = v2[:,None]

    if method=='cov':
        v1_ms = v1 - np.mean(v1, axis=0) # 'mean subtracted'
        v2_ms = v2 - np.mean(v2, axis=0)
        output = (v1_ms.T @ v2_ms) / (v1.shape[0] - ddof)
    if method in ['pearson', 'R']: 
        # Below method should be as fast as numpy.corrcoef . 
        # Output should be same within precision, but 
        # numpy.corrcoef makes a doublewide matrix and can be annoying
        
        # Note: Pearson's R-value can be different than sqrt(EV) 
        # calculated below if the residuals are not orthogonal to the 
        # prediction. Best to use EV for R^2 if unsure
        v1_ms = v1 - np.mean(v1, axis=0) # 'mean subtracted'
        v2_ms = v2 - np.mean(v2, axis=0)
        output = (v1_ms.T @ v2_ms) / np.sqrt(np.sum(v1_ms**2, axis=0, keepdims=True).T * np.sum(v2_ms**2, axis=0, keepdims=True))
    if method=='cosine_similarity':    
        output = (v1 / (np.linalg.norm(v1 , axis=0, keepdims=True))).T  @ (v2  / np.linalg.norm(v2 , axis=0, keepdims=True))
    return output


def batched_covariance(X, batch_size=1000, device='cpu'):
    """
    Batched covariance matrix calculation.
    Allows for large datasets to be processed in batches on GPU.
    RH 2022

    Args:
        X (np.ndarray or torch.Tensor):
            2D array of shape (n_samples, n_features)
        batch_size (int):
            Number of samples to process at a time.
        device (str):
            Device to use for computation.
            
    Returns:
        cov (np.ndarray):
    """
    X_dl1 = list(indexing.make_batches(np.arange(X.shape[1]), batch_size=batch_size, return_idx=True))
    X_dl2 = list(indexing.make_batches(np.arange(X.shape[1]), batch_size=batch_size, return_idx=True))

    if torch.is_tensor(X):
        X_cov = torch.zeros(X.shape[1], X.shape[1], device=device)
    else:
        X_cov = np.zeros((X.shape[1], X.shape[1]))
    
    n_batches = X.shape[1] // batch_size
    for ii, (X_batch_i, idx_batch_i) in enumerate(tqdm(X_dl1, total=n_batches, leave=False, desc='outer loop')):
        for jj, (X_batch_j, idx_batch_j) in enumerate(tqdm(X_dl2, total=n_batches, leave=False, desc='inner loop')):
            x_t = X[:,idx_batch_i[0]:idx_batch_i[-1]].T
            x   = X[:,idx_batch_j[0]:idx_batch_j[-1]]
            if device != 'cpu':
                x_t = x_t.to(device)
                x = x.to(device)

            X_cov[idx_batch_i[0]:idx_batch_i[-1], idx_batch_j[0]:idx_batch_j[-1]] = x_t @ x
    return X_cov


def batched_matrix_multiply(X1, X2, batch_size1=1000, batch_size2=1000, device='cpu'):
    """
    Batched matrix multiplication of two matrices.
    Allows for multiplying huge matrices together on a GPU.
    RH 2022

    Args:
        X1 (np.ndarray or torch.Tensor):
            first matrix. shape (n_samples1, n_features1).
        X2 (np.ndarray or torch.Tensor):
            second matrix. shape (n_samples2, n_features2).
        batch_size1 (int):
            batch size for first matrix.
        batch_size2 (int):
            batch size for second matrix
        device (str):
            device to use for computation and output

    Returns:
        X1_X2 (np.ndarray or torch.Tensor):
    """
    X1_dl = indexing.make_batches(np.arange(X1.shape[1]), batch_size=batch_size1, return_idx=True)
    X2_dl = indexing.make_batches(np.arange(X2.shape[1]), batch_size=batch_size2, return_idx=True)

    if torch.is_tensor(X1):
        Y = torch.zeros(X1.shape[1], X2.shape[1], device=device)
    else:
        Y = np.zeros((X1.shape[1], X2.shape[1]))
    
    n_batches1 = X1.shape[1] // batch_size1
    n_batches2 = X2.shape[1] // batch_size2
    for ii, (X_batch_i, idx_batch_i) in enumerate(tqdm(X1_dl, total=n_batches1, leave=False, desc='outer loop')):
        for jj, (X_batch_j, idx_batch_j) in enumerate(X2_dl):
            x1_t = X1[:,idx_batch_i[0]:idx_batch_i[-1]].T
            x2   = X2[:,idx_batch_j[0]:idx_batch_j[-1]]
            if device != 'cpu':
                x1_t = x1_t.to(device)
                x2 = x2.to(device)

            Y[idx_batch_i[0]:idx_batch_i[-1], idx_batch_j[0]:idx_batch_j[-1]] = x1_t @ x2
    return Y
    # X1_dl = indexing.make_batches(X1.T, batch_size=batch_size1, return_idx=True)
    # X2_dl = indexing.make_batches(X2.T, batch_size=batch_size2, return_idx=True)

    # if torch.is_tensor(X1):
    #     Y = torch.zeros(X1.shape[1], X2.shape[1], device=device)
    # else:
    #     Y = np.zeros((X1.shape[1], X2.shape[1]))
    
    # n_batches1 = X1.shape[1] // batch_size1
    # n_batches2 = X2.shape[1] // batch_size2
    # for ii, (X_batch_i, idx_batch_i) in enumerate(tqdm(X1_dl, total=n_batches1, leave=False, desc='outer loop')):
    #     for jj, (X_batch_j, idx_batch_j) in enumerate(X2_dl):
    #         Y[idx_batch_i[0]:idx_batch_i[-1], idx_batch_j[0]:idx_batch_j[-1]] = X_batch_i @ X_batch_j.T
    # return Y


def similarity_to_distance(x, fn_toUse=1, a=1, b=0, eps=0):
    """
    Convert similarity metric to distance metric.
    RH 2022

    Args:
        x (value or array):
            similarity metric.
        fn_toUse (int from 1 to 7):
            Sets the function to use.
        a (float):
            Scaling parameter.
        b (float):
            Shifting parameter.
        eps (float):
            Small value to add to output.

    """
    if fn_toUse == 1:
        d = ((b+1) / (x**a)) -1 # fn 1: 1/x
    if fn_toUse == 2:
        d = np.exp(((-x+b)**a)) # fn 2: exp(-x)
    if fn_toUse == 3:
        d = np.arctan(a*(-x+b)) # fn 3: arctan(-x)
    if fn_toUse == 4:
        d = b - x**a # fn 4: 1-x
    if fn_toUse == 5:
        d = np.sqrt(1-(x+b))**a # fn 5: sqrt(1-x)
    if fn_toUse == 6:
        d = -np.log((x*a)+b) # fn 6: -log(x)
    
    return d + eps
    

def cp_reconstruction_EV(tensor_dense, tensor_CP):
    """
    Explained variance of a reconstructed tensor using
    by a CP tensor (similar to kruskal tensor).
    RH 2023

    Args:
        tensor_dense (np.ndarray or torch.Tensor):
            Dense tensor to be reconstructed. shape (n_samples, n_features)
        tensor_CP (tensorly CPTensor or list of np.ndarray/torch.Tensor):
            CP tensor.
            If a list of factors, then each factor should be a 2D array of shape
            (n_samples, rank).
            Can also be a tensorly CPTensor object.
    """
    tensor_rec = None
    try:
        import tensorly as tl
        if isinstance(tensor_CP, tl.cp_tensor.CPTensor):
            tensor_rec = tl.cp_to_tensor(tensor_CP)
    except ImportError as e:
        raise ImportError('tensorly not installed. Please install tensorly or provide a list of factors as the tensor_CP argument.')
    
    if tensor_rec is None:
        assert isinstance(tensor_CP, list), 'tensor_CP must be a list of factors'
        assert all([isinstance(f, (np.ndarray, torch.Tensor)) for f in tensor_CP]), 'tensor_CP must be a list of factors'
        tensor_rec = indexing.cp_to_dense(tensor_CP)

    if isinstance(tensor_dense, torch.Tensor):
        var = torch.var
    elif isinstance(tensor_dense, np.ndarray):
        var = np.var
    ev = 1 - (var(tensor_dense - tensor_rec) / var(tensor_dense))
    return ev
    

##########################################    
########### Linear Assignment ############
##########################################

def best_permutation(mat1 , mat2 , method='pearson'):
    '''
    This function compares the representations of two sets of vectors (columns
     of mat1 and columns of mat2).
    We assume that the vectors in mat1 and mat2 are similar up to a permutation.
    We therefore find the 'best' permutation that maximizes the similarity
     between the sets of vectors
    RH 2021
    
    Args:
        mat1 (np.ndarray): 
            a 2D array where the columns are vectors we wish to match with mat2
        mat2 (np.ndarray): 
            a 2D array where the columns are vectors we wish to match with mat1
        method (string)  : 
            defines method of calculating pairwise similarity between vectors:
                'pearson' or 'cosine_similarity'
        
    Returns:
        sim_avg (double): 
            the average similarity between matched vectors. Units depend on 
            method
        sim_matched (double): 
            the similarity between each pair of matched vectors.
        ind1 (int): 
            indices of vectors in mat1 matched to ind2 in mat2 (usually just 
            sequential for ind1)
        ind2 (int): 
            indices of vectors in mat2 matched to ind1 in mat1
    '''
    corr = mat1.T @ mat2
    ind1 , ind2 = scipy.optimize.linear_sum_assignment(corr, maximize=True)
    sim_matched = np.zeros(len(ind1))
    for ii in range(len(ind1)):
        if method=='pearson':
            sim_matched[ii] = np.corrcoef(mat1[:,ind1[ii]] , mat2[:,ind2[ii]])[0][1]
        if method=='cosine_similarity':
            sim_matched[ii] = pairwise_similarity( mat1[:,ind1[ii]] , mat2[:,ind2[ii]] , 'cosine_similarity')

    sim_avg = np.mean(sim_matched)
    return sim_avg , sim_matched , ind1.astype('int64') , ind2.astype('int64')

def self_similarity_pairwise(mat_set , method='pearson'):
    '''
    This function compares sets of 2-D matrices within a 3-D array using the 
    'best_permutation' function.
    We assume that the vectors within the matrices are similar up to a 
    permutation.
    We therefore find the 'best' permutation that maximizes the similarity 
    between the sets of vectors within each matrix.
    RH 2021
    
    Args:
        mat_set (np.ndarray): 
            a 3D array where the columns within the first two dims are vectors
             we wish to match with the columns from matrices from other slices
             in the third dimension
        method (string): 
            defines method of calculating pairwise similarity between vectors:
             'pearson' or 'cosine_similarity'

    Returns:
        same as 'best_permutation', but over each combo
        combos: combinations of pairwise comparisons
    '''
    
    import itertools

    n_repeats = mat_set.shape[2]
    n_components = mat_set.shape[1]

    combos = np.array(list(itertools.combinations(np.arange(n_repeats),2)))
    n_combos = len(combos)

    corr_avg = np.zeros((n_combos))
    corr_matched = np.zeros((n_components , n_combos))
    ind1 = np.zeros((n_components , n_combos), dtype='int64')
    ind2 = np.zeros((n_components , n_combos), dtype='int64')
    for i_combo , combo in enumerate(combos):
        corr_avg[i_combo] , corr_matched[:,i_combo] , ind1[:,i_combo] , ind2[:,i_combo]  =  best_permutation(mat_set[:,:,combo[0]]  ,  mat_set[:,:,combo[1]] , method)
    # print(corr_avg)
    return corr_avg, corr_matched, ind1, ind2, combos


def enumerate_paths(edges):
    """
    From Caleb Weinreb 2022

    Create a list of all paths in a directed graph
    
    Parameters
    ----------
    edges: list of tuples
        Edges in the graph as tuples (i,j) for each edge i->j. The
        edges are assumed to be in topopological order, meaning
        edge (i,j) is listed prior to edge (k,l) whenever j==k.
        
    Returns
    -------
    paths: list of tuples
        All directed paths as tuples of node indexes
    """
    child_map = {parent:[] for parent,child in edges}
    for parent,child in edges: child_map[parent].append(child)
    leaf_nodes = [child for _,child in edges if not child in child_map]
    sub_paths = {leaf:[[leaf]] for leaf in leaf_nodes}
    for parent,_ in edges[::-1]:
        if not parent in sub_paths:
            sub_paths[parent] = [[parent]]
            for child in child_map[parent]:
                for path in sub_paths[child]:
                    sub_paths[parent].append([parent]+path)
    return [tuple(p) for p in sum(sub_paths.values(),[])]
def maximum_directed_matching(weights, partition):
    """
    From Caleb Weinreb 2022

    Find the "maximum directed matching" in a weighted n-partite graph. 
    
    Let $G$ be a directed graph with nodes $\{1,...,N\}$ and weighted 
    edges $E_{ij}$. Assume that the nodes of $G$ are partitioned into 
    groups, where $P_i$ denotes the group-label for node $i$, and that 
    all the edges $E_{ij}$ satisfy $P_i < P_j$. In other words, all 
    edges go from lower to higher partition index. 
    
    We define a collection of node-sets as a "directed matching" if 
    each set forms a connected component in $G$ and no node belongs to
    more than one set. The maximum directed matching is defined as the 
    directed matching that maximizes the sum edge weights across all 
    edges that originate and terminate in the same node-set. 

    Parameters
    ----------
    weights : sparse matrix, shape=(N,N)
        The set of edge weights in the graph where N is the total 
        number of nodes. ``weights[i,j]`` is the weight of the edge 
        (i->j). A weight of 0 implies no edge. 

    partition : ndarray, shape=(N,)
        The partition label for each node in the graph. We require
        ``partition[i] < partition[j]`` whenever ``weights[i,j] != 0``
            
    Returns
    -------
    matching: list of tuples
        Maximum directed matching as a list of node-index tuples. 
    """
    import pulp
    
    # check that the edges are compatible with the partition
    sources,targets = weights.nonzero()
    assert np.all(partition[sources] < partition[targets]), 'Invalid weights/partition combination'
    
    # enumerate all possible node groupings and their score
    edge_order = np.argsort(partition[sources])
    sorted_edges = list(zip(sources[edge_order], targets[edge_order]))
    paths = enumerate_paths(sorted_edges)
    path_weights = [sum([weights[i,j] for i in p for j in p]) for p in paths]
    
    # configure an ILP solver with pulp
    problem = pulp.LpProblem('matching', pulp.LpMaximize)
    
    # initialize variables
    path_vars = pulp.LpVariable.dicts('paths', paths, cat=pulp.LpBinary)
    problem += pulp.lpSum([w*path_vars[p] for p,w in zip(paths, path_weights)])

    # set constraints
    path_membership = {n:[] for n in range(weights.shape[0])}
    for path in paths:
        for n in path: path_membership[n].append(path)
    for n in path_membership: 
        problem += pulp.lpSum(path_vars[path] for path in path_membership[n]) <= 1
        
    # solve and extract results
    problem.solve()
    matching = [p for p,v in path_vars.items() if v.value()==1]
    return matching
