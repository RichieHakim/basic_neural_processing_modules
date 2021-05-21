import numpy as np
from numpy.linalg import norm, qr
from scipy.stats import zscore
import copy
import sklearn.decomposition

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
    
    Returns:
        proj_vec (ndarray): 
            vector set 1 projected onto vector set 2. 
            shape: (v1.shape[0], v1.shape[1], v2.shape[1])
        proj_score (ndarray or scalar): 
            projection scores. shape: (v1.shape[1], v2.shape[1])
    '''
    if v1.ndim < 2:
        v1 = v1[:,None]
    if v2.ndim < 2:
        v2 = v2[:,None]

    u = v2 / norm(v2, axis=0)
    proj_score = v1.T @ u
    # this einsum can probably be optimized better
    proj_vec = np.einsum('ik,jk->ijk', u, proj_score)

    return proj_vec , proj_score


def orthogonalize(v1, v2):
    '''
    Orthogonalizes one or more vectors (columns of v1) relative to a single 
    vector (v2). Subtracts the projection of v1 onto v2 off of v1.
    RH 2021

    Args:
        v1 (ndarray): 
            vector set 1. Either a single vector or a 2-D array where the columns
             are the vectors
        v2 (ndarray): 
            vector set 2. Either a single vector or a 2-D array where the columns
             are the vectors
    
    Returns:
        v1_orth (ndarray): 
            vector set 1 with the projections onto vector set 2 subtracted off.
             Same size as v1.
        v2_PCs (ndarray):
            PCA is performed on v2 to orthogonalize it, so these are the PCs
        EVR (ndarray): 
            Explained Variance Ratio for each column of v1. 
            Amount of variance that all the vectors in v2 can explain for each 
            vector in v1.
            When v1 is z-scored, EVR is equivalent to pearsons R^2; 
            as in pairwise_similarity(OLS(v2, v1)[1] , v1)**2
        EVR_total (scalar):
            total amount of variance explained in v1 by v2
    '''
    if v1.ndim < 2:
        v1 = v1[:,None]
    if v2.ndim < 2:
        v2 = v2[:,None]
    
    # I'm pretty sure using PCA is fine for this, but it might be a good idea
    # to look into QR decompositions, basic SVD, and whether mean subtracting
    # actually matters to the outcome. Pretty sure this is fine though.
    decomp = sklearn.decomposition.PCA(n_components=v2.shape[1])
    v2_PCs = decomp.fit_transform(v2)

    # Serial orthogonalization. I think doing it serially isn't necessary 
    # since we are orthogonalizing the v2 vectors. This method might have
    # some numerical instability issues given it's similarity to a 
    # Gram-Schmidt process, but I doubt it because v2 is orthogonal.
    # I'll leave optimization to a future me.
    v1_orth = copy.deepcopy(v1)
    for ii in range(v2.shape[1]):
        proj_vec = proj(v1_orth , v2_PCs[:,ii])[0]
        v1_orth = np.squeeze(v1_orth) - np.squeeze(proj_vec)

    EVR = 1 - (np.var(v1_orth, axis=0) / np.var(v1, axis=0))
    EVR_total = 1 - ( np.sum(np.var(v1_orth,axis=0),axis=0) / np.sum(np.var(v1,axis=0),axis=0) )

    return v1_orth, v2_PCs, EVR, EVR_total


def OLS(X,y):
    '''
    Ordinary Least Squares regression
    RH 2021

    Args:
        X (ndarray):
            array where columns are vectors to regress against 'y'
        y (ndarray):
            1-D or 2-D array
    Returns:
        theta (ndarray):
            regression coefficents
        y_rec (ndarray):
            y reconstructions
    '''
    if X.ndim==1:
        X = X[:,None]
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    y_rec = X @ theta
    return theta, y_rec


def pairwise_similarity(v1 , v2=None , method='pearson' , ddof=1):
    '''
    Computes similarity matrices between two sets of vectors (columns within
    2-D arrays) using either covariance, Pearson correlation, EV/R^2, or 
    cosine_similarity.

    If method=='EV' or 'R^2':
        - v1 is original data or y_true (total variance)
        - v2 is prediction or reconstruction or y_pred (regressed variance)
        - v1.shape[1] must equal v2.shape[1]
    
    Think of this function as a more general version of np.corrcoef
    RH 2021

    Args:
        v1 (ndarray): 
            2-D array of column vectors.
        v2 (ndarray): 
            2-D array of column vectors to compare to vector_set1. If None, then
             the function is a type of autosimilarity matrix
            If method=='EV' or 'R^2':
                v1 is prediction or reconstruction
                then v1.shape[1] must equal v2.shape[1]
        ddof (scalar/int):
            Used if method=='cov'. Define the degrees of freedom. Set to 1 for
            unbiased calculation, 0 for biased calculation.
    Returns:
        ouput (ndarray):
            For methods: 'cov', 'pearson'/'R'
                Output is a similarity matrix
            For methods: 'EV'/'R^2', 'cosine_similarity'
                Output is a simialrity vector (since inputs must be paired)
    '''

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
        
        # Note: this Pearson's R-value can be different than sqrt(EV) 
        # calculated below if the residuals are not orthogonal to the 
        # prediction. Best to use EV for R^2 if unsure
        v1_ms = v1 - np.mean(v1, axis=0) # 'mean subtracted'
        v2_ms = v2 - np.mean(v2, axis=0)
        output = (v1_ms.T @ v2_ms) / np.sqrt(np.tile(np.sum(v1_ms**2, axis=0),(v2.shape[1],1)).T * np.tile(np.sum(v2_ms**2, axis=0), (v1.shape[1],1)))
    if method in ['EV', 'R^2']:
        # Calculating as 1 - SS_residual_variance / SS_original_variance
        # Should be exactly equivalent to 
        # sklearn.metrics.r2_score(v1,v2, multioutput='raw_values') but slightly faster

        # Should make it a matrix output like other methods in future.
        output = 1 - np.sum((v1 - v2)**2, axis=0) / np.sum((v1 - np.mean(v1, axis=0))**2, axis=0)
    if method=='cosine_similarity':    
        output = (v1 / (np.expand_dims(norm(v1 , axis=0) , axis=0))).T  @ (v2  / np.expand_dims(norm(v2 , axis=0) , axis=0))
    return output

def best_permutation(mat1 , mat2 , method):
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
            defines method of calculating pairwise similarity between vectors
        
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
        if method=='R^2':
            sim_matched[ii] = (np.corrcoef(mat1[:,ind1[ii]] , mat2[:,ind2[ii]])[0][1])**2
        if method=='cosine_similarity':
            sim_matched[ii] = pairwise_similarity( mat1[:,ind1[ii]] , mat2[:,ind2[ii]] , 'cosine_similarity')

    sim_avg = np.mean(sim_matched)
    return sim_avg , sim_matched , ind1 , ind2

def self_similarity_pairwise(mat_set , method):
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
             we wish to match with the columns from matrices from other slices in the third dimension
        method (string): 
            defines method of calculating pairwise similarity between vectors

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
    ind1 = np.zeros((n_components , n_combos))
    ind2 = np.zeros((n_components , n_combos))
    for i_combo , combo in enumerate(combos):
        corr_avg[i_combo] , corr_matched[:,i_combo] , ind1[:,i_combo] , ind2[:,i_combo]  =  best_permutation(mat_set[:,:,combo[0]]  ,  mat_set[:,:,combo[1]] , method)
    # print(corr_avg)
    return corr_avg, corr_matched, ind1, ind2, combos
