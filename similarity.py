import numpy as np
from numpy.linalg import norm
from scipy.stats import zscore
    
def pairwise_similarity(vector_set1 , vector_set2=None , method='pearson'):
    '''
    computes similarity matrices between two sets of vectors (columns within 2-D arrays) using either pearson correlation, R^2, or cosine_similarity
    Think of this function as a more general version of np.corrcoef
    RH 2021

    Args:
        vector_set1 (ndarray): 2-D array of column vectors
        vector_set2 (ndarray): 2-D array of column vectors to compare to vector_set1. If None, then the function is a type of autosimilarity matrix
    Returns:
        ouput (ndarray): similarity matrix
    '''

    if vector_set2 is None:
        vector_set2 = vector_set1
        
    if method=='pearson':
        output = (zscore(vector_set1, axis=0).T @ zscore(vector_set2, axis=0)) / ((vector_set1.shape[0] + vector_set2.shape[0])/2)
    if method=='R^2':
        output = ( (zscore(vector_set1, axis=0).T @ zscore(vector_set2, axis=0)) / ((vector_set1.shape[0] + vector_set2.shape[0])/2) )**2
    if method=='cosine_similarity':    
        output = (vector_set1 / (np.expand_dims(norm(vector_set1 , axis=0) , axis=0))).T  @ (vector_set2  / np.expand_dims(norm(vector_set2 , axis=0) , axis=0))
    return output

def best_permutation(mat1 , mat2 , method):
    '''
    This function compares the representations of two sets of vectors (columns of mat1 and columns of mat2).
    We assume that the vectors in mat1 and mat2 are similar up to a permutation.
    We therefore find the 'best' permutation that maximizes the similarity between the sets of vectors
    RH 2021
    
    Args:
        mat1 (np.ndarray): a 2D array where the columns are vectors we wish to match with mat2
        mat2 (np.ndarray): a 2D array where the columns are vectors we wish to match with mat1
        method (string)  : defines method of calculating pairwise similarity between vectors
        
    Returns:
        sim_avg (double)    : the average similarity between matched vectors. Units depend on method
        sim_matched (double): the similarity between each pair of matched vectors.
        ind1 (int)          : indices of vectors in mat1 matched to ind2 in mat2 (usually just sequential for ind1)
        ind2 (int)          : indices of vectors in mat2 matched to ind1 in mat1
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
    This function compares sets of 2-D matrices within a 3-D array using the 'best_permutation' function.
    We assume that the vectors within the matrices are similar up to a permutation.
    We therefore find the 'best' permutation that maximizes the similarity between the sets of vectors within each matrix
    RH 2021
    
    Args:
        mat_set (np.ndarray): a 3D array where the columns within the first two dims are vectors we wish to match with the columns from matrices from other slices in the third dimension
        method (string)     : defines method of calculating pairwise similarity between vectors

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


def proj(v1, v2):
    '''
    projects one or more vectors (columns of v1) onto a single vector (v2)
    RH 2021

        Args:
            v1 (ndarray): vector set 1. Either a single vector or a 2-D array where the columns are the vectors
            v2 (ndarray): vector 2. A single vector
        
        Returns:
            proj_vec (ndarray): vector set 1 projected onto vector 2. Same size as v1.
            proj_score (ndarray or scalar): projection scores. 1-D of length v1.shape[1]
    '''
    u = v2 / norm(v2)
    proj_score = np.array([v1.T @ u])
    proj_vec = np.squeeze( u[:,None] * proj_score )

    return proj_vec , np.squeeze(proj_score)


def orthogonalize(v1, v2):
        '''
    orthogonalizes one or more vectors (columns of v1) relative to a single vector (v2)
    RH 2021

        Args:
            v1 (ndarray): vector set 1. Either a single vector or a 2-D array where the columns are the vectors
            v2 (ndarray): vector 2. A single vector
        
        Returns:
            output (ndarray): vector set 1 with the projection onto vector 2 subtracted off. Same size as v1.
    '''
    proj_vec,_ = proj(v1, v2)
    return v1 - proj_vec