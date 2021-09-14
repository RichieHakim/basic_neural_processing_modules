import numpy as np
import tensorly
import scipy
import copy
import sklearn


############################################
############ HELPER FUNCTIONS ##############
############################################
# CORE FUNCTION AT BOTTOM

def idx_to_oneHot(arr):
    oneHot = np.zeros((arr.size, arr.max()+1))
    oneHot[np.arange(arr.size), arr] = 1
    return oneHot
def confusion_matrix_prob(probs, y):
    cmat = probs.T @ idx_to_oneHot(y)
    return cmat / np.sum(cmat, axis=0)[None,:]
#     return cmat
def mean_confusion_loss(B, X, y):
    cmat = confusion_matrix_prob(predict_proba(X, B), y)
    return np.mean(np.sum(cmat * (1-np.eye(cmat.shape[0])), axis=0))


def cross_entropy_loss(y_true, y_hat):
    return sklearn.metrics.log_loss(y_true, y_hat)


def predict_proba(X, B):
    return scipy.special.softmax(tensorly.tenalg.inner(X, B, n_modes=B.ndim-1), axis=1)
def predict(X, B_cp, weights=None):
    if weights is None:
        weights = np.ones(B_cp[0].shape[1])
    return np.argmax(predict_proba(X, Bcp_to_B(B_cp, weights)), axis=1)


def score(B, X, y):
    return cross_entropy_loss(y, predict_proba(X,B))
#     return mean_confusion_loss(B, X, y)

def L2_penalty(B_cp):
    return np.sum([np.sqrt(np.sum(comp**2)) for comp in B_cp])

def score_with_Bcp(B_cp, weights, X, y, lambda_L2):
    return score(Bcp_to_B(B_cp, weights), X, y) + lambda_L2 * L2_penalty(B_cp)
def score_with_BcpFlat(B_cp_flat, X, y, weights, B_dims, rank, lambda_L2):
    return score_with_Bcp(BcpFlat_to_Bcp(B_cp_flat, B_dims, rank), weights, X, y, lambda_L2)


def make_BcpInit(B_dims, rank):
    B_cp_init = list([np.random.random_sample((B_dims[0], rank))])
    for ii in range(1,len(B_dims)):
#         B_cp_init.append(np.random.random_sample((B_dims[ii], rank)))
        B_cp_init.append(np.ones((B_dims[ii], rank)))
    return B_cp_init

def BcpFlat_to_Bcp(B_cp_flat, B_dims, rank):
    B_cp = list(np.ones(len(B_dims)))
    last_idx = 0
    for ii in range(len(B_dims)):
        next_idx = last_idx+B_dims[ii]*rank
        B_cp[ii] = np.reshape(B_cp_flat[last_idx:next_idx], (B_dims[ii],rank))
        last_idx = copy.copy(next_idx)
    return B_cp

def Bcp_to_BcpFlat(B_cp):
    return np.concatenate([ii.ravel() for ii in B_cp])

def Bcp_to_B(B_cp, weights=None):
    if weights is None:
        weights = np.ones(B_cp[0].shape[1])
    return tensorly.cp_tensor.cp_to_tensor((weights, B_cp))

#########################################
############ CORE FUNCTION ##############
#########################################

def CP_logitReg(X, y, weights=None, rank=4, lambda_L2=0.1, non_neg_pref=False, **lbfgs_params):
    '''
    Performs a multinomial logistic CP regression.

    Args:
        X (numpy.ndarray):
            First dimension should share same size with y.
            Subsequent dimensions can be of any size.
        y (numpy.ndarray):
            Length should be equal to first dimension of X.
        weights (numpy.ndarray):
            Weights for each of the components.
            Should be of same length as rank.
        rank (int):
            Rank of the CP decomposition / number of 
             components.
        lambda_L2 (float):
            Regularization parameter.
        non_neg_pref (bool):
            If True, the CP decomposition is forced to be
             non-negative.
    Returns:
        B_cp_final (list):
            Final Kruskal tensor. Each entry in the outer
             list is a dimension. Each list entry will be
             a matrix of shape [size of dimension, rank].
            B_cp_final can be made into a dense tensor by
             taking the outer product of each matrix.
            Note that the inner product of X * B_cp_final
             gives a vector of length X.shape[0], which 
             is the same as length of y.
        run_output (dict):
            Dictionary of run information.

    RH 2021
    '''
    if weights is None:
        weights = np.ones((rank))

    n_classes = len(np.unique(y))
    B_dims = np.concatenate((np.array(X.shape[1:]), [n_classes]))
    x0 = Bcp_to_BcpFlat(make_BcpInit(B_dims, rank))

    if non_neg_pref:
        bounds = (1e-6, np.inf)
    else:
        bounds = (-np.inf, np.inf)
    bounds_list = [bounds for ii in range(len(x0))]

    if lbfgs_params is None:
        lbfgs_params = {'approx_grad': True,
                        'm': 10,
                        'factr': 1e7,
                        'pgtol': 1e-5,
                        'epsilon': 1e-8,
                        'iprint': -1,
                        'maxfun': 2e6,
                        'maxiter': 2e6,
                        'disp': True,
                        'maxls': 20}
    run_output = scipy.optimize.fmin_l_bfgs_b(score_with_BcpFlat,
                                                x0=x0,
                                                bounds=bounds_list,
                                                args=(X, y, weights, B_dims, rank, lambda_L2),
                                                **lbfgs_params,
                                               )
    
    B_cp_final = BcpFlat_to_Bcp(run_output[0], B_dims, rank)
    return B_cp_final, run_output