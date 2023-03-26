def modifiedGramSchmidt(A):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors

    found here: https://stackoverflow.com/questions/47349233/modified-gram-schmidt-in-python-for-complex-vectors
    """
    # assuming A is a square matrix
    dim = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:,i], q)
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q


def gram_schmidt(A):
	""" Representation of Gram-Schmidt Process or QR Diagonalization 
		for an mxn system of linear equations. 
        https://github.com/philwilt/gram_schmidt/blob/master/gram_schmidt.py
    """
		
	m = np.shape(A)[0]
	n = np.shape(A)[1]

	Q =  np.zeros((m, m))
	R =  np.zeros((n, n)) 

	for j in xrange(n):
		
		v = A[:,j]
		
		for i in xrange(j):
			
			R[i,j] = Q[:,i].T * A[:,j]

			v = v.squeeze() - (R[i,j] * Q[:,i])

		R[j,j] =  np.linalg.norm(v)
		Q[:,j] = (v / R[j,j]).squeeze()

	return Q, R






"""
Reduced rank regression class.
Requires scipy to be installed.
Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com
Optimal linear 'bottlenecking' or 'multitask learning'.
"""
import numpy as np
from scipy import sparse


# def ideal_data(num, dimX, dimY, rrank, noise=1):
#     """Low rank data"""
#     X = np.random.randn(num, dimX)
#     W = np.dot(np.random.randn(dimX, rrank), np.random.randn(rrank, dimY))
#     Y = np.dot(X, W) + np.random.randn(num, dimY) * noise
#     return X, Y


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, rank, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0
        self.rank = rank

        CXX = np.dot(X.T, X) + reg * sparse.eye(np.size(X, 1))
        CXY = np.dot(X.T, Y)
        _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
        self.W = V[0:rank, :].T
        self.A = np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.dot(X, np.dot(self.A.T, self.W.T))




import numpy as np
import sklearn.linear_model

def _fit_rrr_no_intercept_all_ranks(X: np.ndarray, Y: np.ndarray, alpha: float, solver: str):
    ridge = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=False, solver=solver)
    beta_ridge = ridge.fit(X, Y).coef_
    Lambda = np.eye(X.shape[1]) * np.sqrt(np.sqrt(alpha))
    X_star = np.concatenate((X, Lambda))
    Y_star = X_star @ beta_ridge.T
    _, _, Vt = np.linalg.svd(Y_star, full_matrices=False)
    return beta_ridge, Vt

def _fit_rrr_no_intercept(X: np.ndarray, Y: np.ndarray, alpha: float, rank: int, solver: str, memory=None):
    memory = sklearn.utils.validation.check_memory(memory)
    fit = memory.cache(_fit_rrr_no_intercept_all_ranks)
    beta_ridge, Vt = fit(X, Y, alpha, solver)
    return Vt[:rank, :].T @ (Vt[:rank, :] @ beta_ridge)

class ReducedRankRidge(sklearn.base.MultiOutputMixin, sklearn.base.RegressorMixin, sklearn.linear_model._base.LinearModel):
    def __init__(self, alpha=1.0, fit_intercept=True, rank=None, ridge_solver='auto', memory=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.rank = rank
        self.ridge_solver = ridge_solver
        self.memory = memory

    def fit(self, X, y):
        if self.fit_intercept:
            X_offset = np.average(X, axis=0)
            y_offset = np.average(y, axis=0)
            # doesn't modify inplace, unlike -=
            X = X - X_offset
            y = y - y_offset
        self.coef_ = _fit_rrr_no_intercept(X, y, self.alpha, self.rank, self.ridge_solver, self.memory)
        self.rank_ = np.linalg.matrix_rank(self.coef_)
        if self.fit_intercept:
            self.intercept_ = y_offset - X_offset @ self.coef_.T
        else:
            self.intercept_ = np.zeros(y.shape[1])
        return self




###############################################################################################################
############################################ FROM SKIMAGE #####################################################
###############################################################################################################

import functools

import numpy as np
import scipy.fft

# from .._shared.utils import _supported_float_type


## Fixed the indexing because the skimage nerds drink too much coffee. RH.
def get_nd_butterworth_filter(shape, factor, order, high_pass, real,
                               dtype=np.float64, squared_butterworth=True):
    """Create a N-dimensional Butterworth mask for an FFT
    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image
    squared_butterworth : bool, optional
        When True, the square of the Butterworth filter is used.
    Returns
    -------
    wfilt : ndarray
        The FFT mask.
    """
    ranges = []
    for i, d in enumerate(shape):
        # start and stop ensures center of mask aligns with center of FFT
        # axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        axis = np.arange(-(d - 1) / 2, (d - 1) / 2 + 0.5) / (d * factor)  ## FIXED, RH 2023
        ranges.append(scipy.fft.ifftshift(axis ** 2))
    # for real image FFT, halve the last axis
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    # q2 = squared Euclidean distance grid
    q2 = functools.reduce(
            np.add, np.meshgrid(*ranges, indexing="ij", sparse=True)
            )
    q2 = q2.astype(dtype)
    q2 = np.power(q2, order)
    wfilt = 1 / (1 + q2)
    if high_pass:
        wfilt *= q2
    if not squared_butterworth:
        np.sqrt(wfilt, out=wfilt)
    return wfilt

