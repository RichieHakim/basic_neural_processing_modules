import traceback
import time

import numpy as np
import scipy.signal

from .. import math_functions

from ..featurization import Toeplitz_convolution2d
from ..optimization import Convergence_checker

from ..similarity import pairwise_orthogonalization_torch

from ..clustering import cluster_similarity_matrices

def test_toeplitz_convolution2d():
    """
    Test toeplitz_convolution2d
    Tests for modes, shapes, values, and for sparse matrices against
     scipy.signal.convolve2d.

    RH 2022
    """
    ## test toepltiz convolution

    print(f'testing with batching=False')

    stt = shapes_to_try = np.meshgrid(np.arange(1, 7), np.arange(1, 7), np.arange(1, 7), np.arange(1, 7))
    stt = [s.reshape(-1) for s in stt]

    for mode in ['full', 'same', 'valid']:
        for ii in range(len(stt[0])):
            x = np.random.rand(stt[0][ii], stt[1][ii])
            k = np.random.rand(stt[2][ii], stt[3][ii])
    #         print(stt[0][ii], stt[1][ii], stt[2][ii], stt[3][ii])

            try:
                t = Toeplitz_convolution2d(x_shape=x.shape, k=k, mode=mode, dtype=None)
                out_t2d = t(x, batching=False, mode=mode)
                out_t2d_s = t(scipy.sparse.csr_matrix(x), batching=False, mode=mode)
                out_sp = scipy.signal.convolve2d(x, k, mode=mode)
            except Exception as e:
                if mode == 'valid' and (stt[0][ii] < stt[2][ii] or stt[1][ii] < stt[3][ii]):
                    if 'x must be larger than k' in str(e):
                        continue
                print(f'A) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
                success = False
                break
            try:
                if np.allclose(out_t2d, out_t2d_s.A) and np.allclose(out_t2d, out_sp) and np.allclose(out_sp, out_t2d_s.A):
                    success = True
                    continue
            except Exception as e:
                print(f'B) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
                success = False
                break

            else:
                print(f'C) test failed with batching==False, shapes:  x: {x.shape}, k: {k.shape} and mode: {mode}')
                success = False
                break       

    print(f'testing with batching=True')

    for mode in ['full', 'same', 'valid']:
        for ii in range(len(stt[0])):
            x = np.stack([np.random.rand(stt[0][ii], stt[1][ii]).reshape(-1) for jj in range(3)], axis=0)
            k = np.random.rand(stt[2][ii], stt[3][ii])
    #         print(stt[0][ii], stt[1][ii], stt[2][ii], stt[3][ii])

            try:
                t = Toeplitz_convolution2d(x_shape=(stt[0][ii], stt[1][ii]), k=k, mode=mode, dtype=None)
                out_sp = np.stack([scipy.signal.convolve2d(x_i.reshape(stt[0][ii], stt[1][ii]), k, mode=mode) for x_i in x], axis=0)
                out_t2d = t(x, batching=True, mode=mode).reshape(3, out_sp.shape[1], out_sp.shape[2])
                out_t2d_s = t(scipy.sparse.csr_matrix(x), batching=True, mode=mode).toarray().reshape(3, out_sp.shape[1], out_sp.shape[2])
            except Exception as e:
                if mode == 'valid' and (stt[0][ii] < stt[2][ii] or stt[1][ii] < stt[3][ii]):
                    if 'x must be larger than k' in str(e):
                        continue
                else:
                    print(f'A) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
                success = False
                break
            try:
                if np.allclose(out_t2d, out_t2d_s) and np.allclose(out_t2d, out_sp) and np.allclose(out_sp, out_t2d_s):
                    success = True
                    continue
            except Exception as e:
                print(f'B) test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}  {traceback.format_exc()}')
                success = False
                break

            else:
                print(f'C) test failed with batching==False, shapes:  x: {x.shape}, k: {k.shape} and mode: {mode}')
                print(f"Failure analysis: \n")
                print(f"Shapes: x: {x.shape}, k: {k.shape}, out_t2d: {out_t2d.shape}, out_t2d_s: {out_t2d_s.shape}, out_sp: {out_sp.shape}")
                print(f"out_t2d: {out_t2d}")
                print(f"out_t2d_s: {out_t2d_s}")
                print(f"out_sp: {out_sp}")

                success = False
                break           
    print(f'success with all shapes and modes') if success else None
    assert success, 'test failed'
    # return success



def test_Convergence_checker():
    traces = np.array([
        np.linspace(100,0,1000),
        np.linspace(0,100,1000),
        math_functions.bounded_logspace(0.001,100,1000),
        math_functions.bounded_logspace(100,0.001,1000),
        np.cos(np.linspace(0,2*np.pi,1000))*100,
        np.sin(np.linspace(0,2*np.pi,1000))*100,
        
        # np.linspace(100,0,1000) + np.random.rand(1000)*10,
        # np.linspace(0,100,1000) + np.random.rand(1000)*10,
        # math_functions.bounded_logspace(0.001,100,1000) + np.random.rand(1000)*10,
        # math_functions.bounded_logspace(100,0.001,1000) + np.random.rand(1000)*10,
        # np.cos(np.linspace(0,2*np.pi,1000))*100 + np.random.rand(1000)*10,
        # np.sin(np.linspace(0,2*np.pi,1000))*100 + np.random.rand(1000)*10,
    ])

    win = 100
    iter_converged = np.ones((traces.shape[0]))*np.nan
    diff_window_convergence_all = np.ones_like(traces)*np.nan
    for i_trace,t in enumerate(traces):
        c = convergence_checker = Convergence_checker(
            tol_convergence=-1e-1,
            window_convergence=win,
            mode='greater',
        )
        for ii in range(len(t)):
            t_ii = t[max(ii-win,0):ii]
            diff_window_convergence, loss_smooth, converged = c(t_ii)
            diff_window_convergence_all[i_trace, ii] = diff_window_convergence
            if converged:
                iter_converged[i_trace] = ii
                break

    test_check1 = np.allclose(iter_converged, np.array([ np.nan, 100., 100., 665., 550., 100.]), equal_nan=True)
    assert test_check1, f"'greater' mode failed"


    win = 100
    iter_converged = np.ones((traces.shape[0]))*np.nan
    diff_window_convergence_all = np.ones_like(traces)*np.nan
    for i_trace,t in enumerate(traces):
        c = convergence_checker = Convergence_checker(
            tol_convergence=1e-1,
            window_convergence=win,
            mode='abs_less',
        )
        for ii in range(len(t)):
            t_ii = t[max(ii-win,0):ii]
            diff_window_convergence, loss_smooth, converged = c(t_ii)
            diff_window_convergence_all[i_trace, ii] = diff_window_convergence
            if converged:
                iter_converged[i_trace] = ii
                break

    test_check2 = np.allclose(iter_converged, np.array([ np.nan,  np.nan, 100., 665., 550., 300.]), equal_nan=True)
    assert test_check2, f"'abs_less' mode failed"


    win = 100
    iter_converged = np.ones((traces.shape[0]))*np.nan
    diff_window_convergence_all = np.ones_like(traces)*np.nan
    for i_trace,t in enumerate(traces):
        c = convergence_checker = Convergence_checker(
            tol_convergence=1e-1,
            window_convergence=win,
            mode='less',
        )
        for ii in range(len(t)):
            t_ii = t[max(ii-win,0):ii]
            diff_window_convergence, loss_smooth, converged = c(t_ii)
            diff_window_convergence_all[i_trace, ii] = diff_window_convergence
            if converged:
                iter_converged[i_trace] = ii
                break

    test_check3 = np.allclose(iter_converged, np.array([100.,  np.nan, 100., 100., 100., 300.]), equal_nan=True)
    assert test_check3, f"'less' mode failed"

    # return np.all((test_check1, test_check2, test_check3))


def test_pairwise_orthogonalization_torch():
    import torch
    torch.manual_seed(0)
    
    ## test 1: matrix, matrix, center=True
    ### make two random matrices with correlated columns
    v1 = torch.randn(100, 100)
    v2 = torch.randn(100, 100)
    v2 = v2 + v1*0.5
    v1 = v1 / torch.norm(v1, dim=0)
    v2 = v2 / torch.norm(v2, dim=0)
    ### orthogonalize v2 off of v1. v1_orth should be orthogonal to v2.
    v1_orth, EVR, EVR_total_weighted, EVR_total_unweighted = pairwise_orthogonalization_torch(v1, v2, center=True)

    ## test that columns of v1_orth are orthogonal to columns of v2. Check the pairwise correlation along the columns.
    corr = (v1_orth.T @ v2).diag()
    assert torch.allclose(corr, torch.zeros_like(corr), atol=1e-5), f"columns of v1_orth are not orthogonal to columns of v2. corr={corr}"

    ## test 2: matrix, vector, center=True
    ### make a random matrix and vector with correlated columns. Use a master column that is correlated with all columns.
    master_column = torch.randn(100)
    v1 = torch.randn(100, 100) + master_column[:,None]*0.5
    v2 = torch.randn(100) + master_column*0.5
    v1 = v1 / torch.norm(v1, dim=0)
    v2 = v2 / torch.norm(v2)
    ### orthogonalize v2 off of v1. v1_orth should be orthogonal to v2.
    v1_orth, EVR, EVR_total_weighted, EVR_total_unweighted = pairwise_orthogonalization_torch(v1, v2, center=True)
    
    ## test that columns of v1_orth are orthogonal to v2. Check the pairwise correlation along the columns.
    corr = (v1_orth.T @ v2).diag()
    assert torch.allclose(corr, torch.zeros_like(corr), atol=1e-5), f"columns of v1_orth are not orthogonal to v2. corr={corr}"


def test_cluster_similarity_matrices():
    import numpy as np

    ## test 1: 2 clusters, 2 samples, no sparse values.
    s = np.array([
        [1, 0.5, 0.3, 0.1],
        [0.5, 1, 0.1, 0.3],
        [0.3, 0.1, 1, 0.5],
        [0.1, 0.3, 0.5, 1]
    ])
    l = np.array([0, 0, 1, 1])

    cs_mean_expected = np.array([
        [0.5, 0.2],
        [0.2, 0.5]
    ])

    cs_max_expected = np.array([
        [0.5, 0.3],
        [0.3, 0.5]
    ])

    cs_min_expected = np.array([
        [0.5, 0.1],
        [0.1, 0.5]
    ])

    cs_mean, cs_max, cs_min = cluster_similarity_matrices(s, l)

    assert np.allclose(cs_mean, cs_mean_expected, atol=1e-6)
    assert np.allclose(cs_max, cs_max_expected, atol=1e-6)
    assert np.allclose(cs_min, cs_min_expected, atol=1e-6)

    # Test with sparse input
    s_sparse = scipy.sparse.csr_matrix(s)
    cs_mean, cs_max, cs_min = cluster_similarity_matrices(s_sparse, l)

    assert np.allclose(cs_mean, cs_mean_expected, atol=1e-6)
    assert np.allclose(cs_max, cs_max_expected, atol=1e-6)
    assert np.allclose(cs_min, cs_min_expected, atol=1e-6)

    ## test 2: 2 clusters, 2 samples, with sparse values.
    s_sparse = scipy.sparse.csr_matrix([
    [1,   0,   0.3, 0  ],
    [0,   1,   0,   0  ],
    [0.3, 0,   1,   0.5],
    [0,   0,   0.5, 1  ]
    ])
    l = np.array([0, 0, 1, 1])

    cs_mean_expected = np.array([
        [0, 0.075],
        [0.075, 0.5]
    ])

    cs_max_expected = np.array([
        [0, 0.3],
        [0.3, 0.5]
    ])

    cs_min_expected = np.array([
        [0, 0],
        [0, 0.5]
    ])

    cs_mean, cs_max, cs_min = cluster_similarity_matrices(s_sparse, l)

    assert np.allclose(cs_mean, cs_mean_expected, atol=1e-6)
    assert np.allclose(cs_max, cs_max_expected, atol=1e-6)
    assert np.allclose(cs_min, cs_min_expected, atol=1e-6)
