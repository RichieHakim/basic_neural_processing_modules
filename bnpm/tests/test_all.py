import traceback
import time

import numpy as np
import scipy.signal

import pytest

from .. import math_functions

from ..optimization import Convergence_checker

from ..similarity import pairwise_orthogonalization_torch

from ..clustering import cluster_similarity_matrices

from ..path_helpers import find_date_in_path



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


@pytest.mark.parametrize("path,expected_date,reverse_path_order", [
    (r"/home/user/documents/20220203/report.txt", "20220203", True),
    (r"/home/user/docs/2022_02_03/report.pdf", "2022_02_03", True),
    (r"/home/02_03_22/data.txt", "02_03_22", True),
    (r"/2022-02-03/home/data.txt", "2022-02-03", True),
    (r"/home/data_2_3_2022.txt", "2_3_2022", True),
    (r"/home/docs/data_2-03-22.txt", '2-03-22', True),
    (r"C:\home\docs\data_2-03-22.txt", '2-03-22', True),
    (r"/home/docs/19900101/data_2-03-22.txt", '2-03-22', True),
    (r"/home/docs/19900101/data_2-03-22.txt", '19900101', False),
    (r"/home/docs/_19900101/data_2-03-22.txt", '19900101', False),
    (r"/home/docs/19900101_/data_2-03-22.txt", '19900101', False),
    (r"/home/docs/_19900101_/data_2-03-22.txt", '19900101', False),
    (r"/home/docs/_19900101_/data_020322.txt", '19900101', False),
    (r"/home/docs/_19900101_/data_20230322.txt", '19900101', False),
    (r"/home/docs/_19900101_/data_20230322.txt", '20230322', True),
    (r"/home/docs/_1990010_/data_2023032.txt", None, True),
    (r"/home/docs/_1990010_/data_0 32.txt", None, True),
    (r"/home/docs/_/data_.txt", None, True),
])
def test_known_dates_in_path(path, expected_date, reverse_path_order):
    assert find_date_in_path(path, reverse_path_order=reverse_path_order) == expected_date
