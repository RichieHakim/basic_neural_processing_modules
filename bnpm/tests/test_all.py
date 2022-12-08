import traceback
import time

import numpy as np
import scipy.signal

from .. import math_functions

from ..featurization import Toeplitz_convolution2d
from ..optimization import Convergence_checker

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
                out_t2d = t(x, batching=True, mode=mode)
                out_t2d_s = t(scipy.sparse.csr_matrix(x), batching=True, mode=mode).toarray()
                print(type(out_t2d), type(out_t2d_s))
                out_sp = np.stack([scipy.signal.convolve2d(x_i.reshape(stt[0][ii], stt[1][ii]), k, mode=mode) for x_i in x], axis=0)
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