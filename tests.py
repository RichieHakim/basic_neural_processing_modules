import numpy as np
import scipy.signal

from .featurization import toeplitz_convolution2d

def test_toeplitz_convolution2d():
    """
    Test toeplitz_convolution2d
    Tests for modes, shapes, values, and for sparse matrices against
     scipy.signal.convolve2d.

    RH 2022
    """
    ## test toepltiz convolution

    stt = shapes_to_try = np.meshgrid(np.arange(1, 7), np.arange(1, 7), np.arange(1, 7), np.arange(1, 7))
    stt = [s.reshape(-1) for s in stt]

    for mode in ['full', 'same', 'valid']:
        for ii in range(len(stt[0])):
            x = np.random.rand(stt[0][ii], stt[1][ii])
            k = np.random.rand(stt[2][ii], stt[3][ii])

            try:
                out_t2d = toeplitz_convolution2d(x, k, mode=mode)
                out_t2d_s = toeplitz_convolution2d(scipy.sparse.csr_matrix(x), k, mode=mode)
                out_sp = scipy.signal.convolve2d(x, k, mode=mode)
            except Exception as e:
                if mode == 'valid' and (stt[0][ii] < stt[2][ii] or stt[1][ii] < stt[3][ii]):
                    if 'x must be larger than k' in str(e):
                        continue
                print(f'test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode} and Exception: {e}')
                success = False
                break

            if np.allclose(out_t2d, out_t2d_s.A) and np.allclose(out_t2d, out_sp) and np.allclose(out_sp, out_t2d_s.A):
        #         print(f'success with shapes: x: {x.shape}, k: {k.shape}')
                success = True
                continue
            else:
                success = False
                print(f'test failed with shapes:  x: {x.shape}, k: {k.shape} and mode: {mode}')
                break             

    print(f'success with all shapes and modes') if success else None
    return success
