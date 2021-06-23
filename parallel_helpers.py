from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import numpy as np

def multithreading(func, args, workers):
    with ThreadPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)
def multiprocessing(func, args, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)

def multiprocessing_pool_along_axis(x_in, function, n_workers=None, axis=0, **kwargs):
    pool = mp.Pool(processes=n_workers)
    if axis==0:
        results = pool.map(partial(function , **kwargs), [x_in[ii] for ii in range(x_in.shape[0])])
        pool.close()
        pool.join()
        return np.row_stack(results)
    elif axis==1:
        results = pool.map(partial(function , **kwargs), [x_in[:,ii] for ii in range(x_in.shape[1])])
        pool.close()
        pool.join()
        return np.column_stack(results)