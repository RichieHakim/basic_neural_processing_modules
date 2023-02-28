import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
import gc
import copy

from . import indexing


############################################
############ VARIABLE HELPERS ##############
############################################

def show_all_tensors(globals, sort_by_size_pref=False, data_unit='GB'):
    """
    Show all tensors in a dict.
    RH 2021

    Args:
        globals (dict):
            Dict of global variables.
            Call globals() to get this.
    """
    size = []
    strings = []
    for var in globals:
        if (type(globals[var]) is torch.Tensor):
            size.append(convert_size(globals[var].element_size() * globals[var].nelement(), return_size=data_unit))
            strings.append(f'var: {var},   device:{globals[var].device},   shape: {globals[var].shape},   size: {size[-1]} {data_unit},   requires_grad: {globals[var].requires_grad}')

    if sort_by_size_pref:
        i_sort = np.argsort(size)[::-1]
        strings = [strings[ii] for ii in i_sort]

    for string in strings:
        print(string)

    
    # # prints currently alive Tensors and Variables
    # import torch
    # import gc
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             if str(obj.device) != 'cpu':
    # #                 print(obj.device)
    # #                 print(type(obj), obj.size(), obj.dtype, obj.device)
    #                 sod = misc.estimate_array_size(input_shape=obj.shape)
    #                 if sod > 0.01:
    #                     print(type(obj), obj.size(), obj.dtype, obj.device, sod)
    #     except:
    #         pass


def tensor_sizeOnDisk(tensor, print_pref=True, return_size='GB'):
    """
    Return estimated size of tensor on disk.
    """
    # in MB
    size = convert_size(
        tensor.element_size() * tensor.nelement(),
        return_size=return_size)

    if print_pref:
        print(f'Device: {tensor.device}, Shape: {tensor.shape}, Size: {size} {return_size}')
    return size


######################################
############ CUDA STUFF ##############
######################################

def show_torch_cuda_info():
    """
    Show PyTorch and cuda info.
    """
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION: cannot be directly found with python function. Use `nvcc --version` in terminal or `! nvcc --version in notebook')
    from subprocess import call
    # ! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print ('Available torch cuda devices ', torch.cuda.device_count())
    print ('Current torch cuda device ', torch.cuda.current_device())

def show_cuda_devices():
    """
    Show available cuda devices. Uses pycuda.
    RH 2021
    """
    import pycuda
    import pycuda.driver as drv

    drv.init()
    print("%d device(s) found." % drv.Device.count())
            
    for ordinal in range(drv.Device.count()):
        dev = drv.Device(ordinal)
        print (ordinal, dev.name())

def delete_all_cuda_tensors(globals):
    '''
    Call with: delete_all_cuda_tensors(globals())
    RH 2021

    Args:
        globals (dict):
            Dict of global variables.
            Call globals() to get this.
    '''
    types = [type(ii[1]) for ii in globals.items()]
    keys = list(globals.keys())
    for ii, (i_type, i_key) in enumerate(zip(types, keys)):
        if i_type is torch.Tensor:
            if globals[i_key].device.type == 'cuda':
                print(f'deleting: {i_key}, size: {globals[i_key].element_size() * globals[i_key].nelement()/1000000} MB')
                del(globals[i_key])
    
    clear_cuda_cache()

def clear_cuda_cache():
    """
    Clear cuda cache.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def set_device(use_GPU=True, device_num=0, verbose=True):
    """
    Set torch.cuda device to use.
    Assumes that only one GPU is available or
     that you wish to use cuda:0 only.
    RH 2021

    Args:
        use_GPU (int):
            If 1, use GPU.
            If 0, use CPU.
    """
    if use_GPU:
        print(f'devices available: {[torch.cuda.get_device_properties(ii) for ii in range(torch.cuda.device_count())]}') if verbose else None
        device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print(f"Using device: '{device}': {torch.cuda.get_device_properties(device_num)}") if verbose else None
    else:
        device = "cpu"
        print(f"device: '{device}'") if verbose else None

    return device
    

######################################
############ DATA HELPERS ############
######################################


class Basic_dataset(Dataset):
    """
    demo:
    ds = Basic_dataset(X, device='cuda:0')
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    """
    def __init__(self, 
                 X, 
                 device='cpu',
                 dtype=torch.float32):
        """
        Make a basic dataset.
        RH 2021

        Args:
            X (torch.Tensor or np.array):
                Data to make dataset from.
            device (str):
                Device to use.
            dtype (torch.dtype):
                Data type to use.
        """
        
        self.X = torch.as_tensor(X, dtype=dtype, device=device) # first (0th) dim will be subsampled from
        self.n_samples = self.X.shape[0]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Returns a single sample.

        Args:
            idx (int):
                Index of sample to return.
        """
        return self.X[idx], idx


##################################################################
############# STUFF PYTORCH SHOULD ALREADY HAVE ##################
##################################################################

def nanmean(arr, dim=None, keepdim=False):
    """
    Compute the mean of an array ignoring any NaNs.
    RH 2021
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, 0)
    sum = torch.sum(arr_no_nan, **kwargs)
    num = torch.sum(torch.logical_not(nan_mask), **kwargs)
    return sum / num

def nanvar(arr, dim=None, keepdim=False, ddof=0, ret_mean=False):
    """
    Compute the standard deviation of an array ignoring any NaNs.
    RH 2022
    """
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, 0)
    cnt = torch.sum(torch.logical_not(nan_mask), dim=dim, keepdim=True)
    avg = torch.sum(arr_no_nan, dim=dim, keepdim=True) / cnt
    dev_sqrd = (arr_no_nan - avg)**2
    var = torch.sum(dev_sqrd, dim=dim, keepdim=True) / cnt
    dof = cnt - ddof
    out = var * cnt / dof

    out = squeeze_multiple_dims(out, dims=dim) if keepdim==False else out
    return out if ret_mean==False else (out, avg)

def nanstd(arr, dim=None, keepdim=False, ddof=0, ret_mean=False):
    """
    Compute the standard deviation of an array ignoring any NaNs.
    RH 2022
    """
    var = nanvar(arr, dim=dim, keepdim=keepdim, ddof=ddof, ret_mean=ret_mean)
    if ret_mean:
        var, avg = var
    std = torch.sqrt(var)
    return std if ret_mean==False else (std, avg)
        

def nansum(arr, dim=None, keepdim=False):
    """
    Compute the sum of an array ignoring any NaNs.
    RH 2021
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, 0)
    return torch.sum(arr_no_nan, **kwargs)

def nanmax(arr, dim=None, keepdim=False):
    """
    Compute the max of an array ignoring any NaNs.
    RH 2021
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, float('-inf'))
    return torch.max(arr_no_nan, **kwargs)

def nanmin(arr, dim=None, keepdim=False):
    """
    Compute the min of an array ignoring any NaNs.
    RH 2021
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, float('inf'))
    return torch.min(arr_no_nan, **kwargs)

def unravel_index(index, shape):
    out = []
    for dim in shape[::-1]:
        out.append(index % dim)
        index = index // dim
    return tuple(out[::-1])

def squeeze_multiple_dims(arr, dims=(0, 1)):
    """
    Squeeze multiple dimensions of a tensor.
    RH 2022
    """
    assert all([arr.shape[d] == 1 for d in dims])
    ## make custom slices to squeeze out the dims
    slices = [slice(None)] * arr.ndim
    for d in dims:
        slices[d] = 0
    return arr[tuple(slices)]

def multiply_elementwise_sparse_dense(s, d):
    """
    Multiply a sparse tensor (s) with a dense tensor (d).
    Warning: This creates an intermediate dense tensor with the
     same shape as s.
    RH 2022
    """
    return torch.mul(d.expand(s.shape).sparse_mask(s.coalesce()), s)

def permute_sparse(input, dims):
    """
    Permute the dimensions of a sparse tensor.
    found here: https://github.com/pytorch/pytorch/issues/78422
    """
    dims = torch.LongTensor(dims)
    return torch.sparse_coo_tensor(indices=input._indices()[dims], values=input._values(), size=torch.Size(torch.tensor(input.size())[dims]))

def roll_sparse(X, shifts, dims):
    """
    Roll a sparse tensor along the specified dimensions.
    RH 2022
    """
    if type(shifts) is not tuple:
        shifts = (shifts,)
    if type(dims) is not tuple:
        dims = (dims,)

    X_out = copy.copy(X)
    for shift, dim in zip(shifts, dims):
        idx = X_out._indices()
        idx[dim] = (idx[dim] + shift) % X_out.shape[dim]
        X_out = torch.sparse_coo_tensor(indices=idx, values=X.coalesce().values(), size=X_out.shape)
    return X_out

def diag_sparse(x, return_sparse_vals=True):
    """
    Get the diagonal of a sparse tensor.
    RH 2022

    Args:
        x (torch.sparse.FloatTensor):
            Pytorch sparse tensor.
        return_sparse_vals (bool):
            If True, returns 'sparse' (zeroed) values as well.
            If False, returns only specified (non-sparsed out) values.
    """
    if return_sparse_vals is False:
        row, col = x.indices()
        values = x.values()
        return values[row == col]
    else:
        x_ts = indexing.torch_to_torchSparse(x)
        return x_ts.get_diag()

class Diag_sparse:
    """
    Get the diagonal of a sparse tensor.
    Class version. The  indices are kept as a property
     of the class for faster access.
    RH 2022
    """
    def __init__(self, x):
        """
        Args:
            x (torch.sparse.Tensor):
                Sparse tensor to get diagonal of.
                For the __init__, only the indices of the
                 diagonal are kept as a property. Subsequent
                 calls of the class can use any x with the
                 same indices.
        """
        row, col = x.indices()
        self.idx = (row == col)
    def __call__(self, x):
        values = x.values()
        return values[self.idx]


def zscore(X, dim=None, ddof=0, nan_policy='propagate', axis=None):
    """
    Compute the z-score of a tensor.
    RH 2022
    """
    dim = axis if (axis is not None) and (dim is None) else dim
    assert dim is not None, 'Must specify dimension to compute z-score over.'

    if nan_policy == 'omit':
        std, mean = nanstd(X, dim=dim, ddof=ddof, keepdim=True, ret_mean=True)
    elif nan_policy == 'propagate':
        std, mean = torch.std_mean(X, dim=dim, keepdim=True, unbiased=ddof==1)
    else:
        raise ValueError('nan_policy must be "omit" or "propagate".')
    return (X - mean) / std


#########################################################
############ INTRA-MODULE HELPER FUNCTIONS ##############
#########################################################

def convert_size(size, return_size='GB'):
    """
    Convert size to GB, MB, KB, from B.
    RH 2021

    Args:
        size (int or float):
            Size in bytes.
        return_size (str):
            Size unit to return.
            Options: 'TB', 'GB', 'MB', or 'KB'
        
    Returns:
        out_size (float):
            Size in specified unit.      
    """

    if return_size == 'TB':
        out_size = size / 1000000000000
    elif return_size == 'GB':
        out_size = size / 1000000000
    elif return_size == 'MB':
        out_size = size / 1000000
    elif return_size == 'KB':
        out_size = size / 1000
    elif return_size == 'B':
        out_size = size / 1

    return out_size

