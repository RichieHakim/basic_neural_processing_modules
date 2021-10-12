import torch

import pycuda
import pycuda.driver as drv

import sys
import gc

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
    drv.init()
    print("%d device(s) found." % drv.Device.count())
            
    for ordinal in range(drv.Device.count()):
        dev = drv.Device(ordinal)
        print (ordinal, dev.name())


def show_all_tensors(globals):
    """
    Show all tensors in a dict.
    RH 2021

    Args:
        globals (dict):
            Dict of global variables.
            Call globals() to get this.
    """
    var = []
    for var in globals:
        if (type(globals[var]) is torch.Tensor):
            print(f'var: {var}, device:{globals[var].device}, shape: {globals[var].shape}, size: {globals[var].element_size() * globals[var].nelement()/1000000} MB')           

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
    gc.collect()
    torch.cuda.empty_cache()


def tensor_sizeOnDisk(tensor, print_pref=True, return_size='GB'):
    """
    Return estimated size of tensor on disk.
    """
    # in MB
    size = tensor.element_size() * tensor.nelement()

    if return_size == 'GB':
        size = size / 1000000000
    elif return_size == 'MB':
        size = size / 1000000
    elif return_size == 'KB':
        size = size / 1000

    if print_pref:
        print(f'{tensor.device}, {tensor.shape}, {return_size}')
    return size

def set_device(use_GPU=True, verbose=True):
    """
    Set torch.cuda device to use.
    RH 2021

    Args:
        use_GPU (int):
            If 1, use GPU.
            If 0, use CPU.
    """
    if use_GPU:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print(f"device: '{device}'") if verbose else None
    else:
        device = "cpu"
        print(f"device: '{device}'") if verbose else None

    return device
    