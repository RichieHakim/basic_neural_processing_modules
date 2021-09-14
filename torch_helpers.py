import torch

import pycuda
import pycuda.driver as drv

import sys
import gc

def show_torch_cuda_info():
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
    drv.init()
    print("%d device(s) found." % drv.Device.count())
            
    for ordinal in range(drv.Device.count()):
        dev = drv.Device(ordinal)
        print (ordinal, dev.name())


def show_all_tensors(globals):
    var = []
    for var in globals:
        if (type(globals[var]) is torch.Tensor):
            print(f'var: {var}, device:{globals[var].device}, shape: {globals[var].shape}, size: {globals[var].element_size() * globals[var].nelement()/1000000} MB')           

def delete_all_cuda_tensors(globals):
    '''
    Call with: delete_all_cuda_tensors(globals())
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


def tensor_sizeOnDisk(tensor, print_pref=True):
    # in MB
    size = tensor.element_size() * tensor.nelement()
    if print_pref:
        print(f'{tensor.device}, {tensor.shape}, {size/1000000} MB')
    return size