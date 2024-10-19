def set_device(device_num=0, verbose=True):
    """
    Set the device to use.
    RH 2021

    Args:
        device_num (int): 
            The device number to use.
            Optional. Default is 0.
        verbose (bool):
            Whether to print the device name.
    """
    import cupy
    
    if cupy.cuda.runtime.getDeviceCount() > 0:
        if device_num is None:
            DEVICE=0
        else:
            DEVICE=device_num
        cupy.cuda.Device(DEVICE).use()
        if verbose:
            print(f"using device: {cupy.cuda.runtime.getDeviceProperties(DEVICE)['name']}")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        print("no CUDA devices found")
    if cupy.cuda.runtime.getDeviceCount() > 1 and device_num==None:
        print('RH Warning: number of cupy devices is greater than 1 and device_num note specified')

    return DEVICE