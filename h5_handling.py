import gc
import h5py
import hdfdict
from pathlib import Path


def close_all_h5():
    '''
    Closes all h5 objects in workspace. Not tested thoroughly.
    from here: https://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file
    also see: pytables is able to do this with a simple function
        ```
        import tables
        tables.file._open_files.close_all()
        ```
    '''
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed
    gc.collect()



def show_group_items(hObj):
    '''
    Simple function that displays all the items and groups in the
     highest hierarchical level of an h5 object or python dict.
    See 'show_item_tree' to view the whole tree
    RH 2021

    Args:
        hObj: 'hierarchical Object' hdf5 object or subgroup object OR python dictionary
    
    ##############

    example usage:
        with h5py.File(path , 'r') as f:
            h5_handling.show_group_items(f)
    '''
    for ii,val in enumerate(list(iter(hObj))):
        if isinstance(hObj[val] , h5py.Group) or isinstance(hObj[val], hdfdict.hdfdict.LazyHdfDict):
            print(f'{ii+1}. {val}:----------------')
        if isinstance(hObj[val] , dict):
            print(f'{ii+1}. {val}:----------------')
        else:
            if hasattr(hObj[val] , 'shape') and hasattr(hObj[val] , 'dtype'):
                print(f'{ii+1}. {val}:   shape={hObj[val].shape} , dtype={hObj[val].dtype}')
            else:
                print(f'{ii+1}. {val}:   type={type(hObj[val])}')



def show_item_tree(hObj=None , path=None, show_metadata=True , print_metadata=False, indent_level=0):
    '''
    Recursive function that displays all the items 
     and groups in an h5 object or python dict
    RH 2021

    Args:
        hObj:
            'hierarchical Object'. hdf5 object OR python dictionary
        path (Path or string):
            If not None, then path to h5 object is used instead of hObj
        show_metadata (bool): 
            whether or not to list metadata with items
        print_metadata (bool): 
            whether or not to show values of metadata items
        indent_level: 
            used internally to function. User should leave blank

    ##############
    
    example usage:
        with h5py.File(path , 'r') as f:
            h5_handling.show_item_tree(f)
    '''

    if path is not None:
        with h5py.File(path , 'r') as f:
            show_item_tree(hObj=f, path=None, show_metadata=show_metadata, print_metadata=print_metadata, indent_level=indent_level)
    else:
        indent = f'  '*indent_level
        if hasattr(hObj, 'attrs') and show_metadata:
            for ii,val in enumerate(list(hObj.attrs.keys()) ):
                if print_metadata:
                    print(f'{indent}METADATA: {val}: {hObj.attrs[val]}')
                else:
                    print(f'{indent}METADATA: {val}: shape={hObj.attrs[val].shape} , dtype={hObj.attrs[val].dtype}')
        
        for ii,val in enumerate(list(iter(hObj))):
            if isinstance(hObj[val] , h5py.Group) or isinstance(hObj[val], hdfdict.hdfdict.LazyHdfDict):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            elif isinstance(hObj[val] , dict):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            else:
                if hasattr(hObj[val] , 'shape') and hasattr(hObj[val] , 'dtype'):
                    print(f'{indent}{ii+1}. {val}:   shape={hObj[val].shape} , dtype={hObj[val].dtype}')
                else:
                    print(f'{indent}{ii+1}. {val}:   type={type(hObj[val])}')


def make_h5_tree(dict_obj , h5_obj , group_string=''):
    '''
    This function is meant to be called by write_dict_to_h5. It probably shouldn't be called alone.
    This function creates an h5 group and dataset tree structure based on the hierarchy and values within a python dict.
    There is a recursion in this function.
    RH 2021
    '''
    for ii,(key,val) in enumerate(dict_obj.items()):
        if group_string=='':
            group_string='/'
        if isinstance(val , dict):
            # print(f'making group:  {key}')
            h5_obj[group_string].create_group(key)
            make_h5_tree(val , h5_obj[group_string] , f'{group_string}/{key}')
        else:
            # print(f'saving:  {group_string}: {key}')
            h5_obj[group_string].create_dataset(key , data=val)
def write_dict_to_h5(path_save , input_dict , write_mode='w-', show_item_tree_pref=True):
    '''
    Writes an h5 file that matches the hierarchy and data within a python dict.
    This function calls the function 'make_h5_tree'
    RH 2021
   
    Args:
        path_save (string or Path): 
            Full path name of file to write
        input_dict (dict): 
            Dict containing only variables that can be written as a 'dataset' in an h5 file (generally np.ndarrays and strings)
        write_mode ('w' or 'w-'): 
            The priveleges of the h5 file object. 'w' will overwrite. 'w-' will not overwrite
        show_item_tree_pref (bool): 
            Whether you'd like to print the item tree or not
    '''
    with h5py.File(path_save , write_mode) as hf:
        make_h5_tree(input_dict , hf , '')
        if show_item_tree_pref:
            print(f'==== Successfully wrote h5 file. Displaying h5 hierarchy ====')
            show_item_tree(hf)


def simple_load(path=None, directory=None, fileName=None, verbose=False):
    """
    Returns a lazy dictionary object (specific
    to hdfdict package) containing the groups
    as keys and the datasets as values from
    given hdf file.
    RH 2021

    Args:
        path (string or Path): 
            Full path name of file to read.
            If None, then directory and fileName must be specified.
        directory (string): 
            Directory of file to read.
            Used if path is None.
        fileName (string):
            Name of file to read.
            Used if path is None.
        verbose (bool):
            Whether or not to print out the h5 file hierarchy.
    
    Returns:
        h5_dict (LazyHdfDict):
            LazyHdfDict object containing the groups
    """

    if path is None:
        directory = Path(directory).resolve()
        fileName_load = fileName
        path = directory / fileName_load

    h5Obj = hdfdict.load(str(path), **{'mode': 'r'})
    
    if verbose:
        show_item_tree(hObj=h5Obj)
    
    return h5Obj


def simple_save(dict_to_save, path=None, directory=None, fileName=None, write_mode='w-', verbose=False):
    """
    Saves a python dict to an hdf file.
    Also allows for adding new data to
     an existing hdf file.
    RH 2021

    Args:
        dict_to_save (dict):
            Python dict to save to hdf file.
        path (string or Path):
            Full path name of file to write.
            If None, then directory and fileName must be specified.
        directory (string):
            Directory of file to write.
            Used if path is None.
        fileName (string):
            Name of file to write.
            Used if path is None.
        write_mode ('w', 'w-', 'a'):
            The priveleges of the h5 file object.
            'w' will overwrite.
            'w-' will not overwrite.
            'a' will append/add a new dataset to the h5 file.
        verbose (bool):
            Whether or not to print out the h5 file hierarchy.
    """

    if path is None:
        directory = Path(directory).resolve()
        fileName_load = fileName
        path = directory / fileName_load

    write_dict_to_h5(path, dict_to_save, write_mode=write_mode, show_item_tree_pref=verbose)


def h5py_dataset_iterator(g, prefix=''):
    '''
    Made by Akshay. More general version of above. Could be useful.
    '''
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)


def dump_nwb(nwb_path):
    """
    Print out nwb contents

    Args:
        nwb_path (str): path to the nwb file

    Returns:
    """
    import pynwb
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        for interface in nwbfile.processing['Face Rhythm'].data_interfaces:
            print(interface)
            time_series_list = list(nwbfile.processing['Face Rhythm'][interface].time_series.keys())
            for ii, time_series in enumerate(time_series_list):
                data_tmp = nwbfile.processing['Face Rhythm'][interface][time_series].data
                print(f"     {time_series}:    {data_tmp.shape}   ,  {data_tmp.dtype}   ,   {round((data_tmp.size * data_tmp.dtype.itemsize)/1000000000, 6)} GB")
