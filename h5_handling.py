import gc
import h5py
def close_all_h5():
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed
    gc.collect()



def show_group_items(hObj):
    '''
    simple function that displays all the items and groups in an h5 object or python dict
    args:
        hObj: 'hierarchical Object' hdf5 object or subgroup object OR python dictionary
    returns: NONE

    ##############

    example usage:
        with h5py.File(path , 'r') as f:
            h5_handling.show_group_items(f)

    RH 2021
    '''
    for ii,val in enumerate(list(iter(hObj))):
        if isinstance(hObj[val] , h5py.Group):
            print(f'{ii+1}. {val}:   <group>')
        if isinstance(hObj[val] , dict):
            print(f'{ii+1}. {val}:   <dict>')
        else:
            if hasattr(hObj[val] , 'shape') and hasattr(hObj[val] , 'dtype'):
                print(f'{ii+1}. {val}:   shape={hObj[val].shape} , dtype={hObj[val].dtype}')
            else:
                print(f'{ii+1}. {val}:   type={type(hObj[val])}')



def show_item_tree(hObj , show_metadata=True , print_metadata=False, indent_level=0):
    '''
    recurive function that displays all the items and groups in an h5 object or python dict
    args:
        hObj: 'hierarchical Object'. hdf5 object OR python dictionary
        show_metadata (bool): whether or not to list metadata with items
        print_metadata (bool): whether or not to show values of metadata items
        indent_level: used internally to function. User should leave blank
    returns: NONE

    ##############
    
    example usage:
        with h5py.File(path , 'r') as f:
            h5_handling.show_item_tree(f)

    RH 2021
    '''
    indent = f'  '*indent_level
    if hasattr(hObj, 'attrs') and show_metadata:
        for ii,val in enumerate(list(hObj.attrs.keys()) ):
            if print_metadata:
                print(f'{indent}METADATA: {val}: {hObj.attrs[val]}')
            else:
                print(f'{indent}METADATA: {val}: shape={hObj.attrs[val].shape} , dtype={hObj.attrs[val].dtype}')
    
    for ii,val in enumerate(list(iter(hObj))):
        if isinstance(hObj[val] , h5py.Group):
            print(f'{indent}{ii+1}. {val}:   <group>')
            show_item_tree(hObj[val], show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
        if isinstance(hObj[val] , dict):
            print(f'{indent}{ii+1}. {val}:   <dict>')
            show_item_tree(hObj[val], show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
        else:
            if hasattr(hObj[val] , 'shape') and hasattr(hObj[val] , 'dtype'):
                print(f'{indent}{ii+1}. {val}:   shape={hObj[val].shape} , dtype={hObj[val].dtype}')
            else:
                print(f'{indent}{ii+1}. {val}:   type={type(hObj[val])}')
