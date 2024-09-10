"""
Custom system for saving and loading arrays to binary files.
Useful for saving large arrays to disk and loading them quickly as memory-mapped
torch tensors.

CONVENTIONS:
    - Arrays are saved as binary files using the tensor.tofile() method.
    - Filepaths to arrays have ending + suffix '[filename].array.bin'.
    - Metadata for each array is saved as a JSON file in the same directory and
      with the same name as the array, but with the ending + suffix
      '[filename].metadata.json'.
    - Metadata files are saved in the same directory as the array files.
    - Metadata files contain a dictionary with the following fields:
        - size: int = Number of elements in the array.
        - shape: List[int] = Shape of the array.
        - dtype: str = Data type of the array as a string (e.g. 'float32').
        - type: str = Type of array ('numpy' or 'torch').
        - hash_type: str = Hash type used to hash the array file (e.g. 'MD5').
        - hash: str = Hash of the array file.

USAGE:
    1. You have a set of arrays that you want to save to disk:
    2. Make a dictionary with the desired filepaths and arrays:
        paths_arrays = {
            '[filepath_1].array.bin': array_1,
            '[filepath_2].array.bin': array_2,
            ...
        }
    3. Call the save_arrays_to_binary() function to save the arrays to disk:
        paths_arrays_out, paths_metadata_out = save_arrays_to_binary(paths_arrays=paths_arrays)
    4. You can now load the arrays from disk using the load_arrays_from_binary() function:
        arrays = load_arrays_from_binary(paths_arrays=paths_arrays_out, paths_metadata=paths_metadata_out)
        Note that if the metadata filepaths are not provided, they are generated
        from the array filepaths by replacing '.array.bin' with '.metadata.json'.
"""

from typing import List, Optional

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import tensordict

from . import file_helpers

def save_arrays_to_binary(
    paths_arrays: dict,
    mkdir: bool = False,
    allow_overwrite: bool = False,
    verbose: bool = True,
):
    """
    Saves arrays (numpy arrays or torch tensors) to binary files.
    RH 2024

    Args:
        paths_arrays (dict):
            Dictionary of paths to save arrays to. The suffix will be overwritten
            to be '.array.bin'.
            Each entry should be:
                {'filepath': array}
        mkdir (bool):
            If True, creates parent directory of path_save if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.
        verbose (bool):
            If True, a tqdm progress bar is displayed with the array name.

    Returns:
        tuple:
            paths_arrays_out (List[str]):
                List of paths to the saved arrays.
            paths_metadata_out (List[str]):
                List of paths to the saved metadata files.
    """

    def to_np(arr):
        return arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr
        
    paths_arrays_out, paths_metadata_out = [], []
    for path, array in tqdm(paths_arrays.items(), disable=(verbose==False), desc='Saving arrays to binary files', unit=' arrays'):
        path = file_helpers.prepare_filepath_for_saving(path, mkdir=mkdir, allow_overwrite=allow_overwrite)
        ## Save array
        ### If filename already ends in '.array.suffix', then only replace the suffix
        suffix_new = '.bin' if Path(path).with_suffix('').name.endswith('.array') else '.array.bin'
        path_array = str(Path(path).with_suffix(suffix_new))
        to_np(array).tofile(path_array)
        ## Save metadata: path.metadata.json
        path_metadata = str(Path(path).with_suffix('.metadata.json'))
        metadata = {
            'size': int(np.prod(array.shape)),
            'shape': list(array.shape),
            'dtype': str(array.numpy().dtype),
            'type': 'torch' if isinstance(array, torch.Tensor) else 'numpy',
            'hash_type': 'MD5',
            'hash': file_helpers.hash_file(path_array, type_hash='MD5'),
        }
        file_helpers.json_save(
            metadata,
            path_metadata,
            indent=4,
            mode='w',
            mkdir=mkdir,
            allow_overwrite=allow_overwrite,
        )
        paths_arrays_out.append(path_array)
        paths_metadata_out.append(path_metadata)
    return paths_arrays_out, paths_metadata_out
        


def load_arrays_from_binary(
    paths_arrays: List[str],
    paths_metadata: Optional[List[str]] = None,
    check_hash: bool = True,
    memory_map: bool = True,
    shared_memory: bool = False,
    pin_memory: bool = False,
    verbose: bool = True,
):
    """
    Loads arrays (numpy arrays or torch tensors) from binary files.
    Optionally, uses PyTorch's memory mapping function .from_file() to load and
    memory map the array.
    Optionally, checks the integrity of the file using a hash.
    RH 2024

    Args:
        paths_arrays (List[str]):
            List of paths to load arrays from. Each path should end with
            '.array.bin'.
        paths_metadata (Optional[List[str]]):
            List of paths to load metadata from.
            If None, then paths_metadata is generated from paths_arrays by
            replacing '.array.bin' with '.metadata.json'.
        memory_map (bool):
            If True, memory maps the array. Loading time is nearly instantaneous,
            but operations on the array become 'lazy'.
            If False, loads the array into memory.
        shared_memory (bool):
            If True, sets `shared=True` in tensor.from_file() function. This
            allows the tensor to be shared between processes if the tensor is
            memory mapped.
        pin_memory (bool):
            If True, sets `pin_memory=True` in tensor.from_file() function. This
            allows the tensor to be pinned to CUDA memory if the tensor is memory
            mapped.
        check_hash (bool):
            Hash type to check the integrity of the file. Slows down the loading
            process significantly for large files.
            If None, no hash is checked.
            If not None, the file is hashed and compared using the ['hash_type']
            and ['hash'] fields in the metadata file. Metadata file is required.
        verbose (bool):
            If True, a tqdm progress bar is displayed with the array name.

    Returns:
        List[torch.Tensor]:
            List of loaded arrays.
    """
    ## Get paths and metadata
    def check_paths_arrays(paths_arrays):
        assert isinstance(paths_arrays, list), "paths_arrays must be a list."
        assert all([isinstance(p, str) for p in paths_arrays]), "paths_arrays must be a list of strings."
        ## Assert that all paths end with '.array.bin'
        assert all([p.endswith('.array.bin') for p in paths_arrays]), "All paths must end with '.array.bin'."
    def check_paths_metadata(paths_metadata):
        assert isinstance(paths_metadata, list), "paths_metadata must be a list."
        assert all([isinstance(p, str) for p in paths_metadata]), "paths_metadata must be a list of strings."
        ## Assert that all paths end with '.metadata.json'
        assert all([p.endswith('.metadata.json') for p in paths_metadata]), "All paths must end with '.metadata.json'."
        assert len(paths_arrays) == len(paths_metadata), "Both paths_arrays and paths_metadata must have the same length."

    def check_metadata(metadata):
        assert isinstance(metadata, dict), "metadata must be a dictionary."
        keys = ['size', 'shape', 'dtype', 'type', 'hash_type', 'hash']
        assert all([key in metadata for key in keys]), f"metadata must have the following keys: {keys}. Found: {metadata.keys()}."  
        return metadata

    check_paths_arrays(paths_arrays)
    paths_metadata = [p.replace('.array.bin', '.metadata.json') for p in paths_arrays] if paths_metadata is None else paths_metadata
    check_paths_metadata(paths_metadata)
    metadatas = [check_metadata(file_helpers.json_load(p)) for p in paths_metadata]
    
    def check_and_load(path_array, meta, memory_map, check_hash):
        ## Check hash
        if check_hash:
            hash_file = file_helpers.hash_file(path=path_array, type_hash=meta['hash_type'])
            assert hash_file == meta['hash'], f"Hash mismatch for {path_array}."
        ## Load array
        array = torch.from_file(
            filename=path_array, 
            shared=shared_memory, 
            size=meta['size'], 
            # dtype=torch_helpers.numpy_to_torch_dtype_dict[np.dtype(meta['dtype']).type],
            dtype=torch.float32,
            pin_memory=pin_memory,
        ).reshape(*tuple(meta['shape']))
        if not memory_map:
            array = array.clone()
        return array
    
    arrays = [check_and_load(path_array, meta, memory_map, check_hash) for path_array, meta in tqdm(zip(paths_arrays, metadatas), total=len(paths_arrays), disable=(verbose==False), desc='Loading arrays from binary files', unit=' arrays')]
    return arrays


class Dataset_Arrays(torch.utils.data.Dataset):
    """
    Dataset class for loading arrays from binary files.
    RH 2024

    Args:
        arrays (List[torch.Tensor]):
            List of arrays to load.
        
    """
    def __init__(self, arrays):
        super(Dataset_Arrays, self).__init__()

        assert isinstance(arrays, list), "arrays must be a list."
        assert all([isinstance(arr, torch.Tensor) for arr in arrays]), "arrays must be a list of torch tensors."
        
        self.arrays = arrays
        self.shapes_arrays = [arr.shape for arr in arrays]
        self.n_samples = sum([int(shape[0]) for shape in self.shapes_arrays])
        if self.arrays[0].ndim == 1:
            self.shape = [self.n_samples]
        else:
            assert all([shape[1:] == self.shapes_arrays[0][1:] for shape in self.shapes_arrays]), "All arrays must have the same shape except for the first dimension."
        self.shape = [self.n_samples] + list(self.shapes_arrays[0][1:])


class Dataset_TensorDict_concatenated(torch.utils.data.Dataset):
    """
    Dataset class for loading slices from a TensorDict.
    Input is a TensorDict containing arrays with similar shapes. Output is a
    Dataset where the queried index pulls slices from the concatenated first
    dimension indices of all the input arrays.
    RH 2024

    Args:
        tensor_dict (TensorDict):
            A TensorDict with the following organization: \n
                * One hierarchical level of fields.
                * Each field is a tensor.
                * Each tensor may have different first dimension sizes, but all
                  other dimensions must be the same.
                * Example:
                    TensorDict(
                        fields={
                            'array_1': tensor_1 (shape=[X1, M, N, P, ...]),
                            'array_2': tensor_2 (shape=[X2, M, N, P, ...]),
                            ...
                        },
                        batch_size=torch.Size([]),
                    )
    """
    def __init__(self, tensor_dict: tensordict.TensorDict):
        super(Dataset_TensorDict_concatenated, self).__init__()

        assert isinstance(tensor_dict, tensordict.TensorDict), "tensor_dict must be a TensorDict."
        self.tensor_dict = tensor_dict
        
        ## Check that all arrays have the same shape except for the first dimension
        shapes = [arr.shape for arr in tensor_dict.values()]
        check_shape = lambda shape1, shape2: shape1[1:] == shape2[1:] if len(shape1) > 1 else shape1[0] == shape2[0]
        assert all([check_shape(shape, shapes[0]) for shape in shapes]), "All arrays must have the same shape except for the first dimension."
        self.n_samples = sum([shape[0] for shape in shapes])
        self.shape = [self.n_samples] + list(shapes[0][1:])
        self.fields = list(tensor_dict.keys())

        ## Create an index to field mapping
        ### Use a binary search to find the field for a given index using the cumsum of the first dimensions
        self.cumsum = torch.cumsum(torch.as_tensor([0] + [shape[0] for shape in shapes], dtype=torch.int64), dim=0)
        self.idx_to_fieldIdx = lambda idx: torch.searchsorted(self.cumsum, idx, side='right') - 1
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if (idx < 0):
            idx = self.n_samples + idx
        elif (idx >= self.n_samples):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.n_samples}.")
        
        fieldIdx = self.idx_to_fieldIdx(idx)
        field = self.fields[fieldIdx]
        idx_field = idx - self.cumsum[fieldIdx]
        return self.tensor_dict[field][idx_field]
    


class Dataset_TensorDict_concatenated2(torch.utils.data.Dataset):
    """
    Dataset class for loading slices from arrays within a TensorDict.\n
    Input is a TensorDict containing arrays with similar shapes.\n
    Output is a Dataset where the queried index pulls slices from the
    concatenated first dimension indices of all the input arrays.\n
    RH 2024

    Args:
        tensor_dict (TensorDict):
            A TensorDict with the following organization: \n
                * One hierarchical level of fields.
                * Each field is a tensor.
                * Each tensor may have different first dimension sizes, but all
                  other dimensions must be the same.
                * Example:
                    TensorDict(
                        fields={
                            'array_1': tensor_1 (shape=[X1, M, N, P, ...]),
                            'array_2': tensor_2 (shape=[X2, M, N, P, ...]),
                            ...
                        },
                        batch_size=torch.Size([]),
                    )
        verbose (bool):
            If ``True``, displays progress bar.
    """
    def __init__(self, tensor_dict: tensordict.TensorDict, verbose: bool = True):
        super(Dataset_TensorDict_concatenated, self).__init__()

        self.verbose = verbose

        assert isinstance(tensor_dict, tensordict.TensorDict), "tensor_dict must be a TensorDict."
        self.tensor_dict = tensor_dict
        self.load()

        ## Check that all arrays have the same shape except for the first dimension
        shapes = [arr.shape for arr in self.td_mm.values()]
        check_shape = lambda shape1, shape2: shape1[1:] == shape2[1:] if len(shape1) > 1 else shape1[0] == shape2[0]
        assert all([check_shape(shape, shapes[0]) for shape in shapes]), "All arrays must have the same shape except for the first dimension."
        self.n_samples = sum([shape[0] for shape in shapes])
        self.shape = [self.n_samples] + list(shapes[0][1:])
        self.fields = list(self.td_mm.keys())
        self.dtype = self.td_mm[self.fields[0]].dtype
        
        ## Create an index to field mapping
        ### Use a binary search to find the field for a given index using the cumsum of the first dimensions
        self.cumsum = torch.cumsum(torch.as_tensor([0] + [shape[0] for shape in shapes], dtype=torch.int64), dim=0)
        self.idx_to_fieldIdx = lambda idx: torch.searchsorted(self.cumsum, idx, side='right') - 1

    def load(self):
        # self.td_mm = {key: tensordict.MemoryMappedTensor.from_tensor(td, copy_data=False) for key, td in self.tensor_dict.items()}
        self.td_mm = self.tensor_dict

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self.__init__(self.tensor_dict, verbose=self.verbose)
        if (idx < 0):
            idx = self.n_samples + idx
        elif (idx >= self.n_samples):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {self.n_samples}.")
        
        fieldIdx = self.idx_to_fieldIdx(idx)
        field = self.fields[fieldIdx]
        idx_field = idx - self.cumsum[fieldIdx]
        sample = self.td_mm[field][idx_field]

        return sample
