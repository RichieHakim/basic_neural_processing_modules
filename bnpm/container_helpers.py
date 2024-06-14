import re
import copy
from collections.abc import MutableMapping
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Iterable, Iterator, Type
import itertools

import toolz

"""
This module is intended to have minimal dependencies.
It is called by server.py which is intended to run
 outside of a specialized environment, so no weird
 libraries should be required.
RH 2022
"""

class lazy_repeat_obj(itertools.repeat):
    """
    Makes a lazy iterator that repeats an object.
    Makes a subclass of itertools.repeat that has a length.
    RH 2021

    Args:
        obj (Any):
            Object to repeat.
        pseudo_length (int):
            length of the iterator. Don't use keyword argument, just pass the
            length as the second argument.
    """
    def __init__(
        self, 
        obj: Any, 
        pseudo_length: Optional[int] = None,
    ):
        """
        Initializes the lazy iterator.
        """
        self.obj = obj
        self.pseudo_length = pseudo_length
        self.count = 0

    def __iter__(self):
        """
        Returns the iterator.
        """
        return self
    
    def __next__(self):
        """
        Returns the next object.
        """
        if self.pseudo_length is not None:
            if self.count >= self.pseudo_length:
                raise StopIteration
            self.count += 1
        return self.obj
    
    def __len__(self):
        """
        Returns the length of the iterator.
        """
        return self.pseudo_length
    
    def __repr__(self):
        """
        Returns the representation of the object.
        """
        return f"lazy_repeat_obj({self.obj}, pseudo_length={self.pseudo_length})"


def flatten_list(irregular_list):
    """
    Flattens a list of lists into a single list.
    Stolen from https://stackabuse.com/python-how-to-flatten-list-of-lists/
    RH 2022

    Args:
        irregular_list (list):
            list of lists to flatten

    Returns:
        output (list):
            flattened list
    """
    helper = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]
    return helper(irregular_list)


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    """
    Flattens a dictionary of dictionaries into a 
     single dictionary.
    NOTE: Turns all keys into strings.
    Stolen from https://stackoverflow.com/a/6027615
    RH 2022

    Args:
        d (Dict):
            dictionary to flatten
        parent_key (str):
            key to prepend to flattened keys
            USED INTERNALLY FOR RECURSION
        sep (str):
            separator to use between keys
            USED INTERNALLY FOR RECURSION

    Returns:
        output (Dict):
            flattened dictionary
    """

    items = []
    for k, v in d.items():
        new_key = str(parent_key) + str(sep) + str(k) if parent_key else str(k)
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def deep_update_dict(dictionary, key, new_val=None, new_key=None, in_place=False):
    """
    Updates a dictionary with a new value.
    RH 2022

    Args:
        dictionary (Dict):
            dictionary to update
        key (list of str):
            Key to update
            List elements should be strings.
            Each element should be a hierarchical
             level of the dictionary.
            DEMO:
                deep_update_dict(params, ['dataloader_kwargs', 'prefetch_factor'], val)
        new_val (any):
            If not None, the value to update with this
            If None, then new_key must be specified and will only
             be used to update the key.
        new_key (str):
            If not None, the key will be updated with this key.
             [key[-1]] will be deleted and replaced with new_key.
            If None, then [key[-1]] will be updated with new_val.
             
        in_place (bool):
            whether to update in place

    Returns:
        output (Dict):
            updated dictionary
    """
    def helper_deep_update_dict(d, key):
        if type(key) is str:
            key = [key]

        assert key[0] in d, f"RH ERROR, key: '{key[0]}' is not found"

        if type(key) is list:
            if len(key) > 1:
                helper_deep_update_dict(d[key[0]], key[1:])
            elif len(key) == 1:
                val = d[key[0]] if new_val is None else new_val
                if new_key is None:
                    d[key[0]] = val
                else:
                    d[new_key] = val
                    del d[key[0]]

    if in_place:
        helper_deep_update_dict(dictionary, key)
    else:
        d = copy.deepcopy(dictionary)
        helper_deep_update_dict(d, key)
        return d


def dict_shared_items(
    d1, 
    d2, 
    fn_check_equality=None, 
    error_if_equality_check_fails=True, 
    verbose=False
):
    """
    Returns the matching items between two dictionaries.
    Searches for d1 items within d2.
    RH 2023

    Args:
        d1 (Dict):
            first dictionary
        d2 (Dict):
            second dictionary
        fn_check_equality (function):
            function to use to check equality.
            If None, then equality is checked with ==.
            If not None, then the function should take two
             arguments and return True if they are equal.
        error_if_equality_check_fails (bool):
            whether to raise an error if the equality check fails.
            If False and verbose==True, then a warning is printed
             instead.
        verbose (bool):
            whether to print warnings if the equality check fails.

    Returns:
        output (Dict):
            dictionary of matching items
    """
    def check_equality(v1, v2):
        try:
            if fn_check_equality is not None:
                out = fn_check_equality(v1, v2)
            else:
                if v1 == v2:
                    out = True
                else:
                    out = False
        except ValueError as e:
            if error_if_equality_check_fails:
                raise e
            else:
                print(f'WARNING: {e}') if verbose else None
                out = False
        return out
    
    return {k: d1[k] for k in d1 if k in d2 and check_equality(d1[k], d2[k])}
def dict_diff_items(d1, d2):
    """
    Returns the differing items between two dictionaries.
    Searches for d1 items within d2.
    RH 2022
    """
    return {k: d1[k] for k in d1 if k in d2 and d1[k] != d2[k]}
def dict_missing_keys(d1, d2):
    """
    Returns the keys in d1 that are missing in d2.
    Symmetric difference between d1 and d2 keys.
    RH 2022
    """
    return set(d1.keys()).symmetric_difference(set(d2.keys()))


def find_differences_across_dictionaries(dicts):
    """
    Finds differences across many dictionaries.
    RH 2022

    Args:
        dicts (List):
            List of dictionaries to compare.

    Returns:
        params_unchanging (list of dicts):
            List of dictionary items that are the 
             same across all dictionaries.
        params_changing (list of dicts):
            List of dictionary items that are 
             different in at least one dictionary.
    """
    import numpy as np

    def get_binary_search_combos(n):
        combos = list(np.arange(n))
        if len(combos)%2 == 1:
            combos.append(combos[-1])
        combos = np.array(combos).reshape(len(combos)//2, 2)
        return combos

    ## flatten params to ease matching functions
    params_flat = [flatten_dict(param) for param in dicts]

    ## assert that set of all keys in each dict is exactly the same
    sets_of_keys = set([tuple(sorted(param.keys())) for param in params_flat])
    if len(sets_of_keys) > 1:
        import itertools
        ## use itertools to compare all combinations of sets_of_keys and find the differences
        combos = list(itertools.combinations(sets_of_keys, 2))
        diffs = []
        for combo in combos:
             diffs.append(set(combo[0]).symmetric_difference(set(combo[1])))
        ## collapse diffs into a single set
        diffs = set([item for sublist in diffs for item in sublist])
    assert len(sets_of_keys) == 1, f"RH ERROR, the following keys are not the same across all dictionaries: {diffs}"

    ## find unchanging params
    params_unchanging = copy.deepcopy(params_flat)
    while len(params_unchanging) > 1:
        combos = get_binary_search_combos(len(params_unchanging))
        params_unchanging = [dict_shared_items(params_unchanging[combo[0]], params_unchanging[combo[1]]) for combo in combos]
    params_unchanging = params_unchanging[0]

    ## find keys that are not unchanging
    mk = dict_missing_keys(params_flat[0], params_unchanging)

    ## make list dicts of changing params
    params_changing = [{k: params[k] for k in mk} for params in params_flat]
    
    return params_unchanging, params_changing


def find_subDict_key(d: dict, s: str, max_depth: int=9999999):
    """
    Recursively search for a sub-dictionary that contains the given string.
    Yield the result.

    Args:
        d (dict):
            dictionary to search
        s (str):
            string of the key to search for using regex
        max_depth (int):
            maximum depth to search.
            1 means only search the keys in the top level of
             the dictionary. 2 means the first and second level.

    Returns:
        k_all (list of tuples):
            List of 2-tuples: (list of tuples containing:
             list of strings of keys to sub-dictionary, value
             of sub-dictionary)
    """
    def helper_find_subDict_key(d, s, depth=999, _k_all=[]):
        """
        _k_all: 
            Used for recursion. List of keys. Set to [] on first call.
        depth: 
            Used for recursion. Decrements by 1 each call. At 0, stops
             recursion.
        """
        if depth > 0:    
            depth -= 1
            for k, v in d.items():
                if re.search(s, k):
                    yield _k_all + [k], v
                if isinstance(v, dict):
                    yield from helper_find_subDict_key(v, s, depth, _k_all + [k])
    return list(helper_find_subDict_key(d, s, depth=max_depth, _k_all=[]))


def merge_dicts(dicts: List[Dict]) -> Dict:
    """
    Merges a list of dictionaries into a single dictionary.
    If there are sub-dictionaries, then they are merged recursively.
    RH 2023

    Args:
        dicts (List):
            List of dictionaries to merge.

    Returns:
        output (Dict):
            merged dictionary
    """
    assert isinstance(dicts, list), f"RH ERROR, dicts must be a list, not {type(dicts)}"

    merged_dict = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                merged_dict[k] = merge_dicts([merged_dict.get(k, {}), v])
            else:
                merged_dict[k] = v

    return merged_dict


def invert_dict(d: Dict) -> Dict:
    """
    Inverts a dictionary. Requires that values are hashable. 
    Warning: if values are not unique, then only the last key will be kept.
    RH 2024

    Args:
        d (Dict):
            dictionary to invert

    Returns:
        output (Dict):
            inverted dictionary
    """
    return {v: k for k, v in d.items()}


def serializable_dict(
    obj: Any,
    allowed_libraries: Optional[List[str]] = [
        'bnpm',
        'builtins',
        'collections',
        'datetime',
        'itertools',
        'math',
        'numbers',
        'os',
        'pathlib',
        'string',
        'time',
        'numpy',
        'scipy',
        'sklearn',
    ],
) -> Dict[str, Any]:
    """
    Returns a serializable dictionary that can be saved to disk. This method
    goes through all items in self.__dict__ and checks if they are
    serializable. If they are, add them to a dictionary to be returned.

    Args:
        obj (Any):
            Object to serialize.

    Returns:
        (Dict[str, Any]): 
            serializable_dict (Dict[str, Any]): 
                Dictionary containing serializable items.
    """
    from functools import partial
    ## Go through all items in self.__dict__ and check if they are serializable.
    ### If they are, add them to a dictionary to be returned.
    import pickle

    ## Define a list of libraries and classes that are allowed to be serialized.
    # allowed_libraries = 
    def is_library_allowed(obj):
        try:
            try:
                module_name = obj.__module__.split('.')[0]
            except:
                success = False
            try:
                module_name = obj.__class__.__module__.split('.')[0]
            except:
                success = False
        except:
            success = False
        else:
            ## Check if the module_name is in the allowed_libraries list.
            if module_name in allowed_libraries:
                success = True
            else:
                success = False
        return success

    def make_serializable_dict(obj, depth=0, max_depth=100, name=None):
        """
        Recursively go through all items in self.__dict__ and check if they are serializable.
        """
        msd_partial = partial(make_serializable_dict, depth=depth+1, max_depth=max_depth)
        if depth > max_depth:
            raise Exception(f'RH ERROR: max_depth of {max_depth} reached with object: {obj}')

        serializable_dict = {}
        if hasattr(obj, '__dict__') and is_library_allowed(obj):
            for key, val in obj.__dict__.items():
                try:
                    serializable_dict[key] = msd_partial(val, name=key)
                except:
                    pass

        elif isinstance(obj, (list, tuple, set, frozenset)):
            serializable_dict = [msd_partial(v, name=f'{name}_{ii}') for ii,v in enumerate(obj)]
        elif isinstance(obj, dict):
            serializable_dict = {k: msd_partial(v, name=f'{name}_{k}') for k,v in obj.items()}
        else:
            try:
                assert is_library_allowed(obj), f'RH ERROR: object {obj} is not serializable'
                pickle.dumps(obj)
            except:
                return {'__repr__': repr(obj)} if hasattr(obj, '__repr__') else {'__str__': str(obj)} if hasattr(obj, '__str__') else None

            serializable_dict = obj

        return serializable_dict

    serializable_dict = make_serializable_dict(obj, depth=0, max_depth=100, name='object')
    return serializable_dict


def recursive_valmap(
    func: Callable[[Any], Any],
    d: Dict,
    depth: int = -1,
):
    """
    Apply a function to all values in a nested dictionary. \n
    If item is a dictionary, then apply the function to all values in the
    dictionary. \n
    Uses toolz.dicttoolz.valmap. \n
    RH 2024

    Args:
        func (Callable):
            Function to apply to all values.
        d (Dict):
            Dictionary to apply the function to.
        factory (Type[Dict]):
            Type of dictionary to use for the output.
        depth (int):
            Maximum depth to apply the function.
            If -1, then apply to all depths.
            If 1, then apply to the top level only.
            If 0, then return the dictionary as is.

    Returns:
        output (Dict):
            Dictionary with the function applied to all values.
    """
    def helper_recursive_valmap(func, d, depth):
        if depth == 0:
            return d
        return toolz.valmap(
            lambda x: helper_recursive_valmap(func, x, depth-1) if isinstance(x, dict) else func(x),
            d
        )

    return helper_recursive_valmap(func, d, depth)


############################################################
################# PARAMETER DICTIONARIES ###################
############################################################


def fill_in_dict(
    d: Dict, 
    defaults: Dict,
    verbose: bool = True,
    hierarchy: List[str] = ['dict'], 
):
    """
    In-place. Fills in dictionary ``d`` with values from ``defaults`` if they
    are missing. Works hierachically.
    RH 2023

    Args:
        d (Dict):
            Dictionary to fill in.
            In-place.
        defaults (Dict):
            Dictionary of defaults.
        verbose (bool):
            Whether to print messages.
        hierarchy (List[str]):
            Used internally for recursion.
            Hierarchy of keys to d.
    """
    from copy import deepcopy
    for key in defaults:
        if key not in d:
            print(f"Key '{key}' not found in params dictionary: {' > '.join([f'{str(h)}' for h in hierarchy])}. Using default value: {defaults[key]}") if verbose else None
            d.update({key: deepcopy(defaults[key])})
        elif isinstance(defaults[key], dict):
            assert isinstance(d[key], dict), f"Key '{key}' is a dict in defaults, but not in params. {' > '.join([f'{str(h)}' for h in hierarchy])}."
            fill_in_dict(d[key], defaults[key], hierarchy=hierarchy+[key])
            

def check_keys_subset(d, default_dict, hierarchy=['defaults']):
    """
    Checks that the keys in d are all in default_dict. Raises an error if not.
    RH 2023

    Args:
        d (Dict):
            Dictionary to check.
        default_dict (Dict):
            Dictionary containing the keys to check against.
        hierarchy (List[str]):
            Used internally for recursion.
            Hierarchy of keys to d.
    """
    default_keys = list(default_dict.keys())
    for key in d.keys():
        assert key in default_keys, f"Parameter '{key}' not found in defaults dictionary: {' > '.join([f'{str(h)}' for h in hierarchy])}."
        if isinstance(default_dict[key], dict) and isinstance(d[key], dict):
            check_keys_subset(d[key], default_dict[key], hierarchy=hierarchy+[key])


def prepare_params(params, defaults, verbose=True):
    """
    Does the following:
        * Checks that all keys in ``params`` are in ``defaults``.
        * Fills in any missing keys in ``params`` with values from ``defaults``.
        * Returns a deepcopy of the filled-in ``params``.

    Args:
        params (Dict):
            Dictionary of parameters.
        defaults (Dict):
            Dictionary of defaults.
        verbose (bool):
            Whether to print messages.
    """
    from copy import deepcopy
    ## Check inputs
    assert isinstance(params, dict), f"p must be a dict. Got {type(params)} instead."
    ## Make sure all the keys in p are valid
    check_keys_subset(params, defaults)
    ## Fill in any missing keys with defaults
    params_out = deepcopy(params)
    fill_in_dict(params_out, defaults, verbose=verbose)

    return params_out


def make_grid_search_dicts(search_space):
    """
    Makes a grid sweep over the search space.
    RH 2024

    Args:
        search_space (dict):
            Dictionary with keys as parameter names and values as lists of
            parameter values.\n
            Example: {'lr': [0.1, 0.01, 0.001], 'batch_size': [32, 64]}

    Returns:
        ss_comb_dicts (list):
            List of dictionaries, where each dictionary is a combination of
            parameter values from the search space.\n
            Example: [{'lr': 0.1, 'batch_size': 32}, {'lr': 0.1, 'batch_size': 64}, ...]
    """
    import itertools
    vals_comb = list(itertools.product(*search_space.values()))
    ss_comb_dicts = [{k: val for k, val in zip(search_space.keys(), vals)} for vals in vals_comb]
    return ss_comb_dicts