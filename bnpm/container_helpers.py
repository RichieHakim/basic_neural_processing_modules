import re
import copy
from collections.abc import MutableMapping

import numpy as np

"""
This module is intended to have minimal dependencies.
It is called by server.py which is intended to run
 outside of a specialized environment, so no weird
 libraries should be required.
RH 2022
"""

class Lazy_repeat_item():
    """
    Makes a lazy iterator that repeats an item.
    Very similar to itertools.repeat, but allows
     for a pseudo length to be specified.
     RH 2021
    """
    def __init__(self, item, pseudo_length=None):
        """
        Args:
            item (any object):
                item to repeat
            pseudo_length (int):
                length of the iterator.
        """
        super().__init__()
        self.item = item
        self.pseudo_length = pseudo_length

    def __getitem__(self, i):
        """
        Args:
            i (int):
                index of item to return.
                Ignored if pseudo_length is None.
        """
        if self.pseudo_length is None:
            return self.item
        elif i < self.pseudo_length:
            return self.item
        else:
            raise IndexError('Index out of bounds')


    def __len__(self):
        return self.pseudo_length

    def __repr__(self):
        return repr(self.item)


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
            IGNORE: USED INTERNALLY FOR RECURSION
        sep (str):
            separator to use between keys
            IGNORE: USED INTERNALLY FOR RECURSION

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