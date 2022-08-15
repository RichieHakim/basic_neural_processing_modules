import numpy as np
import copy
from collections.abc import MutableMapping
import itertools

"""
This module is intended to have minimal dependencies.
It is called by server.py which is intended to run
 outside of a specialized environment, so no weird
 libraries should be required.
RH 2022
"""

class lazy_repeat_item():
    """
    Makes a lazy iterator that repeats an item.
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
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def deep_update_dict(dictionary, key, val, in_place=False):
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
        val (any):
            Value to update with
        in_place (bool):
            whether to update in place

    Returns:
        output (Dict):
            updated dictionary
    """
    def helper_deep_update_dict(d, key, val):
        if type(key) is str:
            key = [key]

        assert key[0] in d, f"RH ERROR, key: '{key[0]}' is not found"

        if type(key) is list:
            if len(key) > 1:
                helper_deep_update_dict(d[key[0]], key[1:], val)
            elif len(key) == 1:
                key = key[0]
                d.update({key:val})

    if in_place:
        helper_deep_update_dict(dictionary, key, val)
    else:
        d = copy.deepcopy(dictionary)
        helper_deep_update_dict(d, key, val)
        return d


def dict_shared_items(d1, d2):
    """
    Returns the matching items between two dictionaries.
    Searches for d1 items within d2.
    RH 2022
    """
    return {k: d1[k] for k in d1 if k in d2 and d1[k] == d2[k]}
def dict_diff_items(d1, d2):
    """
    Returns the differing items between two dictionaries.
    Searches for d1 items within d2.
    RH 2022
    """    
    return {k: d1[k] for k in d1 if k in d2 and d1[k] != d2[k]}
def dict_missing_keys(d1, d2):
    """
    Returns the keys in d1 that are missing in d2
    RH 2022
    """    
    return {k for k,v in d1.items() if k not in d2}


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


