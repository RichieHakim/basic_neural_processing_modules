from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

class Equivalence_checker():
    """
    Class for checking if all items are equivalent or allclose (almost equal) in
    two complex data structures. Can check nested lists, dicts, and other data
    structures. Can also optionally assert (raise errors) if all items are not
    equivalent. 
    RH 2023

    Attributes:
        _kwargs_allclose (Optional[dict]): 
            Keyword arguments for the `numpy.allclose` function.
        _assert_mode (bool):
            Whether to raise an assertion error if items are not close.

    Args:
        kwargs_allclose (Optional[dict]): 
            Keyword arguments for the `numpy.allclose` function. (Default is
            ``{'rtol': 1e-7, 'equal_nan': True}``)
        assert_mode (bool): 
            Whether to raise an assertion error if items are not close.
        verbose (bool):
            How much information to print out:
                * ``False`` / ``0``: No information printed out.
                * ``True`` / ``1``: Mismatched items only.
                * ``2``: All items printed out.
    """
    def __init__(
        self,
        kwargs_allclose: Optional[dict] = {'rtol': 1e-7, 'equal_nan': True},
        assert_mode=False,
        verbose=False,
    ) -> None:
        """
        Initializes the Allclose_checker.
        """
        self._kwargs_allclose = kwargs_allclose
        self._assert_mode = assert_mode
        self._verbose = verbose
        
    def _checker(
        self, 
        test: Any,
        true: Any, 
        path: Optional[List[str]] = None,
    ) -> bool:
        """
        Compares the test and true values using numpy's allclose function.

        Args:
            test (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                Test value to compare.
            true (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                True value to compare.
            path (Optional[List[str]]): 
                The path of the data structure that is currently being compared.
                (Default is ``None``)

        Returns:
            (bool): 
                result (bool): 
                    Returns True if all elements in test and true are close.
                    Otherwise, returns False.
        """
        try:
            ## If the dtype is a kind of string (or byte string) or object, then allclose will raise an error. In this case, just check if the values are equal.
            if np.issubdtype(test.dtype, np.str_) or np.issubdtype(test.dtype, np.bytes_) or test.dtype == np.object_:
                out = bool(np.all(test == true))
            else:
                out = np.allclose(test, true, **self._kwargs_allclose)
        except Exception as e:
            out = None  ## This is not False because sometimes allclose will raise an error if the arrays have a weird dtype among other reasons.
            warnings.warn(f"WARNING. Equivalence check failed. Path: {path}. Error: {e}") if self._verbose else None
            
        if out == False:
            if self._assert_mode:
                raise AssertionError(f"Equivalence check failed. Path: {path}.")
            if self._verbose:
                ## Come up with a way to describe the difference between the two values. Something like the following:
                ### IF the arrays are numeric, then calculate the relative difference
                dtypes_numeric = (np.number, np.bool_, np.integer, np.floating, np.complexfloating)
                if any([np.issubdtype(test.dtype, dtype) and np.issubdtype(true.dtype, dtype) for dtype in dtypes_numeric]):
                    diff = np.abs(test - true)
                    at = np.abs(true)
                    r_diff = diff / at if np.all(at != 0) else np.inf
                    r_diff_mean, r_diff_max, any_nan = np.nanmean(r_diff), np.nanmax(r_diff), np.any(np.isnan(r_diff))
                    ## fraction of mismatches
                    n_elements = np.prod(test.shape)
                    n_mismatches = np.sum(diff > 0)
                    frac_mismatches = n_mismatches / n_elements
                    ## Use scientific notation and round to 3 decimal places
                    reason = f"Equivalence: Relative difference: mean={r_diff_mean:.3e}, max={r_diff_max:.3e}, any_nan={any_nan}, n_elements={n_elements}, n_mismatches={n_mismatches}, frac_mismatches={frac_mismatches:.3e}"
                else:
                    reason = f"Values are not numpy numeric types. types: {test.dtype}, {true.dtype}"
        else:
            reason = "equivlance"

        return out, reason

    def __call__(
        self,
        test: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None], 
        true: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None], 
        path: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Compares the test and true values and returns the comparison result.
        Handles various data types including dictionaries, iterables,
        np.ndarray, scalars, strings, numbers, bool, and None.

        Args:
            test (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                Test value to compare.
            true (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                True value to compare.
            path (Optional[List[str]]): 
                The path of the data structure that is currently being compared.
                (Default is ``None``)

        Returns:
            Dict[Tuple[bool, str]]: 
                result Dict[Tuple[bool, str]]: 
                    The comparison result as a dictionary or a tuple depending
                    on the data types of test and true.
        """
        if path is None:
            path = ['']

        if len(path) > 0:
            if path[-1].startswith('_'):
                return (None, 'excluded from testing')

        ## NP.NDARRAY
        if isinstance(true, np.ndarray):
            result = self._checker(test, true, path)
        ## NP.SCALAR
        elif np.isscalar(true):
            if isinstance(test, (int, float, complex, np.number)):
                result = self._checker(np.array(test), np.array(true), path)
            else:
                result = (test == true, 'equivalence')
        ## NUMBER
        elif isinstance(true, (int, float, complex)):
            result = self._checker(test, true, path)
        ## DICT
        elif isinstance(true, dict):
            result = {}
            for key in true:
                if key not in test:
                    result[str(key)] = (False, 'key not found')
                else:
                    result[str(key)] = self.__call__(test[key], true[key], path=path + [str(key)])
        ## ITERATABLE
        elif isinstance(true, (list, tuple, set)):
            if len(true) != len(test):
                result = (False, 'length_mismatch')
            else:
                if all([isinstance(i, (int, float, complex, np.number)) for i in true]):
                    result = self._checker(np.array(test), np.array(true), path)
                else:
                    result = {}
                    for idx, (i, j) in enumerate(zip(test, true)):
                        result[str(idx)] = self.__call__(i, j, path=path + [str(idx)])
        ## STRING
        elif isinstance(true, str):
            result = (test == true, 'equivalence')
        ## BOOL
        elif isinstance(true, bool):
            result = (test == true, 'equivalence')
        ## NONE
        elif true is None:
            result = (test is None, 'equivalence')

        ## OBJECT with __dict__
        elif hasattr(true, '__dict__'):
            result = {}
            for key in true.__dict__:
                if key.startswith('_'):
                    continue
                if not hasattr(test, key):
                    result[str(key)] = (False, 'key not found')
                else:
                    result[str(key)] = self.__call__(getattr(test, key), getattr(true, key), path=path + [str(key)])
        ## N/A
        else:
            result = (None, 'not tested')

        if isinstance(result, tuple):
            if self._assert_mode:
                assert (result[0] != False), f"Equivalence check failed. Path: {path}. Reason: {result[1]}"

            if self._verbose > 0:
                ## Print False results
                if result[0] == False:
                    print(f"Equivalence check failed. Path: {path}. Reason: {result[1]}")
            if self._verbose > 1:
                ## Print True results
                if result[0] == True:
                    print(f"Equivalence check passed. Path: {path}. Reason: {result[1]}")
                elif result[0] is None:
                    print(f"Equivalence check not tested. Path: {path}. Reason: {result[1]}")

        return result