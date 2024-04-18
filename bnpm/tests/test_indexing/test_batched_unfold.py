import contextlib

import pytest
import hypothesis
from hypothesis import given, strategies as st

import torch
import numpy as np

from ...indexing import batched_unfold


def generate_tensor(ndim, n_samples_in_dim, dim):
    torch.manual_seed(0)
    shape_out = np.random.randint(low=1, high=10, size=ndim)
    shape_out[dim] = n_samples_in_dim
    return torch.rand(size=tuple(shape_out))


@hypothesis.given(
    st.integers(min_value=1, max_value=4),  # ndim
    st.integers(min_value=1, max_value=30),  # n_samples in dimension
    st.integers(min_value=1, max_value=30),  # size
    st.integers(min_value=1, max_value=30),  # step
    st.sampled_from([1, 2, 3, 4, 5, 25, 30, 31]),  # batch_size
    ## Either float or None
    # st.sampled_from([np.nan, None]), # padding
)
@hypothesis.settings(max_examples=1000, deadline=2000)
def test_batched_unfold(
    ndim, 
    n_samples_in_dim, 
    size, 
    step, 
    batch_size, 
    # padding,
):
    """
    NOTE: No tests for padding yet
    """
    np.random.seed(0)
    np.random.randint(low=0, high=ndim)  ## For some reason, you need to call this once so that dim is not always 0. Maybe hypothesis is caching the value?

    size = min(size, n_samples_in_dim)

    dim = np.random.randint(low=0, high=ndim)
    x = generate_tensor(ndim, n_samples_in_dim, dim)

    expected_error = None
    if step >= n_samples_in_dim:
        expected_error = pytest.raises(ValueError)
    if size > n_samples_in_dim:
        expected_error = pytest.raises(ValueError)

    @contextlib.contextmanager
    def nullcontext(enter_result=None):
        yield enter_result
    expected_error = nullcontext() if expected_error is None else expected_error
    
    with expected_error:
        x_unfolded = batched_unfold(
            tensor=x,
            dimension=dim,
            size=size,
            step=step,
            batch_size=batch_size,
            # pad_value=padding,
        )
        x_unfolded_cat = torch.cat(list(x_unfolded), dim=dim)

        x_unfolded_ref = x.unfold(dim, size, step)

        params = {
            'dim': dim,
            'size': size,
            'step': step,
            'batch_size': batch_size,
            # 'padding': padding,
            'ndim': ndim,
            'x.shape': x.shape,
            'n_samples_in_dim': n_samples_in_dim,
            'x_unfolded_cat.shape': x_unfolded_cat.shape,
            'x_unfolded_ref.shape': x_unfolded_ref.shape,
        }

        # if padding is None:
        assert torch.allclose(x_unfolded_cat, x_unfolded_ref), f"Test failed with params: {params}"
        # else:
        #     idx_nan = torch.isnan(x_unfolded_cat)
        #     ## Sum along all dims except dim
        #     idx_nan_dim = (idx_nan.sum(dim=tuple(set(range(ndim + 1)) - {dim})) > 0).type(torch.bool)
        #     x_unfolded_cat_narrow = torch.narrow(x_unfolded_cat, dim, 0, (~idx_nan_dim).sum())

        #     ## Shapes must match
        #     assert x_unfolded_cat_narrow.shape == x_unfolded_ref.shape, f"Shapes do not match with params: {params}"
        #     ## Values must match where padding is not applied
        #     assert torch.allclose(x_unfolded_cat_narrow, x_unfolded_ref), f"Test failed with params: {params}"
        #     ## Make sure last value from x is the same as max value from x_unfolded_cat, respecting the step size
        #     ### Can't figure it out
            