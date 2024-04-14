import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import numpy as np
from hypothesis import given, strategies as st
import hypothesis

from ..linear_regression import OLS

# Basic functionality and accuracy
def test_basic_functionality():
    X, y = make_regression(n_samples=100, n_features=90, noise=0.1)
    model_sklearn = LinearRegression().fit(X, y)
    model_ols = OLS().fit(X, y)

    np.testing.assert_allclose(model_sklearn.coef_, model_ols.coef_, rtol=1e-5)
    np.testing.assert_allclose(model_sklearn.intercept_, model_ols.intercept_, rtol=1e-5)
    predictions_sklearn = model_sklearn.predict(X)
    predictions_ols = model_ols.predict(X)
    assert mean_squared_error(predictions_sklearn, predictions_ols) < 1e-5

# High dimensionality
def test_high_dimensionality():
    """
    OLS solution is expected to diverge when n_features >= n_samples. Expect a
    warning.
    """
    X, y = make_regression(n_samples=100, n_features=100, noise=0.1)
    with pytest.warns(UserWarning):
        model_ols = OLS().fit(X, y)

# Single feature test
def test_single_feature():
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    model_sklearn = LinearRegression().fit(X, y)
    model_ols = OLS().fit(X, y)

    np.testing.assert_allclose(model_sklearn.coef_, model_ols.coef_, rtol=1e-5)
    np.testing.assert_allclose(model_sklearn.intercept_, model_ols.intercept_, rtol=1e-5)

# Zero variation test
def test_zero_variation():
    X = np.ones((100, 1))  # no variation in X
    y = np.random.randn(100)
    with pytest.raises(Exception):
        model_ols = OLS().fit(X, y)

# Extreme values test
def test_extreme_values():
    X = np.array([
        [1e10, 1e11, 1e12], 
        [1e-10, 1e-11, 1e-12],
        [1e-10, 1e11, 1e12]
    ])
    y = np.array([1, 2, 3])
                    
    model_sklearn = LinearRegression().fit(X, y)
    model_ols = OLS().fit(X, y)

    np.testing.assert_allclose(model_sklearn.coef_, model_ols.coef_, atol=1e-5)
    np.testing.assert_allclose(model_sklearn.intercept_, model_ols.intercept_, atol=1e-5)

# Hypothesis test
@given(
    n=st.integers(min_value=2, max_value=100),
    m=st.integers(min_value=2, max_value=99),
    noise=st.floats(min_value=0.01, max_value=1.0),
)
@hypothesis.settings(max_examples=10)
def test_hypothesis(n, m, noise):
    X, y = make_regression(n_samples=n, n_features=m, noise=noise)
    model_sklearn = LinearRegression().fit(X, y)
    if m >= n:
        with pytest.warns(UserWarning):
            model_ols = OLS().fit(X, y)
    else:
        model_ols = OLS().fit(X, y)

        np.testing.assert_allclose(model_sklearn.coef_, model_ols.coef_, rtol=1e-5)
        np.testing.assert_allclose(model_sklearn.intercept_, model_ols.intercept_, rtol=1e-5)
        predictions_sklearn = model_sklearn.predict(X)
        predictions_ols = model_ols.predict(X)
        assert mean_squared_error(predictions_sklearn, predictions_ols) < 1e-5