from typing import Dict, Type, Any, Union, Optional, Callable, Tuple, List
import gc
import functools

import numpy as np
import sklearn
import sklearn.linear_model
import torch


def ols(X,y, add_bias_terms=False):
    '''
    Ordinary Least Squares regression.
    This method works great and is fast under most conditions.
    It tends to do poorly when X.shape[1] is small or too big 
     (overfitting can occur). Using OLS + EV is probably
     better than the 'orthogonalize' function.
    y_rec = X @ theta + bias
    RH 2021

    Args:
        X (ndarray):
            array where columns are vectors to regress against 'y'
        y (ndarray):
            1-D or 2-D array
        add_bias_terms (bool):
            If True, add a column of ones to X.
            Theta will not contain the bias term.
    Returns:
        theta (ndarray):
            regression coefficents
        bias (ndarray):
            bias terms
    '''
    if X.ndim==1:
        X = X[:,None]

    if type(X) == np.ndarray:
        inv = np.linalg.inv
        eye = np.eye(X.shape[1] + 1*add_bias_terms, dtype=X.dtype)
        ones = functools.partial(np.ones, dtype=X.dtype)
        zeros = functools.partial(np.zeros, dtype=X.dtype)
        cat = np.concatenate
    elif type(X) == torch.Tensor:
        inv = torch.inverse
        eye = torch.eye(X.shape[1] + 1*add_bias_terms, device=X.device, dtype=X.dtype)
        ones = functools.partial(torch.ones, device=X.device, dtype=X.dtype)
        zeros = functools.partial(torch.zeros, device=X.device, dtype=X.dtype)
        cat = torch.cat

    if add_bias_terms:
        X = cat((X, ones((X.shape[0],1))), axis=1)

    theta = inv(X.T @ X) @ X.T @ y

    if add_bias_terms:
        bias = theta[-1]
        theta = theta[:-1]
    else:
        bias = zeros((y.shape[1],))

    return theta, bias


# @njit
def ridge(X, y, lam=1, add_bias_terms=False):
    '''
    Ridge regression.
    This method works great and is fast under most conditions.
    Lambda often ~1e5
    y_rec = X @ theta + bias
    RH 2021

    Args:
        X (ndarray):
            array where columns are vectors to regress against 'y'
        y (ndarray):
            1-D or 2-D array
        lam (float):
            regularization parameter
        add_bias_terms (bool):
            If True, add a column of ones to X.
            Theta will not contain the bias term.
    Returns:
        theta (ndarray):
            regression coefficents
        bias (ndarray):
            bias terms
    '''
    if X.ndim==1:
        X = X[:, None]

    if type(X) == np.ndarray:
        inv = np.linalg.inv
        eye = np.eye(X.shape[1] + 1*add_bias_terms, dtype=X.dtype)
        ones = functools.partial(np.ones, dtype=X.dtype)
        zeros = functools.partial(np.zeros, dtype=X.dtype)
        cat = np.concatenate
    elif type(X) == torch.Tensor:
        inv = torch.inverse
        eye = torch.eye(X.shape[1] + 1*add_bias_terms, device=X.device, dtype=X.dtype)
        ones = functools.partial(torch.ones, device=X.device, dtype=X.dtype)
        zeros = functools.partial(torch.zeros, device=X.device, dtype=X.dtype)
        cat = torch.cat

    if add_bias_terms:
        X = cat((X, ones((X.shape[0],1))), axis=1)

    theta = inv(X.T @ X + lam*eye) @ X.T @ y

    if add_bias_terms:
        bias = theta[-1]
        theta = theta[:-1]
    else:
        bias = zeros((y.shape[1],))

    return theta, bias


class LinearRegression_sk(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    My own version of a base linear regression estimator using sklearn's format.
    RH 2023
    """

    def get_backend_namespace(self, X, y=None):
        if isinstance(X, torch.Tensor):
            backend = 'torch'
            if y is not None:
                assert X.device == y.device, 'X and y must be on the same device'
            device = X.device
        elif isinstance(X, np.ndarray):
            backend = 'numpy'
        else:
            raise NotImplementedError
        
        if y is not None:
            assert isinstance(y, type(X)), 'X and y must be the same type'
        
        dtype = X.dtype

        ns = {}

        if backend == 'numpy':
            ns['inv'] = np.linalg.inv
            ns['eye'] = functools.partial(np.eye, dtype=dtype)
            ns['ones'] = functools.partial(np.ones, dtype=dtype)
            ns['zeros'] = functools.partial(np.zeros, dtype=dtype)
            ns['cat'] = np.concatenate
        elif backend == 'torch':
            ns['inv'] = torch.inverse
            ns['eye'] = functools.partial(torch.eye, device=device, dtype=dtype)
            ns['ones'] = functools.partial(torch.ones, device=device, dtype=dtype)
            ns['zeros'] = functools.partial(torch.zeros, device=device, dtype=dtype)
            ns['cat'] = torch.cat
        else:
            raise NotImplementedError
        
        return ns

    def predict(self, X):

        if X.ndim==1:
            X = X[:, None]

        return X @ self.coef_ + self.intercept_

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if isinstance(X, torch.Tensor):
            if sample_weight is not None:
                assert sample_weight.ndim == 1
                assert isinstance(sample_weight, torch.Tensor)
                weight = sample_weight[:, None]
            else:
                weight, sample_weight = 1.0, 1.0

            numerator = torch.sum(weight * (y - y_pred) ** 2, dim=0)
            denominator = (weight * (y - torch.mean(y * sample_weight, dim=0)) ** 2).sum(dim=0)

            output_scores = 1 - (numerator / denominator)
            output_scores[denominator == 0.0] = 0.0
            return torch.mean(output_scores) ## 'uniform_average' in sklearn
        
        elif isinstance(X, np.ndarray):
            return sklearn.metrics.r2_score(
                y_true=y, 
                y_pred=y_pred, 
                sample_weight=sample_weight, 
                multioutput='uniform_average',
            )
        
        else:
            raise NotImplementedError
        
    def to(self, device):
        self.coef_ = torch.as_tensor(self.coef_, device=device)
        self.intercept_ = torch.as_tensor(self.intercept_, device=device)
        return self
    def cpu(self):
        return self.to('cpu')
    def numpy(self):
        def check_and_convert(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            else:
                return x
        self.coef_, self.intercept_ = (check_and_convert(x) for x in (self.coef_, self.intercept_))
        return self


class Ridge(LinearRegression_sk):
    """
    Estimator class for Ridge regression.
    Based on the Cholesky's closed form solution.

    Args:
        alpha (float):
            Regularization parameter. Usually ~1e5
        fit_intercept (bool):
            If True, add a column of ones to X.
            Theta will not contain the bias term.
    """
    def __init__(
        self, 
        alpha=1, 
        fit_intercept=False,
        X_precompute=None,
        **kwargs,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        self.X_precompute = True if X_precompute is not None else False

        if X_precompute is not None:
            self.iXTXaeXT = self.prefit(X_precompute)
        else:
            self.iXTXaeXT = None

    def prefit(self, X):
        ns = self.get_backend_namespace(X=X)
        cat, eye, inv, ones, zeros = (ns[key] for key in ['cat', 'eye', 'inv', 'ones', 'zeros'])
        if self.fit_intercept:
            X = cat((X, ones((X.shape[0], 1))), axis=1)
        
        ## There is a bug in torch's inverse function that causes it to fail. This is a workaround that should be removed once the bug is fixed.
        if isinstance(X, torch.Tensor):
            if X.device.type != 'cpu':
                inv(torch.ones((1,1), device=X.device)) ## This line is a hack to force the GPU to initialize the CUDA context. Otherwise, the next line will fail.
        
        inv_XT_X_plus_alpha_eye_XT = inv(X.T @ X + self.alpha*eye(X.shape[1])) @ X.T
        return inv_XT_X_plus_alpha_eye_XT

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        ns = self.get_backend_namespace(X=X, y=y)
        zeros = ns['zeros']

        ## Regression algorithm
        inv_XT_X_plus_alpha_eye_XT = self.prefit(X) if self.iXTXaeXT is None else self.iXTXaeXT
        theta = inv_XT_X_plus_alpha_eye_XT @ y

        ### Extract bias terms
        if self.fit_intercept:
            self.intercept_ = theta[-1]
            self.coef_ = theta[:-1]
        else:
            self.intercept_ = zeros((y.shape[1],)) if y.ndim == 2 else 0
            self.coef_ = theta

        return self

class OLS(LinearRegression_sk):
    """
    Estimator class for OLS regression.
    Based on the closed form OLS solution.

    Args:
        fit_intercept (bool):
            If True, add a column of ones to X.
            Theta will not contain the bias term.
    """

    def __init__(
        self,
        fit_intercept=False,
        **kwargs,
    ):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        ns = self.get_backend_namespace(X=X, y=y)
        cat, eye, inv, ones, zeros = (ns[key] for key in ['cat', 'eye', 'inv', 'ones', 'zeros'])

        ## Regression algorithm
        ### Append bias terms
        if self.fit_intercept:
            X = cat((X, ones((X.shape[0], 1))), axis=1)

        ### OLS
        theta = inv(X.T @ X) @ X.T @ y

        ### Extract bias terms
        if self.fit_intercept:
            self.intercept_ = theta[-1]
            self.coef_ = theta[:-1]
        else:
            self.intercept_ = zeros((y.shape[1],))
            self.coef_ = theta

        return self
    
        
def LinearRegression_sweep(X_in,
                            y_in,
                            cv_idx,
                            alphas=0,
                            l1_ratios=0.5,
                            rolls=0,
                            method_package='sklearn',
                            method_model='LinearRegression',
                            compute_preds=False,
                            verbose=True,
                            theta_inPlace=None, 
                            intercept_inPlace=None, 
                            EV_train_inPlace=None, 
                            EV_test_inPlace=None,
                            preds_inPlace=None,
                            **model_params,
                            ):
    '''
    Performs linear regression over a sweep of input parameters.
    For every input parameter, each entry linearly increases the
    size of the parameter grid. The number of regression runs and
    shape of the outputs will be:
        n_y * n_alphas * n_l1_ratios * n_cv_idx * n_rolls.

    This function is similar to using sklearn's CV-GridSearch
    functions. Benefits to this function are that you can use
    a variety of different models, including CuML's functions.
    Also allows for in-place computation so you can stop it early
    and tune the model easier.
    RH 2021

    Args:
        X (ndarray): 
            Predictors. Shape: (n_samples, n_features).
        y (ndarray): 
            Outputs. Regression performed on one column at a time.
            Shape: (n_samples, n_outputs).
        alphas (ndarray): 
            Alpha parameters. Sets strength of regularization.
        l1_ratios (ndarray): 
            L1 to L2 ratio parameters. 0:L2 only, 1:L1 only
        cv_idx (list of lists of 1-D arrays): 
            Cross-validation indices. Each entry in the outer list is
            for a different run. Each entry in the outer list should 
            contain 2 lists, the first one containing the training 
            indices, and the second one containing the test indices.
            The function 'cross_validation.make_cv_indices' can make
            this variable easily.
        rolls (ndarray of ints):
            1-D array containing index shifts to apply.
        method_package (string):
            Can be any of the following:
                'cuml': 
                    Uses Rapids' cuml.linear_model library (requires GPU and CuPy)
                'sklearn':
                    Uses sklearn's sklearn.linear_model library
        method_model (string):
            Can be any of the following functions:
                'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'
            Note that not all models use the same parameters (eg LinearRegression
            does not use l1_ratio or alpha). So, if there are multiple
            values for an unused parameter, then the regression will be
            unnecessarily computed multiple times.
        compute_preds (bool):
            If True, will compute the predictions for the model.
             preds = theta @ X.T + intercept
            This is currently done on CPU so it is very slow.
        verbose (int 0-2):
            Preference of whether to print reconstruction scores.
            0: Print nothing
            1: Print for every y input
            2: Print at every regression step
        theta_inPlace (ndarray): (optional)
            Allows for stopping the function while it is running and 
            recovering the data up to that point. 
            Input a pre-allocated array with appropriate shape (see function below)
        intercept_inPlace (ndarray): (optional)
            Allows for stopping the function while it is running and 
            recovering the data up to that point. 
            Input a pre-allocated array with appropriate shape (see function below)
        EV_train_inPlace (ndarray): (optional)
            Allows for stopping the function while it is running and 
            recovering the data up to that point. 
            Input a pre-allocated array with appropriate shape (see function below)
        EV_test_inPlace (ndarray): (optional)
            Allows for stopping the function while it is running and 
            recovering the data up to that point. 
            Input a pre-allocated array with appropriate shape (see function below)
        **model_params (**kwargs):
            Pass into this function as **params.
            Different models require different parameters beyond alphas and 
            l1_ratios. You should input those parameters here.
            Check documentation for sklearn/CuML's documentation for what to
            put in here.

    Returns:
        theta (ndarray):
            Regression coefficients.
            Shape: (n_y, n_splits, n_rolls, n_alphas, n_l1Ratios, X.shape[1])
        intercept (ndarray):
            Intercept values
            Shape: (n_y, n_splits, n_rolls, n_alphas, n_l1Ratios)
        EV_test (ndarray):
            Pearson correlation scores between test data and reconstructed
            test data.
            Shape: (n_y, n_splits, n_rolls, n_alphas, n_l1Ratios)
        EV_train (ndarray):
            Pearson correlation scores between train data and reconstructed
            train data.
            Shape: (n_y, n_splits, n_rolls, n_alphas, n_l1Ratios)

    ============ DEMO ===============

    from basic_neural_processing_modules import cross_validation

    from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                    StratifiedKFold, GroupShuffleSplit,
                                    GroupKFold, StratifiedShuffleSplit)
    group_len = 60*2 * Fs # seconds * Fs
    n_splits = 10
    test_size = 0.3
    groups = np.arange(X.shape[0])//group_len
    n_groups = np.max(groups)
    cv = GroupShuffleSplit(n_splits, test_size=test_size)
    cv_idx = cross_validation.make_cv_indices(cv,
                                            groups,
                                            lw=10,
                                            plot_pref=True)

    OR:

    n_splits = 4
    cv = StratifiedKFold(n_splits, shuffle=False, random_state=None, )
    cv_idx = list(cv.split(X=scores.T, y=trial_types_aligned, groups=trial_types_aligned))


    from basic_neural_processing_modules.linear_regression import LinearRegression_sweep

    n_nonzero_rolls = 2
    min_roll = 60*10*Fs
    max_roll = X.shape[0] - min_roll
    rolls = np.concatenate(([0] , np.random.randint(min_roll, max_roll, n_nonzero_rolls)))
    n_rolls = n_nonzero_rolls + 1

    model_params_cuml_ElasticNet = {
            'fit_intercept': True,
            'normalize': False,
            'max_iter': 1000,
            'tol': 0.0001,
            'selection': 'cyclic',
    }

    l1_ratios = np.array([0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])
    alphas = np.array([0.0001, 0.001, 0.01, 0.1, 1])

    # prepare output variables for in-place computations
    n_y = y.shape[1]
    n_alphas = len(alphas)
    n_l1Ratios = len(l1_ratios)

    theta = np.ones((n_y , n_splits , n_rolls , n_alphas , n_l1Ratios, X.shape[1]))
    intercept = np.ones((n_y , n_splits , n_rolls , n_alphas , n_l1Ratios))
    EV_train  = np.zeros((n_y, n_splits , n_rolls , n_alphas , n_l1Ratios))
    EV_test   = np.zeros((n_y, n_splits , n_rolls , n_alphas , n_l1Ratios))

    # Run regression sweep
    theta, intercept, EV_train, EV_test, preds = LinearRegression_sweep(   X,
                                                                    y,
                                                                    cv_idx,
                                                                    alphas=alphas,
                                                                    l1_ratios=l1_ratios,
                                                                    rolls=rolls,
                                                                    method_package='cuml',
                                                                    method_model='ElasticNet',
                                                                    compute_preds=True,
                                                                    verbose=True,
                                                                    theta_inPlace=theta, 
                                                                    intercept_inPlace=intercept,
                                                                    EV_train_inPlace=EV_train,
                                                                    EV_test_inPlace=EV_test,
                                                                    **model_params_cuml_ElasticNet
                                                                    )
'''

    gc.collect()
    
    if method_package in ['cuml']:
        import cudf
        import cuml
        import cupy

        print('making cupy arrays')    
        X = cupy.asarray(X_in)
        y = cupy.asarray(y_in)

        # I thought this might help with the memory leak issue,
        # but it doesn't seem to help that much
        model_params['handle'] = cuml.Handle()
    else:
        X = np.array(X_in)
        y = np.array(y_in)
    

    n_alphas = len(alphas)
    n_l1Ratios = len(l1_ratios)

    n_splits = len(cv_idx)
    len_train = len(cv_idx[0][0])
    len_test = len(cv_idx[0][1])

    n_rolls = len(rolls)

    n_y = y.shape[1]

    if method_model in ['LogisticRegression']:
        n_classes = len(np.unique(y))
    else:
        n_classes = 1


    if theta_inPlace is not None:
        theta = theta_inPlace
    else:
        theta = np.ones((n_y , n_splits , n_rolls , n_alphas , n_l1Ratios, n_classes, X.shape[1]))
    
    if preds_inPlace is not None:
        preds = preds_inPlace
    else:
        preds = np.zeros((n_y, n_splits , n_rolls , n_alphas , n_l1Ratios, n_classes, X.shape[0]))


    if intercept_inPlace is not None:
        intercept = intercept_inPlace
    else:
        intercept = np.ones((n_y , n_splits , n_rolls , n_alphas , n_l1Ratios))
    
    if EV_train_inPlace is not None:
        EV_train = EV_train_inPlace
    else:
        EV_train  = np.zeros((n_y, n_splits , n_rolls , n_alphas , n_l1Ratios))
    
    if EV_test_inPlace is not None:
        EV_test = EV_test_inPlace
    else:
        EV_test   = np.zeros((n_y, n_splits , n_rolls , n_alphas , n_l1Ratios))

    for iter_factor in range(n_y):
        for iter_roll in range(n_rolls):
            y_iter = np.roll(y[:,iter_factor] , rolls[iter_roll])
            
            for iter_cv, (idx_train , idx_test) in enumerate(cv_idx):
                X_train = X[idx_train,:]
                y_train = y_iter[idx_train]
                X_test = X[idx_test,:]
                y_test = y_iter[idx_test]

                for iter_alpha , alpha in enumerate(alphas):
                    for iter_l1Ratio , l1_ratio in enumerate(l1_ratios):
                        if method_model in ['LinearRegression']:
                            clf = eval(f'{method_package}.linear_model.{method_model}')(**model_params)
                        
                        if method_model in  ['Lasso', 'Ridge']:
                            clf = eval(f'{method_package}.linear_model.{method_model}')(alpha=alpha,
                                                                                        **model_params)
                        
                        if method_model in ['ElasticNet']:
                            clf = eval(f'{method_package}.linear_model.{method_model}')(alpha=alpha,
                                                                                        l1_ratio=l1_ratio,
                                                                                        **model_params)

                        if method_model in ['LogisticRegression']:
                            clf = eval(f'{method_package}.linear_model.{method_model}')(C=1/alpha,
                                                                                        l1_ratio=l1_ratio if model_params['penalty'] == 'elasticnet' else None,
                                                                                        **model_params)

                        clf.fit(X_train , y_train )

                        if method_package=='cuml':
                            # theta[:, iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = cupy.asnumpy(clf.coef_)
                            # intercept[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = cupy.asnumpy(clf.intercept_)
                            theta[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = cupy.asnumpy(clf.coef_)
                            intercept[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = cupy.asnumpy(clf.intercept_)
                        else:
                            # theta[:, iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = clf.coef_
                            # intercept[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = clf.intercept_
                            theta[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = clf.coef_
                            intercept[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = clf.intercept_

                        EV_train_tmp = clf.score(X_train, y_train)
                        EV_test_tmp = clf.score(X_test, y_test)
                        
                        EV_train[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = EV_train_tmp
                        EV_test[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = EV_test_tmp       
                        
                        if compute_preds:
                            if method_package=='cuml':
                                preds[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = ((clf.coef_ @ X.T) + clf.intercept_).get()
                            else:
                                preds[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = ((clf.coef_ @ X.T) + clf.intercept_)
                        else:
                            preds = np.nan

                        if verbose==2:
                            print(f'y #: {iter_factor} , Roll iter: {iter_roll} , CV repeat #: {iter_cv} , alpha val: {alpha} , l1_ratio: {l1_ratio} , train R^2: {round(EV_train_tmp,3)}')
                            print(f'y #: {iter_factor} , Roll iter: {iter_roll} , CV repeat #: {iter_cv} , alpha val: {alpha} , l1_ratio: {l1_ratio} , test  R^2: {round(EV_test_tmp,3)} \n')
        if verbose==1:
            print(f'computed y #: {iter_factor}')

            # This is pretty important when using CuML on GPU. 
            # Move this line of code up or down within the for-loops to make it more or less frequent.
            # Generally move it as far up the for-loop hierarchy as possible without running out of memory
            gc.collect()

    return theta, intercept, EV_train, EV_test, preds