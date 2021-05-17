import gc
import numpy as np
import sklearn
import time

from . import similarity

def LinearRegression_sweep(X,
                            y,
                            alphas,
                            l1_ratios,
                            cv_idx,
                            rolls,
                            method_package='sklearn',
                            method_model='LinearRegression',
                            verbose=True,
                            **model_params,
                            ):
    '''
    Performs linear regression over a sweep of input parameters.
    For every input parameter, each entry linearly increases the
    size of the parameter grid. The number of regression runs and
    shape of the outputs will be:
        n_y * n_alphas * n_l1_ratios * n_cv_idx * n_rolls.

    Args:
        X (ndarray): 
            Predictors
        y (ndarray): 
            Outputs. Regression performed on one column at a time.
        alphas (ndarray): 
            Alpha parameters. Sets strength of regularization.
        l1_ratios (ndarray): 
            L1 to L2 ratio parameters.
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
        **model_params (**kwargs):
            Pass into this function as **params.
            Different models require different parameters beyond alphas and 
            l1_ratios. You may input those parameters here.
        verbose (bool):
            Preference of whether to print reconstruction scores

    Returns:
        theta (ndarray):
            Regression coefficients
        intercept (ndarray):
            Intercept values
        R_test (ndarray):
            Pearson correlation scores between test data and reconstructed
            test data.
        R_train (ndarray):
            Pearson correlation scores between train data and reconstructed
            train data.
    '''

    if method_package in ['cuml']:
        import cudf
        import cuml
        import cupy

        print('making cupy arrays')    
        X = cupy.asarray(X)
        y = cupy.asarray(y)
    

    n_alphas = len(alphas)
    n_l1Ratios = len(l1_ratios)

    n_splits = len(cv_idx)
    len_train = len(cv_idx[0][0])
    len_test = len(cv_idx[0][1])

    n_rolls = len(rolls)

    n_y = y.shape[1]


    theta = np.ones((X.shape[1] , n_y , n_splits , n_rolls , n_alphas , n_l1Ratios))
    intercept = np.ones((n_y , n_splits , n_rolls , n_alphas , n_l1Ratios))
    R_train = np.zeros((n_y, n_splits , n_rolls , n_alphas , n_l1Ratios))
    R_test = np.zeros((n_y,n_splits , n_rolls , n_alphas , n_l1Ratios))

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

                        clf.fit(X_train , y_train )

                        if method_package=='cuml':
                            theta[:, iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = cupy.asnumpy(clf.coef_)
                            intercept[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = cupy.asnumpy(clf.intercept_)
                        else:
                            theta[:, iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = clf.coef_
                            intercept[iter_factor, iter_cv, iter_roll, iter_alpha, iter_l1Ratio] = clf.intercept_

                        tic = time.time()
                        y_reconstructed = clf.predict(X)
                        y_train_reconstructed = clf.predict(X_train)
                        y_test_reconstructed  = clf.predict(X_test)

                        R_train = (np.corrcoef(y_train_reconstructed, y_train))[1,0]
                        R_test = (np.corrcoef(y_test_reconstructed, y_test))[1,0]
                            
                        if method_package=='cuml':
                            y_reconstructed = cupy.asnumpy(y_reconstructed)
                            y_train_reconstructed = cupy.asnumpy(y_train_reconstructed)
                            y_test_reconstructed  = cupy.asnumpy(y_test_reconstructed)
                            R_train = cupy.asnumpy(R_train)
                            R_test = cupy.asnumpy(R_test)


                        if verbose:
                            print(np.round(R_train**2,5))
                            print(np.round(clf.score(X_train,y_train),5))
                            print(np.round(similarity.pairwise_similarity(cupy.asnumpy(y_train) , cupy.asnumpy(y_train_reconstructed), 'R^2'),5))
                            # print(f'factor #: {iter_factor} , Roll iter: {iter_roll} , CV repeat #: {iter_cv} , alpha val: {alpha} , l1_ratio: {l1_ratio} , train  R: {round(R_train,3)} , train R^2: {round(R_train**2,3)}')
                            # print(f'factor #: {iter_factor} , Roll iter: {iter_roll} , CV repeat #: {iter_cv} , alpha val: {alpha} , l1_ratio: {l1_ratio} , test   R: {round(R_test,3)} , test  R^2: {round(R_test**2,3)} \n')
                        print(time.time()-tic)

            # gc.collect()

    return theta, intercept, R_train, R_test