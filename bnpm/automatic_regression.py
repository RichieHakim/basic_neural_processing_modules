from typing import Dict, Type, Any, Union, Optional, Callable, Tuple, List
import time
import warnings
import functools
import datetime

import numpy as np
import torch
import sklearn
import sklearn.model_selection
import optuna

from . import path_helpers
from . import optimization
from . import linear_regression


class Autotuner_BaseEstimator:
    """
    A class for automatic hyperparameter tuning and training of a regression
    model.
    RH 2023
    
    Attributes:
        model_class (Type[sklearn.base.BaseEstimator]):
            A Scikit-Learn estimator class.
            Must have: \n
                * Method: ``fit(X, y)``
                * Method: ``predict_proba(X)`` (for classifiers) or ``predict(X)`` (for continuous regressors) \n
        X (np.ndarray):
            Input data.
            Shape: *(n_samples, n_features)*
        y (np.ndarray):
            Output data.
            Shape: *(n_samples,)*
        params (Dict[str, Dict[str, Any]]):
            A dictionary of hyperparameters with their names, types, and bounds.
        cv (Type[sklearn.model_selection._split.BaseCrossValidator]):
            A Scikit-Learn cross-validator class.
            Must have: \n
                * Call signature: ``idx_train, idx_test = next(self.cv.split(self.X, self.y))``
        fn_loss (Callable):
            Function to compute the loss.
            Must have: \n
                * Call signature: ``loss, loss_train, loss_test =
                  fn_loss(y_train_pred, y_test_pred, y_true_train, y_true_test,
                  sample_weight_train, sample_weight_test)`` \n
        params_linked (Dict):
            Dictionary of parameters that are linked. \n
            Each item in the dictionary is of the form: \n
                * ``{'param_name': (param_name_linked, lambda x: f(x))}`` \n
            Where ``'param_name_linked'`` matches the name of a parameter in
            ``params_dynamic`` and ``lambda x: f(x)`` is a function that takes
            the value of that parameter and returns the value to be used by
            ``'param_name'``.
        groups (Optional[np.ndarray]):
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used if ``cv`` is a GroupShuffleSplit.
        n_jobs_optuna (int):
            Number of jobs for Optuna. Set to ``-1`` to use all cores.
            Note that some ``'solver'`` options are already parallelized (like
            ``'lbfgs'``). Set ``n_jobs_optuna`` to ``1`` for these solvers.
        n_startup (int):
            The number of startup trials for the optuna pruner and sampler.
        kwargs_convergence (Dict[str, Union[int, float]]):
            Convergence settings for the optimization. Includes: \n
                * ``'n_patience'`` (int): The number of trials to wait for
                  convergence before stopping the optimization.
                * ``'tol_frac'`` (float): The fractional tolerance for
                  convergence. After ``n_patience`` trials, the optimization
                  will stop if the loss has not improved by at least
                  ``tol_frac``.
                * ``'max_trials'`` (int): The maximum number of trials to run.
                * ``'max_duration'`` (int): The maximum duration of the
                  optimization in seconds. \n
        sample_weight (Optional[np.ndarray]):
            Weights for the samples, equal to ones_like(y) if None.
        catch_convergence_warnings (bool):
            If ``True``, ignore ConvergenceWarning during model fitting.
        verbose (bool):
            If ``True``, show progress bar and print running results.
        optuna_storage_url (Optional[str]):
            The path or URL to a SQL database. If ``None``, then the study is
            stored in memory. See optuna documentation for details. \n
        optuna_storage_name (Optional[str]):
            The name of the study in the storage. Should only be ``None`` if
            ``optuna_storage_url`` is ``None``, else it should be a string to a
            valid study name. 
        optuna_engine_kwargs (Optional[Dict[str, Any]]):
            Additional keyword arguments to pass to the SQLAlchemy engine.
        wandb_project (Optional[str]):
            The name of the Weights and Biases project to log to. If ``None``,
            then no logging is done. Assumes that a wandb.init() has already
            been called. \n


    Example:
        .. highlight:: python
        .. code-block:: python
    
        params = {
            'C':             {'type': 'real',        'kwargs': {'log': True, 'low': 1e-4, 'high': 1e4}},
            'penalty':       {'type': 'categorical', 'kwargs': {'choices': ['l1', 'l2']}},
        }
    """
    def __init__(
        self, 
        model_class: Type[sklearn.base.BaseEstimator], 
        X: Any, 
        y: Any, 
        params: Dict[str, Dict[str, Any]], 
        cv: Any, 
        fn_loss: Callable, 
        params_linked: Dict = {},
        groups: Optional[np.ndarray] = None,
        n_jobs_optuna: int = -1,
        n_startup: int = 15,
        kwargs_convergence = {
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
            'use_internal_timer': True,
        }, 
        n_repeats: int = 1,
        fn_reduce_repeats: Callable = np.nanmedian,
        sample_weight: Optional[Any] = None, 
        catch_convergence_warnings: bool = True,
        verbose=True,
        optuna_storage_url: Optional[str] = None,
        optuna_storage_name: Optional[str] = None,
        optuna_engine_kwargs: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
    ):
        """
        Initializes the AutotunerRegression with the given model class, parameters, data, and settings.
        """
        ## Set model variables
        self.X = X  ## shape (n_samples, n_features)
        self.y = y  ## shape (n_samples,)
        self.model_class = model_class  ## sklearn estimator class
        if isinstance(self.y, torch.Tensor):
            assert sample_weight is None, 'sample weights not supported for torch tensors.'
            self.sample_weight = None
        elif isinstance(self.y, np.ndarray):
            self.sample_weight = np.ones_like(self.y)
        self.cv = cv  ## sklearn cross-validator object with split method
        self.groups = groups  ## group labels for the samples used while splitting the dataset into train/test set

        ## Set optuna variables
        self.n_startup = n_startup
        self.params = params
        self.params_linked = params_linked
        self.n_jobs_optuna = n_jobs_optuna
        self.fn_loss = fn_loss
        self.catch_convergence_warnings = catch_convergence_warnings

        ## Set repeat variables
        self.n_repeats = n_repeats
        self.fn_reduce_repeats = fn_reduce_repeats


        self.kwargs_convergence = kwargs_convergence

        self.verbose = verbose
        self.optuna_storage_url = optuna_storage_url
        self.optuna_storage_name = optuna_storage_name
        self.optuna_engine_kwargs = optuna_engine_kwargs

        self.wandb_project = wandb_project

        # Initialize a convergence checker
        self.checker = optimization.Convergence_checker_optuna(verbose=verbose, **self.kwargs_convergence)

        # Initialize wandb callback
        if self.wandb_project is not None:
            import logging
            import sys
            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            config = {
                'model_class': self.model_class.func.__name__ if hasattr(self.model_class, 'func') else 'unknown',
                'params': self.params,
                'n_repeats': self.n_repeats,
                'n_startup': self.n_startup,
                'kwargs_convergence': self.kwargs_convergence,
                'n_jobs_optuna': self.n_jobs_optuna,
            }
            callback_wandb = optuna.integration.WeightsAndBiasesCallback(
                metric_name="loss",
                wandb_kwargs={
                    'project': self.wandb_project,
                    'config': config,
                },
                as_multirun=False,
            )
            ## Make a safe version of the callback by putting it in a try-except block
            def safe_callback(study, trial):
                try:
                    callback_wandb(study, trial)
                except Exception as e:
                    print(f'Error in wandb callback: {e}')
            self.callback_wandb = safe_callback
        else:
            self.callback_wandb = None

        # Set verbosity
        if int(self.verbose) <= 1:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif int(self.verbose) == 2:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        elif int(self.verbose) > 2:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

        # Initialize an Optuna study
        if self.optuna_storage_url is not None:
            storage = optuna.storages.RDBStorage(
                url=self.optuna_storage_url,
                engine_kwargs=self.optuna_engine_kwargs,
            )
        else:
            storage = None
        self.study = optuna.create_study(
            direction="minimize", 
            pruner=optuna.pruners.MedianPruner(n_startup_trials=self.n_startup), 
            sampler=optuna.samplers.TPESampler(n_startup_trials=self.n_startup),
            study_name='Autotuner' if self.optuna_storage_name is None else self.optuna_storage_name,
            storage=storage,
            load_if_exists=True,
        )
        
        # Initialize variables to store loss and best model
        self.loss_running_train = []
        self.loss_running_test = []
        self.loss_running = []
        self.loss_repeats_train = []
        self.loss_repeats_test = []
        self.loss_repeats = []
        self.params_running = []
        self.model_best = None
        self.loss_best = np.inf
        self.params_best = None

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Define the objective function for Optuna to optimize.

        Args:
            trial (optuna.trial.Trial): 
                An Optuna trial object.

        Returns:
            (float): 
                loss (float):
                    The score of the model.
        """
        # Make a lookup table for the suggest methods
        LUT_suggest = {
            'categorical': trial.suggest_categorical,
            'real': trial.suggest_float,
            'int': trial.suggest_int,
            'discrete_uniform': trial.suggest_discrete_uniform,
            'loguniform': trial.suggest_loguniform,
            'uniform': trial.suggest_uniform,
        }

        # Suggest hyperparameters using optuna
        kwargs_model = {}
        for name, config in self.params.items():
            kwargs_model[name] = LUT_suggest[config['type']](name, **config['kwargs'])
        ## Add linked parameters
        for param, (param_linked, fn) in self.params_linked.items():
            kwargs_model[param] = fn(kwargs_model[param_linked])

        # Train the model
        loss_train_all, loss_test_all, loss_all = [], [], []
        for ii in range(self.n_repeats):
            # Split the data
            idx_train, idx_test = next(self.cv.split(self.X, self.y, **({'groups': self.groups} if self.groups is not None else {}) ))
            X_train, y_train_true, X_test, y_test_true = self.X[idx_train], self.y[idx_train], self.X[idx_test], self.y[idx_test]
            if self.sample_weight is not None:
                sample_weight_train, sample_weight_test = self.sample_weight[idx_train], self.sample_weight[idx_test]
            else:
                sample_weight_train, sample_weight_test = None, None

            # Initialize the model with the suggested hyperparameters
            model = self.model_class(**kwargs_model)

            # Train the model
            ## Turn off ConvergenceWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning) if self.catch_convergence_warnings else None
                model.fit(X_train, y_train_true)

            # Transform the training data
            if hasattr(model, 'predict_proba'):
                y_train_pred = model.predict_proba(X_train)
                y_test_pred = model.predict_proba(X_test)
            elif hasattr(model, 'predict'):
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
            else:
                raise ValueError('Model must have either a predict_proba or predict method.')

            # Evaluate the model using the scoring method
            loss, loss_train, loss_test = self.fn_loss(
                y_train_pred=y_train_pred, 
                y_test_pred=y_test_pred,
                y_train_true=y_train_true,
                y_test_true=y_test_true,
                sample_weight_train=sample_weight_train,
                sample_weight_test=sample_weight_test,
            )

            loss_train_all.append(loss_train)
            loss_test_all.append(loss_test)
            loss_all.append(loss)

        # Reduce the loss
        if len(loss_train_all) == 1:
            loss_train, loss_test, loss = loss_train_all[0], loss_test_all[0], loss_all[0]        
        else:
            if isinstance(loss_train_all[0], (np.ndarray, np.generic, float, int)):
                stack = np.stack
            elif isinstance(loss_train_all[0], torch.Tensor):
                stack = torch.stack
            elif loss_train_all[0] is None:
                stack = lambda x: x
            else:
                raise ValueError(f'loss_train_all[0] must be either np.ndarray, np.generic, torch.Tensor, or None. Got {type(loss_train_all[0])}')
            loss_train, loss_test, loss = (self.fn_reduce_repeats(stack(l)) for l in [loss_train_all, loss_test_all, loss_all])


        # Save the running loss
        self.loss_running_train.append(loss_train)
        self.loss_running_test.append(loss_test)
        self.loss_running.append(loss)
        self.loss_repeats_train.append(loss_train_all)
        self.loss_repeats_test.append(loss_test_all)
        self.loss_repeats.append(loss_all)
        self.params_running.append(kwargs_model)

        # Update the bests
        loss = np.nan if loss is None else loss
        if loss < self.loss_best:
            self.loss_best = loss
            self.model_best = model
            self.params_best = kwargs_model

        # Log the trial results in optuna: losses
        trial.set_user_attr('loss', loss)
        trial.set_user_attr('loss_train', loss_train)
        trial.set_user_attr('loss_test', loss_test)

        return loss

    def fit(self) -> Union[sklearn.base.BaseEstimator, Optional[Dict[str, Any]]]:
        """
        Fit and tune the hyperparameters and train the model.

        Returns:
            (Union[sklearn.base.BaseEstimator, Optional[Dict[str, Any]]): 
                best_model (sklearn.base.BaseEstimator):
                    The best estimator obtained from hyperparameter tuning.
                best_params (Optional[Dict[str, Any]]):
                    The best parameters obtained from hyperparameter tuning.
        """
        # Optimize the study
        callbacks = [self.checker.check] + ([self.callback_wandb] if self.callback_wandb is not None else [])
        self.study.optimize(
            self._objective, 
            n_jobs=self.n_jobs_optuna, 
            callbacks=callbacks, 
            n_trials=self.kwargs_convergence['max_trials'],
            show_progress_bar=self.verbose,
        )

        # Print reason for convergence
        if self.verbose > 0:
            print(f'Convergence reason: {self.checker.reason_converged}')

        # Retrieve the best parameters and the best model
        self.best_params = self.study.best_params
        self.model_best = self.model_class(**self.best_params)
        
        # Train the model on the full data set
        self.model_best.fit(self.X, self.y)

        # End wandb run
        if self.wandb_project is not None:
            import wandb
            wandb.finish()

        return self.model_best, self.best_params
    
    def save_model(
        self,
        filepath: Optional[str]=None,
        allow_overwrite: bool=False,
    ):
        """
        Uses ONNX to save the best model as a binary file.

        Args:
            filepath (str): 
                The path to save the model to.
                If None, then the model will not be saved.
            allow_overwrite (bool):
                Whether to allow overwriting of existing files.

        Returns:
            (onnx.ModelProto):
                The ONNX model.
        """
        import datetime
        try:
            import onnx
            import skl2onnx
        except ImportError as e:
            raise ImportError(f'You need to (pip) install onnx and skl2onnx to use this method. {e}')
        
        ## Make sure we have what we need
        assert self.model_best is not None, 'You need to fit the model first.'

        # Convert the model to ONNX format
        ## Prepare initial types
        initial_types = skl2onnx.common.data_types.guess_data_type(self.X)[0][1]
        initial_types.shape = tuple([None] + list(initial_types.shape[1:]))
        initial_types = [('input', initial_types)]
        ### Convert the model
        model_onnx = skl2onnx.convert_sklearn(
            model=self.model_best,
            name=str(self.model_best.__class__),
            initial_types=initial_types,
            doc_string=f"Created by Autotuner. Saved on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        )
        
        # Save the model
        if filepath is not None:
            path_helpers.prepare_filepath_for_saving(filepath=filepath, mkdir=True, allow_overwrite=allow_overwrite)
            onnx.save(
                proto=model_onnx,
                f=filepath,
            )
        return model_onnx
    
    def plot_param_curve(
        self,
        param='alpha',
        xscale='linear',
        yscale='linear',
        jitter=0.01,
    ):
        """
        Makes a scatter plot of a selected values vs loss values.

        Args:
            param (str):
                The parameter to plot.
            xscale (str):
                The scale of the x-axis. Either ``'linear'`` or ``'log'``.
        """
        def prep(x):
            """to numpy"""
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x).detach().cpu().numpy()
            else:
                return np.stack(x)

        import matplotlib.pyplot as plt
        results = {
            param: prep([t[param] for t in self.params_running]), 
            'value': prep(self.loss_running), 
            'loss_train': prep(self.loss_running_train), 
            'loss_test': prep(self.loss_running_test),
        }
       
        fig = plt.figure()
        isnumeric = np.issubdtype(results[param].dtype, np.number)
        n_vals = len(results[param])
        range_vals = np.ptp(results['value'])
        jitter_scaling = np.ptp(results[param]) if isnumeric else n_vals
        jitter_vals = np.random.uniform(-jitter, jitter, size=n_vals) * jitter * jitter_scaling
        xaxis = results[param] if isnumeric else np.unique(results[param], return_inverse=True)[1]
        xaxis_unique, xaxis_counts = np.unique(xaxis, return_counts=True)
        results_means = np.array([(u, np.mean(results['value'][xaxis == u])) for u in xaxis_unique])
        scatter = functools.partial(
            plt.scatter, 
            x=xaxis + jitter_vals,
            alpha=0.3, 
            s=10
        )
        scatter(y=results['loss_train'], color='blue')
        scatter(y=results['loss_test'], color='orange')
        scatter(y=results['value'], color='k')
        plt.scatter(results_means[:,0], results_means[:,1], s=70, color='k') if any(xaxis_counts > 1) else None
        plt.scatter(x=xaxis[np.argmin(results['value'])], y=np.min(results['value']), color='r', s=100, alpha=1)

        plt.xlabel(param)
        plt.ylabel('loss')
        if isnumeric:
            plt.xscale(xscale)
            plt.yscale(yscale)
        else:
            plt.xticks(xaxis, results[param], rotation=90)
            plt.xlim(-1, len(xaxis_unique))
            plt.ylim(np.min(results['value']) + (range_vals * -0.1), np.max(results['value']) + (range_vals * 0.1))
        plt.legend([
            'loss_train',
            'loss_test',
            'loss_withPenalty',
            'loss_withPenalty_mean',
            'best model',
        ])

        return fig

    def to(self, device):
        if isinstance(self.model_best, linear_regression.LinearRegression_sk):
            self.model_best.to(device)
        return self
    def cpu(self):
        return self.to('cpu')
    def numpy(self):
        if isinstance(self.model_best, linear_regression.LinearRegression_sk):
            self.model_best.numpy()
        return self

class LossFunction_CrossEntropy_CV():
    """
    Calculates the cross-entropy loss of a classifier using cross-validation. 
    RH 2023

    Args:
        penalty_testTrainRatio (float): 
            The amount of penalty for the test loss to the train loss. 
            Penalty is applied with formula: 
            ``loss = loss_test_or_train * ((loss_test / loss_train) ** penalty_testTrainRatio)``.
        labels (Optional[Union[List, np.ndarray]]): 
            A list or ndarray of labels. 
            Shape: *(n_samples,)*.
        test_or_train (str): 
            A string indicating whether to apply the penalty to the test or
            train loss.
            It should be either ``'test'`` or ``'train'``. 
    """
    def __init__(
        self,
        penalty_testTrainRatio: float = 1.0,
        labels: Optional[Union[List, np.ndarray]] = None,
        test_or_train: str = 'test',
    ) -> None:
        """
        Initializes the LossFunctionCrossEntropyCV with the given penalty, labels, and test_or_train setting.
        """
        self.labels = labels
        self.penalty_testTrainRatio = penalty_testTrainRatio
        ## Set the penalty function
        if test_or_train == 'test':
            self.fn_penalty_testTrainRatio = lambda test, train: test * ((test  / train) ** self.penalty_testTrainRatio)
        elif test_or_train == 'train':
            self.fn_penalty_testTrainRatio = lambda test, train: train * ((train / test) ** self.penalty_testTrainRatio)
        else:
            raise ValueError('test_or_train must be either "test" or "train".')

    
    def __call__(
        self,
        y_train_pred: np.ndarray, 
        y_test_pred: np.ndarray,
        y_train_true: np.ndarray,
        y_test_true: np.ndarray,
        sample_weight_train: Optional[List[float]] = None,
        sample_weight_test: Optional[List[float]] = None,
    ):
        """
        Calculates the cross-entropy loss using cross-validation.

        Args:
            y_train_pred (np.ndarray): 
                Predicted output probabilities for the training set. (shape:
                *(n_samples,)*)
            y_test_pred (np.ndarray): 
                Predicted output probabilities for the test set. (shape:
                *(n_samples,)*)
            y_train_true (np.ndarray): 
                True output probabilities for the training set. (shape:
                *(n_samples,)*)
            y_test_true (np.ndarray): 
                True output probabilities for the test set. (shape:
                *(n_samples,)*)
            sample_weight_train (Optional[List[float]]): 
                Weights assigned to each training sample. 
            sample_weight_test (Optional[List[float]]): 
                Weights assigned to each test sample.

        Returns:
            (tuple): tuple containing:
                loss (float): 
                    The calculated loss after applying the penalty.
                loss_train (float): 
                    The cross-entropy loss of the training set.
                loss_test (float): 
                    The cross-entropy loss of the test set.
        """
        # Calculate the cross-entropy loss using cross-validation.
        from sklearn.metrics import log_loss
        loss_train = log_loss(y_train_true, y_train_pred, sample_weight=sample_weight_train, labels=self.labels)
        loss_test =  log_loss(y_test_true,  y_test_pred,  sample_weight=sample_weight_test,  labels=self.labels)
        loss = self.fn_penalty_testTrainRatio(loss_test, loss_train)

        return loss, loss_train, loss_test


class LossFunction_MSE_CV():
    """
    Calculates the mean-squared error loss of a model using
    cross-validation. 
    RH 2023

    Args:
        penalty_testTrainRatio (float): 
            The amount of penalty for the test loss to the train loss. 
            Penalty is applied with formula: 
            ``loss = loss_test_or_train * ((loss_test / loss_train) ** penalty_testTrainRatio)``.
        test_or_train (str): 
            A string indicating whether to apply the penalty to the test or
            train loss.
            It should be either ``'test'`` or ``'train'``. 
    """
    def __init__(
        self,
        penalty_testTrainRatio: float = 1.0,
        test_or_train: str = 'test',
    ) -> None:
        """
        Initializes the class with the given penalty, and test_or_train setting.
        """
        self.penalty_testTrainRatio = penalty_testTrainRatio
        ## Set the penalty function
        if test_or_train == 'test':
            self.fn_penalty_testTrainRatio = lambda test, train: test * ((test  / train) ** self.penalty_testTrainRatio)
        elif test_or_train == 'train':
            self.fn_penalty_testTrainRatio = lambda test, train: train * ((train / test) ** self.penalty_testTrainRatio)
        else:
            raise ValueError('test_or_train must be either "test" or "train".')

    
    def __call__(
        self,
        y_train_pred: np.ndarray, 
        y_test_pred: np.ndarray,
        y_train_true: np.ndarray,
        y_test_true: np.ndarray,
        sample_weight_train: Optional[List[float]] = None,
        sample_weight_test: Optional[List[float]] = None,
    ):
        """
        Calculates the mean-squared error loss using cross-validation.

        Args:
            y_train_pred (np.ndarray): 
                Predicted output data for the training set. (shape:
                *(n_samples,)*)
            y_test_pred (np.ndarray): 
                Predicted output data for the test set. (shape: *(n_samples,)*)
            y_train_true (np.ndarray): 
                True output data for the training set. (shape: *(n_samples,)*)
            y_test_true (np.ndarray): 
                True output data for the test set. (shape: *(n_samples,)*)
            sample_weight_train (Optional[List[float]]): 
                Weights assigned to each training sample. 
            sample_weight_test (Optional[List[float]]): 
                Weights assigned to each test sample.

        Returns:
            (tuple): tuple containing:
                loss (float): 
                    The calculated loss after applying the penalty.
                loss_train (float): 
                    The mean-squared error loss of the training set.
                loss_test (float): 
                    The mean-squared error loss of the test set.
        """
        # Normalize the y values such that the variance of the true values is 1.
        y_train_pred = y_train_pred / y_train_true.std()
        y_test_pred = y_test_pred / y_test_true.std()
        y_train_true = y_train_true / y_train_true.std()
        y_test_true = y_test_true / y_test_true.std()

        # Calculate the mean-squared error loss using cross-validation.
        if isinstance(y_train_pred, np.ndarray):
            from sklearn.metrics import mean_squared_error
            loss_train = mean_squared_error(y_train_true, y_train_pred, sample_weight=sample_weight_train)
            loss_test =  mean_squared_error(y_test_true,  y_test_pred,  sample_weight=sample_weight_test)
            loss = self.fn_penalty_testTrainRatio(loss_test, loss_train)
        elif isinstance(y_train_pred, torch.Tensor):
            from torch.nn.functional import mse_loss
            assert (sample_weight_test is None) and (sample_weight_train is None), 'sample weights not supported for torch tensors.'
            loss_train = mse_loss(y_train_pred, y_train_true, reduction='mean').item()
            loss_test =  mse_loss(y_test_pred,  y_test_true,  reduction='mean').item()
            loss = self.fn_penalty_testTrainRatio(loss_test, loss_train)
        else:
            raise ValueError(f'Expected y_train_pred to be of type np.ndarray or torch.Tensor, but got type {type(y_train_pred)}.')
        
        return loss, loss_train, loss_test


class LossFunction_EVR_CV():
    """
    Calculates the explained variance ratio loss of a model using
    cross-validation. Output loss is 1 - EVR.
    RH 2024

    Args:
        penalty_testTrainRatio (float): 
            The amount of penalty for the test loss to the train loss. 
            Penalty is applied with formula: 
            ``loss = loss_test_or_train * ((loss_test / loss_train) ** penalty_testTrainRatio)``.
        test_or_train (str): 
            A string indicating whether to apply the penalty to the test or
            train loss.
            It should be either ``'test'`` or ``'train'``. 
    """
    def __init__(
        self,
        penalty_testTrainRatio: float = 1.0,
        test_or_train: str = 'test',
    ) -> None:
        """
        Initializes the class with the given penalty, and test_or_train setting.
        """
        self.penalty_testTrainRatio = penalty_testTrainRatio
        ## Set the penalty function
        if test_or_train == 'test':
            self.fn_penalty_testTrainRatio = lambda test, train: test * ((test  / train) ** self.penalty_testTrainRatio)
        elif test_or_train == 'train':
            self.fn_penalty_testTrainRatio = lambda test, train: train * ((train / test) ** self.penalty_testTrainRatio)
        else:
            raise ValueError('test_or_train must be either "test" or "train".')

    def explainable_variance_ratio(self, v1, v2, sample_weight=None):
        if isinstance(v1, torch.Tensor):
            v1 = v1 - torch.nanmean(v1, dim=0)
            v2 = v2 - torch.nanmean(v2, dim=0)

            v1_orth = v1 - (torch.nansum(v1 * v2, dim=0) / torch.nansum(v2 * v2, dim=0) )*v2

            v1_var = torch.var(v1, dim=0)
            EVR = 1 - (torch.var(v1_orth, dim=0) / v1_var)
            if sample_weight is not None:
                EVR = EVR * (sample_weight / torch.mean(sample_weight))

            EVR_total_weighted = torch.nansum(v1_var * EVR) / torch.sum(v1_var)
            return EVR_total_weighted
        elif isinstance(v1, np.ndarray):
            v1 = v1 - np.nanmean(v1, axis=0)
            v2 = v2 - np.nanmean(v2, axis=0)

            v1_orth = v1 - (np.nansum(v1 * v2, axis=0) / np.nansum(v2 * v2, axis=0) )*v2

            v1_var = np.var(v1, axis=0)
            EVR = 1 - (np.var(v1_orth, axis=0) / v1_var)
            if sample_weight is not None:
                EVR = EVR * (sample_weight / np.mean(sample_weight))

            EVR_total_weighted = np.nansum(v1_var * EVR) / np.sum(v1_var)
            return EVR_total_weighted
        else:
            raise ValueError(f'Expected v1 to be of type np.ndarray or torch.Tensor, but got type {type(v1)}.')

    def __call__(
        self,
        y_train_pred: np.ndarray, 
        y_test_pred: np.ndarray,
        y_train_true: np.ndarray,
        y_test_true: np.ndarray,
        sample_weight_train: Optional[List[float]] = None,
        sample_weight_test: Optional[List[float]] = None,
    ):
        """
        Calculates the explained variance ratio loss using cross-validation.
        Args:
            y_train_pred (np.ndarray): 
                Predicted output data for the training set. (shape:
                *(n_samples,)*)
            y_test_pred (np.ndarray): 
                Predicted output data for the test set. (shape: *(n_samples,)*)
            y_train_true (np.ndarray): 
                True output data for the training set. (shape: *(n_samples,)*)
            y_test_true (np.ndarray): 
                True output data for the test set. (shape: *(n_samples,)*)
            sample_weight_train (Optional[List[float]]): 
                Weights assigned to each training sample. 
            sample_weight_test (Optional[List[float]]): 
                Weights assigned to each test sample.

        Returns:
            (tuple): tuple containing:
                loss (float): 
                    The calculated loss after applying the penalty.
                loss_train (float): 
                    1 - explained variance ratio loss of the training set.
                loss_test (float): 
                    1 - explained variance ratio loss of the test set.
        """
        # Calculate 1 - explained variance ratio loss using cross-validation.
        
        loss_train = 1 - self.explainable_variance_ratio(y_train_true, y_train_pred, sample_weight_train)
        loss_test =  1 - self.explainable_variance_ratio(y_test_true, y_test_pred, sample_weight_test)
        loss = self.fn_penalty_testTrainRatio(loss_test, loss_train)
        
        return loss, loss_train, loss_test    


class Auto_Classifier(Autotuner_BaseEstimator):
    """
    Implements automatic hyperparameter tuning for a user defined classification model.
    RH 2023

    Args:
        X (np.ndarray):
            Training data. (shape: *(n_samples, n_features)*)
        y (np.ndarray):
            Target variable. (shape: *(n_samples, n_features)*)
        Model (class):
            A class to be used as the model. 
            Must have: \n
                * Method: ``fit(X, y)``
                * Method: ``predict_proba(X)`` (for classifiers) \n            
        params (Dict):
            Dictionary of model class initialization parameters. 
            For each item in the dictionary if item is: \n
                * ``list``: The parameter is tuned. If the values are numbers,
                  then the list wil be the bounds [low, high] to search over. If
                  the values are strings, then the list will be the categorical
                  values to search over. Note that the type of the parameter
                  (int, float, str) will be used to determine the type of the
                  optuna suggest.
                * **not** a ``list``: The parameter is fixed to the given value. \n
        params_log (Dict):
            Dictionary of parameters that vary logarithmically: \n
                * If True: parameter is set to logaritmically varying. \n
                * If False: parameter is set to linearly varying. \n
                * If not specified: parameter is set to logaritmically varying.
                  \n
        params_linked (Dict):
            Dictionary of parameters that are linked. \n
            Each item in the dictionary is of the form: \n
                * ``{'param_name': (param_name_linked, lambda x: f(x))}`` \n
            Where ``'param_name_linked'`` matches the name of a parameter in
            ``params_dynamic`` and ``lambda x: f(x)`` is a function that takes
            the value of that parameter and returns the value to be used by
            ``'param_name'``.
        n_startup (int):
            Number of startup trials.
        kwargs_convergence (Dict[str, Union[int, float]]):
            Convergence settings for the optimization. Includes: \n
                * ``'n_patience'`` (int): The number of trials to wait for
                  convergence before stopping the optimization.
                * ``'tol_frac'`` (float): The fractional tolerance for
                  convergence. After ``n_patience`` trials, the optimization
                  will stop if the loss has not improved by at least
                  ``tol_frac``.
                * ``'max_trials'`` (int): The maximum number of trials to run.
                * ``'max_duration'`` (int): The maximum duration of the
                  optimization in seconds. \n
        n_repeats (int):
            Number of repetitions for each trial.
        fn_reduce_repeats (Callable):
            Function to aggregate results from repetitions. (Default:
            `np.median`)
        n_jobs_optuna (int):
            Number of jobs for Optuna. Set to ``-1`` to use all cores.
            Note that some ``'solver'`` options are already parallelized (like
            ``'lbfgs'``). Set ``n_jobs_optuna`` to ``1`` for these solvers.
        penalty_testTrainRatio (float):
            Penalty ratio for test and train. 
        sample_weight (Optional[List[float]]):
            Sample weights. Applied only during cross-validation train/test
            accuracy, not model training. \n
            See `LogisticRegression
            <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
            for more details.
        cv (Optional[sklearn.model_selection._split.BaseCrossValidator]):
            An Scikit-Learn cross-validator class.
            If not ``None``, then must have: \n
                * Call signature: ``idx_train, idx_test =
                  next(self.cv.split(self.X, self.y))`` \n
            If ``None``, then a StratifiedShuffleSplit cross-validator will be
            used.
        groups (Optional[np.ndarray]):
            Group labels for the samples used while splitting the dataset into
            train/test set. \n
        test_size (float):
            Test set ratio.
            Only used if ``cv`` is ``None``.
        loss_function (Optional[Callable]):
            A custom loss function. If ``None``, then the default loss function
            is used (CrossEntropy_CV).
        verbose (bool):
            Whether to print progress messages.
        optuna_storage_url (Optional[str]):
            The path or URL to a SQL database. If ``None``, then the study is
            stored in memory. See optuna documentation for details. \n
        optuna_storage_name (Optional[str]):
            The name of the study in the storage. Should only be ``None`` if
            ``optuna_storage_url`` is ``None``, else it should be a string to a
            valid study name. 
        optuna_engine_kwargs (Optional[Dict[str, Any]]):
            Additional keyword arguments to pass to the SQLAlchemy engine.
        wandb_project (Optional[str]):
            The name of the Weights and Biases project to log to. If ``None``,
            then no logging is done. Assumes that a wandb.init() has already
            been called. \n

    Demo:
        .. code-block:: python

            ## Initialize sklearn LogisticRegression class with NO TUNING. All
            parameters are fixed.
            autoclassifier = Auto_Classifier(
                X, 
                y, 
                params_dynamic={},
                params_static={
                    'C': 1e-14,
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                },
            )

            ## Initialize with TUNING 'C', 'penalty', and 'l1_ratio'. 'solver' is fixed.
            autoclassifier = Auto_Classifier(
                X,
                y,
                params_dynamic={
                    'C': [1e-14, 1e3],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'l1_ratio': [0.0, 1.0],
                },
                params_static={
                    'solver': 'lbfgs',
                },
                params_log={
                    'C': True,
                    'l1_ratio': False,
                },
            )
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Model: Callable,
        params_dynamic: Dict = {
            'alpha': [1e-1, 1e5],
        },
        params_static: Dict = {
            'fit_intercept': True,
            'max_iter': 1000,
            'tol': 0.0001,
            'solver': 'auto',
        },
        params_log: Dict = {
            'C': True,
            'max_iter': False,
        },
        params_linked: Dict = {},
        n_startup: int = 15,
        kwargs_convergence: Dict = {
            'n_patience': 50,
            'tol_frac': 0.05,
            'max_trials': 150,
            'max_duration': 60*10,
        }, 
        n_repeats: int = 1,
        fn_reduce_repeats: Callable = np.median,
        n_jobs_optuna: int = 1,
        penalty_testTrainRatio: float = 1.0,
        sample_weight: Optional[List[float]] = None,
        cv: Optional[sklearn.model_selection._split.BaseCrossValidator] = None,
        groups: Optional[np.ndarray] = None,
        test_size: float = 0.3,
        loss_function: Optional[Callable] = None,
        verbose: bool = True,
        optuna_storage_url: Optional[str] = None,
        optuna_storage_name: Optional[str] = None,
        optuna_engine_kwargs: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
    ) -> None:
        """
        Initializes Auto_Classifier with the given parameters and data.
        """
        assert y.ndim == 1, f'y.ndim must be 1. Found {y.ndim}.'

        ## Prepare sample weights
        self.sample_weight = sklearn.utils.class_weight.compute_sample_weight(
            class_weight=sample_weight, 
            y=y,
        )

        ## Prepare the loss function
        if loss_function is None:
            self.fn_loss = LossFunction_CrossEntropy_CV(
                penalty_testTrainRatio=penalty_testTrainRatio,
                labels=y,
                test_or_train='test',
            )
        else:
            self.fn_loss = loss_function

        ## Prepare the cross-validation
        self.cv = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
        ) if cv is None else cv

        ## Prepare dynamic parameters for optuna
        params = _infer_params_types(params_dynamic, params_log)

        ## Prepare the model class
        self.model_class = functools.partial(
            Model,
            **params_static,
        )

        ## Initialize the Autotuner superclass
        super().__init__(
            model_class=self.model_class,
            X=X,
            y=y,
            params=params,
            params_linked=params_linked,
            n_startup=n_startup,
            kwargs_convergence=kwargs_convergence,
            n_repeats=n_repeats,
            fn_reduce_repeats=fn_reduce_repeats,
            n_jobs_optuna=n_jobs_optuna,
            cv=self.cv,
            groups=groups,
            fn_loss=self.fn_loss,
            catch_convergence_warnings=True,
            verbose=verbose,
            optuna_storage_url=optuna_storage_url,
            optuna_storage_name=optuna_storage_name,
            optuna_engine_kwargs=optuna_engine_kwargs,
            wandb_project=wandb_project,
        )

    def evaluate_model(
        self, 
        model: Optional[object] = None,
        X: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[List[float]] = None,
    ) -> Tuple[float, np.array]:
        """
        Evaluates the given model on the given data. Makes label predictions,
        then computes the accuracy and confusion matrix.

        Args:
            model (Optional[object]):
                The model to evaluate.
                If None, then self.model_best is used.
            X (np.ndarray):
                The data to evaluate on.
                If None, then self.X is used.
            y (np.ndarray):
                The labels to evaluate on.
                If None, then self.y is used.
            sample_weight (List[float]):
                The sample weights to evaluate on.
                If None, then self.sample_weight is used.

        Returns:
            (tuple): Tuple containing:
                accuracy (float):
                    The accuracy of the model on the given data.
                confusion_matrix (np.array):
                    The confusion matrix of the model on the given data.
        """
        model = self.model_best if model is None else model
        X = self.X if X is None else X
        y = self.y if y is None else y
        sample_weight = self.sample_weight if sample_weight is None else sample_weight

        y_pred = model.predict(X)
        
        accuracy = sklearn.metrics.accuracy_score(
            y_true=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            normalize=True,
        )

        confusion_matrix = sklearn.metrics.confusion_matrix(
            y_true=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            labels=self.classes,
            normalize='true',
        ).T

        return accuracy, confusion_matrix
    

class Auto_Regression(Autotuner_BaseEstimator):
    """
    Implements automatic hyperparameter tuning for a user defined classification model.
    RH 2024

    Args:
        X (np.ndarray):
            Training data. (shape: *(n_samples, n_features)*)
        y (np.ndarray):
            Target variable. (shape: *(n_samples, n_features)*)
        Model (class):
            A class to be used as the model. 
            Must have: \n
                * Method: ``fit(X, y)``
                * Method: ``predict(X)`` \n
        params (Dict):
            Dictionary of model class initialization parameters. 
            For each item in the dictionary if item is: \n
                * ``list``: The parameter is tuned. If the values are numbers,
                  then the list wil be the bounds [low, high] to search over. If
                  the values are strings, then the list will be the categorical
                  values to search over. Note that the type of the parameter
                  (int, float, str) will be used to determine the type of the
                  optuna suggest.
                * **not** a ``list``: The parameter is fixed to the given value. \n
        params_log (Dict):
            Dictionary of parameters that vary logarithmically: \n
                * If True: parameter is set to logaritmically varying. \n
                * If False: parameter is set to linearly varying. \n
                * If not specified: parameter is set to logaritmically varying.
                  \n
        params_linked (Dict):
            Dictionary of parameters that are linked. \n
            Each item in the dictionary is of the form: \n
                * ``{'param_name': (param_name_linked, lambda x: f(x))}`` \n
            Where ``'param_name_linked'`` matches the name of a parameter in
            ``params_dynamic`` and ``lambda x: f(x)`` is a function that takes
            the value of that parameter and returns the value to be used by
            ``'param_name'``.
        n_startup (int):
            Number of startup trials.
        kwargs_convergence (Dict[str, Union[int, float]]):
            Convergence settings for the optimization. Includes: \n
                * ``'n_patience'`` (int): The number of trials to wait for
                  convergence before stopping the optimization.
                * ``'tol_frac'`` (float): The fractional tolerance for
                  convergence. After ``n_patience`` trials, the optimization
                  will stop if the loss has not improved by at least
                  ``tol_frac``.
                * ``'max_trials'`` (int): The maximum number of trials to run.
                * ``'max_duration'`` (int): The maximum duration of the
                  optimization in seconds. \n
        n_repeats (int):
            Number of repetitions for each trial.
        fn_reduce_repeats (Callable):
            Function to aggregate results from repetitions. (Default:
            `np.median`)
        n_jobs_optuna (int):
            Number of jobs for Optuna. Set to ``-1`` to use all cores.
            Note that some ``'solver'`` options are already parallelized (like
            ``'lbfgs'``). Set ``n_jobs_optuna`` to ``1`` for these solvers.
        penalty_testTrainRatio (float):
            Penalty ratio for test and train. 
        sample_weight (Optional[List[float]]):
            Sample weights. Applied only during cross-validation train/test
            accuracy, not model training. \n
            See `RidgeRegression
            <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeRegression.html>`_
            for more details.
        cv (Optional[sklearn.model_selection._split.BaseCrossValidator]):
            An Scikit-Learn cross-validator class.
            If not ``None``, then must have: \n
                * Call signature: ``idx_train, idx_test =
                  next(self.cv.split(self.X, self.y))`` \n
            If ``None``, then a ShuffleSplit cross-validator will be
            used.
        groups (Optional[np.ndarray]):
            Group labels for the samples used while splitting the dataset into
            train/test set. \n
        test_size (float):
            Test set ratio.
            Only used if ``cv`` is ``None``.
        loss_function (Optional[Callable]):
            A custom loss function. If ``None``, then the default loss function
            is used (LossFunction_MSE_CV).\n
        verbose (bool):
            Whether to print progress messages.
        optuna_storage_url (Optional[str]):
            The path or URL to a SQL database. If ``None``, then the study is
            stored in memory. See optuna documentation for details. \n
        optuna_storage_name (Optional[str]):
            The name of the study in the storage. Should only be ``None`` if
            ``optuna_storage_url`` is ``None``, else it should be a string to a
            valid study name. 
        optuna_engine_kwargs (Optional[Dict[str, Any]]):
            Additional keyword arguments to pass to the SQLAlchemy engine.
        wandb_project (Optional[str]):
            The name of the Weights and Biases project to log to. If ``None``,
            then no logging is done. Assumes that a wandb.init() has already
            been called. \n


    Demo:
        .. code-block:: python

            ## Initialize sklearn RidgeRegression class with NO TUNING. All
            parameters are fixed.
            autoregressor = Auto_Regression(
                X,
                y,
                Model=sklearn.linear_model.Ridge,
                params_dynamic={},
                params_static={
                    'alpha': 1e-14,
                    'solver': 'lbfgs',
                },
            )

            ## Initialize with TUNING 'alpha', 'solver', and 'positive'. 'tol' is fixed.
            autoregressor = Auto_Regression(
                X,
                y,
                Model=sklearn.linear_model.Ridge,
                params_dynamic={
                    'alpha': [1e-1, 1e3],
                    'solver': ['lbfgs', 'sag', 'saga'],
                    'positive': [True, False],
                },
                params_static={
                    'tol': 0.0001,
                },
            )
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        Model: Callable,
        params_dynamic: Dict = {
            'alpha': [1e-1, 1e5],
            'fit_intercept': [True, False],
        },
        params_static: Dict = {
            'max_iter': 1000,
            'tol': 0.0001,
            'solver': 'auto',
        },
        params_log: Dict = {
            'alpha': True,
        },
        params_linked: Dict = {},
        n_startup: int = 15,
        kwargs_convergence: Dict = {
            'n_patience': 50,
            'tol_frac': 0.05,
            'max_trials': 150,
            'max_duration': 60*10,
        }, 
        n_repeats: int = 1,
        fn_reduce_repeats: Callable = np.median,
        n_jobs_optuna: int = 1,
        penalty_testTrainRatio: float = 1.0,
        sample_weight: Optional[List[float]] = None,
        cv: Optional[sklearn.model_selection._split.BaseCrossValidator] = None,
        groups: Optional[np.ndarray] = None,
        test_size: float = 0.3,
        loss_function: Optional[Callable] = None,
        verbose: bool = True,
        optuna_storage_url: Optional[str] = None,
        optuna_storage_name: Optional[str] = None,
        optuna_engine_kwargs: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
    ) -> None:
        """
        Initializes Auto_Regression with the given parameters and data.
        """
        ## Prepare sample weights
        self.sample_weight = sample_weight

        ## Prepare the loss function
        if loss_function is None:
            self.fn_loss = LossFunction_MSE_CV(
                penalty_testTrainRatio=penalty_testTrainRatio,
            )
        else:
            self.fn_loss = loss_function

        ## Prepare the cross-validation
        self.cv = sklearn.model_selection.ShuffleSplit(
            n_splits=1,
            test_size=test_size,
        ) if cv is None else cv

        ## Prepare dynamic parameters for optuna
        params = _infer_params_types(params_dynamic, params_log)

        ## Prepare the model class
        self.model_class = functools.partial(
            Model,
            **params_static,
        )

        ## Initialize the Autotuner superclass
        super().__init__(
            model_class=self.model_class,
            X=X,
            y=y,
            params=params,
            params_linked=params_linked,
            n_startup=n_startup,
            kwargs_convergence=kwargs_convergence,
            n_repeats=n_repeats,
            fn_reduce_repeats=fn_reduce_repeats,
            n_jobs_optuna=n_jobs_optuna,
            cv=self.cv,
            groups=groups,
            fn_loss=self.fn_loss,
            catch_convergence_warnings=True,
            verbose=verbose,
            optuna_storage_url=optuna_storage_url,
            optuna_storage_name=optuna_storage_name,
            optuna_engine_kwargs=optuna_engine_kwargs,
            wandb_project=wandb_project,
        )


def _infer_params_types(
    params, 
    params_log={}, 
    log_default=True,
):
    """
    Infers the type of each parameter in params.
    RH 2023

    Args:
        params_dynamic (Dict):
            Dictionary of model class initialization parameters. 
            For each item in the dictionary if item is: \n
                * ``list``: The parameter is tuned. If the values are numbers,
                  then the list wil be the bounds [low, high] to search over. If
                  the values are strings, then the list will be the categorical
                  values to search over. Note that the type of the parameter
                  (int, float, str) will be used to determine the type of the
                  optuna suggest.
                * **not** a ``list``: The parameter is fixed to the given value. \n
        params_log (Dict):
            Dictionary of parameters that vary logarithmically: \n
                * If True: parameter is set to logaritmically varying. \n
                * If False: parameter is set to linearly varying. \n
                * If not specified: parameter is set to logaritmically varying.
                  \n
    
    Returns:
        (Dict)
            params_dynamic:
                Dictionary of parameters to be tuned. \n
                Formatted as: \n
                .. code-block:: python
                    {
                        'param1': {
                            'type': 'categorical' or 'real' or 'int',
                            'kwargs': {
                                'choices': [val1, val2, ...] if categorical,
                                'low': val1,
                                'high': val2,
                                'step': val3,
                                'log': True or False,
                            }
                        },
                        'param2': {
                            ...
                        },
                        ...
                    }                    
    """
    params_dynamic = {}
    for key, val in params.items():
        ## Dynamic parameters
        if isinstance(val, list):
            ## Categorical parameters
            if isinstance(val[0], str):
                params_dynamic[key] = {
                    'type': 'categorical', 
                    'kwargs': {}
                }
                params_dynamic[key]['kwargs']['choices'] = val
            ## Numerical parameters
            elif isinstance(val[0], (int, float)):
                assert len(val) >= 2, f'Parameter "{key}" must have at least 2 values.'
                assert all([isinstance(v, type(val[0])) for v in val]), f'Parameter "{key}" must have all values of the same type. Found types {[type(v) for v in val]}.'
                kwargs = ['low', 'high', 'step', 'log']
                params_dynamic[key] = {
                    'type': 'real' if isinstance(val[0], float) else 'int',
                    'kwargs': {kwargs[ii]: val for ii, val in enumerate(val)},
                }
                params_dynamic[key]['kwargs']['log'] = params_log.get(key, log_default)
            else:
                raise ValueError(f'Parameter "{key}" has type {type(val[0])}, which is not supported.')
        ## Static parameters
        else:
            raise ValueError(f'Parameter "{key}" must be a list. Found type {type(val)}.')
    return params_dynamic