from typing import Dict, Type, Any, Union, Optional, Callable, Tuple, List
import time
import warnings

import numpy as np
import torch

class Convergence_checker:
    """
    Checks for convergence during an optimization process. Uses Ordinary Least Squares (OLS) to 
     fit a line to the last 'window_convergence' number of iterations.

    RH 2022
    """
    def __init__(
        self,
        tol_convergence=-1e-2,
        fractional=False,
        window_convergence=100,
        mode='greater',
        max_iter=None,
        max_time=None,
    ):
        """
        Initialize the convergence checker.
        
        Args:
            tol_convergence (float): 
                Tolerance for convergence.
                Corresponds to the slope of the line that is fit.
                If fractional==True, then tol_convergence is the fractional
                 change in loss over the window_convergence
            fractional (bool):
                If True, then tol_convergence is the fractional change in loss
                 over the window_convergence. ie: delta(lossWin) / mean(lossWin)
            window_convergence (int):
                Number of iterations to use for fitting the line.
            mode (str):
                Where deltaLoss = loss[current] - loss[window convergence steps ago]
                For typical loss curves, deltaLoss should be negative. So common
                 modes are: 'greater' with tol_convergence = -1e-x, and 'less' with
                 tol_convergence = 1e-x.
                Mode for how criterion is defined.
                'less': converged = deltaLoss < tol_convergence (default)
                'abs_less': converged = abs(deltaLoss) < tol_convergence
                'greater': converged = deltaLoss > tol_convergence
                'abs_greater': converged = abs(deltaLoss) > tol_convergence
                'between': converged = tol_convergence[0] < deltaLoss < tol_convergence[1]
                    (tol_convergence must be a list or tuple, if mode='between')
            max_iter (int):
                Maximum number of iterations to run for.
                If None, then no maximum.
            max_time (float):
                Maximum time to run for (in seconds).
                If None, then no maximum.
        """
        self.window_convergence = window_convergence
        self.tol_convergence = tol_convergence
        self.fractional = fractional

        self.line_regressor = torch.cat((torch.linspace(0,1,window_convergence)[:,None], torch.ones((window_convergence,1))), dim=1)

        if mode=='less':          self.fn_criterion = (lambda diff: diff < self.tol_convergence)
        elif mode=='abs_less':    self.fn_criterion = (lambda diff: abs(diff) < self.tol_convergence)
        elif mode=='greater':     self.fn_criterion = (lambda diff: diff > self.tol_convergence)
        elif mode=='abs_greater': self.fn_criterion = (lambda diff: abs(diff) > self.tol_convergence)
        elif mode=='between':     self.fn_criterion = (lambda diff: self.tol_convergence[0] < diff < self.tol_convergence[1])
        assert self.fn_criterion is not None, f"mode '{mode}' not recognized"

        self.max_iter = max_iter
        self.max_time = max_time

        self.iter = -1

    def OLS(self, y):
        """
        Ordinary least squares.
        Fits a line and bias term (stored in self.line_regressor)
         to y input.
        """
        X = self.line_regressor
        theta = torch.inverse(X.T @ X) @ X.T @ y
        y_rec = X @ theta
        bias = theta[-1]
        theta = theta[:-1]

        return theta, y_rec, bias

    def __call__(
        self,
        loss_history=None,
        loss_single=None,
    ):
        """
        Forward pass of the convergence checker.
        Checks if the last 'window_convergence' number of iterations are
         within 'tol_convergence' of the line fit.

        Args:
            loss_history (list or array):
                List of loss values for entire optimization process.
                If None, then internally tracked loss_history is used.
            loss_single (float):
                Single loss value for current iteration.

        Returns:
            delta_window_convergence (float):
                Difference between first and last element of the fit line
                 over the range of 'window_convergence'.
                 diff_window_convergence = (y_rec[-1] - y_rec[0])
            loss_smooth (float):
                The mean loss over 'window_convergence'.
            converged (bool):
                True if the 'diff_window_convergence' is less than
                 'tol_convergence'.
        """
        if self.iter == 0:
            self.t0 = time.time()
        self.iter += 1

        if loss_history is None:
            if not hasattr(self, 'loss_history'):
                assert loss_single is not None, "loss_history and loss_single are both None"
                self.loss_history = []
            self.loss_history.append(loss_single)
            loss_history = self.loss_history

        if len(loss_history) < self.window_convergence:
            return torch.nan, torch.nan, False
        loss_window = torch.as_tensor(loss_history[-self.window_convergence:], device='cpu', dtype=torch.float32)
        loss_smooth = loss_window.mean()

        theta, y_rec, bias = self.OLS(y=loss_window)

        delta_window_convergence = (y_rec[-1] - y_rec[0]) if not self.fractional else (y_rec[-1] - y_rec[0]) / ((y_rec[-1] + y_rec[0])/2)
        converged = self.fn_criterion(delta_window_convergence)

        if self.max_iter is not None:
            converged = converged or (len(loss_history) >= self.max_iter)
        if self.max_time is not None:
            converged = converged or (time.time() - self.t0 > self.max_time)
        
        return delta_window_convergence.item(), loss_smooth.item(), converged
    

class Convergence_checker_optuna:
    """
    Checks if the optuna optimization has converged.
    RH 2023

    Args:
        n_patience (int): 
            Number of trials to look back to check for convergence. 
            Also the minimum number of trials that must be completed 
            before starting to check for convergence. 
            (Default is *10*)
        tol_frac (float): 
            Fractional tolerance for convergence. 
            The best output value must change by less than this 
            fractional amount to be considered converged. 
            (Default is *0.05*)
        max_trials (int): 
            Maximum number of trials to run before stopping. 
            (Default is *350*)
        max_duration (float): 
            Maximum number of seconds to run before stopping. 
            (Default is *600*)
        verbose (bool): 
            If ``True``, print messages. 
            (Default is ``True``)

    Attributes:
        bests (List[float]):
            List to hold the best values obtained in the trials.
        best (float):
            Best value obtained among the trials. Initialized with infinity.

    Example:
        .. highlight:: python
        .. code-block:: python

            # Create a ConvergenceChecker instance
            convergence_checker = ConvergenceChecker(
                n_patience=15, 
                tol_frac=0.01, 
                max_trials=500, 
                max_duration=60*20, 
                verbose=True
            )
            
            # Assume we have a study and trial objects from optuna
            # Use the check method in the callback
            study.optimize(objective, n_trials=100, callbacks=[convergence_checker.check])    
    """
    def __init__(
        self, 
        n_patience: int = 10, 
        tol_frac: float = 0.05, 
        max_trials: int = 350, 
        max_duration: float = 60*10, 
        verbose: bool = True,
    ):
        """
        Initializes the ConvergenceChecker with the given parameters.
        """
        self.bests = []
        self.best = np.inf
        self.n_patience = n_patience
        self.tol_frac = tol_frac
        self.max_trials = max_trials
        self.max_duration = max_duration
        self.num_trial = 0
        self.verbose = verbose
        
    def check(
        self, 
        study: object, 
        trial: object,
    ):
        """
        Checks if the optuna optimization has converged. This function should be
        used as the callback function for the optuna study.

        Args:
            study (optuna.study.Study): 
                Optuna study object.
            trial (optuna.trial.FrozenTrial): 
                Optuna trial object.
        """
        dur_first, dur_last = study.trials[0].datetime_complete, trial.datetime_complete
        if (dur_first is not None) and (dur_last is not None):
            duration = (dur_last - dur_first).total_seconds()
        else:
            duration = 0
        
        if trial.value is not None:
            if trial.value < self.best:
                self.best = trial.value
        self.bests.append(self.best)
            
        bests_recent = np.unique(self.bests[-self.n_patience:])
        if self.num_trial > self.n_patience and ((np.abs(bests_recent.max() - bests_recent.min())/np.abs(self.best)) < self.tol_frac):
            print(f'Stopping. Convergence reached. Best value ({self.best*10000}) over last ({self.n_patience}) trials fractionally changed less than ({self.tol_frac})') if self.verbose else None
            study.stop()
        if self.num_trial >= self.max_trials:
            print(f'Stopping. Trial number limit reached. num_trial={self.num_trial}, max_trials={self.max_trials}.') if self.verbose else None
            study.stop()
        if duration > self.max_duration:
            print(f'Stopping. Duration limit reached. study.duration={duration}, max_duration={self.max_duration}.') if self.verbose else None
            study.stop()
            
        if self.verbose:
            print(f'Trial num: {self.num_trial}. Duration: {duration:.3f}s. Best value: {self.best:3e}. Current value:{trial.value:3e}') if self.verbose else None
        self.num_trial += 1
