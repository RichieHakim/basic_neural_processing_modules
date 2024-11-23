from typing import Dict, Type, Any, Union, Optional, Callable, Tuple, List
import time
import logging
import os

import numpy as np
import torch
import optuna
from tqdm.auto import tqdm

class Convergence_checker:
    """
    Checks for convergence during an optimization process. Uses Ordinary Least
    Squares (OLS) to fit a line to the last 'window_convergence' number of
    iterations.\n
    RH 2022
    
    Args:
        tol_convergence (float): 
            Tolerance for convergence.\n
            Corresponds to the slope of the line that is fit.\n
            If fractional==True, then tol_convergence is the fractional
            change in loss over the window_convergence
        fractional (bool):
            If True, then tol_convergence is the fractional change in loss
            over the window_convergence. ie: delta(lossWin) / mean(lossWin)
        window_convergence (int):
            Number of iterations to use for fitting the line.
        mode (str):
            Where deltaLoss = loss[current] - loss[window convergence steps ago]\n
            For typical loss curves, deltaLoss should be negative. So common
            modes are: 'greater' with tol_convergence = -1e-x, and 'less'
            with tol_convergence = 1e-x.
            Mode for how criterion is defined:\n
                * 'less': converged = deltaLoss < tol_convergence (default)
                * 'abs_less': converged = abs(deltaLoss) < tol_convergence
                * 'greater': converged = deltaLoss > tol_convergence
                * 'abs_greater': converged = abs(deltaLoss) > tol_convergence
                * 'between': converged = tol_convergence[0] < deltaLoss < tol_convergence[1]\n
                    (tol_convergence must be a list or tuple, if mode='between')
        max_iter (int):
            Maximum number of iterations to run for.\n
            If None, then no maximum.
        max_time (float):
            Maximum time to run for (in seconds).\n
            If None, then no maximum.
        nan_policy (str):
            Policy for handling NaNs in the loss history.\n
                * 'omit': (default) Ignore NaNs in the loss history.
                * 'halt': Return converged=True if NaNs are found in the loss history.
                * 'allow': Allow NaNs in the loss history.
                * 'raise': Raise an error if NaNs are found in the loss history.
        explode_tolerance (float):
            Tolerance for exploding loss. If the ratio in loss between the
            current and previous iteration (current / previous) is greater than
            this value, then the optimization is considered to have exploded. If
            None, then no explosion check is done.
        explode_patience (int):
            Number of iterations to wait before checking for explosion. If None, then
            no patience is used.

    """
    def __init__(
        self,
        tol_convergence=-1e-2,
        fractional=False,
        window_convergence=100,
        mode='greater',
        max_iter=None,
        max_time=None,
        nan_policy='omit',
        explode_tolerance=None,
        explode_patience=10,
    ):
        """
        Initialize the convergence checker.
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
        self.nan_policy = nan_policy
        self.explode_tolerance = explode_tolerance
        self.explode_patience = explode_patience

        self.iter = -1

    def OLS(self, y, X=None):
        """
        Ordinary least squares.
        Fits a line and bias term (stored in self.line_regressor) to y input.
        """
        X = self.line_regressor if X is None else X
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
        ## Initialize timer t0
        if self.iter == 0:
            self.t0 = time.time()
        
        ## Increment iter
        self.iter += 1

        ## Parse args and prepare loss_history
        if loss_history is None:
            if not hasattr(self, 'loss_history'):
                assert loss_single is not None, "loss_history and loss_single are both None"
                self.loss_history = []
            self.loss_history.append(loss_single)
            loss_history = self.loss_history

        ## Check for explosion
        if (self.explode_tolerance is not None) and (self.iter > self.explode_patience):
            if abs(loss_history[-1] / loss_history[-2]) > self.explode_tolerance:
                return torch.nan, torch.nan, True

        ## Prepare loss_window
        loss_window = torch.as_tensor(loss_history[-self.window_convergence:], device='cpu', dtype=torch.float32)

        ## Handle nan_policy, fit line, and make loss_smooth
        if self.nan_policy=='omit':
            loss_window = loss_window[~torch.isnan(loss_window)]
            loss_smooth = torch.nanmean(loss_window)

            ## Wait until window_convergence number of iterations
            if len(loss_history) < self.window_convergence:
                return torch.nan, loss_smooth, False

            theta, y_rec, bias = self.OLS(y=loss_window, X=self.line_regressor[~torch.isnan(loss_window)])
        else:
            if self.nan_policy=='halt':
                if torch.isnan(loss_window).any():
                    return torch.nan, torch.nan, True
            elif self.nan_policy=='raise':
                if torch.isnan(loss_window).any():
                    raise ValueError("NaNs found in loss history")
            elif self.nan_policy=='allow':
                pass

            ## Wait until window_convergence number of iterations
            if len(loss_history) < self.window_convergence:
                return torch.nan, torch.nan, False
            
            loss_smooth = torch.mean(loss_window)
            theta, y_rec, bias = self.OLS(y=loss_window)

        ## Check for convergence
        delta_window_convergence = (y_rec[-1] - y_rec[0]) if not self.fractional else (y_rec[-1] - y_rec[0]) / ((y_rec[-1] + y_rec[0])/2)
        converged = self.fn_criterion(delta_window_convergence)

        ## Check for max_iter and max_time
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
        value_stop (Optional[float]):
            Value at which to stop the optimization. If the best value is equal
            to or less than this value, the optimization will stop.
            (Default is *None*)
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
        value_stop: Optional[float] = None,
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
        self.value_stop = value_stop
        self.num_trial = 0
        self.verbose = verbose

        self.reason_converged = 'not converged'
        
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
        
        if trial.value < self.best:
            self.best = trial.value
        self.bests.append(self.best)
            
        bests_recent = np.unique(self.bests[-self.n_patience:])
        if self.best == 0:
            print(f'Stopping. Best value is 0.') if self.verbose else None
            self.reason_converged = 'best value is 0'
            study.stop()
        elif self.num_trial > self.n_patience and ((np.abs(bests_recent.max() - bests_recent.min()) / np.abs(self.best)) < self.tol_frac):
            print(f'Stopping. Convergence reached. Best value ({self.best*10000}) over last ({self.n_patience}) trials fractionally changed less than ({self.tol_frac})') if self.verbose else None
            self.reason_converged = 'fractional change in best value'
            study.stop()
        elif self.num_trial >= self.max_trials:
            print(f'Stopping. Trial number limit reached. num_trial={self.num_trial}, max_trials={self.max_trials}.') if self.verbose else None
            self.reason_converged = 'max trials reached'
            study.stop()
        elif duration > self.max_duration:
            print(f'Stopping. Duration limit reached. study.duration={duration}, max_duration={self.max_duration}.') if self.verbose else None
            self.reason_converged = 'max duration reached'
            study.stop()

        if self.value_stop is not None:
            if self.best <= self.value_stop:
                print(f'Stopping. Best value ({self.best}) is less than or equal to value_stop ({self.value_stop}).') if self.verbose else None
                self.reason_converged = 'value stop'
                study.stop()
            
        if self.verbose:
            print(f'Trial num: {self.num_trial}. Duration: {duration:.3f}s. Best value: {self.best:3e}. Current value:{trial.value:3e}') if self.verbose else None
        self.num_trial += 1


class OptunaProgressBar:
    """
    A customizable progress bar for Optuna's study.optimize().

    Args:
        n_trials (int, optional): 
            The number of trials. Required if timeout is not set.
        timeout (float, optional): 
            The maximum time to run in seconds. Required if n_trials is not set.
        tqdm_kwargs (dict, optional): 
            Additional keyword arguments to pass to tqdm.
    """

    def __init__(self, n_trials=None, timeout=None, **tqdm_kwargs):
        from optuna import logging as optuna_logging

        self._n_trials = n_trials
        self._timeout = timeout
        self._tqdm_kwargs = tqdm_kwargs

        # Read TQDM environment variables
        self._load_env_variables()

        # Initialize progress bar
        self._progress_bar = None
        self._last_elapsed_seconds = 0.0

        # Setup logging handler to redirect Optuna logs to tqdm
        self._tqdm_handler = _TqdmLoggingHandler()
        self._tqdm_handler.setLevel(logging.INFO)
        self._tqdm_handler.setFormatter(optuna_logging.create_default_formatter())
        optuna_logging.disable_default_handler()
        optuna_logging._get_library_root_logger().addHandler(self._tqdm_handler)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # Initialize progress bar on first call
        if self._progress_bar is None:
            if self._n_trials is not None:
                self._progress_bar = tqdm(total=self._n_trials, **self._tqdm_kwargs)
            elif self._timeout is not None:
                total = tqdm.format_interval(self._timeout)
                fmt = "{desc} {percentage:3.0f}%|{bar}| {elapsed}/" + total
                self._progress_bar = tqdm(total=self._timeout, bar_format=fmt, **self._tqdm_kwargs)
            else:
                raise ValueError("Either n_trials or timeout must be set.")

        # Update progress bar
        if self._n_trials is not None:
            self._progress_bar.update(1)
        elif self._timeout is not None:
            elapsed = study._stop_watch.elapsed_time()
            time_diff = elapsed - self._last_elapsed_seconds
            self._progress_bar.update(time_diff)
            self._last_elapsed_seconds = elapsed

        # Update description with best trial information
        if not study._is_multi_objective():
            try:
                best_value = study.best_value
                self._progress_bar.set_description(f"Best value: {best_value:.6g}")
            except ValueError:
                pass  # No trials completed yet

    def close(self):
        from optuna import logging as optuna_logging
        if self._progress_bar is not None:
            self._progress_bar.close()
        # Restore Optuna's default logging handler
        optuna_logging._get_library_root_logger().removeHandler(self._tqdm_handler)
        optuna_logging.enable_default_handler()

    def _load_env_variables(self):
        # Load TQDM environment variables
        env_vars = {
            'disable': os.environ.get('TQDM_DISABLE', '').lower() == 'true',
            'mininterval': float(os.environ.get('TQDM_MININTERVAL', 0.1)),
            'maxinterval': float(os.environ.get('TQDM_MAXINTERVAL', 10.0)),
            'miniters': int(os.environ.get('TQDM_MINITERS', 1)),
            'smoothing': float(os.environ.get('TQDM_SMOOTHING', 0.3)),
        }
        # Update tqdm_kwargs with environment variables if not already set
        for key, value in env_vars.items():
            if key not in self._tqdm_kwargs:
                self._tqdm_kwargs[key] = value
class _TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler to redirect Optuna logs to tqdm.write()."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)